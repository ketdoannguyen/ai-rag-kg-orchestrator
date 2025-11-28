"""Refiner for merging textbook paragraphs using LLM with MinIO support."""

import asyncio
import json
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple

from config.settings import settings
from loguru import logger
from workflows.models import Image, Paragraph, Section, Table

from llm_gateway_client import AIGatewayClient
from llm_gateway_client.models import GatewayChatTurn, GatewayInvokeRequest
from minio_client import StorageClient
from workflows.dependencies import get_minio_client
from services.prompts.refinement import build_content_merge_prompt, get_system_prompt

from .helpers import (
    create_refined_paragraphs,
    extract_json_from_response,
    load_json_file,
    write_file,
)
from .models import RefineJsonOutput


class TextbookRefiner:
    """Refines textbook content by merging paragraphs using LLM."""

    def __init__(
        self,
        llm_client: Optional[AIGatewayClient],
        model: Optional[str],
        bucket: str = settings.MINIO_BUCKET,
        storage: Optional[StorageClient] = None,
    ):
        """
        Initialize TextbookRefiner.

        Args:
            llm_client: LLM gateway client for API calls
            model: Model name to use for refinement
            bucket: MinIO bucket name for storage
            storage: Optional MinIO storage client (uses get_minio_client() if None)
        """
        self.llm_client = llm_client
        self.model = model
        self.bucket = bucket
        self.storage = storage or get_minio_client()
        self._date_prefix = datetime.utcnow().strftime("%d-%m-%Y")
        self._job_id = datetime.utcnow().strftime("%H-%M-%S")
        self._object_root = f"textbook/{self._date_prefix}/{self._job_id}"

    async def refine(self, input_path: str, output_path: str, llm_timeout: float = 60.0) -> RefineJsonOutput:
        """
        Refine textbook from input file and save to output file.

        Args:
            input_path: Path to input JSON file
            output_path: Path to output JSON file
            llm_timeout: Timeout in seconds for LLM calls

        Returns:
            RefineJsonOutput with refinement results
        """
        if not self.llm_client or not self.model:
            logger.warning(
                "LLM client or model not provided, skipping refinement",
                extra={"input_path": input_path}
            )
            return RefineJsonOutput(
                success=False,
                output_path=output_path,
                error="LLM client or model not provided"
            )

        try:
            # Load input file asynchronously
            data = await asyncio.to_thread(load_json_file, input_path)
            root = Section.model_validate(data)
        except FileNotFoundError as e:
            logger.error("Input file not found", extra={"input_path": input_path, "error": str(e)})
            return RefineJsonOutput(
                success=False,
                output_path=output_path,
                error=f"Input file not found: {e}"
            )
        except json.JSONDecodeError as e:
            logger.error("Invalid JSON in input file", extra={"input_path": input_path, "error": str(e)})
            return RefineJsonOutput(
                success=False,
                output_path=output_path,
                error=f"Invalid JSON: {e}"
            )
        except Exception as e:
            logger.error("Failed to load input file", extra={"input_path": input_path, "error": str(e)})
            return RefineJsonOutput(
                success=False,
                output_path=output_path,
                error=f"Failed to load input: {e}"
            )

        # Refine content
        stats = await self._refine_node(root, [root.title], llm_timeout)

        # Save output
        result = await self._save_json(root, output_path)
        result.buffers_processed = stats.get("sections_processed", 0)
        result.paragraphs_refined = stats.get("paragraphs_refined", 0)
        return result

    async def _save_json(self, root: Section, output_path: str) -> RefineJsonOutput:
        """
        Save refined JSON to local file and MinIO.

        Args:
            root: Root Section to save
            output_path: Local file path

        Returns:
            RefineJsonOutput with save results
        """
        try:
            # Save locally asynchronously
            json_content = root.model_dump_json(indent=2, exclude_none=True)
            await asyncio.to_thread(write_file, output_path, json_content)
            
            logger.info(
                "Saved refined JSON locally",
                extra={"output_path": output_path, "size_bytes": len(json_content)}
            )

            # Upload to MinIO
            minio_object = None
            if self.storage:
                try:
                    object_name = f"{self._object_root}/refined.json"
                    await self.storage.upload_file(
                        bucket_name=self.bucket,
                        object_name=object_name,
                        file_path=Path(output_path),
                        content_type="application/json"
                    )
                    minio_object = object_name
                    logger.info(
                        "Uploaded refined JSON to MinIO",
                        extra={"object_name": object_name, "bucket": self.bucket}
                    )
                except Exception as e:
                    logger.error(
                        "Failed to upload to MinIO",
                        extra={"object_name": object_name, "error": str(e)}
                    )

            return RefineJsonOutput(
                success=True,
                output_path=output_path,
                minio_object=minio_object
            )
        except Exception as e:
            logger.error("Failed to save JSON", extra={"output_path": output_path, "error": str(e)})
            return RefineJsonOutput(
                success=False,
                output_path=output_path,
                error=f"Failed to save: {e}"
            )

    def _collect_section_children(self, node: Section) -> Tuple[List[Paragraph], List[Tuple[int, Section | Image | Table]]]:
        """
        Collect all direct Paragraphs and track positions of non-Paragraph children.

        Args:
            node: Section node to collect from

        Returns:
            Tuple of (paragraphs, other_children_with_positions) where positions are original indices
        """
        paragraphs: List[Paragraph] = []
        other_children: List[Tuple[int, Section | Image | Table]] = []

        for idx, child in enumerate(node.children):
            if isinstance(child, Paragraph):
                paragraphs.append(child)
            else:
                other_children.append((idx, child))

        return paragraphs, other_children

    async def _refine_node(
        self, 
        node: Section, 
        breadcrumb: List[str], 
        llm_timeout: float
    ) -> dict:
        """
        Recursively refine a Section node by collecting all its Paragraphs and refining them together.

        Args:
            node: Section node to refine
            breadcrumb: Breadcrumb path for context
            llm_timeout: Timeout for LLM calls

        Returns:
            Dictionary with stats: {"sections_processed": int, "paragraphs_refined": int}
        """
        if not node.children:
            return {"sections_processed": 0, "paragraphs_refined": 0}

        # Collect all direct Paragraphs and track positions of other children
        paragraphs, other_children_with_pos = self._collect_section_children(node)

        # Refine all Paragraphs of this Section together
        refined_paragraphs = paragraphs
        paragraphs_were_refined = False
        if len(paragraphs) > 1:
            refined_paragraphs = await self._refine_paragraphs(paragraphs, breadcrumb, llm_timeout)
            paragraphs_were_refined = refined_paragraphs != paragraphs

        # Process all child Sections concurrently
        sections_to_refine: List[Tuple[Section, List[str]]] = []
        for _, child in other_children_with_pos:
            if isinstance(child, Section):
                child_breadcrumb = breadcrumb + [child.title]
                sections_to_refine.append((child, child_breadcrumb))

        # Refine all child Sections concurrently
        total_sections = len(sections_to_refine)
        total_refined = len(refined_paragraphs) if paragraphs_were_refined else 0

        if sections_to_refine:
            results = await asyncio.gather(*[
                self._refine_node(section, bc, llm_timeout)
                for section, bc in sections_to_refine
            ], return_exceptions=True)

            for result in results:
                if isinstance(result, Exception):
                    logger.error(
                        "Error refining child section",
                        extra={"error": str(result), "breadcrumb": breadcrumb}
                    )
                    continue
                total_refined += result.get("paragraphs_refined", 0)

        # Rebuild children: place all refined paragraphs first, then other children in original order
        # This is simpler and ensures all paragraphs are refined together
        new_children: List[Section | Paragraph | Image | Table] = []
        
        # Add all refined paragraphs first
        new_children.extend(refined_paragraphs)
        
        # Then add all non-paragraph children in their original order
        for _, child in other_children_with_pos:
            new_children.append(child)

        node.children = new_children

        return {
            "sections_processed": total_sections + 1,  # +1 for current section
            "paragraphs_refined": total_refined
        }

    async def _refine_paragraphs(
        self, 
        paragraphs: List[Paragraph], 
        breadcrumb: List[str],
        llm_timeout: float
    ) -> List[Paragraph]:
        """
        Refine all paragraphs of a Section together using LLM.

        Args:
            paragraphs: List of paragraphs to refine (all from same Section)
            breadcrumb: Context breadcrumb path
            llm_timeout: Timeout for LLM call

        Returns:
            List of refined/merged paragraphs
        """
        if not paragraphs:
            return []

        if len(paragraphs) == 1:
            return paragraphs

        prompt = build_content_merge_prompt(paragraphs, breadcrumb)
        context_str = " > ".join(breadcrumb)
        logger.info(
            "Refining all paragraphs of section",
            extra={
                "paragraph_count": len(paragraphs),
                "section": context_str
            }
        )

        try:
            request = GatewayInvokeRequest(
                model=self.model,
                messages=[
                    GatewayChatTurn(role="system", content=get_system_prompt()),
                    GatewayChatTurn(role="user", content=prompt),
                ],
                provider_options={"temperature": 0.0},
            )

            # Call LLM with timeout
            response = await asyncio.wait_for(
                self.llm_client.invoke(request),
                timeout=llm_timeout
            )
            content = response.message.content

            # Parse JSON from response
            parsed_content = extract_json_from_response(content)
            data = json.loads(parsed_content)
            segments = data.get("segments", [])

            # Create refined paragraphs
            refined_paragraphs = create_refined_paragraphs(segments, paragraphs)

            logger.info(
                "Section paragraphs refinement completed",
                extra={
                    "input_paragraphs": len(paragraphs),
                    "output_segments": len(refined_paragraphs),
                    "section": context_str
                }
            )
            return refined_paragraphs

        except asyncio.TimeoutError:
            logger.error(
                "LLM call timed out, returning original paragraphs",
                extra={"paragraph_count": len(paragraphs), "timeout": llm_timeout, "section": context_str}
            )
            return paragraphs
        except json.JSONDecodeError as e:
            logger.error(
                "Failed to parse LLM response as JSON",
                extra={"error": str(e), "section": context_str}
            )
            return paragraphs
        except Exception as e:
            logger.error(
                "Error refining section paragraphs",
                extra={"error": str(e), "paragraph_count": len(paragraphs), "section": context_str}
            )
            return paragraphs
