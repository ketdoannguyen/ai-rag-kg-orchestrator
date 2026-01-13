from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple

from loguru import logger
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from llm_gateway_client import AIGatewayClient
from llm_gateway_client.models import GatewayChatTurn, GatewayInvokeRequest

from arango_client import (
    KnowledgeNode,
    ParagraphNode,
    StructureNode,
    TableNode,
    ImageNode,
)

from workflows.shared.file_utils import extract_json_from_response

from services.prompts.enrichment import (
    build_detail_merge_prompt,
    build_structure_enrich_prompt,
)

from .helper import (
    collect_detail_items,
    count_paragraphs,
    create_refined_paragraphs_from_segments,
    get_title,
    aggregate_children_metadata,
)

from .models import (
    EnrichKnowledgeTreeInput,
    EnrichKnowledgeTreeOutput,
    EnrichStatistics,
    ChangesSummary,
)


@dataclass
class _EnrichState:
    """Mutable state shared across the traversal."""

    node_summaries: Dict[str, str] = field(default_factory=dict)
    stats: EnrichStatistics = field(default_factory=EnrichStatistics)
    changes: ChangesSummary = field(default_factory=ChangesSummary)


class KnowledgeTreeEnricher:
    """Enriches a KnowledgeNode tree by refining detail nodes and summarising structure nodes."""

    def __init__(self, llm_client: AIGatewayClient) -> None:
        self.llm_client = llm_client
    
    # ---------------------------------------------------------------------
    # Low-level LLM invocation with retry
    # ---------------------------------------------------------------------
    @retry(
        retry=retry_if_exception_type(Exception),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=8),
        reraise=True,
    )
    async def _invoke_with_retry(self, request: GatewayInvokeRequest):
        """Invoke LLM with simple exponential backoff retry."""
        return await self.llm_client.invoke(request)

    async def enrich(self, workflow_input: EnrichKnowledgeTreeInput) -> EnrichKnowledgeTreeOutput:
        """Run the enrich_knowledge_tree workflow."""
        # Expect a Pydantic KnowledgeNode/StructureNode as input
        root = workflow_input.knowledge_tree

        # Basic validation
        self._validate_root(root)

        state = _EnrichState()

        # Pre-compute simple statistics: paragraphs_before
        state.stats.paragraphs_before = count_paragraphs(root)

        # Global semaphore to limit concurrent section processing / LLM calls
        semaphore = asyncio.Semaphore(
            workflow_input.refine_config.max_concurrent_sections
        )

        # Bottom-up traversal: process all Structure nodes; children first, parent after.
        await self._process_structure_node(
            node=root,
            breadcrumb=[get_title(root)],
            state=state,
            workflow_input=workflow_input,
            semaphore=semaphore,
        )

        # Final statistics
        state.stats.paragraphs_after = count_paragraphs(root)

        # Root summary preview
        root_desc = None
        if isinstance(root, StructureNode):
            root_desc = root.attributes.description
        if root_desc:
            state.changes.root_summary_preview = root_desc[:300]

        return EnrichKnowledgeTreeOutput(
            enriched_tree=root,
            statistics=state.stats,
            changes_summary=state.changes,
        )

    # ---------------------------------------------------------------------
    # Validation & helpers
    # ---------------------------------------------------------------------
    def _validate_root(self, root: KnowledgeNode) -> None:
        if root is None:
            raise ValueError("knowledge_tree cannot be None")
        if not isinstance(root, StructureNode):
            raise ValueError("Root knowledge_tree must be a StructureNode")
        if root.parent_id is not None:
            logger.warning(
                "Root StructureNode has non-null parent_id; continuing but this is unusual",
                extra={"root_id": str(root.id), "parent_id": str(root.parent_id)},
            )
        if not root.children:
            logger.info("Root StructureNode has no children; nothing to enrich")

    # ---------------------------------------------------------------------
    # Core traversal
    # ---------------------------------------------------------------------
    async def _process_structure_node(
        self,
        node: StructureNode,
        breadcrumb: List[str],
        state: _EnrichState,
        workflow_input: EnrichKnowledgeTreeInput,
        semaphore: asyncio.Semaphore,
    ) -> None:
        """Post-order processing for a single StructureNode."""
        if not isinstance(node, StructureNode):
            return

        # Acquire global section semaphore to limit concurrency
        async with semaphore:
            await self._process_structure_node_inner(
                node=node,
                breadcrumb=breadcrumb,
                state=state,
                workflow_input=workflow_input,
                semaphore=semaphore,
            )

    async def _process_structure_node_inner(
        self,
        node: StructureNode,
        breadcrumb: List[str],
        state: _EnrichState,
        workflow_input: EnrichKnowledgeTreeInput,
        semaphore: asyncio.Semaphore,
    ) -> None:
        """Actual logic for processing a StructureNode (expects semaphore held)."""
        if not isinstance(node, StructureNode):
            return

        # 1) Process all STRUCTURE children concurrently (bottom-up)
        structure_children: List[StructureNode] = [
            c for c in node.children if isinstance(c, StructureNode)
        ]

        tasks = []
        for child in structure_children:
            child_breadcrumb = breadcrumb + [get_title(child)]
            tasks.append(
                self._process_structure_node(
                    node=child,
                    breadcrumb=child_breadcrumb,
                    state=state,
                    workflow_input=workflow_input,
                    semaphore=semaphore,
                )
            )

        if tasks:
            await asyncio.gather(*tasks, return_exceptions=False)

        # 2) Refine DETAIL children of current node (paragraphs/tables)
        refined_children, section_stats, detail_section_summary = await self._refine_detail_children(
            node=node,
            breadcrumb=breadcrumb,
            state=state,
            workflow_input=workflow_input,
        )

        # Replace children with refined version (images + refined paragraphs/tables + structure children)
        node.children = refined_children

        # 2.5) Aggregate domain_metadata from all children (paragraphs + structure nodes)
        aggregated_metadata = aggregate_children_metadata(node.children)
        if aggregated_metadata:
            node.metadata.domain_metadata = aggregated_metadata

        # 3) Enrich current StructureNode with summary that takes into account:
        #    - summaries from STRUCTURE children (stored in state.node_summaries)
        # If this node has no STRUCTURE children and detail-level LLM already
        # produced a section_summary, reuse it to avoid an extra LLM call.
        if not structure_children and detail_section_summary:
            section_summary = detail_section_summary
        else:
            section_summary = await self._enrich_structure_node(
                node=node,
                breadcrumb=breadcrumb,
                state=state,
                workflow_input=workflow_input,
            )

        # Attach summary to node attributes
        if section_summary:
            node.attributes.description = section_summary
            state.node_summaries[str(node.id)] = section_summary
            state.stats.structures_enriched += 1

        # Track per-section changes for changes_summary
        state.changes.refined_sections[str(node.id)] = {
            "title": get_title(node),
            "original_detail_count": section_stats.get("original_detail_count", 0),
            "refined_detail_count": section_stats.get("refined_detail_count", 0),
        }


    # ---------------------------------------------------------------------
    # Detail refinement
    # ---------------------------------------------------------------------
    async def _refine_detail_children(
        self,
        node: StructureNode,
        breadcrumb: List[str],
        state: _EnrichState,
        workflow_input: EnrichKnowledgeTreeInput,
    ) -> Tuple[List[KnowledgeNode], Dict[str, int], str | None]:
        """Refine DETAIL children (Paragraph/Table) of a StructureNode.

        Returns:
            (new_children, section_stats, detail_section_summary)
        """
        detail_nodes: List[KnowledgeNode] = []
        structure_children: List[KnowledgeNode] = []
        image_children: List[KnowledgeNode] = []

        for child in node.children:
            if isinstance(child, (ParagraphNode, TableNode)):
                detail_nodes.append(child)
            elif isinstance(child, StructureNode):
                structure_children.append(child)
            elif isinstance(child, ImageNode):
                image_children.append(child)
            else:
                # Preserve unknown nodes as-is (defensive)
                structure_children.append(child)

        original_detail_count = len(detail_nodes)
        if original_detail_count <= 1:
            # Children order: details (unchanged) + images + structure children
            new_children: List[KnowledgeNode] = [
                *detail_nodes,
                *image_children,
                *structure_children,
            ]
            # _enrich_structure_node will read full paragraph text from children directly.
            return new_children, {
                "original_detail_count": original_detail_count,
                "refined_detail_count": original_detail_count,
            }, None

        # Case: >=2 detail nodes → call LLM to merge/refine
        logger.info(
            "Refining detail children of section",
            extra={
                "section": " > ".join(breadcrumb),
                "detail_count": len(detail_nodes),
                "structure_children": len(structure_children),
            },
        )

        # Build text items for LLM
        collected_items = collect_detail_items(detail_nodes)

        if not collected_items:
            # Nothing textual to refine
            new_children: List[KnowledgeNode] = [
                *detail_nodes,
                *image_children,
                *structure_children,
            ]
            return new_children, {
                "original_detail_count": original_detail_count,
                "refined_detail_count": original_detail_count,
            }, None

        detail_result = await self._invoke_detail_llm(
            items=collected_items,
            breadcrumb=breadcrumb,
            workflow_input=workflow_input,
            state=state,
        )

        segments = detail_result.get("segments", [])
        detail_section_summary = detail_result.get("section_summary")

        refined_paragraphs = create_refined_paragraphs_from_segments(
            segments=segments,
            items=collected_items,
            parent=node,
            domain_schema=workflow_input.domain_schema,
        )

        refined_detail_count = len(refined_paragraphs)

        # Rebuild children: refined paragraphs + images + structure children
        new_children: List[KnowledgeNode] = [
            *refined_paragraphs,
            *image_children,
            *structure_children,
        ]

        return new_children, {
            "original_detail_count": original_detail_count,
            "refined_detail_count": refined_detail_count,
        }, detail_section_summary

    async def _invoke_detail_llm(
        self,
        items: List[Dict[str, Any]],
        breadcrumb: List[str],
        workflow_input: EnrichKnowledgeTreeInput,
        state: _EnrichState,
    ) -> Dict[str, Any]:
        """Call LLM to merge/refine detail items.

        Returns dict with:
            - segments: list of segment dicts
            - section_summary: optional summary string for the whole section
        """
        prompt = build_detail_merge_prompt(
            items, breadcrumb, domain_schema=workflow_input.domain_schema
        )
        section = " > ".join(breadcrumb)
        logger.debug(
            "Detail LLM prompt",
            extra={
                "section": section,
                "item_count": len(items),
                "prompt": prompt,
            },
        )
        # Single combined prompt used as system message
        request = GatewayInvokeRequest(
            model=workflow_input.refine_config.llm_model,
            messages=[
                GatewayChatTurn(
                    role="system",
                    content=prompt,
                )
            ],
            provider_options={"temperature": workflow_input.refine_config.llm_temperature},
        )

        try:
            response = await self._invoke_with_retry(request)
            state.stats.detail_llm_calls += 1
        except Exception as e:  # noqa: BLE001
            logger.error(
                "Detail-level LLM call failed; keeping original detail nodes",
                extra={"error": str(e)},
            )
            return {"segments": [], "section_summary": None}

        try:
            content = response.message.content
            parsed = extract_json_from_response(content)
            data = json.loads(parsed)
            segments = data.get("segments", [])
            if not isinstance(segments, list):
                logger.error("Invalid segments format in detail LLM response")
                return {"segments": [], "section_summary": None}
            section_summary = (data.get("section_summary") or "").strip() or None
            logger.debug(
                "Detail LLM response",
                extra={
                    "section": section,
                    "raw_response": content,
                    "segments": segments,
                },
            )
            return {"segments": segments, "section_summary": section_summary}
        except Exception as e:  # noqa: BLE001
            logger.error(
                "Failed to parse detail LLM response; keeping original detail nodes",
                extra={"error": str(e)},
            )
            return {"segments": [], "section_summary": None}

    # ---------------------------------------------------------------------
    # Structure enrichment
    # ---------------------------------------------------------------------
    async def _enrich_structure_node(
        self,
        node: StructureNode,
        breadcrumb: List[str],
        state: _EnrichState,
        workflow_input: EnrichKnowledgeTreeInput,
    ) -> str | None:
        """Generate a full summary for a StructureNode, leveraging children summaries."""
        children_content: List[Dict[str, Any]] = []

        for child in node.children:
            if isinstance(child, ParagraphNode):
                text = (child.attributes.text or "").strip()
                if not text:
                    continue
                children_content.append(
                    {
                        "type": "paragraph",
                        "title": None,
                        "content": text,
                    }
                )
            elif isinstance(child, StructureNode):
                child_summary = state.node_summaries.get(str(child.id))
                if not child_summary:
                    continue
                children_content.append(
                    {
                        "type": "section",
                        "title": get_title(child),
                        "content": child_summary,
                    }
                )

        if not children_content:
            # No textual content to summarize
            return None

        prompt = build_structure_enrich_prompt(
            title=get_title(node),
            children_content=children_content,
            summary_style=workflow_input.enrich_config.summary_style,
            short=False,
        )
        section_label = " > ".join(breadcrumb)
        logger.debug(
            "Structure LLM prompt",
            extra={
                "section": section_label,
                "child_blocks": len(children_content),
                "prompt": prompt,
                "max_words": workflow_input.enrich_config.max_summary_length,
            },
        )

        return await self._invoke_structure_llm_for_summary(
            prompt=prompt,
            workflow_input=workflow_input,
            state=state,
            section_label=section_label,
        )

    async def _invoke_structure_llm_for_summary(
        self,
        prompt: str,
        workflow_input: EnrichKnowledgeTreeInput,
        state: _EnrichState,
        section_label: str | None = None,
    ) -> str | None:
        """Call LLM for structure-level enrichment, expecting a JSON with 'summary'."""
        request = GatewayInvokeRequest(
            model=workflow_input.enrich_config.llm_model,
            messages=[
                GatewayChatTurn(
                    role="system",
                    content=prompt,
                )
            ],
            provider_options={"temperature": workflow_input.enrich_config.llm_temperature},
        )

        try:
            response = await self._invoke_with_retry(request)
            state.stats.structure_llm_calls += 1
        except Exception as e:  # noqa: BLE001
            logger.error(
                "Structure-level LLM call failed; falling back to None summary",
                extra={"error": str(e)},
            )
            return None

        try:
            content = response.message.content
            parsed = extract_json_from_response(content)
            data = json.loads(parsed)
            summary = (data.get("summary") or "").strip()
            if not summary:
                return None
            logger.debug(
                "Structure LLM response",
                extra={
                    "section": section_label,
                    "raw_response": content,
                    "summary": summary,
                },
            )
            return summary
        except Exception as e:  # noqa: BLE001
            logger.error(
                "Failed to parse structure LLM response; returning None",
                extra={"error": str(e)},
            )
            return None


