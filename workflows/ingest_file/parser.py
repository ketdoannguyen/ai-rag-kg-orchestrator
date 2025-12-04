"""Parser for converting raw textbook JSON to standardized models with MinIO support."""

import base64
import imghdr
import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from config.settings import settings
from loguru import logger
from postgres_client.models import (
    ImageNode,
    ParagraphNode,
    StructureNode,
)
from postgres_client.models.attributes_metadata import (
    ImageAttributes,
    ImageMetadata,
    ParagraphAttributes,
    ParagraphMetadata,
    StructureAttributes,
    StructureMetadata,
)
from postgres_client import set_parent_ids, set_positions

from workflows.dependencies import get_minio_client


class TextbookParser:
    """Parses raw textbook JSON into a structured Section hierarchy."""

    def __init__(
        self,
        input_path: str,
        output_path: str,
        page_offset: int = 5,
        extract_images: bool = True,
        bucket: str = settings.MINIO_BUCKET,
    ):
        self.input_path = input_path
        self.output_path = output_path
        self.page_offset = page_offset
        self.extract_images = extract_images
        self.bucket = bucket
        self.storage = get_minio_client()

        self.raw_data = self._load_raw_json()
        self._date_prefix = datetime.utcnow().strftime("%d-%m-%Y")
        self._job_id = datetime.utcnow().strftime("%H-%M-%S")
        self._object_root = f"textbook/{self._date_prefix}/{self._job_id}"

        # Root node is a StructureNode with level_down=1 (Textbook)
        # level_up will be recomputed later based on the maximum depth of the tree.
        self.root = StructureNode(
            node_type="structure",
            level_down=1,
            level_up=1,
            attributes=StructureAttributes(title="SGK Vật Lý 12 - Kết nối tri thức"),
            metadata=StructureMetadata(start_page=0),
        )

        # Stack used to maintain the current parent chain while walking pages/blocks
        self.parent_stack: List[StructureNode] = [self.root]

    def _load_raw_json(self) -> Dict[str, Any]:
        try:
            with open(self.input_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Error loading JSON: {e}")
            return {}

    def _clean_html(self, html: str) -> str:
        if not html:
            return ""
        text = re.sub(r"<li[^>]*>", "\n* ", html)
        text = re.sub(r"<[^>]+>", "", text)
        text = text.replace("&gt;", ">").replace("&lt;", "<").replace("&amp;", "&")
        return text.strip()

    def _get_bbox(self, block: Dict[str, Any]) -> List[int]:
        return block.get("bbox", [])

    async def _extract_and_upload_image(
        self, base64_data: str, image_id: str
    ) -> Optional[str]:
        """Extracts base64 image, saves locally, uploads to MinIO, returns MinIO path."""
        if not base64_data:
            return None

        try:
            # 1. Decode and Detect Format
            if "," in base64_data:
                header, base64_data = base64_data.split(",", 1)

            image_bytes = base64.b64decode(base64_data)
            image_format = imghdr.what(None, h=image_bytes) or "png"
            if image_format == "jpg":
                image_format = "jpeg"

            filename = f"{image_id}.{image_format}"

            # 2. Save Locally (Optional, but good for debugging/cache)
            local_dir = Path("data/process_raw_json/images") / self._job_id
            local_dir.mkdir(parents=True, exist_ok=True)
            local_path = local_dir / filename
            with open(local_path, "wb") as f:
                f.write(image_bytes)

            # 3. Upload to MinIO
            if self.storage:
                object_name = f"{self._object_root}/images/{filename}"
                await self.storage.upload_file(
                    bucket_name=self.bucket,
                    object_name=object_name,
                    file_path=local_path,
                    content_type=f"image/{image_format}",
                )
                return object_name

            return str(local_path)

        except Exception as e:
            logger.warning(f"Failed to extract/upload image {image_id}: {e}")
            return None

    def _classify_header(self, block: Dict, next_blocks: List[Dict]) -> Optional[Dict]:
        block_type = block.get("block_type")
        raw_html = block.get("html", "")
        clean_text = self._clean_html(raw_html)

        # Chapter -> Level 2
        if (
            block_type == "SectionHeader"
            and "<h1>" in raw_html
            and "CHƯƠNG" in clean_text.upper()
        ):
            match = re.match(
                r"^(CHƯƠNG\s+[IVXLCDM]+)\s*(.*)", clean_text.upper(), re.DOTALL
            )
            number = match.group(1).strip() if match else "Chương"
            title = match.group(2).strip() if match else clean_text

            blocks_consumed = 1
            if not title and next_blocks and "<h1>" in next_blocks[0].get("html", ""):
                title = self._clean_html(next_blocks[0]["html"])
                blocks_consumed = 2

            return {
                "type": "Chapter",
                "title": f"{number}: {title}".upper(),
                "level": 2,
                "blocks_consumed": blocks_consumed,
            }

        # Lesson (Bài) -> Level 3
        if block_type == "PageHeader" and clean_text.upper() == "BÀI":
            if len(next_blocks) >= 2:
                num_block = next_blocks[0]
                title_block = next_blocks[1]
                num_str = self._clean_html(num_block.get("html", ""))

                if (
                    num_block.get("block_type") == "PageHeader"
                    and num_str.isdigit()
                ):
                    title = self._clean_html(title_block.get("html", ""))
                    title = re.sub(r"\s+", " ", title)
                    return {
                        "type": "Lesson",
                        "title": f"Bài {num_str}: {title.upper()}",
                        "level": 3,
                        "blocks_consumed": 3,
                    }

        # Section
        if block_type == "SectionHeader":
            # Level 4: I, II, III...
            match_I = re.match(r"^\s*([IVXLCDM]+)\.\s+(.*)", clean_text, re.DOTALL)
            if match_I and "ĐỊNH LUẬT" not in match_I.group(1).upper():
                return {
                    "type": "Section",
                    "title": match_I.group(2).strip(),
                    "level": 4,
                    "blocks_consumed": 1,
                }

            # Level 5: 1, 2, 3...
            match_1 = re.match(r"^\s*(\d+)\.\s+(.*)", clean_text, re.DOTALL)
            if match_1:
                return {
                    "type": "Section",
                    "title": match_1.group(2).strip(),
                    "level": 5,
                    "blocks_consumed": 1,
                }

            # Level 6: a, b, c...
            match_a = re.match(r"^\s*([a-z])\)\s+(.*)", clean_text, re.DOTALL)
            if match_a:
                return {
                    "type": "Section",
                    "title": match_a.group(2).strip(),
                    "level": 6,
                    "blocks_consumed": 1,
                }

            # Level 7+: Others
            special = ["EM ĐÃ HỌC", "EM CÓ THỂ", "EM CÓ BIẾT", "NỘI DUNG"]
            if clean_text.upper() in special:
                return {
                    "type": "Section",
                    "title": clean_text,
                    "level": 7,
                    "blocks_consumed": 1,
                }

        return None

    async def parse(self):
        if not self.raw_data:
            return

        pages = self.raw_data.get("children", [])
        for i, page in enumerate(pages):
            page_idx = page.get("page", i)
            page_num = page_idx + self.page_offset

            blocks = page.get("children", [])
            j = 0
            while j < len(blocks):
                block = blocks[j]

                header_info = self._classify_header(block, blocks[j + 1 :])

                if header_info:
                    level = header_info["level"]

                    # Pop until we find a parent with smaller level_down
                    while (
                        self.parent_stack
                        and getattr(self.parent_stack[-1], "level_down", 1) >= level
                    ):
                        self.parent_stack.pop()

                    if not self.parent_stack:
                        self.parent_stack.append(self.root)

                    # Create a new StructureNode for this header
                    new_node = StructureNode(
                        node_type="structure",
                        level_down=level,
                        level_up=1,  # Will be recomputed later based on tree depth
                        attributes=StructureAttributes(
                            title=header_info["title"]
                        ),
                        metadata=StructureMetadata(start_page=page_num),
                    )

                    self.parent_stack[-1].children.append(new_node)
                    self.parent_stack.append(new_node)
                    j += header_info["blocks_consumed"]
                    continue

                # Content Blocks
                bbox = self._get_bbox(block)
                block_type = block.get("block_type")

                content_node = None
                if block_type in ["Text", "ListGroup", "Caption"]:
                    text = self._clean_html(block.get("html", ""))
                    if text:
                        content_node = ParagraphNode(
                            attributes=ParagraphAttributes(text=text),
                            metadata=ParagraphMetadata(
                                page=page_num,
                                bbox=bbox if bbox else [],
                            ),
                        )
                elif block_type in ["Picture", "Figure"]:
                    alt_match = re.search(
                        r'alt="([^"]*)"', block.get("html", "")
                    )
                    alt_text = alt_match.group(1) if alt_match else None

                    # Handle Image Extraction
                    image_url = None
                    if self.extract_images:
                        # Check for base64 in 'src' or 'base64_data' field if it existed (raw json structure varies)
                        # Assuming raw_json might have base64 in html src="data:image..."
                        src_match = re.search(
                            r'src="(data:image/[^;]+;base64,[^"]+)"',
                            block.get("html", ""),
                        )
                        if src_match:
                            base64_data = src_match.group(1)
                            image_id = f"img_{page_num}_{j}"
                            image_url = await self._extract_and_upload_image(
                                base64_data, image_id
                            )

                    # Only create ImageNode if we actually have a URL or file_path.
                    # Otherwise, skip the image block to avoid invalid nodes that fail validation.
                    if image_url:
                        content_node = ImageNode(
                            attributes=ImageAttributes(
                                alt_text=alt_text,
                                url=image_url,
                            ),
                            metadata=ImageMetadata(
                                page=page_num,
                                bbox=bbox if bbox else [],
                            ),
                        )

                if content_node:
                    self.parent_stack[-1].children.append(content_node)

                j += 1

    async def save_json(self):
        # Set parent_id and position for all nodes in the tree before saving
        set_parent_ids(self.root)
        set_positions(self.root)

        # Save locally
        Path(self.output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(self.output_path, "w", encoding="utf-8") as f:
            f.write(self.root.model_dump_json(indent=2, exclude_none=False))
        logger.info(f"Saved processed JSON locally to {self.output_path}")

        # Upload to MinIO
        if self.storage:
            try:
                object_name = f"{self._object_root}/processed.json"
                await self.storage.upload_file(
                    bucket_name=self.bucket,
                    object_name=object_name,
                    file_path=Path(self.output_path),
                    content_type="application/json",
                )
                logger.info(
                    f"Uploaded processed JSON to MinIO: {object_name}"
                )
            except Exception as e:
                logger.error(
                    f"Failed to upload processed JSON to MinIO: {e}"
                )
