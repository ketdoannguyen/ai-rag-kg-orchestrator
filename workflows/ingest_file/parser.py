"""Parser for converting raw textbook JSON to standardized models with MinIO support."""

import base64
import imghdr
import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any

from arango_client import (
    ImageAttributes,
    ImageMetadata,
    ImageNode,
    ParagraphAttributes,
    ParagraphMetadata,
    ParagraphNode,
    StructureAttributes,
    StructureMetadata,
    StructureNode,
    set_parent_ids,
    set_positions,
)
from loguru import logger

from config.settings import settings
from workflows.dependencies import get_minio_client


class TextbookParser:
    """Parses raw textbook JSON into a structured Section hierarchy.

    Uses a dynamic level mapping system that:
    1. Tracks h-tag to level mappings learned during parsing
    2. Uses section_hierarchy from Marker to understand relative positions
    3. Look-ahead to determine level of unclassified headers
    """

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

        # Root node is a StructureNode with level=1 (Textbook)
        self.root = StructureNode(
            node_type="structure",
            level=1,
            attributes=StructureAttributes(title="Lịch sử 12"),
            metadata=StructureMetadata(start_page=0),
        )

        # Stack used to maintain the current parent chain while walking pages/blocks
        self.parent_stack: list[StructureNode] = [self.root]

        # Dynamic h-tag to level mapping - learned during parsing
        # Key: h-tag number (1, 2, 4), Value: assigned level
        self.h_tag_level_map: dict[int, int] = {}

        # Pre-scan to build initial level mapping
        self._build_level_mapping()

    def _load_raw_json(self) -> dict[str, Any]:
        try:
            with open(self.input_path, encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Error loading JSON: {e}")
            return {}

    def _get_h_tag(self, html: str) -> int | None:
        """Extract h-tag number from HTML (e.g., <h1> -> 1, <h2> -> 2)."""
        match = re.search(r"<h(\d)>", html)
        return int(match.group(1)) if match else None

    def _build_level_mapping(self) -> None:
        """Pre-scan all SectionHeaders to build h-tag to level mapping.

        Strategy:
        1. Collect all SectionHeaders with their h-tags and patterns
        2. Assign levels based on known patterns (CHỦ ĐỀ, Bài, etc.)
        3. Use section_hierarchy to infer relative levels
        4. Build a consistent h-tag -> level mapping
        """
        if not self.raw_data:
            return

        # Collect all headers with metadata
        headers: list[dict[str, Any]] = []
        for page in self.raw_data.get("children", []):
            for block in page.get("children", []):
                if block.get("block_type") == "SectionHeader":
                    html = block.get("html", "")
                    h_tag = self._get_h_tag(html)
                    clean_text = self._clean_html(html)
                    hierarchy = block.get("section_hierarchy", {})
                    hierarchy_depth = len(hierarchy)

                    headers.append(
                        {
                            "h_tag": h_tag,
                            "text": clean_text,
                            "hierarchy_depth": hierarchy_depth,
                            "hierarchy": hierarchy,
                            "pattern": self._detect_pattern(clean_text),
                        }
                    )

        # Build level mapping from known patterns
        pattern_levels = {
            "chu_de": 2,  # CHỦ ĐỀ -> Level 2
            "chuong": 2,  # CHƯƠNG -> Level 2
            "bai": 3,  # Bài -> Level 3
            "section_num": 4,  # 1., 2. in h2 -> Level 4
            "section_roman": 4,  # I., II. -> Level 4
            "subsection": 5,  # 1., 2. in h4 -> Level 5
            "letter": 6,  # a), b) -> Level 6
        }

        # Assign h-tag levels based on observed patterns
        h_tag_patterns: dict[int, set[str]] = {}
        for header in headers:
            h_tag = header["h_tag"]
            pattern = header["pattern"]
            if h_tag and pattern:
                if h_tag not in h_tag_patterns:
                    h_tag_patterns[h_tag] = set()
                h_tag_patterns[h_tag].add(pattern)

        # Determine level for each h-tag
        for h_tag, patterns in h_tag_patterns.items():
            # Find the highest priority (lowest level number) pattern
            min_level = 99
            for pattern in patterns:
                if pattern in pattern_levels:
                    min_level = min(min_level, pattern_levels[pattern])
            if min_level < 99:
                self.h_tag_level_map[h_tag] = min_level

        # Fill gaps using hierarchy depth
        # If h1 is used for both Level 2 and Level 3, use hierarchy depth
        logger.info(f"Built h-tag level mapping: {self.h_tag_level_map}")

    def _detect_pattern(self, text: str) -> str | None:
        """Detect the structural pattern of a header text."""
        text_upper = text.upper()

        # CHỦ ĐỀ pattern
        if re.match(r"^CHỦ\s*(ĐỀ|ĐỂ)\s+\d+", text, re.IGNORECASE):
            return "chu_de"

        # CHƯƠNG pattern
        if "CHƯƠNG" in text_upper and re.match(r"^CHƯƠNG\s+[IVXLCDM]+", text_upper):
            return "chuong"

        # Bài pattern
        if re.match(r"^Bài\s+\d+", text, re.IGNORECASE):
            return "bai"

        # Roman numeral section (I., II., III.)
        if re.match(r"^\s*[IVXLCDM]+\.\s+", text):
            return "section_roman"

        # Numbered section (1., 2., 3.)
        if re.match(r"^\s*\d+\.\s+", text):
            return "section_num"

        # Letter section (a), b), c))
        if re.match(r"^\s*[a-z]\)\s+", text):
            return "letter"

        return None

    def _is_question_or_instruction(self, text: str) -> bool:
        """Check if text is a question or instruction that should be content, not structure.

        Questions and instructions should be treated as content (ParagraphNode),
        not as structure headers.

        Examples of questions/instructions to filter:
        - "? Trình bày khái niệm đa cực."
        - "Nêu bối cảnh lịch sử và quá trình hình thành Liên hợp quốc."
        - "Trình bày quá trình hình thành của Trật tự thế giới hai cực I-an-ta."
        """
        if not text:
            return False

        text = text.strip()

        # Starts with question mark
        if text.startswith("?"):
            return True

        # Common question/instruction verbs at the start (Vietnamese)
        instruction_patterns = [
            r"^Nêu\s+",  # Nêu ...
            r"^Trình bày\s+",  # Trình bày ...
            r"^Giải thích\s+",  # Giải thích ...
            r"^Phân tích\s+",  # Phân tích ...
            r"^So sánh\s+",  # So sánh ...
            r"^Liệt kê\s+",  # Liệt kê ...
            r"^Cho biết\s+",  # Cho biết ...
            r"^Hãy\s+",  # Hãy ...
            r"^Em hãy\s+",  # Em hãy ...
            r"^Tại sao\s+",  # Tại sao ...
            r"^Vì sao\s+",  # Vì sao ...
            r"^Làm thế nào\s+",  # Làm thế nào ...
            r"^Như thế nào\s+",  # Như thế nào ...
        ]

        for pattern in instruction_patterns:
            if re.match(pattern, text, re.IGNORECASE):
                return True

        # Ends with question mark
        if text.endswith("?"):
            return True

        return False

    def _clean_html(self, html: str) -> str:
        if not html:
            return ""
        text = re.sub(r"<li[^>]*>", "\n* ", html)
        text = re.sub(r"<[^>]+>", "", text)
        text = text.replace("&gt;", ">").replace("&lt;", "<").replace("&amp;", "&")
        return text.strip()

    def _get_bbox(self, block: dict[str, Any]) -> list[int]:
        return block.get("bbox", [])

    async def _extract_and_upload_image(self, base64_data: str, image_id: str) -> str | None:
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

    # Special standalone topic titles that should be Level 2
    TOPIC_TITLES = [
        "ASEAN: NHỮNG CHẶNG ĐƯỜNG LỊCH SỬ",
        "VIỆT NAM VÀ ASEAN",
    ]

    def _get_next_section_header(self, blocks: list[dict], start_idx: int) -> dict | None:
        """Look ahead to find the next SectionHeader block."""
        for i in range(start_idx, len(blocks)):
            if blocks[i].get("block_type") == "SectionHeader":
                return blocks[i]
        return None

    def _infer_level_from_context(
        self,
        block: dict,
        next_blocks: list[dict],
        h_tag: int | None,
    ) -> int | None:
        """Infer level for unclassified headers using context.

        Strategy:
        1. If h-tag has a known level mapping, use it
        2. Look at next SectionHeader - if it has a known level, this one is level-1
        3. Use section_hierarchy depth as fallback
        """
        # Check h-tag mapping first
        if h_tag and h_tag in self.h_tag_level_map:
            return self.h_tag_level_map[h_tag]

        # Look ahead to next SectionHeader
        next_header = self._get_next_section_header(next_blocks, 0)
        if next_header:
            next_html = next_header.get("html", "")
            next_text = self._clean_html(next_html)
            next_pattern = self._detect_pattern(next_text)

            # Map pattern to level
            pattern_to_level = {
                "chu_de": 2,
                "chuong": 2,
                "bai": 3,
                "section_num": 4,
                "section_roman": 4,
                "subsection": 5,
                "letter": 6,
            }

            if next_pattern in pattern_to_level:
                next_level = pattern_to_level[next_pattern]
                # This header is one level above the next
                inferred_level = next_level - 1
                if inferred_level >= 2:
                    logger.info(
                        f"Inferred level {inferred_level} from next header pattern '{next_pattern}'"
                    )
                    return inferred_level

        # Fallback: use section_hierarchy depth
        hierarchy = block.get("section_hierarchy", {})
        if hierarchy:
            depth = len(hierarchy)
            # Map depth to level: depth 1 -> level 2 or 3, depth 2 -> level 4, etc.
            inferred_level = depth + 1
            logger.debug(f"Inferred level {inferred_level} from hierarchy depth {depth}")
            return inferred_level

        return None

    def _classify_header(self, block: dict, next_blocks: list[dict]) -> dict | None:
        """Classify a header block and determine its level.

        Uses a combination of:
        1. Known patterns (CHỦ ĐỀ, CHƯƠNG, Bài, etc.)
        2. H-tag level mapping (learned during pre-scan)
        3. Look-ahead to next SectionHeader
        4. Section hierarchy from Marker
        """
        block_type = block.get("block_type")
        raw_html = block.get("html", "")
        clean_text = self._clean_html(raw_html)
        block_id = block.get("id", "unknown")
        h_tag = self._get_h_tag(raw_html)

        # DEBUG: Log every SectionHeader block for analysis
        if block_type == "SectionHeader":
            logger.debug(
                "Processing SectionHeader",
                block_id=block_id,
                h_tag=h_tag,
                clean_text=clean_text[:80] if clean_text else "",
            )

        # ============================================================
        # FILTER: Skip questions/instructions - treat as content, not structure
        # ============================================================
        if block_type == "SectionHeader" and self._is_question_or_instruction(clean_text):
            logger.info(f"Skipping question/instruction (will be content): {clean_text[:60]}")
            return None

        # ============================================================
        # LEVEL 2: CHỦ ĐỀ / CHƯƠNG / Special Topic Titles
        # ============================================================

        # Pattern 1: CHƯƠNG (Chapter) -> Level 2
        if block_type == "SectionHeader" and "<h1>" in raw_html and "CHƯƠNG" in clean_text.upper():
            match = re.match(r"^(CHƯƠNG\s+[IVXLCDM]+)\s*(.*)", clean_text.upper(), re.DOTALL)
            number = match.group(1).strip() if match else "Chương"
            title = match.group(2).strip() if match else clean_text

            blocks_consumed = 1
            if not title and next_blocks and "<h1>" in next_blocks[0].get("html", ""):
                title = self._clean_html(next_blocks[0]["html"])
                blocks_consumed = 2

            logger.info(f"Classified CHƯƠNG (Level 2): {number} - {title}")
            return {
                "type": "Chapter",
                "title": f"{number}: {title}".upper(),
                "code": number,
                "level": 2,
                "blocks_consumed": blocks_consumed,
            }

        # Pattern 2: CHỦ ĐỀ (Topic) -> Level 2
        if block_type == "SectionHeader" and "<h1>" in raw_html:
            # Match: "CHỦ ĐỀ 1 TITLE" or "CHỦ ĐỂ 1 TITLE" (typo in source)
            match = re.match(
                r"^(CHỦ\s*ĐỀ\s+\d+|CHỦ\s*ĐỂ\s+\d+)\s*(.*)", clean_text, re.IGNORECASE | re.DOTALL
            )
            if match:
                number = match.group(1).strip().upper().replace("ĐỂ", "ĐỀ")  # Fix typo
                title = match.group(2).strip()
                logger.info(f"Classified CHỦ ĐỀ (Level 2): {number} - {title}")
                return {
                    "type": "Topic",
                    "title": f"{number}: {title}".upper() if title else number.upper(),
                    "code": number,
                    "level": 2,
                    "blocks_consumed": 1,
                }

        # Pattern 3: Special standalone titles -> Level 2
        if block_type == "SectionHeader" and "<h1>" in raw_html:
            for topic in self.TOPIC_TITLES:
                if topic.upper() in clean_text.upper():
                    logger.info(f"Classified special topic (Level 2): {clean_text}")
                    return {
                        "type": "Topic",
                        "title": clean_text.upper(),
                        "code": None,
                        "level": 2,
                        "blocks_consumed": 1,
                    }

        # Pattern 4: Unclassified h1 after PageHeader - infer from next header
        if block_type == "SectionHeader" and "<h1>" in raw_html:
            # Check if next header is a "Bài" pattern
            next_header = self._get_next_section_header(next_blocks, 0)
            if next_header:
                next_text = self._clean_html(next_header.get("html", ""))
                if re.match(r"^Bài\s+\d+", next_text, re.IGNORECASE):
                    # This is a topic title before a Bài -> Level 2
                    logger.info(f"Inferred as topic (Level 2) from next Bài: {clean_text}")
                    return {
                        "type": "Topic",
                        "title": clean_text.upper(),
                        "code": None,
                        "level": 2,
                        "blocks_consumed": 1,
                    }

        # ============================================================
        # LEVEL 3: Bài (Lesson)
        # ============================================================

        # Pattern 1: Bài X in SectionHeader with <h1> -> Level 3
        if block_type == "SectionHeader" and "<h1>" in raw_html:
            match_bai = re.match(r"^(Bài\s+\d+)\s+(.*)", clean_text, re.IGNORECASE | re.DOTALL)
            if match_bai:
                code = match_bai.group(1).strip()
                title = match_bai.group(2).strip()
                logger.info(f"Classified Bài (Level 3): {code} - {title}")
                return {
                    "type": "Lesson",
                    "title": f"{code}: {title}".upper(),
                    "code": code,
                    "level": 3,
                    "blocks_consumed": 1,
                }

        # Pattern 2: Bài in PageHeader (original pattern, multi-block)
        if block_type == "PageHeader" and clean_text.upper() == "BÀI":
            if len(next_blocks) >= 2:
                num_block = next_blocks[0]
                title_block = next_blocks[1]
                num_str = self._clean_html(num_block.get("html", ""))

                if num_block.get("block_type") == "PageHeader" and num_str.isdigit():
                    title = self._clean_html(title_block.get("html", ""))
                    title = re.sub(r"\s+", " ", title)
                    logger.info(f"Classified Bài (PageHeader, Level 3): Bài {num_str} - {title}")
                    return {
                        "type": "Lesson",
                        "title": f"Bài {num_str}: {title.upper()}",
                        "code": f"Bài {num_str}",
                        "level": 3,
                        "blocks_consumed": 3,
                    }

        # ============================================================
        # LEVEL 4-7: Sections based on CODE PATTERN (not h-tag)
        # RULE: h3/h4/h5+ MUST have a code pattern to be structure
        # No code = content (return None)
        #
        # Level mapping by CODE PATTERN:
        #   - I., II., III. (Roman numerals) → Level 4
        #   - 1., 2., 3. (Arabic numerals)   → Level 5
        #   - a), b), c) (Letters)           → Level 6
        # ============================================================
        if block_type == "SectionHeader":
            # Level 4: Roman numerals - I., II., III...
            match_roman = re.match(r"^\s*([IVXLCDM]+)\.\s+(.*)", clean_text, re.DOTALL)
            if match_roman and "ĐỊNH LUẬT" not in match_roman.group(1).upper():
                logger.info(f"Classified Roman numeral (Level 4): {match_roman.group(1)}")
                return {
                    "type": "Section",
                    "title": match_roman.group(2).strip(),
                    "code": match_roman.group(1).upper(),
                    "level": 4,
                    "blocks_consumed": 1,
                }

            # Level 5: Arabic numerals - 1., 2., 3...
            match_num = re.match(r"^\s*(\d+)\.\s+(.*)", clean_text, re.DOTALL)
            if match_num:
                code = match_num.group(1)
                title = match_num.group(2).strip()
                logger.info(f"Classified numbered section (Level 5): {code}. {title}")
                return {
                    "type": "Section",
                    "title": title,
                    "code": code,
                    "level": 5,
                    "blocks_consumed": 1,
                }

            # Level 6: Lowercase letters - a), b), c)
            match_letter = re.match(r"^\s*([a-z])\)\s+(.*)", clean_text, re.DOTALL)
            if match_letter:
                logger.info(f"Classified letter section (Level 6): {match_letter.group(1)})")
                return {
                    "type": "Section",
                    "title": match_letter.group(2).strip(),
                    "code": match_letter.group(1).lower(),
                    "level": 6,
                    "blocks_consumed": 1,
                }

            # ============================================================
            # NO CODE PATTERN FOUND for h3/h4/h5+
            # These should be treated as CONTENT, not structure
            # ============================================================
            if h_tag and h_tag >= 3:
                # h3, h4, h5... without code pattern -> content (skip as structure)
                logger.info(
                    f"Skipping h{h_tag} without code pattern (will be content): {clean_text[:60]}"
                )
                return None

            # For h1/h2 without known patterns: log warning but don't infer
            logger.warning(
                "SectionHeader not classified - no pattern matched",
                block_id=block_id,
                h_tag=h_tag,
                clean_text=clean_text[:80] if clean_text else "",
            )

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

                    # Pop until we find a parent with smaller level
                    while (
                        self.parent_stack and getattr(self.parent_stack[-1], "level", 1) >= level
                    ):
                        self.parent_stack.pop()

                    if not self.parent_stack:
                        self.parent_stack.append(self.root)

                    # Create a new StructureNode for this header
                    new_node = StructureNode(
                        node_type="structure",
                        level=level,
                        attributes=StructureAttributes(
                            title=header_info["title"], code=header_info.get("code")
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
                    alt_match = re.search(r'alt="([^"]*)"', block.get("html", ""))
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
                            image_url = await self._extract_and_upload_image(base64_data, image_id)

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
                logger.info(f"Uploaded processed JSON to MinIO: {object_name}")
            except Exception as e:
                logger.error(f"Failed to upload processed JSON to MinIO: {e}")
