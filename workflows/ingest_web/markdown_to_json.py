"""Convert markdown to structured hierarchical JSON using common models."""

from __future__ import annotations

import re
from re import Match
from typing import Any

from arango_client import (
    ParagraphAttributes,
    ParagraphMetadata,
    ParagraphNode,
    SourceType,
    StructureAttributes,
    StructureMetadata,
    StructureNode,
)

from .code_extraction import clean_title_from_code, extract_code_from_context
from .semantic_splitter import SemanticParagraphSplitter

# Code-based semantic level mapping
# Level 1: ROOT
# Level 2-3: Parent sections (headers before Roman numerals)
# Level 4: Roman numerals (I, II, III, IV, V...)
# Level 5: Numbers (1, 2, 3, 4, 5...)
# Level 6: Letters (a, b, c, d...)
# Level 7: No code (misc headers)
CODE_TYPE_ROMAN = "roman"
CODE_TYPE_NUMBER = "number"
CODE_TYPE_LETTER = "letter"
CODE_TYPE_NONE = "none"

CODE_TO_LEVEL = {
    CODE_TYPE_ROMAN: 4,
    CODE_TYPE_NUMBER: 5,
    CODE_TYPE_LETTER: 6,
    CODE_TYPE_NONE: 7,
}


def get_code_type(code: str | None) -> str:
    """Determine code type: 'roman', 'number', 'letter', or 'none'.

    Args:
        code: The extracted code string

    Returns:
        One of: CODE_TYPE_ROMAN, CODE_TYPE_NUMBER, CODE_TYPE_LETTER, CODE_TYPE_NONE
    """
    if not code:
        return CODE_TYPE_NONE

    code = code.strip()

    # Check for compound codes like "Bài 1", "Chương I" - extract the actual identifier
    compound_match = re.match(
        r"^(?:Bài|BÀI|Chương|CHƯƠNG|Part|Section|Mục)\s+(.+)$", code, re.IGNORECASE
    )
    if compound_match:
        code = compound_match.group(1).strip()

    # Roman numerals (I, II, III, IV, V, VI, VII, VIII, IX, X, etc.)
    if re.match(r"^[IVXLCDM]+$", code.upper()):
        return CODE_TYPE_ROMAN

    # Numbers (1, 2, 3, 4, 5...)
    if re.match(r"^\d+$", code):
        return CODE_TYPE_NUMBER

    # Letters (a, b, c, d...) - single letter only
    if re.match(r"^[a-zA-Z]$", code):
        return CODE_TYPE_LETTER

    return CODE_TYPE_NONE


class MarkdownToJSONConverter:
    """Convert markdown to structured hierarchical JSON based on headers."""

    def __init__(
        self,
        enable_sentence_splitting: bool = True,
    ) -> None:
        """
        Initialize the converter.

        Args:
            enable_sentence_splitting: Split paragraphs into sentences (each sentence = 1 paragraph) (default: True)
        """
        self._splitter = SemanticParagraphSplitter() if enable_sentence_splitting else None

    def convert_to_section(
        self,
        markdown: str,
        source_type: SourceType = SourceType.WEB,
        url: str | None = None,
        file_path: str | None = None,
        title: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> StructureNode:
        """Convert markdown to Section with hierarchical structure.

        Uses two-pass algorithm:
        - Pass 1: Discover heading-to-level mapping based on code patterns
        - Pass 2: Build tree with semantic levels

        If markdown has no headings, uses synthetic structure generation.
        """
        lines = markdown.split("\n")
        metadata = metadata or {}

        # Extract title from first H1 or first line or bold text
        if not title:
            title = self._extract_title(lines) or "Untitled Document"

        # Check if markdown has any headings
        if not self._has_headings(lines):
            # Use synthetic structure generation for flat content
            return self._build_tree_from_flat_content(
                lines=lines,
                title=title,
                url=url,
                file_path=file_path,
                metadata=metadata,
            )

        # Pass 1: Discover heading level mapping based on codes
        heading_level_map = self._discover_heading_level_mapping(lines)

        # Pass 2: Build hierarchical structure using semantic levels
        root_children: list[StructureNode | ParagraphNode] = []
        section_stack: list[tuple[int, StructureNode]] = []  # Stack of (semantic_level, section)

        current_paragraph_lines: list[str] = []
        processed_lines: list[str] = []  # Track processed lines for context

        for line_idx, line in enumerate(lines):
            line = line.strip()
            if not line:
                continue

            # Check for headers
            header_match = re.match(r"^(#{1,6})\s+(.+)$", line)
            if header_match:
                # Save current paragraph if exists
                if current_paragraph_lines:
                    paragraphs = self._create_paragraph("\n".join(current_paragraph_lines))
                    if paragraphs:
                        self._add_to_current_section(paragraphs, section_stack, root_children)
                    current_paragraph_lines = []

                # Process header
                markdown_level = len(header_match.group(1))
                header_text = self._clean_text(header_match.group(2))
                if not header_text:
                    # Skip headings that become empty after cleaning
                    continue

                # Extract code from header text or previous lines
                previous_lines = processed_lines[-3:] if len(processed_lines) > 0 else []
                extracted_code = extract_code_from_context(
                    current_text=header_text, previous_items=previous_lines
                )

                # Determine semantic level based on code or heading mapping
                code_type = get_code_type(extracted_code)
                if code_type != CODE_TYPE_NONE:
                    # Code found - use fixed level based on code type
                    semantic_level = CODE_TO_LEVEL[code_type]
                else:
                    # No code - use discovered mapping or fallback
                    semantic_level = heading_level_map.get(markdown_level, 7)

                # Adjust stack based on semantic level
                section_stack = self._adjust_section_stack_by_level(section_stack, semantic_level)

                # Create new section with semantic level
                final_title = header_text
                if extracted_code:
                    final_title = clean_title_from_code(header_text, extracted_code)

                attrs = StructureAttributes(title=final_title)
                if extracted_code and not attrs.code:
                    attrs.code = extracted_code

                new_section = self._create_section_with_attrs(final_title, semantic_level, attrs)

                # Add to parent or root
                if section_stack:
                    parent = section_stack[-1][1]
                    parent.children.append(new_section)
                else:
                    root_children.append(new_section)

                # Push new section with its semantic level
                section_stack.append((semantic_level, new_section))

                # Track processed line
                processed_lines.append(line)
            else:
                # Regular content line - add to current paragraph
                current_paragraph_lines.append(line)
                processed_lines.append(line)

        # Save last paragraph if exists
        if current_paragraph_lines:
            paragraphs = self._create_paragraph("\n".join(current_paragraph_lines))
            if paragraphs:
                self._add_to_current_section(paragraphs, section_stack, root_children)

        # Create root section with source metadata
        root_metadata = StructureMetadata(
            url=url,
            file_path=file_path,
            **metadata if metadata else {},
        )
        root_section = StructureNode(
            node_type="structure",
            level=1,
            attributes=StructureAttributes(title=title),
            metadata=root_metadata,
            children=root_children,
        )

        return root_section

    def _has_headings(self, lines: list[str]) -> bool:
        """Check if markdown has any proper headings (#)."""
        for line in lines:
            if re.match(r"^#{1,6}\s+.+$", line.strip()):
                return True
        return False

    def _build_tree_from_flat_content(
        self,
        lines: list[str],
        title: str,
        url: str | None,
        file_path: str | None,
        metadata: dict[str, Any],
    ) -> StructureNode:
        """Build tree from flat content without headings.

        Uses synthetic structure based on:
        - Bold text (**text**) as potential section titles
        - List item groups as sections
        - Paragraph chunking for long content
        """
        root_children: list[StructureNode | ParagraphNode] = []
        current_section: StructureNode | None = None
        current_paragraph_lines: list[str] = []

        def flush_paragraph():
            nonlocal current_paragraph_lines
            if current_paragraph_lines:
                text = "\n".join(current_paragraph_lines)
                paragraphs = self._create_paragraph(text)
                if paragraphs:
                    if current_section:
                        current_section.children.extend(paragraphs)
                    else:
                        root_children.extend(paragraphs)
                current_paragraph_lines = []

        for line in lines:
            line = line.strip()
            if not line:
                continue

            # Check for bold text that could be a section title
            bold_match = re.match(r"^\*\*(.+?)\*\*\s*:?\s*(.*)$", line)
            if bold_match:
                bold_text = bold_match.group(1).strip()
                rest = bold_match.group(2).strip()

                # If bold text is substantial and at start, treat as section
                if len(bold_text) >= 3 and len(bold_text) <= 100:
                    flush_paragraph()

                    # Create new section
                    current_section = StructureNode(
                        node_type="structure",
                        level=2,
                        attributes=StructureAttributes(title=bold_text),
                        metadata=StructureMetadata(),
                        children=[],
                    )
                    root_children.append(current_section)

                    # Add rest as paragraph if exists
                    if rest:
                        current_paragraph_lines.append(rest)
                    continue

            # Check if line is a list item
            list_match = re.match(r"^[\*\-\+]\s+(.+)$", line)
            numbered_match = re.match(r"^\d+[\.\)]\s+(.+)$", line)

            if list_match or numbered_match:
                # Add to current paragraph (list items)
                current_paragraph_lines.append(line)
            else:
                # Regular content
                current_paragraph_lines.append(line)

                # Chunk if paragraph is getting long
                if len("\n".join(current_paragraph_lines)) > 1000:
                    flush_paragraph()

        # Flush remaining content
        flush_paragraph()

        # If no sections were created, chunk content semantically
        if not any(isinstance(child, StructureNode) for child in root_children):
            root_children = self._chunk_flat_content(root_children)

        # Create root section
        root_metadata = StructureMetadata(
            url=url,
            file_path=file_path,
            **metadata if metadata else {},
        )
        return StructureNode(
            node_type="structure",
            level=1,
            attributes=StructureAttributes(title=title),
            metadata=root_metadata,
            children=root_children,
        )

    def _chunk_flat_content(
        self,
        children: list[StructureNode | ParagraphNode],
    ) -> list[StructureNode | ParagraphNode]:
        """Chunk flat paragraphs into semantic groups."""
        if len(children) <= 5:
            return children

        # Group paragraphs into chunks of ~5 items
        chunks: list[list[ParagraphNode]] = []
        current_chunk: list[ParagraphNode] = []

        for child in children:
            if isinstance(child, ParagraphNode):
                current_chunk.append(child)
                if len(current_chunk) >= 5:
                    chunks.append(current_chunk)
                    current_chunk = []
            else:
                # Keep structure nodes as-is
                if current_chunk:
                    chunks.append(current_chunk)
                    current_chunk = []
                chunks.append([child])  # type: ignore

        if current_chunk:
            chunks.append(current_chunk)

        # Convert chunks to sections if we have multiple
        if len(chunks) <= 1:
            return children

        result: list[StructureNode | ParagraphNode] = []
        for i, chunk in enumerate(chunks):
            if len(chunk) == 1 and isinstance(chunk[0], StructureNode):
                result.append(chunk[0])
            else:
                section = StructureNode(
                    node_type="structure",
                    level=2,
                    attributes=StructureAttributes(title=f"Section {i + 1}"),
                    metadata=StructureMetadata(),
                    children=list(chunk),
                )
                result.append(section)

        return result

    def _discover_heading_level_mapping(self, lines: list[str]) -> dict[int, int]:
        """Pass 1: Discover heading-to-semantic-level mapping based on codes.

        Scans all headers to find which markdown heading levels contain which code types,
        then builds a mapping from markdown level to semantic level.

        Returns:
            dict mapping markdown_level (1-6) to semantic_level (2-7)
        """
        heading_codes: dict[int, set[str]] = {}  # {markdown_level: set(code_types)}

        for line in lines:
            line = line.strip()
            if not line:
                continue

            header_match = re.match(r"^(#{1,6})\s+(.+)$", line)
            if header_match:
                md_level = len(header_match.group(1))
                header_text = self._clean_text(header_match.group(2))
                if not header_text:
                    continue

                # Extract code from header text
                extracted_code = extract_code_from_context(current_text=header_text)
                code_type = get_code_type(extracted_code)

                if md_level not in heading_codes:
                    heading_codes[md_level] = set()
                heading_codes[md_level].add(code_type)

        # Build semantic level mapping
        return self._build_semantic_level_map(heading_codes)

    def _build_semantic_level_map(self, heading_codes: dict[int, set[str]]) -> dict[int, int]:
        """Build mapping from markdown level to semantic level.

        Uses code types as anchors:
        - Roman numerals (I, II, III) -> Level 4
        - Numbers (1, 2, 3) -> Level 5
        - Letters (a, b, c) -> Level 6
        - Headings before Roman level -> Level 2 or 3
        - No code -> Level 7
        """
        # Find which md_level has which code type (first occurrence takes priority)
        roman_level: int | None = None
        number_level: int | None = None
        letter_level: int | None = None

        for md_level in sorted(heading_codes.keys()):
            code_types = heading_codes[md_level]
            if CODE_TYPE_ROMAN in code_types and roman_level is None:
                roman_level = md_level
            if CODE_TYPE_NUMBER in code_types and number_level is None:
                number_level = md_level
            if CODE_TYPE_LETTER in code_types and letter_level is None:
                letter_level = md_level

        # Build mapping
        mapping: dict[int, int] = {}
        for md_level in heading_codes:
            if md_level == roman_level:
                mapping[md_level] = 4
            elif md_level == number_level:
                mapping[md_level] = 5
            elif md_level == letter_level:
                mapping[md_level] = 6
            elif roman_level and md_level < roman_level:
                # Heading before roman level -> calculate relative level
                # E.g., if roman is at h3, then h2 -> 3, h1 -> 2
                mapping[md_level] = max(2, 4 - (roman_level - md_level))
            else:
                # No code or after letter level
                mapping[md_level] = 7

        return mapping

    def _adjust_section_stack_by_level(
        self, stack: list[tuple[int, StructureNode]], new_semantic_level: int
    ) -> list[tuple[int, StructureNode]]:
        """Adjust section stack based on new header's semantic level."""
        # Pop sections until we find a parent with lower semantic level
        while stack and stack[-1][0] >= new_semantic_level:
            stack.pop()
        return stack

    def _extract_title(self, lines: list[str]) -> str | None:
        """Extract title from first H1 or first meaningful line."""
        # Pass 1: look for explicit H1
        for line in lines:
            stripped = line.strip()
            if not stripped:
                continue

            h1_match = re.match(r"^#\s+(.+)$", stripped)
            if h1_match:
                title = self._clean_text(h1_match.group(1))
                if title:
                    return title

        # Pass 2: use first meaningful non-bullet line
        for line in lines:
            stripped = line.strip()
            if not stripped:
                continue

            # Skip list-like lines when falling back
            if re.match(r"^[\-\*\u2022]", stripped):
                continue

            clean_line = self._clean_text(stripped)
            if clean_line and len(clean_line) > 5:
                return clean_line

        return None

    def _create_section_with_attrs(
        self, title: str, level: int, attrs: StructureAttributes
    ) -> StructureNode:
        """Create a new Section with provided attributes."""
        actual_level = min(level, 10)  # Cap at 10

        return StructureNode(
            node_type="structure",
            level=actual_level,
            attributes=attrs,
        )

    def _create_paragraph(self, text: str, page: int | None = None) -> list[ParagraphNode] | None:
        """Create Paragraph(s) from text, splitting if too long."""
        # Clean up text
        text = re.sub(r"\n{3,}", "\n\n", text).strip()
        text = self._clean_text(text)
        if not text:
            return None

        # Skip very short lines that are likely artifacts
        if len(text) < 3:
            return None

        # Split paragraph into sentences (each sentence = 1 paragraph)
        if self._splitter:
            sentences = self._splitter.split_paragraph(text)
            cleaned_sentences = [sentence.strip() for sentence in sentences if sentence.strip()]
            if len(cleaned_sentences) > 1:
                # Return multiple paragraphs (one per sentence)
                return [
                    ParagraphNode(
                        attributes=ParagraphAttributes(text=sentence),
                        metadata=ParagraphMetadata(page=page),
                    )
                    for sentence in cleaned_sentences
                ]
            elif len(cleaned_sentences) == 1:
                # Single sentence, return as one paragraph
                return [
                    ParagraphNode(
                        attributes=ParagraphAttributes(text=cleaned_sentences[0]),
                        metadata=ParagraphMetadata(page=page),
                    )
                ]

        # Return single paragraph (if splitting disabled)
        return [
            ParagraphNode(
                attributes=ParagraphAttributes(text=text), metadata=ParagraphMetadata(page=page)
            )
        ]

    def _adjust_section_stack(
        self, stack: list[tuple[int, StructureNode]], new_markdown_level: int
    ) -> list[tuple[int, StructureNode]]:
        """Adjust section stack based on new header markdown level."""
        # Pop sections until we find a parent with lower markdown level
        while stack and stack[-1][0] >= new_markdown_level:
            stack.pop()
        return stack

    def _add_to_current_section(
        self,
        paragraphs: list[ParagraphNode] | None,
        section_stack: list[tuple[int, StructureNode]],
        root_children: list[StructureNode | ParagraphNode],
    ) -> None:
        """Add paragraph(s) to current section or root."""
        if not paragraphs:
            return

        target = section_stack[-1][1].children if section_stack else root_children
        target.extend(paragraphs)

    def replace_image_urls(self, markdown: str, image_mappings: dict[str, str]) -> str:
        """Replace image URLs in markdown with MinIO object paths or extracted text."""
        result = markdown
        for original_url, replacement in image_mappings.items():
            # Replace markdown image syntax
            pattern = rf"!\[.*?\]\({re.escape(original_url)}\)"
            result = re.sub(pattern, replacement, result)
        return result

    def _strip_markdown_links(self, text: str) -> str:
        """Replace markdown links [label](url) with plain label."""
        if not text:
            return ""

        def _replace(match: Match[str]) -> str:
            label = match.group(1).strip()
            return label

        return re.sub(r"\[(.*?)\]\((.*?)\)", _replace, text)

    def _clean_text(self, text: str) -> str:
        """Normalize text by stripping markdown links and removing empty markers."""
        if not text:
            return ""

        original = text
        text = self._strip_markdown_links(text)

        original_lines = original.splitlines() or [original]
        cleaned_lines: list[str] = []

        for idx, line in enumerate(text.splitlines() or [text]):
            line = line.strip()
            if not line:
                continue

            original_line = original_lines[idx] if idx < len(original_lines) else line
            original_line = original_line.strip()

            # If the original line was just a bullet + link, drop the bullet marker
            if re.fullmatch(r"^[\-\*\u2022]\s*\[.*?\]\(.*?\)\s*$", original_line):
                line = line.lstrip("-*•").strip()

            # Drop lines that only contain bullet markers
            if not line or re.fullmatch(r"^[\-\*\u2022]+\s*$", line):
                continue

            cleaned_lines.append(line)

        cleaned_text = "\n".join(cleaned_lines).strip()

        # Remove markdown formatting (*, _, `) from headings/titles only if they wrap the whole line
        # But here we want to be safe, so let's just remove bold/italic markers if they look like artifacts
        # E.g. "**Title**" -> "Title"
        cleaned_text = re.sub(r"^\*\*(.+?)\*\*$", r"\1", cleaned_text)
        cleaned_text = re.sub(r"^\*(.+?)\*$", r"\1", cleaned_text)
        cleaned_text = re.sub(r"^__(.+?)__$", r"\1", cleaned_text)

        return cleaned_text
