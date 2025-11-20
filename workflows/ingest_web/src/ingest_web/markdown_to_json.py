"""Convert markdown to structured hierarchical JSON using common models."""

from __future__ import annotations

import re
from typing import Any, Match

from workflows.models import Paragraph, Section, Source, SourceType

from .semantic_splitter import SemanticParagraphSplitter


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
        self._splitter = (
            SemanticParagraphSplitter()
            if enable_sentence_splitting
            else None
        )

    def convert_to_section(
        self,
        markdown: str,
        source_type: SourceType = SourceType.WEB,
        url: str | None = None,
        file_path: str | None = None,
        title: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> Section:
        """Convert markdown to Section with hierarchical structure."""
        lines = markdown.split("\n")
        metadata = metadata or {}

        # Extract title from first H1 or first line
        if not title:
            title = self._extract_title(lines) or "Untitled Document"

        # Create source reference
        source = Source(
            source_type=source_type,
            url=url,
            file_path=file_path,
            metadata=metadata,
        )

        # Build hierarchical structure
        root_children: list[Section | Paragraph] = []
        section_stack: list[tuple[int, Section]] = []  # Stack of (markdown_level, section)

        current_paragraph_lines: list[str] = []

        for line in lines:
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

                # Adjust stack based on markdown level (not actual level)
                # This handles cases where markdown levels jump (e.g., # to ###)
                section_stack = self._adjust_section_stack(section_stack, markdown_level)

                # Calculate actual hierarchical level based on stack depth
                actual_level = len(section_stack) + 1

                # Create new section (without source - only root has source)
                new_section = self._create_section(header_text, actual_level)

                # Add to parent or root
                if section_stack:
                    # Parent is last section on stack
                    parent = section_stack[-1][1]
                    parent.children.append(new_section)
                else:
                    # No parent - add to root
                    root_children.append(new_section)

                # Push new section with its markdown level
                section_stack.append((markdown_level, new_section))
            else:
                # Regular content line - add to current paragraph
                current_paragraph_lines.append(line)

        # Save last paragraph if exists
        if current_paragraph_lines:
            paragraphs = self._create_paragraph("\n".join(current_paragraph_lines))
            if paragraphs:
                self._add_to_current_section(paragraphs, section_stack, root_children)

        # Create root section
        root_section = Section(
            title=title,
            level=1,
            children=root_children,
            source=source,
        )

        return root_section

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

    def _create_section(self, title: str, level: int) -> Section:
        """Create a new Section."""
        actual_level = min(level, 10)  # Cap at 10

        return Section(
            title=title,
            level=actual_level,
        )

    def _create_paragraph(self, text: str, page: int | None = None) -> list[Paragraph] | None:
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
                return [Paragraph(text=sentence, page=page) for sentence in cleaned_sentences]
            elif len(cleaned_sentences) == 1:
                # Single sentence, return as one paragraph
                return [Paragraph(text=cleaned_sentences[0], page=page)]

        # Return single paragraph (if splitting disabled)
        return [Paragraph(text=text, page=page)]

    def _adjust_section_stack(
        self, stack: list[tuple[int, Section]], new_markdown_level: int
    ) -> list[tuple[int, Section]]:
        """Adjust section stack based on new header markdown level."""
        # Pop sections until we find a parent with lower markdown level
        while stack and stack[-1][0] >= new_markdown_level:
            stack.pop()
        return stack

    def _add_to_current_section(
        self,
        paragraphs: list[Paragraph] | None,
        section_stack: list[tuple[int, Section]],
        root_children: list[Section | Paragraph],
    ) -> None:
        """Add paragraph(s) to current section or root."""
        if not paragraphs:
            return

        target = section_stack[-1][1].children if section_stack else root_children
        target.extend(paragraphs)


    def replace_image_urls(
        self, markdown: str, image_mappings: dict[str, str]
    ) -> str:
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
        return cleaned_text
