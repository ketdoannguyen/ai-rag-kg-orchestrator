"""Deterministic normalization of vision OCR pages into the knowledge tree.

The model only proposes typed blocks. This module decides what becomes a
section, how deep it nests, and what becomes content, so the final tree never
depends on the model's own idea of the hierarchy.
"""

from __future__ import annotations

import re
from collections import Counter
from typing import Any

from arango_client import (
    ParagraphAttributes,
    ParagraphMetadata,
    ParagraphNode,
    RootNode,
    StructureAttributes,
    StructureMetadata,
    StructureNode,
    TableAttributes,
    TableMetadata,
    TableNode,
)

from config.chunking import MAX_CHUNK_SIZE

from .vision_ocr import VisionBlock, VisionPage

CALLOUT_LABELS = ("EM ĐÃ HỌC", "EM CÓ THỂ", "EM CÓ BIẾT")

INSTRUCTION_PREFIXES = (
    "nêu ",
    "trình bày ",
    "giải thích ",
    "phân tích ",
    "so sánh ",
    "liệt kê ",
    "cho biết ",
    "hãy ",
    "em hãy ",
    "tại sao ",
    "vì sao ",
    "làm thế nào ",
    "như thế nào ",
)

FRAGMENT_CHARS = 150
"""Groups shorter than this are fragments rather than retrieval units, so they are absorbed."""

TAIL_CHARS = 200
"""Characters of the open group echoed to the model so it can continue it across a page."""

_SENTENCE_RE = re.compile(r"(?<=[.!?:])\s+")
_TABLE_SEPARATOR_RE = re.compile(r"^\|?[\s:|-]+\|?$")


def _is_callout_label(text: str) -> bool:
    """True for the fixed Vietnamese callout labels the books use."""
    upper = text.upper()
    return any(upper.startswith(label) for label in CALLOUT_LABELS)


def _is_question(text: str, marker: str | None) -> bool:
    """Questions are content, not structure.

    A marker-carrying heading is trusted as structure unless it is literally a
    question, because books number real sub-sections after an instruction verb
    (``b) Giải thích sự nóng chảy ...``). Unmarked lines that read like an
    instruction are almost always questions the model promoted by mistake.
    """
    stripped = text.strip()
    if stripped.startswith("?") or stripped.endswith("?"):
        return True
    return marker is None and stripped.lower().startswith(INSTRUCTION_PREFIXES)


def _units(text: str) -> list[str]:
    """Break text on line edges, then sentence edges, into budget-sized pieces."""
    units: list[str] = []
    for line in text.splitlines():
        if not line.strip():
            continue
        if len(line) <= MAX_CHUNK_SIZE:
            units.append(line)
            continue
        for sentence in _SENTENCE_RE.split(line):
            if not sentence.strip():
                continue
            while len(sentence) > MAX_CHUNK_SIZE:
                units.append(sentence[:MAX_CHUNK_SIZE])
                sentence = sentence[MAX_CHUNK_SIZE:]
            units.append(sentence)
    return units


def _split_oversized(text: str) -> list[str]:
    """Pack a group into chunks that stay within ``MAX_CHUNK_SIZE``."""
    if len(text) <= MAX_CHUNK_SIZE:
        return [text]
    pieces: list[str] = []
    current = ""
    for unit in _units(text):
        if current and len(current) + len(unit) + 1 > MAX_CHUNK_SIZE:
            pieces.append(current)
            current = ""
        current = f"{current}\n{unit}" if current else unit
    if current:
        pieces.append(current)
    return pieces


def _parse_table_rows(text: str) -> list[list[str]]:
    """Parse pipe/tab separated rows, ignoring markdown separator lines."""
    rows: list[list[str]] = []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped or _TABLE_SEPARATOR_RE.match(stripped):
            continue
        if "|" in stripped:
            rows.append([cell.strip() for cell in stripped.strip("|").split("|")])
        elif "\t" in stripped:
            rows.append([cell.strip() for cell in stripped.split("\t")])
        else:
            rows.append([stripped])
    return rows


class VisionDocumentParser:
    """Builds the canonical knowledge tree from accepted vision pages.

    Heading depth is relative to the document root: the model's ``depth`` of 1
    is the top-level section (level 2 below the root). The open heading chain
    survives page boundaries, so a heading continued on the next page nests
    under the same parent.
    """

    MAX_LEVEL = 8

    def __init__(self, title: str) -> None:
        self.root = RootNode(attributes=StructureAttributes(title=title))
        self._stack: list[StructureNode] = [self.root]
        self._page = 0
        self._first_page: int | None = None
        self._buffer: list[str] = []
        self._buffer_group: int | None = None
        self._buffer_page = 0
        self._carried_group: int | None = None
        self._last_chunk: ParagraphNode | None = None
        self._last_chunk_section: StructureNode | None = None
        self.blocks_by_kind: Counter[str] = Counter()
        self.normalization: Counter[str] = Counter()

    @property
    def state(self) -> dict[str, Any]:
        """Open heading chain and the group still being built, for the OCR prompt.

        ``open_group`` is the label the next page must reuse to continue that
        group; without it the model can only restart numbering and no page
        boundary would ever join.
        """
        return {
            "open_headings": [
                {
                    "marker": node.attributes.code,
                    "depth": node.level - 1,
                    "text": node.attributes.title,
                }
                for node in self._stack[1:]
            ],
            "open_group": self._buffer_group,
            "open_group_tail": "\n".join(self._buffer)[-TAIL_CHARS:],
        }

    def add_page(self, page: VisionPage, *, page_number: int) -> None:
        """Normalize and append one accepted page.

        Only accepted pages reach this method, so a failed page can never
        mutate the hierarchy state.
        """
        self._page = page_number
        if self._first_page is None:
            self._first_page = page_number
            self.root.metadata.start_page = page_number

        # A page that claims to continue keeps the open group alive. The join
        # only happens if the model then reuses that label (see ``_group``).
        if page.continues_previous and self._buffer:
            self.normalization["cross_page_continues"] += 1
            self._carried_group = self._buffer_group
        else:
            self._flush()

        for block in page.blocks:
            if not block.text:
                # An empty figure is a picture with no visible text: there is no
                # text-first payload to index, so it is counted and dropped.
                self.normalization["textless_figures_dropped"] += 1
                continue
            kind, text, depth = self._normalize(block)
            self.blocks_by_kind[kind] += 1
            if kind == "heading":
                self._flush()
                self._add_heading(text, block.marker, depth, page_number)
            elif kind == "table":
                self._flush()
                self._stack[-1].children.append(self._build_table(text, page_number))
                self._last_chunk = None
                self._last_chunk_section = None
            else:
                self._group(block.group, text, page_number)

    def finish(self) -> StructureNode:
        """Close every open section and return the document root."""
        self._flush()
        for node in self._stack[1:]:
            self._close(node)
        self.root.metadata.end_page = self._page
        return self.root

    def _normalize(self, block: VisionBlock) -> tuple[str, str, int]:
        """Return the effective ``(kind, text, depth)`` after the semantic rules."""
        depth = block.depth or 1
        if block.kind != "heading":
            return block.kind, block.text, depth
        if _is_callout_label(block.text):
            self.normalization["callouts_reclassified"] += 1
            return "callout", block.text, depth
        if _is_question(block.text, block.marker):
            self.normalization["questions_reclassified"] += 1
            return "question", block.text, depth
        return "heading", block.text, depth

    def _add_heading(
        self,
        text: str,
        marker: str | None,
        depth: int,
        page_number: int,
    ) -> None:
        target = max(2, min(depth + 1, self.MAX_LEVEL))
        open_node = self._stack[-1]

        if open_node.level == target and open_node.attributes.title == text:
            self.normalization["duplicate_headings"] += 1
            return
        if target > open_node.level + 1:
            self.normalization["depth_clamped"] += 1
            target = open_node.level + 1

        while self._stack[-1].level >= target:
            self._close(self._stack.pop())

        node = StructureNode(
            level=target,
            attributes=StructureAttributes(title=text, code=marker),
            metadata=StructureMetadata(start_page=page_number),
        )
        self._stack[-1].children.append(node)
        self._stack.append(node)
        # A heading ends the previous idea, so nothing after it may join back.
        self._last_chunk = None
        self._last_chunk_section = None

    def _group(self, group: int, text: str, page_number: int) -> None:
        """Add a content block to the open group, starting a new one when the label changes."""
        if group != self._buffer_group:
            self._flush()
            self._buffer_group = group
            self._buffer_page = page_number
        elif self._carried_group == group:
            # First block on this page reuses the group left open on the last
            # one, so the two pages really are one chunk.
            self.normalization["cross_page_joins"] += 1
            self._carried_group = None
        self._buffer.append(text)

    def _flush(self) -> None:
        """Turn the open group into chunk nodes under the current section."""
        if not self._buffer:
            return
        text = "\n".join(self._buffer)
        page_number = self._buffer_page
        self._buffer = []
        self._buffer_group = None
        self._carried_group = None
        self.normalization["groups_created"] += 1

        parent = self._stack[-1]
        pieces = _split_oversized(text)
        self.normalization["groups_split"] += len(pieces) - 1
        for piece in pieces:
            if len(piece) < FRAGMENT_CHARS and self._join_fragment(piece, parent):
                continue
            node = self._paragraph(piece, page_number)
            parent.children.append(node)
            self._last_chunk = node
            self._last_chunk_section = parent

    def _join_fragment(self, piece: str, parent: StructureNode) -> bool:
        """Absorb a too-short chunk into the previous one when they sit together."""
        previous = self._last_chunk
        if previous is None or self._last_chunk_section is not parent:
            return False
        held = previous.attributes.text
        if len(held) >= FRAGMENT_CHARS or len(held) + len(piece) + 1 > MAX_CHUNK_SIZE:
            return False
        previous.attributes.text = f"{held}\n{piece}"
        self.normalization["short_groups_merged"] += 1
        return True

    def _paragraph(self, text: str, page_number: int) -> ParagraphNode:
        return ParagraphNode(
            attributes=ParagraphAttributes(text=text),
            metadata=ParagraphMetadata(page=page_number),
        )

    def _build_table(self, text: str, page_number: int) -> TableNode:
        rows = _parse_table_rows(text)
        if not rows:
            self.normalization["tables_fallback"] += 1
            rows = [[text]]
        return TableNode(
            attributes=TableAttributes(rows=rows, extracted_text=text),
            metadata=TableMetadata(page=page_number),
        )

    def _close(self, node: StructureNode) -> None:
        if node.metadata.end_page is None:
            node.metadata.end_page = self._page
