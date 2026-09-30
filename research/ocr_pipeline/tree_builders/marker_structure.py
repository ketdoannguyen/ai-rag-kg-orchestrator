"""Text-first reconstruction of the Marker baseline, and its comparison with the atomic tree.

Marker decides structure from markup it no longer carries once the artifact is
neutral: the ``section_hierarchy`` and the ``h1``/``h2``/``h4`` tags are gone, and
what is left is the text of each block. This builder therefore reconstructs the
structure the way the historical ``TextbookParser`` did after its pre-scan -- from
the heading text itself -- with the frozen converter's depth used when the adapter
could keep it, and the v1 fallback (top-level section, counted) when nothing else
identifies the heading.

Everything downstream of that decision is inherited from the atomic builder, so the
two trees share chunking, size splitting and detail rules, and the comparison in
this module measures the structure decision alone:

* :func:`heading_boundaries` lists the sections each tree opened;
* :func:`compare_headings` reports where two trees agree and where a title moved;
* :func:`text_fidelity` (package level) reports text the builder dropped, invented
  or reordered, which is how running heads and folios show up as a real cost.

Both are measurements on the artifact's own text, never accuracy: only
human-checked gold can say whether a boundary is correct.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from arango_client import StructureNode

from workflows.ingest_file.vision_ocr import VisionBlock

from .atomic import AtomicTreeBuilder
from .vision_groups import (
    BuildResult,
    ReplayError,
    page_index,
    raw_kind,
    request_page,
    runtime_block,
)

MARKER_BLOCK_KINDS: dict[str, str | None] = {
    # Marker's JSONRenderer block types mapped into the runtime vocabulary; a
    # ``None`` marks a kind Marker emits that carries no payload to index.
    "SectionHeader": "heading",
    "Text": "p",
    "ListGroup": "list",
    "Table": "table",
    "Picture": "figure",
    "Figure": "figure",
    "Caption": "p",
    "Equation": "equation",
    "PageHeader": None,
    "PageFooter": None,
}
"""Marker mapping v1: the historical converter's vocabulary, plus the running heads."""

CONTENT = "content"
DROPPED = "dropped"
UNKNOWN = "unknown"

FALLBACK_DEPTH = 1
"""Rule v1 for a heading no pattern identifies: a top-level section, counted."""

_HEADING_PATTERNS: tuple[tuple[re.Pattern[str], int], ...] = (
    # The levels the historical pre-scan assigned to these patterns.
    (re.compile(r"^CHỦ\s*(?:ĐỀ|ĐỂ)\s+\d+", re.IGNORECASE), 2),
    (re.compile(r"^CHƯƠNG\s+[IVXLCDM]+", re.IGNORECASE), 2),
    (re.compile(r"^Bài\s+\d+", re.IGNORECASE), 3),
    (re.compile(r"^\s*[IVXLCDM]+\.\s+", re.IGNORECASE), 4),
    (re.compile(r"^\s*\d+\.\s+", re.IGNORECASE), 4),
    (re.compile(r"^\s*[a-z]\)\s+", re.IGNORECASE), 6),
)


class MarkerStructureError(ReplayError):
    """The artifact cannot be rebuilt as a Marker tree."""


@dataclass(frozen=True)
class MarkerCompatibility:
    """Whether an artifact looks like Marker output at all."""

    compatible: bool
    recognized: int
    unknown_kinds: tuple[str, ...]


@dataclass(frozen=True)
class HeadingBoundary:
    """One section a builder decided to open."""

    level: int
    title: str
    code: str | None
    start_page: int | None


@dataclass(frozen=True)
class HeadingComparison:
    """Where two trees' sections agree.

    Titles are compared as a set, so a title repeated inside one tree counts once
    and the level reported for it is the last one seen. Precision and recall are
    not computed: they belong to scoring against human-checked spans, and these
    two trees were built from the same artifact.
    """

    shared: tuple[str, ...]
    only_in_first: tuple[str, ...]
    only_in_second: tuple[str, ...]
    level_mismatches: tuple[tuple[str, int, int], ...]
    jaccard: float | None


def classify_marker_kind(raw: str) -> tuple[str, str]:
    """Return ``(mapped runtime kind, accounting class)`` for one raw Marker kind.

    An unknown kind keeps its text as a paragraph so the words are never silently
    dropped; the accounting class is what makes the gap visible in the counters.
    """
    if raw not in MARKER_BLOCK_KINDS:
        return "p", UNKNOWN
    mapped = MARKER_BLOCK_KINDS[raw]
    if mapped is None:
        return "p", DROPPED
    return mapped, CONTENT


def text_heading_level(text: str) -> int | None:
    """The historical level a heading's own text implies, or ``None``.

    Text-first mode cannot tell a numbered sub-section from a numbered
    sub-sub-section the way the old h-tag map could, so every numbered heading
    lands on the level of its pattern.
    """
    for pattern, level in _HEADING_PATTERNS:
        if pattern.match(text.strip()):
            return level
    return None


def marker_compatibility(pages: Sequence[Any]) -> MarkerCompatibility:
    """Check that the artifact is Marker output before rebuilding it as such."""
    recognized = 0
    unknown: set[str] = set()
    for page in pages:
        for block in getattr(page, "blocks", ()) or ():
            kind = raw_kind(block)
            if kind in MARKER_BLOCK_KINDS:
                recognized += 1
            else:
                unknown.add(kind)
    return MarkerCompatibility(
        compatible=recognized > 0,
        recognized=recognized,
        unknown_kinds=tuple(sorted(unknown)),
    )


class MarkerStructureBuilder(AtomicTreeBuilder):
    """Marker's structure decisions on top of the atomic builder's chunking."""

    NAME = "marker_structure"

    def add_page(self, page: Any, *, page_number: int) -> None:
        """Ingest one neutral Marker page.

        The parameter is the neutral page rather than the runtime one this class
        inherits: the structure decision needs the engine's raw block kind, which
        the runtime block no longer carries.
        """
        self._open_page(page_number)
        for index, block in enumerate(page.blocks):
            raw = raw_kind(block)
            mapped, accounting = classify_marker_kind(raw)
            self.counters[f"marker_{accounting}_blocks"] += 1
            if accounting == DROPPED:
                continue
            if accounting == UNKNOWN:
                self.counters[f"marker_unknown_kind:{raw}"] += 1

            depth = self._marker_depth(block) if mapped == "heading" else None
            runtime = runtime_block(
                block,
                kind=mapped,
                group=index + 1,
                depth=depth,
                where=f"page_index={page_index(page)} block {index}",
            )
            if mapped == "list":
                self._add_list(runtime, page_number)
            else:
                self._add_block(runtime, page_number)

    def _marker_depth(self, block: Any) -> int:
        """The depth a ``SectionHeader`` gets, text-first.

        A depth on the block means the adapter kept the converter's h-tag decision
        and is trusted over any guess from the words.
        """
        explicit = getattr(block, "depth", None)
        if isinstance(explicit, int) and not isinstance(explicit, bool) and explicit >= 1:
            self.counters["marker_depth_from_artifact"] += 1
            return explicit
        level = text_heading_level(getattr(block, "text", "") or "")
        if level is None:
            self.counters["marker_depth_fallback"] += 1
            return FALLBACK_DEPTH
        self.counters["marker_depth_from_text"] += 1
        return max(1, level - 1)

    def _add_list(self, block: VisionBlock, page_number: int) -> None:
        """One detail per list item, in reading order.

        Marker groups a list's items into one block, so its lines are the items;
        the runtime model already emits one item per line.
        """
        lines = [line.strip() for line in block.text.splitlines() if line.strip()]
        if len(lines) <= 1:
            self.counters["marker_list_items"] += 1
            self._add_block(block, page_number)
            return
        for line in lines:
            self.counters["marker_list_items"] += 1
            self._add_block(VisionBlock(kind="list", group=block.group, text=line), page_number)


def build_marker_structure(pages: Sequence[Any], title: str) -> BuildResult:
    """Build the text-first Marker reconstruction for one artifact.

    A page set without a single recognized Marker block is rejected rather than
    rebuilt as something it is not.
    """
    compatibility = marker_compatibility(pages)
    if not compatibility.compatible:
        seen = ", ".join(compatibility.unknown_kinds) or "no blocks"
        raise MarkerStructureError(
            f"no raw Marker block kind in this artifact (kinds seen: {seen}); "
            "the text-first reconstruction only applies to Marker output"
        )

    builder = MarkerStructureBuilder(title)
    for page in pages:
        builder.add_page(page, page_number=request_page(page))
    root = builder.finish()
    return BuildResult(
        root=root,
        builder=MarkerStructureBuilder.NAME,
        pages_processed=len(pages),
        blocks_by_kind=dict(builder.blocks_by_kind),
        counters={**builder.normalization, **builder.counters},
    )


def heading_boundaries(root: StructureNode) -> tuple[HeadingBoundary, ...]:
    """Every section below the root, in document order.

    The root is the document title rather than a heading a builder chose, so it
    stays out of the comparison.
    """
    found: list[HeadingBoundary] = []

    def walk(node: StructureNode) -> None:
        for child in node.children:
            if getattr(child, "node_type", None) != "structure":
                continue
            found.append(
                HeadingBoundary(
                    level=child.level,
                    title=child.attributes.title,
                    code=child.attributes.code,
                    start_page=child.metadata.start_page,
                )
            )
            walk(child)

    walk(root)
    return tuple(found)


def compare_headings(first: StructureNode, second: StructureNode) -> HeadingComparison:
    """Compare the sections two trees opened on the same artifact."""
    first_levels = {boundary.title: boundary.level for boundary in heading_boundaries(first)}
    second_levels = {boundary.title: boundary.level for boundary in heading_boundaries(second)}
    shared = sorted(set(first_levels) & set(second_levels))
    union = set(first_levels) | set(second_levels)
    return HeadingComparison(
        shared=tuple(shared),
        only_in_first=tuple(sorted(set(first_levels) - set(second_levels))),
        only_in_second=tuple(sorted(set(second_levels) - set(first_levels))),
        level_mismatches=tuple(
            (title, first_levels[title], second_levels[title])
            for title in shared
            if first_levels[title] != second_levels[title]
        ),
        jaccard=len(shared) / len(union) if union else None,
    )
