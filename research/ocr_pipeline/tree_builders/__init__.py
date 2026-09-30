"""Tree builders: one neutral artifact in, one knowledge tree out.

Three strategies over the same approved artifact:

* :func:`build_vision_groups` replays it through the runtime parser and keeps the
  model's own grouping, which is what the production PDF pipeline does;
* :func:`build_atomic` throws grouping away and makes every textual block its own
  detail;
* :func:`build_marker_structure` re-derives the structure from Marker's raw block
  kinds and the heading text itself.

Each returns a :class:`BuildResult`. Two fingerprints and one check then describe
the tree without reference to the run that produced it:

* :func:`semantic_signature` hashes the scored content -- kinds, levels, titles,
  text and child order -- and ignores ids, timestamps, embeddings and summaries, so
  replaying an artifact always lands on the same hash;
* :func:`page_signature` hashes page ranges on their own, because a page can move
  without the text changing;
* :func:`validate_tree` runs the ArangoDB contract and returns tree statistics.

:func:`text_fidelity` compares a tree with the artifact it came from, which is how
a builder's dropped, invented or reordered text becomes a number.
"""

from __future__ import annotations

import difflib
import hashlib
import json
import re
import uuid
from collections import Counter
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass
from typing import Any
from uuid import UUID

from arango_client import (
    StructureNode,
    compute_tree_statistics,
    set_parent_ids,
    set_positions,
    validate_tree_and_attach_source,
)

from .atomic import AtomicTreeBuilder, build_atomic
from .marker_structure import (
    CONTENT,
    DROPPED,
    FALLBACK_DEPTH,
    MARKER_BLOCK_KINDS,
    UNKNOWN,
    HeadingBoundary,
    HeadingComparison,
    MarkerCompatibility,
    MarkerStructureBuilder,
    MarkerStructureError,
    build_marker_structure,
    classify_marker_kind,
    compare_headings,
    heading_boundaries,
    marker_compatibility,
    text_heading_level,
)
from .vision_groups import (
    VISION_BLOCK_KINDS,
    BuildResult,
    ReplayError,
    build_vision_groups,
    mapped_kind,
    page_index,
    raw_kind,
    replay_page,
    replay_pages,
    request_page,
    runtime_block,
)

__all__ = [
    # Builders
    "AtomicTreeBuilder",
    "build_atomic",
    "build_marker_structure",
    "build_vision_groups",
    "MarkerStructureBuilder",
    "BuildResult",
    # Replay
    "ReplayError",
    "VISION_BLOCK_KINDS",
    "mapped_kind",
    "page_index",
    "raw_kind",
    "replay_page",
    "replay_pages",
    "request_page",
    "runtime_block",
    # Marker structure
    "CONTENT",
    "DROPPED",
    "UNKNOWN",
    "FALLBACK_DEPTH",
    "MARKER_BLOCK_KINDS",
    "HeadingBoundary",
    "HeadingComparison",
    "MarkerCompatibility",
    "MarkerStructureError",
    "classify_marker_kind",
    "compare_headings",
    "heading_boundaries",
    "marker_compatibility",
    "text_heading_level",
    # Signatures and checks
    "SYNTHETIC_SOURCE_ID",
    "TextFidelity",
    "normalize_stream",
    "page_signature",
    "semantic_signature",
    "text_fidelity",
    "validate_tree",
]

SYNTHETIC_SOURCE_ID: UUID = uuid.uuid5(
    uuid.NAMESPACE_URL, "research.ocr_pipeline/synthetic-source"
)
"""Stable stand-in for a data source, so offline scoring never invents a real one."""

_WHITESPACE_RE = re.compile(r"\s+")


@dataclass(frozen=True)
class TextFidelity:
    """How much of the artifact's own text survives into the tree.

    The artifact is the reference, so this is not accuracy: it says what a builder
    dropped, invented or moved. A character deleted from one side and inserted on
    the other is counted once, as reordered, rather than as both omitted and added.
    """

    artifact_chars: int
    tree_chars: int
    omitted: int
    added: int
    reordered: int

    @property
    def exact(self) -> bool:
        return self.omitted == 0 and self.added == 0 and self.reordered == 0

    @property
    def omitted_ratio(self) -> float | None:
        """Share of the artifact's characters missing from the tree; ``None`` if empty."""
        if not self.artifact_chars:
            return None
        return self.omitted / self.artifact_chars


def normalize_stream(text: str) -> str:
    """Collapse every whitespace run.

    A chunk boundary only ever lands in whitespace, so collapsing it keeps a
    re-chunked tree comparable with the artifact it came from.
    """
    return _WHITESPACE_RE.sub(" ", text).strip()


def semantic_signature(root: StructureNode) -> str:
    """Hash of the scored tree: kinds, levels, titles, text and child order.

    Node ids, parent ids, source ids, timestamps, positions, embeddings and
    summaries are left out, so the same artifact and builder always hash the same
    and validating or saving the tree never moves the hash.
    """
    return _digest(_semantic_nodes(root))


def page_signature(root: StructureNode) -> str:
    """Hash of the page ranges alone, kept apart from the semantic signature."""
    return _digest(_page_nodes(root))


def validate_tree(
    root: StructureNode,
    *,
    source_id: UUID = SYNTHETIC_SOURCE_ID,
) -> dict[str, Any]:
    """Check the tree against the ArangoDB contract and return its statistics.

    Parent ids, breadcrumb positions and the source id are attached first because
    the contract requires them; none of them reaches the semantic signature.
    """
    set_parent_ids(root)
    set_positions(root)
    validate_tree_and_attach_source(root, source_id)
    return compute_tree_statistics(root)


def text_fidelity(root: StructureNode, pages: Sequence[Any]) -> TextFidelity:
    """Compare the artifact's text with the text the tree kept."""
    artifact = normalize_stream("\n".join(_artifact_text(pages)))
    tree = normalize_stream("\n".join(_tree_text(root)))
    deleted: Counter[str] = Counter()
    inserted: Counter[str] = Counter()
    matcher = difflib.SequenceMatcher(None, artifact, tree, autojunk=False)
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag in ("delete", "replace"):
            deleted.update(artifact[i1:i2])
        if tag in ("insert", "replace"):
            inserted.update(tree[j1:j2])
    # A character deleted on one side and inserted on the other did not vanish, it
    # moved; counting it as both would overstate omitted and added.
    moved = sum(min(count, inserted[char]) for char, count in deleted.items())
    return TextFidelity(
        artifact_chars=len(artifact),
        tree_chars=len(tree),
        omitted=sum(deleted.values()) - moved,
        added=sum(inserted.values()) - moved,
        reordered=moved,
    )


def _digest(rows: Iterable[Any]) -> str:
    payload = json.dumps(list(rows), ensure_ascii=False, separators=(",", ":"), sort_keys=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _semantic_nodes(node: Any) -> Iterator[Any]:
    if getattr(node, "node_type", None) == "structure":
        yield ["s", node.level, node.attributes.code, node.attributes.title]
        for child in node.children:
            yield from _semantic_nodes(child)
    elif getattr(node, "node_type", None) == "table":
        rows = node.attributes.rows
        yield ["t", rows if rows else node.attributes.extracted_text]
    else:
        yield ["p", node.attributes.text]


def _page_nodes(node: Any) -> Iterator[Any]:
    if getattr(node, "node_type", None) == "structure":
        yield [
            "s",
            node.level,
            node.attributes.code,
            node.attributes.title,
            node.metadata.start_page,
            node.metadata.end_page,
        ]
        for child in node.children:
            yield from _page_nodes(child)
    else:
        yield ["d", node.metadata.page]


def _artifact_text(pages: Sequence[Any]) -> Iterator[str]:
    for page in pages:
        for block in getattr(page, "blocks", ()) or ():
            text = getattr(block, "text", "")
            if isinstance(text, str) and text.strip():
                yield text


def _tree_text(node: Any, *, root: bool = True) -> Iterator[str]:
    """Text a tree holds, in document order.

    A section's title is text the artifact also had, so it counts; the root's is
    not, because it is the caller's document title rather than anything the engine
    transcribed.
    """
    if getattr(node, "node_type", None) == "structure":
        title = getattr(node.attributes, "title", None)
        if not root and title:
            yield title
        for child in node.children:
            yield from _tree_text(child, root=False)
        return
    text = _detail_text(node)
    if text:
        yield text


def _detail_text(node: Any) -> str:
    """The text a detail node contributes, whichever payload type it carries."""
    attributes = getattr(node, "attributes", None)
    for name in ("text", "extracted_text"):
        value = getattr(attributes, name, None)
        if isinstance(value, str) and value:
            return value
    rows = getattr(attributes, "rows", None)
    if rows:
        return "\n".join(" | ".join(str(cell) for cell in row) for row in rows)
    return ""
