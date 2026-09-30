"""Neutral extraction artifacts, their replay into runtime vision pages, and the groups builder.

A live vision run and a replay of the stored artifact must produce the same tree, so
replay is a first-class operation here: :func:`replay_pages` turns neutral artifact
pages back into the runtime ``VisionPage`` objects that ``VisionDocumentParser``
consumes, and the parser does the rest.

The neutral artifact is engine-agnostic. A block carries the engine's raw kind
(``kind``), the kind the engine mapping produced (``normalized_kind``, absent for
vision), and ``group=None`` whenever the engine does not group blocks. The runtime
model is stricter: ``VisionBlock.group`` is required and must be >= 1. Replay never
guesses, it validates page identity, kind, text, marker, depth and group, and raises
:class:`ReplayError` naming the page and the block that failed.

Grouping is the model's own output, so the numbers are labels rather than ids, and
this builder keeps the runtime behaviour exactly: cross-page continuation, the
short-fragment safety net, and the same counter vocabulary the runtime reports.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any, get_args

from arango_client import StructureNode  # type: ignore[import-untyped]
from pydantic import ValidationError

from workflows.ingest_file.vision_ocr import BlockKind, VisionBlock, VisionPage
from workflows.ingest_file.vision_parser import VisionDocumentParser

VISION_BLOCK_KINDS: frozenset[str] = frozenset(get_args(BlockKind))
"""The vision model's block vocabulary; a neutral block must map into this set."""


class ReplayError(ValueError):
    """A neutral artifact page cannot be replayed as a runtime vision page."""


@dataclass(frozen=True)
class BuildResult:
    """A built tree plus the accounting a comparison run needs.

    ``counters`` uses the runtime parser's own vocabulary so a replay can be
    diffed against a live run; ``blocks_by_kind`` counts the effective kinds after
    semantic normalization.
    """

    root: StructureNode
    builder: str
    pages_processed: int
    blocks_by_kind: dict[str, int]
    counters: dict[str, int]


def raw_kind(block: Any) -> str:
    """The kind the engine itself reported."""
    kind = getattr(block, "kind", None)
    if not isinstance(kind, str) or not kind:
        raise ReplayError(f"block kind must be a non-empty string, got {kind!r}")
    return kind


def mapped_kind(block: Any) -> str:
    """The engine kind mapped into the runtime vocabulary.

    The vision engine maps nothing, so ``normalized_kind`` is absent and its raw
    kind is already a runtime kind.
    """
    normalized = getattr(block, "normalized_kind", None)
    if isinstance(normalized, str) and normalized:
        return normalized
    return raw_kind(block)


def request_page(page: Any) -> int:
    """The page number the runtime parser must see.

    ``render_pdf_pages`` numbers pages ``index + 1 + VISION_PAGE_OFFSET`` and hands
    that number to ``add_page``, so replaying with ``request_page`` reproduces the
    page metadata of a live run.
    """
    value = getattr(page, "request_page", None)
    if not isinstance(value, int) or isinstance(value, bool) or value < 1:
        raise ReplayError(f"request_page must be an int >= 1, got {value!r}")
    return value


def page_index(page: Any) -> int:
    """The 0-based physical PDF page the artifact recorded."""
    value = getattr(page, "page_index", None)
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ReplayError(f"page_index must be an int >= 0, got {value!r}")
    return value


def runtime_block(
    block: Any,
    *,
    kind: str,
    group: int,
    depth: int | None = None,
    text: str | None = None,
    where: str,
) -> VisionBlock:
    """Build the runtime block for one neutral block, validating every field.

    ``group`` is passed in because the caller decides what a missing group means:
    grouping is required to replay a group-carrying artifact, and irrelevant to a
    builder that never groups.
    """
    if kind not in VISION_BLOCK_KINDS:
        raise ReplayError(
            f"{where}: kind {kind!r} is not a runtime vision kind; "
            f"expected one of {sorted(VISION_BLOCK_KINDS)}"
        )
    if not isinstance(group, int) or isinstance(group, bool) or group < 1:
        raise ReplayError(f"{where}: group must be an int >= 1, got {group!r}")

    body = getattr(block, "text", None) if text is None else text
    if not isinstance(body, str):
        raise ReplayError(f"{where}: text must be a string, got {type(body).__name__}")
    body = body.strip()
    if not body and kind != "figure":
        raise ReplayError(f"{where}: text is blank for kind {kind!r}")

    marker = getattr(block, "marker", None)
    if marker is not None:
        if not isinstance(marker, str) or not marker.strip():
            raise ReplayError(f"{where}: marker must be null or non-blank, got {marker!r}")
        marker = marker.strip()

    effective_depth = getattr(block, "depth", None) if depth is None else depth
    if effective_depth is not None and (
        not isinstance(effective_depth, int) or isinstance(effective_depth, bool)
    ):
        raise ReplayError(f"{where}: depth must be null or an int, got {effective_depth!r}")
    if kind == "heading" and (effective_depth is None or effective_depth < 1):
        raise ReplayError(f"{where}: heading requires depth >= 1, got {effective_depth!r}")

    try:
        return VisionBlock(kind=kind, marker=marker, depth=effective_depth, group=group, text=body)
    except ValidationError as exc:
        raise ReplayError(f"{where}: {exc}") from exc


def replay_page(
    page: Any,
    *,
    require_group: bool,
    kind_of: Callable[[Any], str] = mapped_kind,
) -> VisionPage:
    """Rebuild the runtime page for one artifact page.

    ``require_group`` makes the missing-group case explicit: an artifact whose
    engine does not group blocks cannot drive a grouping builder, and a builder
    that ignores groups may number the blocks instead.
    """
    where = f"page_index={page_index(page)}"
    number = request_page(page)
    continues = getattr(page, "continues_previous", None)
    if not isinstance(continues, bool):
        raise ReplayError(f"{where}: continues_previous must be a bool, got {continues!r}")
    running = getattr(page, "running", "")
    if running is None:
        running = ""
    if not isinstance(running, str):
        raise ReplayError(f"{where}: running must be a string, got {type(running).__name__}")

    blocks = getattr(page, "blocks", None)
    if blocks is None:
        raise ReplayError(f"{where}: blocks are missing")
    runtime: list[VisionBlock] = []
    for index, block in enumerate(blocks):
        at = f"{where} block {index}"
        group = getattr(block, "group", None)
        if group is None:
            if require_group:
                raise ReplayError(
                    f"{at}: group is required to replay grouping; this artifact did not record one"
                )
            group = index + 1
        runtime.append(runtime_block(block, kind=kind_of(block), group=group, where=at))

    return VisionPage(page=number, running=running, continues_previous=continues, blocks=runtime)


def replay_pages(
    pages: Sequence[Any],
    *,
    require_group: bool,
    kind_of: Callable[[Any], str] = mapped_kind,
) -> list[VisionPage]:
    """Replay every page up front, so a rejected artifact never half-mutates a tree."""
    return [replay_page(page, require_group=require_group, kind_of=kind_of) for page in pages]


def build_vision_groups(pages: Sequence[Any], title: str) -> BuildResult:
    """Build the tree the runtime parser would build for the same artifact.

    Grouping is the whole point of this builder, so an artifact without recorded
    groups is rejected instead of being silently re-numbered.
    """
    runtime_pages = replay_pages(pages, require_group=True)
    parser = VisionDocumentParser(title)
    for page, runtime in zip(pages, runtime_pages, strict=True):
        parser.add_page(runtime, page_number=request_page(page))
    root = parser.finish()
    return BuildResult(
        root=root,
        builder="vision_groups",
        pages_processed=len(runtime_pages),
        blocks_by_kind=dict(parser.blocks_by_kind),
        counters=dict(parser.normalization),
    )
