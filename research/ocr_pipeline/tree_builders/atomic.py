"""The atomic builder: one textual block, one detail node, no merging.

The strategy question this builder answers is what the tree looks like when the
engine's grouping is thrown away. Blocks are never joined, so a one-line block
stays a one-line detail, and the only relation left between blocks is the heading
chain they sit under. It reuses the runtime parser's normalization, heading,
table and size-splitting logic rather than forking it, so a comparison against
the grouped builder measures chunking and nothing else.

Two rules of the grouped builder are deliberately dropped:

* a short group is never absorbed into the previous detail;
* ``continues_previous`` is ignored, because joining across a page boundary is
  the merging this builder exists to remove.

Headings still open sections across page boundaries, so both builders share the
same structure and differ only in what becomes one child.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from typing import Any

from workflows.ingest_file.vision_ocr import VisionBlock, VisionPage
from workflows.ingest_file.vision_parser import VisionDocumentParser, _split_oversized

from .vision_groups import BuildResult, replay_pages, request_page


class AtomicTreeBuilder(VisionDocumentParser):
    """Every textual block becomes one detail in the section open at that point.

    Detail text is split until each node fits ``MAX_CHUNK_SIZE``, which the parser
    does with the same sentence-then-character rules it uses for oversized groups.
    """

    NAME = "atomic"

    # First page this builder has seen; typed here because the runtime parser's own
    # declaration is not visible when its module is skipped by the scoped type gate.
    _first_page: int | None

    def __init__(self, title: str) -> None:
        super().__init__(title)
        self.counters: Counter[str] = Counter()

    def add_page(self, page: VisionPage, *, page_number: int) -> None:
        self._open_page(page_number)
        for block in page.blocks:
            self._add_block(block, page_number)

    def _open_page(self, page_number: int) -> None:
        self._page = page_number
        if self._first_page is None:
            self._first_page = page_number
            self.root.metadata.start_page = page_number

    def _add_block(self, block: VisionBlock, page_number: int) -> None:
        """Feed one runtime block through the shared heading/table/detail path."""
        if not block.text:
            # A picture with no visible text has no payload to index.
            self.normalization["textless_figures_dropped"] += 1
            return
        kind, text, depth = self._normalize(block)
        self.blocks_by_kind[kind] += 1
        if kind == "heading":
            self._add_heading(text, block.marker, depth, page_number)
        elif kind == "table":
            self.counters["tables_created"] += 1
            self._stack[-1].children.append(self._build_table(text, page_number))
        else:
            self._add_detail(text, page_number)

    def _add_detail(self, text: str, page_number: int) -> None:
        """Append one detail per input piece; nothing is absorbed, nothing regroups."""
        parent = self._stack[-1]
        pieces = _split_oversized(text)
        self.counters["details_created"] += len(pieces)
        self.counters["details_split"] += len(pieces) - 1
        for piece in pieces:
            parent.children.append(self._paragraph(piece, page_number))


def build_atomic(pages: Sequence[Any], title: str) -> BuildResult:
    """Build the atomic tree for one neutral artifact."""
    runtime_pages = replay_pages(pages, require_group=False)
    builder = AtomicTreeBuilder(title)
    for page, runtime in zip(pages, runtime_pages, strict=True):
        builder.add_page(runtime, page_number=request_page(page))
    root = builder.finish()
    return BuildResult(
        root=root,
        builder=AtomicTreeBuilder.NAME,
        pages_processed=len(runtime_pages),
        blocks_by_kind=dict(builder.blocks_by_kind),
        counters={**builder.normalization, **builder.counters},
    )
