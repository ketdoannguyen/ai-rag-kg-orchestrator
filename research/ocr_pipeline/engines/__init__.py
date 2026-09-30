"""Engine adapters: one PDF (or frozen engine output) -> one neutral artifact.

Every adapter in this package turns a single engine's output into the *same*
neutral artifact, so the downstream builders and metrics can treat engines
interchangeably. The artifact holds no database handle, no image bytes, no
provider error body and no secret; it is plain data.

Adapter protocol
----------------
Each engine module exposes:

``ENGINE_ID``
    Registry id, one of :data:`ENGINE_IDS`.
``probe()``
    Cheap, import-light availability check returning :class:`EngineProbe`. It
    never contacts a network endpoint, never loads model weights and never
    imports the engine package.
``extract(...)``
    Run the engine over a document and return an :class:`EngineArtifact`. A
    missing dependency raises :class:`EngineNotAvailableError`, so a missing engine
    can never be mistaken for a successful run.

Use :func:`load_engine` to import an adapter by its registry id.

Neutral data contract (shared with the S1 ``models.py``)
--------------------------------------------------------
:class:`EngineBlock` carries the engine's own kind in ``kind`` and the neutral
kind in ``normalized_kind``, so the Marker reconstruction can still see
``SectionHeader``/``Text`` while the runtime-vocabulary builders replay the
mapped kind. ``group`` is ``None`` for every engine that does not group blocks,
and ``depth`` is only set when the engine reported a heading level.

:class:`EnginePage` is addressed by its 0-based physical PDF index
(``page_index``) and carries the 1-based number the engine was asked for
(``request_page``), so a replay reproduces the runtime parser's page metadata.
Running heads and folios are lifted into ``running``/``printed_folio`` *and*
kept as ``page_header``/``page_footer`` blocks, so their text stays visible to
the fidelity counters while :func:`content_pages` keeps them out of a tree.

Two normalized kinds have no runtime equivalent:

``caption``
    Kept distinct for accounting, replays as a paragraph.
``unmapped``
    A block kind the adapter does not know; its text is kept, never dropped.

``page_header``/``page_footer`` are non-content: they are recorded for
provenance and separate accounting, lifted to the page, and removed by
:func:`content_pages` before a runtime-vocabulary builder replays the artifact.
``not_available``/``failed`` never describe a returned artifact because no
artifact exists for a run that did not happen; adapters raise instead.

Continuity across a selection with gaps
---------------------------------------
``EnginePage.continues_previous`` claims that the page's first block continues
the group left open on the page before it, which is only meaningful when that
page was read. :func:`follows_previous` is that one policy, :func:`continuity_breaks`
names the pages that begin after an unread region, and :func:`reset_continuity`
applies it to the pages of an artifact -- so a replay of a gapped selection opens
a new group there instead of carrying one across a region nobody read.
"""

from __future__ import annotations

import hashlib
import importlib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, NamedTuple, get_args

from workflows.ingest_file.vision_ocr import BlockKind

ARTIFACT_VERSION = 1
"""Layout version of the neutral artifact produced by every engine."""

MAPPING_VERSION = 1
"""Version of the engine -> neutral kind mapping rules; bump when rules change."""

ENGINE_MODULES: Mapping[str, str] = {
    "vision": "vision",
    "marker": "marker",
    "docling": "docling",
    "native": "native_text",
}
"""Registry id -> adapter module name."""

ENGINE_IDS: tuple[str, ...] = tuple(ENGINE_MODULES)
"""Accepted engine ids, also the ``--engine`` values the CLI takes."""

ISOLATED_ENGINES: tuple[str, ...] = ("marker", "docling", "native")
"""Engines whose run belongs in a worker process of its own.

The three local engines own the process that runs them — they load weights, import
a document stack or read a text layer — so a runner that samples its own tree
samples itself and reports its own heap as the engine's. ``vision`` is not here:
it is a hosted stage whose backend memory is not observable at all.
"""

STATUS_OK = "ok"
STATUS_UNSUPPORTED_MODALITY = "unsupported_modality"
STATUS_NOT_AVAILABLE = "not_available"
STATUS_FAILED = "failed"

RETURNED_STATUSES = (STATUS_OK, STATUS_UNSUPPORTED_MODALITY)
"""Statuses a *returned* artifact may carry.

``not_available`` and ``failed`` are never carried by an artifact because no
artifact exists for a run that did not happen; adapters raise instead.
"""

RUNTIME_BLOCK_KINDS: frozenset[str] = frozenset(get_args(BlockKind))
"""Kinds the runtime ``VisionBlock`` accepts, derived from the one vocabulary."""

ARTIFACT_ONLY_KINDS = ("caption", "page_header", "page_footer", "unmapped")
"""Neutral kinds the runtime vocabulary has no equivalent for."""

NORMALIZED_KINDS: tuple[str, ...] = tuple(sorted(RUNTIME_BLOCK_KINDS)) + ARTIFACT_ONLY_KINDS
"""Closed vocabulary of ``EngineBlock.normalized_kind``."""

USAGE_FIELDS = ("attempts", "prompt_tokens", "completion_tokens")
"""Keys of ``EnginePage.usage``; the only per-page engine accounting worth keeping."""

FALLBACK_DEPTH = 1
"""Depth an engine's heading gets when nothing else identifies its level."""


class EngineProbe(NamedTuple):
    """Result of ``probe()``: whether an engine can run, and how it reports itself."""

    engine_id: str
    available: bool
    version: str | None
    detail: str


class EngineNotAvailableError(RuntimeError):
    """Raised when an engine cannot run because a dependency or model is absent.

    ``resources`` and ``worker`` carry what a raise site already observed, the same
    two fields the isolated worker's error carries; a raise site outside a worker
    has nothing to report and leaves them empty.
    """

    status = STATUS_NOT_AVAILABLE

    def __init__(
        self,
        engine_id: str,
        detail: str,
        *,
        resources: Mapping[str, Any] | None = None,
        worker: Mapping[str, Any] | None = None,
    ) -> None:
        super().__init__(f"{engine_id}: {detail}")
        self.engine_id = engine_id
        self.detail = detail
        self.resources: dict[str, Any] = dict(resources or {})
        self.worker: dict[str, Any] = dict(worker or {})


@dataclass(frozen=True)
class EngineBlock:
    """One engine block, stored in reading order within its page."""

    kind: str
    """The kind the engine itself reported (``SectionHeader``, ``text``, ...)."""
    normalized_kind: str
    """Neutral kind; a runtime kind whenever the engine maps into the vocabulary."""
    text: str
    marker: str | None = None
    depth: int | None = None
    """Structural depth of a heading, when the engine reported one."""
    group: int | None = None
    """Consecutive blocks sharing a group belong to one idea; ``None`` when ungrouped."""
    bbox: tuple[float, float, float, float] | None = None
    """Coordinates in the engine's own space, never normalised across engines."""
    source_id: str | None = None
    """Engine provenance id (a Marker block id, a native line range, ...)."""

    def __post_init__(self) -> None:
        if self.normalized_kind not in NORMALIZED_KINDS:
            raise ValueError(f"unknown normalized_kind: {self.normalized_kind!r}")
        if not self.kind:
            raise ValueError("kind must not be empty")
        if self.normalized_kind == "heading" and (self.depth is None or self.depth < 1):
            raise ValueError("heading requires depth >= 1")
        if self.depth is not None and self.depth < 1:
            raise ValueError("depth must be >= 1 when set")
        if self.group is not None and self.group < 1:
            raise ValueError("group must be >= 1 when set")
        # A picture may be a picture with no visible text; every other kind
        # carries its payload here.
        if not self.text.strip() and self.normalized_kind != "figure":
            raise ValueError("text must not be blank")
        if self.bbox is not None and len(self.bbox) != 4:
            raise ValueError("bbox must have four coordinates")


@dataclass(frozen=True)
class EnginePage:
    """One page of an artifact, addressed by its physical index in the PDF."""

    page_index: int
    request_page: int
    """The 1-based number the engine was asked for: ``index + 1 + page_offset``."""
    blocks: tuple[EngineBlock, ...] = ()
    running: str = ""
    """Running head or footer the engine reported for this page."""
    printed_folio: str | None = None
    """The folio printed on the page, when the engine reported one."""
    continues_previous: bool = False
    """True when the first block continues the group left open on the previous page."""
    status: str = STATUS_OK
    """``ok``, or ``unsupported_modality`` for a page the engine cannot read."""
    usage: Mapping[str, int] | None = None
    """Attempts and tokens the engine actually spent on this page, or ``None``."""

    def __post_init__(self) -> None:
        if self.page_index < 0:
            raise ValueError("page_index must be >= 0")
        if self.request_page < 1:
            raise ValueError("request_page must be >= 1")
        if self.status not in RETURNED_STATUSES:
            raise ValueError(f"unknown page status: {self.status!r}")
        if self.usage is not None:
            if set(self.usage) != set(USAGE_FIELDS):
                raise ValueError(f"usage must carry exactly {USAGE_FIELDS}")
            if any(value < 0 for value in self.usage.values()):
                raise ValueError("usage counters must be >= 0")


@dataclass(frozen=True)
class EngineArtifact:
    """One extraction run over a page subset, independent of the engine that ran it.

    Content-addressing, the manifest and the run directory belong to the store
    (S1); this is the adapter's product, which the runner persists.
    """

    engine_id: str
    engine_version: str | None
    doc_id: str
    pages: tuple[EnginePage, ...]
    status: str = STATUS_OK
    pdf_sha256: str | None = None
    source_sha256: str | None = None
    """Digest of the bytes the engine read, when that is not the PDF itself."""
    render_dpi: int | None = None
    render_page_offset: int | None = None
    prompt_sha256: str | None = None
    public_config: Mapping[str, Any] = field(default_factory=dict)
    """Allowlisted, resolved configuration; never a key, header, or full URL."""
    normalization: Mapping[str, int] = field(default_factory=dict)
    """How the adapter mapped the engine output, counted per rule."""
    notes: tuple[str, ...] = ()
    mapping_version: int = MAPPING_VERSION
    artifact_version: int = ARTIFACT_VERSION

    def __post_init__(self) -> None:
        if self.engine_id not in ENGINE_IDS:
            raise ValueError(f"unknown engine_id: {self.engine_id!r}")
        if self.status not in RETURNED_STATUSES:
            raise ValueError(f"unknown artifact status: {self.status!r}")
        if not self.pages:
            raise ValueError("artifact must contain at least one page")
        indices = [page.page_index for page in self.pages]
        if indices != sorted(set(indices)):
            raise ValueError(f"pages must be in strictly increasing order, got {indices!r}")

    @property
    def physical_pages(self) -> tuple[int, ...]:
        """The 0-based physical pages the run covered, in ascending order."""
        return tuple(page.page_index for page in self.pages)

    @property
    def blocks_by_kind(self) -> dict[str, int]:
        """Blocks per neutral kind, sorted so two runs compare byte for byte."""
        return count_block_kinds(self.pages)


def count_block_kinds(pages: Sequence[EnginePage]) -> dict[str, int]:
    """Count blocks per neutral kind over ``pages``."""
    counts: dict[str, int] = {}
    for page in pages:
        for block in page.blocks:
            counts[block.normalized_kind] = counts.get(block.normalized_kind, 0) + 1
    return dict(sorted(counts.items()))


def content_pages(pages: Sequence[EnginePage]) -> tuple[EnginePage, ...]:
    """The pages as a runtime-vocabulary builder can replay them.

    The runtime ``VisionBlock`` accepts only its own kinds, so this is the one
    legal translation of an artifact into that vocabulary: running heads and
    folios are dropped (they are page furniture, already lifted into
    ``running``/``printed_folio``), a caption or an unmapped engine kind keeps
    its words as a paragraph, and everything else is passed through unchanged.
    """
    translated: list[EnginePage] = []
    for page in pages:
        blocks: list[EngineBlock] = []
        for block in page.blocks:
            if block.normalized_kind in ("page_header", "page_footer"):
                continue
            if block.normalized_kind in RUNTIME_BLOCK_KINDS:
                blocks.append(block)
            else:
                blocks.append(replace(block, normalized_kind="p"))
        translated.append(replace(page, blocks=tuple(blocks)))
    return tuple(translated)


def follows_previous(page_index: int, previous_index: int | None) -> bool:
    """Whether a page directly follows the previous page of the run.

    ``page_index`` is a physical index, so ``False`` means the page begins after a
    region the engine never read -- and nothing the engine had open before that
    region can continue across it. A run's first page has no predecessor, so it
    follows nothing.
    """
    return previous_index is not None and page_index == previous_index + 1


def continuity_breaks(pages: Sequence[EnginePage]) -> tuple[int, ...]:
    """The pages that start a region: their predecessor in the document was not read."""
    breaks: list[int] = []
    for position, page in enumerate(pages):
        previous = pages[position - 1].page_index if position else None
        if previous is not None and not follows_previous(page.page_index, previous):
            breaks.append(page.page_index)
    return tuple(breaks)


def reset_continuity(pages: Sequence[EnginePage]) -> tuple[EnginePage, ...]:
    """The pages with ``continues_previous`` cleared where the document was not read.

    This is what a replay of a scoped selection applies before it feeds the pages
    to a runtime parser: the parser would otherwise carry the group left open
    before the gap into a page whose region was never shown to the model.
    """
    reset: list[EnginePage] = []
    previous_index: int | None = None
    for page in pages:
        if page.continues_previous and not follows_previous(page.page_index, previous_index):
            page = replace(page, continues_previous=False)
        reset.append(page)
        previous_index = page.page_index
    return tuple(reset)


def lift_running(blocks: Sequence[EngineBlock]) -> tuple[str, str | None]:
    """Return ``(running head, printed folio)`` for the blocks of one page.

    A running head or folio is page furniture: its text is the page's, not the
    document's. A digits-only header or footer is the printed folio, the rest is
    the running head, joined in reading order. Either may be empty.
    """
    running: list[str] = []
    folio: str | None = None
    for block in blocks:
        if block.normalized_kind not in ("page_header", "page_footer"):
            continue
        text = block.text.strip()
        if not text:
            continue
        if text.isdigit():
            folio = folio or text
        else:
            running.append(text)
    return " ".join(running), folio


def sha256_bytes(data: bytes) -> str:
    """Hex SHA-256 of ``data``."""
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    """Hex SHA-256 of a file, read in chunks so a large PDF stays out of memory."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_engine(engine_id: str) -> Any:
    """Import and return the adapter module registered under ``engine_id``."""
    key = engine_id.strip().lower().replace("-", "_")
    if key not in ENGINE_MODULES:
        raise ValueError(f"unknown engine_id: {engine_id!r}; expected one of {ENGINE_IDS}")
    return importlib.import_module(f"{__name__}.{ENGINE_MODULES[key]}")
