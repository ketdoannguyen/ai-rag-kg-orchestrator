"""Live Marker: imported only when a run happens, converted through its JSON renderer.

Marker is imported lazily, asked for its JSON renderer, and the rendered document
is mapped to the neutral artifact here -- one mapping, in one place, so a live run
cannot mean anything the rest of the pipeline does not expect.

What a real run costs, and why nothing here runs by default
-----------------------------------------------------------
``create_model_dict`` builds seven predictor objects, and each one downloads its
weights the first time it is used; ``BaseConverter.__init__`` also downloads a
render font. :func:`probe` therefore reads the distribution metadata alone and
never imports Marker, and the run happens inside :func:`extract`, where a caller
has asked for it.
"""

from __future__ import annotations

import html
import importlib.metadata
import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from . import (
    FALLBACK_DEPTH,
    EngineArtifact,
    EngineBlock,
    EngineNotAvailableError,
    EnginePage,
    EngineProbe,
    lift_running,
    sha256_file,
)

ENGINE_ID = "marker"

DISTRIBUTION = "marker-pdf"
"""The distribution that provides the ``marker`` package."""

RENDERER = "marker.renderers.json.JSONRenderer"
"""Renderer Marker resolves by dotted path, so this module imports no Marker class."""

MARKER_KIND_MAP: dict[str, str] = {
    "SectionHeader": "heading",
    "Text": "p",
    "ListGroup": "list",
    "Table": "table",
    "Picture": "figure",
    "Figure": "figure",
    "Caption": "caption",
    "Equation": "equation",
    "PageHeader": "page_header",
    "PageFooter": "page_footer",
}
"""Marker block type -> neutral kind, mapping v1.

``caption`` and the two running-head kinds are artifact-only: the runtime
vocabulary has no equivalent for them, and :func:`..content_pages` translates or
drops them before a runtime builder replays the artifact.
"""

_PAGE_TYPE = "Page"

_PAGE_ID_RE = re.compile(r"^/page/(\d+)/")
_H_TAG_RE = re.compile(r"<h(\d+)[^>]*>", re.IGNORECASE)
_BR_RE = re.compile(r"<br\s*/?>", re.IGNORECASE)
_CELL_ROW_END_RE = re.compile(r"</(?:td|th)>\s*</(?:tr|tbody|table)>", re.IGNORECASE)
_CELL_END_RE = re.compile(r"</(?:td|th)>", re.IGNORECASE)
_BLOCK_END_RE = re.compile(r"</(?:li|p|div|tr|blockquote|h\d)>", re.IGNORECASE)
_TAG_RE = re.compile(r"<[^>]*>")


# --------------------------------------------------------------------------- #
# the engine
# --------------------------------------------------------------------------- #


def probe() -> EngineProbe:
    """Report availability from distribution metadata alone.

    No import, no network, no weight check: ``available`` says the package is
    installed, and only a run can settle whether the weights are on disk.
    """
    version = marker_version()
    if version is None:
        return EngineProbe(ENGINE_ID, False, None, f"{DISTRIBUTION} is not installed")
    return EngineProbe(
        engine_id=ENGINE_ID,
        available=True,
        version=version,
        detail="marker is installed; a run downloads any weights and the render font it lacks",
    )


def marker_version() -> str | None:
    """The installed Marker version, or ``None`` when the package is absent."""
    try:
        return importlib.metadata.version(DISTRIBUTION)
    except importlib.metadata.PackageNotFoundError:
        return None


def build_converter(
    *,
    page_indices: Sequence[int] | None = None,
    config: Mapping[str, Any] | None = None,
) -> Any:
    """Build a real Marker JSON converter, importing Marker for the first time.

    ``page_indices`` becomes Marker's own ``page_range``, so a scoped run never
    processes a page outside the scope. This is the network-touching step: every
    predictor without cached weights downloads them here.
    """
    try:
        from marker.converters.pdf import PdfConverter  # type: ignore[import-untyped]
        from marker.models import create_model_dict  # type: ignore[import-untyped]
    except ImportError as exc:
        raise EngineNotAvailableError(ENGINE_ID, f"marker could not be imported: {exc}") from exc

    settings: dict[str, Any] = dict(config or {})
    if page_indices is not None:
        settings["page_range"] = list(page_indices)
    return PdfConverter(
        artifact_dict=create_model_dict(),
        renderer=RENDERER,
        config=settings,
    )


def render_document(converter: Any, pdf_path: str | Path) -> Mapping[str, Any]:
    """Render one PDF and return the document as the plain mapping the converter reads."""
    output = converter(str(pdf_path))
    if isinstance(output, Mapping):
        return output
    dump = getattr(output, "model_dump", None)
    if callable(dump):
        document = dump(mode="json")
        if isinstance(document, Mapping):
            return document
    raise TypeError(
        f"Marker output must be a mapping or a pydantic model, got {type(output).__name__}"
    )


def extract(
    pdf_path: str | Path,
    *,
    doc_id: str,
    page_indices: Sequence[int] | None = None,
    page_offset: int = 0,
    converter: Any | None = None,
    config: Mapping[str, Any] | None = None,
) -> EngineArtifact:
    """Run Marker over a PDF and convert its JSON output into a neutral artifact.

    A caller that already owns a converter passes it as ``converter`` and nothing
    is imported or downloaded; otherwise this function builds one and the run pays
    for the weights it lacks.

    ``page_indices`` is applied to the rendered dump as well as to a built converter,
    so an injected converter cannot widen the scope: a page the selection names that
    the dump does not hold fails the run instead of shrinking it silently.
    """
    source = Path(pdf_path)
    engine = (
        converter
        if converter is not None
        else build_converter(page_indices=page_indices, config=config)
    )
    document = render_document(engine, source)
    return convert(
        document,
        doc_id=doc_id,
        engine_version=marker_version(),
        pdf_sha256=sha256_file(source),
        page_offset=page_offset,
        page_indices=page_indices,
    )


# --------------------------------------------------------------------------- #
# the mapping: Marker's document dump -> neutral blocks
# --------------------------------------------------------------------------- #


def html_text(markup: Any) -> str:
    """The visible text of one block's html, one line per markup text run.

    Structure becomes newlines (list items, paragraphs, table rows) and a cell
    boundary becomes ``" | "``, which is the row syntax the runtime table parser
    reads back; the last cell of a row ends the line without a separator of its
    own, because a trailing one would be text no cell contains. Tags are removed
    before anything is kept, so an ``<img>`` with a base64 source contributes
    nothing.
    """
    if not isinstance(markup, str):
        return ""
    text = _BR_RE.sub("\n", markup)
    text = _CELL_ROW_END_RE.sub("\n", text)
    text = _CELL_END_RE.sub(" | ", text)
    text = _BLOCK_END_RE.sub("\n", text)
    text = _TAG_RE.sub("", text)
    text = html.unescape(text)
    lines = (" ".join(line.split()) for line in text.splitlines())
    return "\n".join(line for line in lines if line)


def _hierarchy_levels(value: Any) -> tuple[int, ...]:
    """The heading levels in a ``section_hierarchy``, ignoring unusable keys.

    Marker writes the map with integer level keys, which JSON turns into strings;
    anything that is not a mapping has no levels at all.
    """
    if not isinstance(value, Mapping):
        return ()
    levels = []
    for key in value:
        try:
            level = int(key)
        except (TypeError, ValueError):
            continue
        if level >= 1:
            levels.append(level)
    return tuple(levels)


def heading_depth(node: Mapping[str, Any], counters: dict[str, int]) -> int:
    """The depth of one section header, from the most faithful source available.

    ``section_hierarchy`` is ``{level: block id}`` over the headings open at that
    point, and its deepest key is the level Marker itself assigned, so it wins
    when it is not empty; ``h1``..``h9`` is Marker's own structure decision and is
    used when the hierarchy is empty; a header that carries neither takes the v1
    fallback depth and is counted.
    """
    levels = _hierarchy_levels(node.get("section_hierarchy"))
    if levels:
        _bump(counters, "marker_depth_from_hierarchy")
        return max(levels)
    tag = _H_TAG_RE.search(node.get("html") or "")
    if tag is not None and int(tag.group(1)) >= 1:
        _bump(counters, "marker_depth_from_htag")
        return int(tag.group(1))
    _bump(counters, "marker_depth_fallback")
    return FALLBACK_DEPTH


def _bump(counters: dict[str, int], key: str) -> None:
    counters[key] = counters.get(key, 0) + 1


def _bbox(value: Any) -> tuple[float, float, float, float] | None:
    """One Marker bbox as four floats, or ``None`` when it is not one."""
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        return None
    try:
        return (float(value[0]), float(value[1]), float(value[2]), float(value[3]))
    except (TypeError, ValueError):
        return None


def block(node: Mapping[str, Any], counters: dict[str, int]) -> EngineBlock | None:
    """One page child as a neutral block, or ``None`` when it holds no text.

    A kind outside mapping v1 keeps its text as an unmapped block and is counted
    by name, so an unfamiliar Marker version shows up in the counters instead of
    silently losing content. A picture with no text stays, because a picture is
    allowed to have none; an empty text block is dropped and counted rather than
    stored.

    Each block's payload is read from its own html -- ``<h2>`` for a section
    header, ``<p>`` for text, a caption or an equation, one ``<li>`` per list
    item, the assembled ``<table>`` for a table, a bare ``<img>`` for a picture.
    A table's children are its cells, so this never descends into a child:
    ``images`` is never read either, and tags are stripped before a block's text
    is taken, so no base64 payload can reach an artifact or a tracked fixture.
    """
    raw = node.get("block_type")
    if not isinstance(raw, str) or not raw:
        raise ValueError(f"block {node.get('id')!r} has no block_type")
    normalized = MARKER_KIND_MAP.get(raw)
    if normalized is None:
        normalized = "unmapped"
        _bump(counters, "marker_unmapped_blocks")
        _bump(counters, f"marker_unmapped_kind:{raw}")
    text = html_text(node.get("html"))
    if not text and normalized != "figure":
        _bump(counters, "marker_empty_blocks")
        return None
    source_id = node.get("id")
    return EngineBlock(
        kind=raw,
        normalized_kind=normalized,
        text=text,
        marker=None,
        depth=heading_depth(node, counters) if normalized == "heading" else None,
        group=None,
        bbox=_bbox(node.get("bbox")),
        source_id=source_id if isinstance(source_id, str) else None,
    )


def page(
    node: Mapping[str, Any],
    *,
    index: int,
    page_offset: int,
    counters: dict[str, int],
) -> EnginePage:
    """One page child as a neutral page, with its running head and folio lifted."""
    blocks: list[EngineBlock] = []
    for child in _children(node):
        converted = block(child, counters)
        if converted is not None:
            blocks.append(converted)
    running, folio = lift_running(blocks)
    return EnginePage(
        page_index=index,
        request_page=index + 1 + page_offset,
        blocks=tuple(blocks),
        running=running,
        printed_folio=folio,
    )


def _children(node: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    """The block children of one page node, as mappings."""
    children = node.get("children")
    if not isinstance(children, list):
        return []
    return [child for child in children if isinstance(child, Mapping)]


def _page_nodes(document: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    """Every page node of a dump, in document order."""
    children = document.get("children")
    if not isinstance(children, list):
        raise ValueError("Marker dump has no children list")
    nodes = [child for child in children if isinstance(child, Mapping)]
    unexpected = sorted(
        {str(node.get("block_type")) for node in nodes if node.get("block_type") != _PAGE_TYPE}
    )
    if unexpected:
        raise ValueError(f"Marker dump has non-page root children: {unexpected}")
    if not nodes:
        raise ValueError("Marker dump has no pages")
    return nodes


def physical_index(node: Mapping[str, Any], position: int) -> int:
    """The physical PDF index of one page node, read from its Marker block id.

    Marker puts the index in every id (``/page/3/Page/1``) and keeps it the
    physical index even when the run processed a page subset, so a scoped dump
    still numbers its pages the way the PDF does. ``position`` is only the
    fallback for a document whose ids were rewritten.
    """
    match = _PAGE_ID_RE.match(str(node.get("id") or ""))
    if match is None:
        return position
    return int(match.group(1))


def _select(
    nodes: Sequence[Mapping[str, Any]], page_indices: Sequence[int] | None
) -> list[tuple[int, Mapping[str, Any]]]:
    """The ``(physical index, page node)`` pairs a run covers, in the given order.

    ``page_indices`` addresses pages the way every other engine does, by their
    physical index, because a dump from a scoped run holds only the scoped pages
    and each one's id still says where it sits in the PDF.
    """
    indexed = [(physical_index(node, position), node) for position, node in enumerate(nodes)]
    if page_indices is None:
        return indexed
    by_index = dict(indexed)
    selected = []
    for index in page_indices:
        if not isinstance(index, int) or isinstance(index, bool) or index < 0:
            raise ValueError(f"page index must be an int >= 0, got {index!r}")
        if index not in by_index:
            raise ValueError(f"page index {index} is not a page in the dump ({len(nodes)} pages)")
        selected.append((index, by_index[index]))
    return selected


def convert(
    document: Mapping[str, Any],
    *,
    doc_id: str,
    engine_version: str | None = None,
    pdf_sha256: str | None = None,
    source_sha256: str | None = None,
    page_offset: int = 0,
    page_indices: Sequence[int] | None = None,
) -> EngineArtifact:
    """Turn one Marker document dump into a neutral artifact.

    ``page_indices`` names the pages in scope by their physical index in the PDF,
    exactly as the runtime takes them. Each selected page keeps the physical index
    its own block id records, so the artifact of a subset lines up with the
    artifact of the whole document.
    """
    nodes = _page_nodes(document)
    counters: dict[str, int] = {}
    pages = tuple(
        page(node, index=index, page_offset=page_offset, counters=counters)
        for index, node in _select(nodes, page_indices)
    )
    counters["marker_pages"] = len(pages)
    counters["marker_blocks"] = sum(len(item.blocks) for item in pages)
    return EngineArtifact(
        engine_id=ENGINE_ID,
        engine_version=engine_version,
        doc_id=doc_id,
        pages=pages,
        pdf_sha256=pdf_sha256,
        source_sha256=source_sha256,
        render_page_offset=page_offset,
        normalization=dict(sorted(counters.items())),
    )
