"""Docling: a real page scope, a lazy import, and an honest refusal when it is absent.

Docling is not an application dependency: it runs in an interpreter of its own,
named through ``OCR_RESEARCH_ENGINE_PYTHON``, because its pins and the app's
cannot coexist. What it reads was checked against the upstream sources
(``docling-project/docling`` and ``docling-project/docling-core``, ``main``) on
2026-09-28: ``DocumentConverter.convert(source, ..., page_range: PageRange =
DEFAULT_PAGE_RANGE)``, where ``PageRange`` is ``Annotated[Tuple[int, int], ...]``
validated as a 1-based inclusive range; ``ConversionStatus`` values ``success``,
``partial_success``, ``failure`` and ``skipped``; ``Document.pages`` keyed by the
1-based ``int`` page number; ``iterate_items()`` yielding ``(item, level)``;
``TableItem.export_to_markdown(doc=None)``; and the ``DocItemLabel`` vocabulary the
map below covers. The mapping is exercised by the tests with a synthetic document
that follows those shapes.

Two real runs have been through this mapping, both ``docling==2.130.0``: a
synthetic three-page digital PDF, and the ten-page scanned corpus this study
exists for. Both honoured their scope through the converter's own ``page_range``
and returned every page ``ok``; the scan produced 216 blocks and reached
``heading``, ``list``, ``caption``, ``figure`` and ``text`` with nothing left
unmapped. What that does *not* establish is correctness: no page of the corpus
carries human gold, so no label here has been checked against a person's reading
of a page, and ``table``, ``formula``, ``page_header`` and ``page_footer`` have
never been reached at all -- nor has a partial or a multi-run conversion. So
:func:`probe` still reports ``not_available`` where Docling is missing rather
than guessing, every artifact carries the note below, and the probed version
belongs in the run manifest: a version bump must not silently change a mapping
this thinly covered.

Page scope
----------
A scoped run never converts the whole PDF and throws pages away afterwards: the
engine would have read, and been measured on, every page outside the scope. The
converter's own ``convert`` signature is read first. A converter that declares
``page_range`` is asked for one contiguous run of the selection at a time, and each
answer is checked to cover exactly that run; a converter that declares none is
handed a selected-page PDF built for the selection, and its pages are mapped back
to their physical indices. Either way a conversion that does not cover the scope
fails the run, a page the conversion did not produce is ``unsupported_modality``
rather than an empty page, and an item whose provenance names no page is refused
instead of being reported as page 0.
"""

from __future__ import annotations

import importlib.metadata
import inspect
import tempfile
from collections.abc import Collection, Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

from . import (
    STATUS_OK,
    STATUS_UNSUPPORTED_MODALITY,
    EngineArtifact,
    EngineBlock,
    EngineNotAvailableError,
    EnginePage,
    EngineProbe,
    lift_running,
    sha256_file,
)

ENGINE_ID = "docling"

DISTRIBUTION = "docling"
"""The distribution that provides the ``docling`` package."""

PAGE_RANGE_ARGUMENT = "page_range"
"""The ``convert`` parameter a converter must declare to be given a real page range."""

DOCLING_KIND_MAP: dict[str, str] = {
    "title": "heading",
    "section_header": "heading",
    "text": "p",
    "paragraph": "p",
    "list_item": "list",
    "table": "table",
    "picture": "figure",
    "formula": "equation",
    "caption": "caption",
    "page_header": "page_header",
    "page_footer": "page_footer",
}
"""Documented ``DocItemLabel`` value -> neutral kind, mapping v1."""

CONVERSION_PARTIAL = "partial_success"
CONVERSION_FAILURE = "failure"
CONVERSION_SKIPPED = "skipped"
"""Documented ``ConversionStatus`` values that mean the conversion is not complete."""

MAPPING_NOTE = (
    "mapping v1: a real ten-page scan reached heading, list, caption, figure and text "
    "with nothing unmapped; whether each is right is unmeasured -- no human gold"
)


class DoclingConversionError(RuntimeError):
    """Docling reported a conversion it did not complete."""

    def __init__(self, status: str, detail: str = "") -> None:
        super().__init__(f"docling reported a {status} conversion{detail}")
        self.status = status


def probe() -> EngineProbe:
    """Report availability from distribution metadata alone, importing nothing."""
    version = docling_version()
    if version is None:
        return EngineProbe(ENGINE_ID, False, None, f"{DISTRIBUTION} is not installed")
    return EngineProbe(
        engine_id=ENGINE_ID,
        available=True,
        version=version,
        detail="docling is installed; the mapping is v1 and is only as verified as the last run",
    )


def docling_version() -> str | None:
    """The installed Docling version, or ``None`` when the package is absent."""
    try:
        return importlib.metadata.version(DISTRIBUTION)
    except importlib.metadata.PackageNotFoundError:
        return None


def build_converter() -> Any:
    """Build a real ``DocumentConverter``, importing Docling for the first time."""
    try:
        from docling.document_converter import DocumentConverter  # type: ignore[import-not-found]
    except ImportError as exc:
        raise EngineNotAvailableError(ENGINE_ID, f"docling could not be imported: {exc}") from exc
    return DocumentConverter()


# --------------------------------------------------------------------------- #
# reading one document
# --------------------------------------------------------------------------- #


def _label(item: Any) -> str:
    """One item's label as the documented lowercase value, or an empty string."""
    value = getattr(item, "label", None)
    # DocItemLabel is a str enum, so the value is the documented spelling; a
    # plain string label passes through unchanged.
    if value is None:
        return ""
    return str(getattr(value, "value", value)).strip().lower()


def _parameters(callable_: Any) -> dict[str, inspect.Parameter]:
    """The declared parameters of a callable, empty when it declares none."""
    try:
        return dict(inspect.signature(callable_).parameters)
    except (TypeError, ValueError):  # a builtin or a C callable declares nothing
        return {}


def _markdown(item: Any, document: Any) -> str:
    """One item's markdown rendering, or an empty string when it has none.

    Docling's table exporter takes the document the table belongs to -- as an
    optional argument whose absence is the deprecated path -- so the call is
    shaped by the callable's own signature: an exporter that declares a positional
    argument is handed the document, and one that declares none is called with
    nothing rather than with an argument it never asked for.
    """
    export = getattr(item, "export_to_markdown", None)
    if not callable(export):
        return ""
    takes_document = any(
        parameter.kind in (parameter.POSITIONAL_ONLY, parameter.POSITIONAL_OR_KEYWORD)
        for parameter in _parameters(export).values()
    )
    rendered = export(document) if takes_document else export()
    return rendered.strip() if isinstance(rendered, str) else ""


def _text(item: Any, document: Any) -> str:
    """One item's text, falling back to a table's markdown rendering.

    ``text`` carries a text item's payload. A table has no plain text of its own,
    so its markdown rendering is the closest thing to one. An exporter that raises
    is not swallowed: a table this adapter cannot read is not a table it may drop.
    """
    value = getattr(item, "text", None)
    if isinstance(value, str) and value.strip():
        return value.strip()
    return _markdown(item, document)


def _page_number(item: Any) -> int | None:
    """The 1-based page number an item's first provenance entry names, or ``None``.

    Docling numbers provenance pages from 1. ``None`` means the item names no page
    at all -- no provenance, or an entry without a usable ``page_no``.
    """
    prov = getattr(item, "prov", None)
    if not isinstance(prov, Iterable):
        return None
    for entry in prov:
        page_no = getattr(entry, "page_no", None)
        if isinstance(page_no, int) and not isinstance(page_no, bool) and page_no >= 1:
            return page_no
    return None


def _bbox(item: Any) -> tuple[float, float, float, float] | None:
    """One item's first provenance bbox as four floats, or ``None``."""
    prov = getattr(item, "prov", None)
    if not isinstance(prov, Iterable):
        return None
    for entry in prov:
        bbox = getattr(entry, "bbox", None)
        if bbox is None:
            continue
        box = getattr(bbox, "as_tuple", None)
        try:
            values = tuple(float(value) for value in (box() if callable(box) else bbox))
        except (TypeError, ValueError):
            continue
        if len(values) == 4:
            return (values[0], values[1], values[2], values[3])
    return None


def _source_id(item: Any) -> str | None:
    """One item's self reference, when it has one."""
    reference = getattr(item, "self_ref", None)
    return reference if isinstance(reference, str) and reference else None


def iter_items(document: Any) -> list[tuple[Any, int]]:
    """Every item of a Docling document with its level, in the engine's order."""
    iterate = getattr(document, "iterate_items", None)
    if not callable(iterate):
        raise TypeError("Docling document has no iterate_items()")
    items: list[tuple[Any, int]] = []
    for entry in iterate():
        if not isinstance(entry, tuple) or len(entry) != 2:
            raise TypeError(f"iterate_items() must yield (item, level), got {entry!r}")
        item, level = entry
        items.append((item, level if isinstance(level, int) else 0))
    return items


def _bump(counters: dict[str, int], key: str) -> None:
    counters[key] = counters.get(key, 0) + 1


# --------------------------------------------------------------------------- #
# one conversion -> physical pages
# --------------------------------------------------------------------------- #


def _physical_page(page_no: int, page_map: Sequence[int] | None) -> int:
    """The physical page a 1-based Docling page number names under a run's mapping.

    ``page_map`` names the physical page each page of a scoped conversion came
    from, so entry ``k`` is the physical index of the converted page ``k + 1``;
    ``None`` means the conversion ran over the whole PDF and its own numbers are
    the physical ones.
    """
    if page_no < 1:
        raise ValueError(f"a Docling page number must be >= 1, got {page_no}")
    if page_map is None:
        return page_no - 1
    if page_no > len(page_map):
        raise ValueError(
            f"Docling named page {page_no}, but only {len(page_map)} pages were converted"
        )
    return page_map[page_no - 1]


def _physical(item: Any, page_map: Sequence[int] | None) -> int:
    """The physical page one item came from, refused when it names no page."""
    page_no = _page_number(item)
    if page_no is None:
        raise ValueError(
            f"a Docling item carries no provenance page (source_id={_source_id(item)!r}); "
            "refusing to report it as page 0"
        )
    return _physical_page(page_no, page_map)


def _blocks_by_page(
    document: Any,
    *,
    page_map: Sequence[int] | None,
    counters: dict[str, int],
) -> dict[int, list[EngineBlock]]:
    """Every item of one converted document, keyed by the physical page it came from.

    Blocks keep the engine's reading order within their page. An item whose label
    mapping v1 does not know keeps its text as an unmapped block and is counted by
    label; an item with no text at all is dropped and counted.
    """
    by_page: dict[int, list[EngineBlock]] = {}
    for item, level in iter_items(document):
        index = _physical(item, page_map)
        label = _label(item)
        normalized = DOCLING_KIND_MAP.get(label)
        if normalized is None:
            normalized = "unmapped"
            _bump(counters, "docling_unmapped_blocks")
            _bump(counters, f"docling_unmapped_kind:{label or 'none'}")
        text = _text(item, document)
        if not text and normalized != "figure":
            _bump(counters, "docling_empty_blocks")
            continue
        depth = max(1, level) if normalized == "heading" else None
        by_page.setdefault(index, []).append(
            EngineBlock(
                kind=label or "none",
                normalized_kind=normalized,
                text=text,
                marker=None,
                depth=depth,
                group=None,
                bbox=_bbox(item),
                source_id=_source_id(item),
            )
        )
    return by_page


def _converted_pages(
    document: Any,
    *,
    page_map: Sequence[int] | None,
    by_page: Mapping[int, Sequence[EngineBlock]],
) -> set[int]:
    """The physical pages a conversion produced.

    Docling's page table is the engine's own statement of what it converted; a
    document that carries none is read from the items it produced instead. A page
    in scope that is missing here is one the engine did not convert, and the
    artifact says so rather than showing it as a page with no content.
    """
    table = getattr(document, "pages", None)
    if not isinstance(table, Mapping) or not table:
        return set(by_page)
    converted: set[int] = set()
    for key in table:
        if not isinstance(key, int) or isinstance(key, bool):
            raise TypeError(f"a Docling page table must be keyed by page number, got {key!r}")
        converted.add(_physical_page(key, page_map))
    return converted


def _conversion_status(result: Any) -> str | None:
    """Docling's own verdict on the conversion, lowercased; ``None`` when it states none."""
    value = getattr(result, "status", None)
    if value is None:
        return None
    # ConversionStatus is a str enum, so ``value`` is the documented spelling.
    text = str(getattr(value, "value", value)).strip().lower()
    return text or None


def _accept_result(
    result: Any,
    path: Path,
    *,
    counters: dict[str, int],
    notes: list[str],
) -> Any:
    """The document of a conversion that ran, refusing one Docling says it did not complete."""
    document = getattr(result, "document", None)
    if document is None:
        raise TypeError(f"{path}: Docling result has no document: {type(result).__name__}")
    status = _conversion_status(result)
    if status in (CONVERSION_FAILURE, CONVERSION_SKIPPED):
        raise DoclingConversionError(status, f" for {path}")
    if status == CONVERSION_PARTIAL:
        counters["docling_partial_conversion"] = 1
        notes.append(
            f"docling reported a partial conversion (status={status}); a page it did not "
            "produce is unsupported_modality, not a page with no content"
        )
    return document


def _artifact(
    *,
    doc_id: str,
    engine_version: str | None,
    pdf_sha256: str | None,
    page_offset: int,
    by_page: Mapping[int, Sequence[EngineBlock]],
    expected_pages: Sequence[int],
    converted_pages: Collection[int],
    counters: dict[str, int],
    notes: Sequence[str],
) -> EngineArtifact:
    """The neutral artifact of one run: every page in scope, converted or not.

    A page the conversion did not produce is ``unsupported_modality`` with no
    blocks -- explicit about being unread, never silently absent and never filled
    with content the engine did not report.
    """
    pages: list[EnginePage] = []
    for index in sorted(set(expected_pages)):
        blocks = tuple(by_page.get(index, ()))
        status = STATUS_OK
        if index not in converted_pages:
            status = STATUS_UNSUPPORTED_MODALITY
            _bump(counters, "docling_pages_not_converted")
        running, folio = lift_running(blocks)
        pages.append(
            EnginePage(
                page_index=index,
                request_page=index + 1 + page_offset,
                blocks=blocks,
                running=running,
                printed_folio=folio,
                status=status,
            )
        )
    if not pages:
        raise ValueError("Docling document has no pages")
    counters["docling_pages"] = len(pages)
    counters["docling_blocks"] = sum(len(page.blocks) for page in pages)
    return EngineArtifact(
        engine_id=ENGINE_ID,
        engine_version=engine_version,
        doc_id=doc_id,
        pages=tuple(pages),
        pdf_sha256=pdf_sha256,
        render_page_offset=page_offset,
        normalization=dict(sorted(counters.items())),
        notes=(MAPPING_NOTE, *notes),
    )


def convert_document(
    document: Any,
    *,
    doc_id: str,
    engine_version: str | None = None,
    pdf_sha256: str | None = None,
    page_offset: int = 0,
    page_map: Sequence[int] | None = None,
) -> EngineArtifact:
    """Turn one already-converted Docling document into a neutral artifact.

    ``page_map`` names the physical page each page of ``document`` came from, for a
    conversion that ran over a selected-page PDF: entry ``k`` is the physical index
    of the converted page ``k + 1``. ``None`` means the conversion ran over the
    source PDF itself, so the document's own page numbers are the physical ones.
    """
    counters: dict[str, int] = {}
    by_page = _blocks_by_page(document, page_map=page_map, counters=counters)
    converted = _converted_pages(document, page_map=page_map, by_page=by_page)
    return _artifact(
        doc_id=doc_id,
        engine_version=engine_version,
        pdf_sha256=pdf_sha256,
        page_offset=page_offset,
        by_page=by_page,
        expected_pages=sorted(converted),
        converted_pages=converted,
        counters=counters,
        notes=(),
    )


# --------------------------------------------------------------------------- #
# page scope
# --------------------------------------------------------------------------- #


def _pdf_page_count(path: Path) -> int:
    """How many pages the PDF has, read by the same renderer the runtime uses."""
    try:
        import pypdfium2  # type: ignore[import-untyped]
    except ImportError as exc:
        raise EngineNotAvailableError(
            ENGINE_ID, f"pypdfium2 could not be imported: {exc}"
        ) from exc
    try:
        document = pypdfium2.PdfDocument(str(path))
    except Exception as exc:  # a file that is not a PDF cannot be scoped at all
        raise ValueError(f"{path}: not a readable PDF: {exc}") from exc
    try:
        return len(document)
    finally:
        document.close()


def _contiguous_runs(selection: Sequence[int]) -> list[list[int]]:
    """Split a selection into runs of consecutive pages, in selection order."""
    runs: list[list[int]] = []
    for index in selection:
        if runs and index == runs[-1][-1] + 1:
            runs[-1].append(index)
        else:
            runs.append([index])
    return runs


def _validate_selection(path: Path, page_indices: Sequence[int]) -> list[int]:
    """The pages a scoped run covers, checked against the PDF's own page count."""
    count = _pdf_page_count(path)
    selection: list[int] = []
    for index in page_indices:
        if not isinstance(index, int) or isinstance(index, bool) or index < 0:
            raise ValueError(f"page index must be an int >= 0, got {index!r}")
        if index >= count:
            raise ValueError(f"page index {index} is outside the PDF ({count} pages)")
        if index in selection:
            raise ValueError(f"page index {index} is selected twice")
        selection.append(index)
    if not selection:
        raise ValueError("the page selection is empty")
    return selection


def _accepts_page_range(converter: Any) -> bool:
    """Whether this converter's ``convert`` declares a page-range argument.

    Read from the callable's own signature, so a converter that cannot scope is
    given a selected-page PDF instead of an argument it would not understand.
    """
    convert = getattr(converter, "convert", None)
    if not callable(convert):
        raise TypeError(
            f"a Docling converter needs a convert() method, got {type(converter).__name__}"
        )
    return PAGE_RANGE_ARGUMENT in _parameters(convert)


def _selected_pdf(path: Path, selection: Sequence[int], *, directory: Path) -> Path:
    """A PDF holding only ``selection``, in selection order, for a converter that cannot scope.

    Built with the same renderer the runtime uses, so the pages are the pages the
    run would otherwise have read. The caller owns the temporary directory.
    """
    try:
        import pypdfium2
    except ImportError as exc:
        raise EngineNotAvailableError(
            ENGINE_ID, f"pypdfium2 could not be imported: {exc}"
        ) from exc
    source = pypdfium2.PdfDocument(str(path))
    try:
        selected = pypdfium2.PdfDocument.new()
        try:
            selected.import_pages(source, pages=list(selection))
            target = directory / "selected.pdf"
            selected.save(str(target))
        finally:
            selected.close()
    finally:
        source.close()
    return target


def extract(
    source: str | Path,
    *,
    doc_id: str,
    page_indices: Sequence[int] | None = None,
    page_offset: int = 0,
    converter: Any | None = None,
) -> EngineArtifact:
    """Convert a source with Docling and map its document into a neutral artifact.

    Without an installed Docling this raises :class:`EngineNotAvailableError`, so a
    missing engine is never reported as an empty run. ``page_indices`` is a real
    scope: the engine converts only those pages, one contiguous run at a time, and
    a conversion that does not cover them fails the run.
    """
    path = Path(source)
    engine = converter if converter is not None else build_converter()
    version = docling_version()
    digest = sha256_file(path) if path.is_file() else None

    if page_indices is None:
        counters: dict[str, int] = {}
        notes: list[str] = []
        result = engine.convert(str(path))
        document = _accept_result(result, path, counters=counters, notes=notes)
        by_page = _blocks_by_page(document, page_map=None, counters=counters)
        converted = _converted_pages(document, page_map=None, by_page=by_page)
        return _artifact(
            doc_id=doc_id,
            engine_version=version,
            pdf_sha256=digest,
            page_offset=page_offset,
            by_page=by_page,
            expected_pages=sorted(converted),
            converted_pages=converted,
            counters=counters,
            notes=notes,
        )

    selection = _validate_selection(path, page_indices)
    counters = {}
    notes = []
    by_page = {}
    converted = set()
    if _accepts_page_range(engine):
        runs = _contiguous_runs(selection)
        for run in runs:
            result = engine.convert(str(path), page_range=(run[0] + 1, run[-1] + 1))
            document = _accept_result(result, path, counters=counters, notes=notes)
            run_blocks = _blocks_by_page(document, page_map=None, counters=counters)
            covered = _converted_pages(document, page_map=None, by_page=run_blocks)
            if covered != set(run):
                raise ValueError(
                    f"{path}: the converter returned pages {sorted(covered)} for a run of "
                    f"{run}; the page scope was not honoured"
                )
            by_page.update(run_blocks)
            converted |= covered
        counters["docling_page_runs"] = len(runs)
        counters["docling_context_resets"] = len(runs) - 1
        notes.append(
            f"page scope: {len(runs)} contiguous run(s) requested through the converter's own "
            f"{PAGE_RANGE_ARGUMENT} over physical pages {selection}"
        )
    else:
        with tempfile.TemporaryDirectory(prefix="docling-scope-") as work:
            scoped = _selected_pdf(path, selection, directory=Path(work))
            result = engine.convert(str(scoped))
            document = _accept_result(result, path, counters=counters, notes=notes)
            by_page.update(_blocks_by_page(document, page_map=selection, counters=counters))
            converted = _converted_pages(document, page_map=selection, by_page=by_page)
        counters["docling_page_runs"] = 1
        counters["docling_context_resets"] = 0
        notes.append(
            f"page scope: the converter declares no {PAGE_RANGE_ARGUMENT}, so physical pages "
            f"{selection} were converted as a selected-page PDF and mapped back by index"
        )
    return _artifact(
        doc_id=doc_id,
        engine_version=version,
        pdf_sha256=digest,
        page_offset=page_offset,
        by_page=by_page,
        expected_pages=selection,
        converted_pages=converted,
        counters=counters,
        notes=notes,
    )
