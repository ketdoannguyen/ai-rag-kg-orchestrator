"""The PDF's own text layer: paragraphs, and an honest refusal where there is no text.

This engine reads what the PDF already carries instead of recognising it, so it is
free, exact where it works, and simply absent on a scanned page. It never falls
back to OCR: a page without a text layer is reported ``unsupported_modality`` with
no blocks, which is the fact a comparison needs, and the run continues to the next
page instead of pretending the page was empty or filling it with a guess.

A page's text comes back as running lines of a layout, not as a document. A blank
line ends a paragraph and lines inside one are joined with a single space, because
a line break in the text layer is where the typesetter put it rather than where the
sentence ends; printed hyphenation is left exactly as printed.
"""

from __future__ import annotations

import importlib.metadata
from collections.abc import Sequence
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
    sha256_file,
)

ENGINE_ID = "native"

DISTRIBUTION = "pypdfium2"
"""The distribution that provides the PDF text layer used here."""


def probe() -> EngineProbe:
    """Report availability from distribution metadata alone, importing nothing."""
    version = pypdfium2_version()
    if version is None:
        return EngineProbe(ENGINE_ID, False, None, f"{DISTRIBUTION} is not installed")
    return EngineProbe(
        engine_id=ENGINE_ID,
        available=True,
        version=version,
        detail="reads the PDF text layer; no OCR fallback and no model",
    )


def pypdfium2_version() -> str | None:
    """The installed pypdfium2 version, or ``None`` when the package is absent."""
    try:
        return importlib.metadata.version(DISTRIBUTION)
    except importlib.metadata.PackageNotFoundError:
        return None


def paragraphs(text: str) -> list[str]:
    """Split a page's text layer into paragraphs at blank lines."""
    found: list[str] = []
    current: list[str] = []
    for line in text.splitlines():
        stripped = line.strip()
        if stripped:
            current.append(stripped)
        elif current:
            found.append(" ".join(current))
            current = []
    if current:
        found.append(" ".join(current))
    return found


def _bump(counters: dict[str, int], key: str) -> None:
    counters[key] = counters.get(key, 0) + 1


def _page(
    pdfium_page: Any, *, index: int, page_offset: int, counters: dict[str, int]
) -> EnginePage:
    """One page of the text layer, or an explicit unsupported-modality page."""
    number = index + 1 + page_offset
    textpage = pdfium_page.get_textpage()
    try:
        text = textpage.get_text_range()
    finally:
        textpage.close()

    found = paragraphs(text) if isinstance(text, str) else []
    if not found:
        _bump(counters, "native_pages_without_text")
        return EnginePage(
            page_index=index,
            request_page=number,
            status=STATUS_UNSUPPORTED_MODALITY,
        )
    blocks = tuple(
        EngineBlock(kind="text", normalized_kind="p", text=paragraph, source_id=None)
        for paragraph in found
    )
    return EnginePage(
        page_index=index,
        request_page=number,
        blocks=blocks,
        status=STATUS_OK,
    )


def _select(page_count: int, page_indices: Sequence[int] | None) -> list[int]:
    """The pages a run covers, in ascending order and inside the document."""
    if page_count == 0:
        raise ValueError("PDF has no pages")
    if page_indices is None:
        return list(range(page_count))
    selected = []
    for index in page_indices:
        if not isinstance(index, int) or isinstance(index, bool) or index < 0:
            raise ValueError(f"page index must be an int >= 0, got {index!r}")
        if index >= page_count:
            raise ValueError(f"page index {index} is outside the PDF ({page_count} pages)")
        selected.append(index)
    return selected


def extract(
    pdf_path: str | Path,
    *,
    doc_id: str,
    page_indices: Sequence[int] | None = None,
    page_offset: int = 0,
) -> EngineArtifact:
    """Read the text layer of one PDF into a neutral artifact.

    No rendering, no OCR and no network: a page whose text layer is empty is
    recorded as ``unsupported_modality`` so the artifact says which pages this
    engine could not read, rather than leaving a hole a reader cannot see.
    """
    try:
        import pypdfium2  # type: ignore[import-untyped]
    except ImportError as exc:
        raise EngineNotAvailableError(
            ENGINE_ID, f"pypdfium2 could not be imported: {exc}"
        ) from exc

    path = Path(pdf_path)
    counters: dict[str, int] = {}
    document = pypdfium2.PdfDocument(str(path))
    try:
        pages = []
        for index in _select(len(document), page_indices):
            pdfium_page = document[index]
            try:
                pages.append(
                    _page(pdfium_page, index=index, page_offset=page_offset, counters=counters)
                )
            finally:
                pdfium_page.close()
    finally:
        document.close()

    counters["native_pages"] = len(pages)
    counters["native_blocks"] = sum(len(item.blocks) for item in pages)
    return EngineArtifact(
        engine_id=ENGINE_ID,
        engine_version=pypdfium2_version(),
        doc_id=doc_id,
        pages=tuple(pages),
        pdf_sha256=sha256_file(path),
        render_page_offset=page_offset,
        normalization=dict(sorted(counters.items())),
        notes=("no OCR fallback: a page without a text layer stays unsupported_modality",),
    )
