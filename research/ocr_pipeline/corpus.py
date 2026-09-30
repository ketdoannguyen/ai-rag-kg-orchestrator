"""Manifest loading and fail-closed validation of the PDFs a manifest names.

The manifest is public config: it states which PDF sits where, which pages are
in scope, and under which permission. Validation is deliberately shallow — it
hashes the file and asks pypdfium2 how many pages the document has and which of
them carry a text layer — because it must run before any expensive extraction
and must never reach the network.

The declared modality is re-checked against the real text layer, because it
decides whether an engine can be trusted on the page: a document claiming
``scan`` must have no text layer on any page in scope, ``digital`` must have one
everywhere, and ``mixed`` must have both. A manifest that overstates its
documents fails here, before a run spends anything on them.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import pypdfium2  # type: ignore[import-untyped]
from pydantic import ValidationError

from .models import DocumentSpec, ExtractionArtifact, Manifest, Modality

CHUNK_SIZE = 1 << 20
"""Bytes read per iteration while hashing a PDF."""


class CorpusError(RuntimeError):
    """Raised when a manifest cannot be loaded or a document fails validation."""


@dataclass(frozen=True)
class DocumentReport:
    """What validation established about one document."""

    doc_id: str
    pdf_path: Path
    """Resolved path of the PDF that was hashed."""
    page_count: int
    """Physical page count of the PDF according to pypdfium2."""
    page_indices: tuple[int, ...]
    """Validated 0-based page indices in scope, strictly increasing."""
    pages_with_text: tuple[int, ...]
    """Pages in scope whose own text layer is non-empty, strictly increasing."""


def sha256_file(path: str | Path, *, chunk_size: int = CHUNK_SIZE) -> str:
    """Return the sha256 hex digest of a file, read in chunks."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_path(value: str | Path, base_dir: str | Path) -> Path:
    """Resolve a manifest path, treating relative paths as relative to the manifest."""
    candidate = Path(value)
    return candidate if candidate.is_absolute() else Path(base_dir) / candidate


def load_manifest(path: str | Path) -> Manifest:
    """Load a manifest from disk and validate it against the schema.

    Raises:
        CorpusError: if the file cannot be read, is not valid JSON, or does not
            match :class:`~research.ocr_pipeline.models.Manifest`.
    """
    manifest_path = Path(path)
    try:
        raw = manifest_path.read_text(encoding="utf-8")
    except OSError as exc:
        raise CorpusError(f"cannot read manifest {manifest_path}: {exc}") from exc
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise CorpusError(f"manifest {manifest_path} is not valid JSON: {exc}") from exc
    try:
        return Manifest.model_validate(payload)
    except ValidationError as exc:
        locations = [
            ".".join(str(part) for part in error["loc"])
            for error in exc.errors(include_input=False)
        ]
        raise CorpusError(
            f"manifest {manifest_path} does not match the schema at {locations}"
        ) from None


def validate_document(spec: DocumentSpec, *, base_dir: str | Path) -> DocumentReport:
    """Check one document's checksum, page range and declared modality.

    Args:
        spec: Document entry as declared in the manifest.
        base_dir: Directory relative paths in ``spec`` resolve against, normally
            the manifest's own directory.

    Returns:
        The resolved path, physical page count, validated page indices, and the
        pages in scope that carry a text layer.

    Raises:
        CorpusError: if the PDF is missing, hashes to something other than
            ``pdf_sha256``, cannot be opened, does not contain every declared
            page index, or its text layer contradicts ``spec.modality``.
    """
    document_path = resolve_path(spec.pdf_path, base_dir)
    if not document_path.is_file():
        raise CorpusError(f"{spec.doc_id}: pdf not found at {document_path}")

    digest = sha256_file(document_path)
    if digest != spec.pdf_sha256:
        raise CorpusError(
            f"{spec.doc_id}: sha256 mismatch for {document_path}: "
            f"declared {spec.pdf_sha256}, found {digest}"
        )

    try:
        document = pypdfium2.PdfDocument(str(document_path))
    except pypdfium2.PdfiumError as exc:
        raise CorpusError(f"{spec.doc_id}: cannot open {document_path}: {exc}") from exc
    try:
        page_count = len(document)
        too_far = [index for index in spec.page_indices if index >= page_count]
        if too_far:
            raise CorpusError(
                f"{spec.doc_id}: page_indices {too_far} out of range for "
                f"{document_path} with {page_count} page(s)"
            )
        pages_with_text = tuple(
            index for index in spec.page_indices if _page_text(document[index]).strip()
        )
    finally:
        document.close()

    _check_modality(spec, pages_with_text=pages_with_text)
    return DocumentReport(
        doc_id=spec.doc_id,
        pdf_path=document_path,
        page_count=page_count,
        page_indices=spec.page_indices,
        pages_with_text=pages_with_text,
    )


def validate_manifest(manifest: Manifest, *, base_dir: str | Path) -> tuple[DocumentReport, ...]:
    """Validate every document in manifest order.

    Raises:
        CorpusError: on the first document that fails validation.
    """
    return tuple(validate_document(spec, base_dir=base_dir) for spec in manifest.documents)


def verify_artifact(
    artifact: ExtractionArtifact,
    spec: DocumentSpec,
    *,
    expected_pages: Sequence[int] | None = None,
) -> None:
    """Check the artifact's document, PDF revision and requested page selection.

    ``expected_pages`` may be a subset of the manifest selection, but an artifact
    must cover that request exactly; missing pages cannot count as a successful
    extraction or be silently scored as a complete run.
    """
    if artifact.doc_id != spec.doc_id:
        raise CorpusError(
            f"artifact is for {artifact.doc_id!r}, but the manifest document is {spec.doc_id!r}"
        )
    if artifact.pdf_sha256 is None:
        raise CorpusError(
            f"{spec.doc_id}: artifact records no pdf_sha256, so it cannot be tied to a PDF revision"
        )
    if artifact.pdf_sha256 != spec.pdf_sha256:
        raise CorpusError(
            f"{spec.doc_id}: artifact pdf_sha256 {artifact.pdf_sha256} does not match the "
            f"manifest's {spec.pdf_sha256}"
        )
    stray = sorted(set(artifact.physical_pages) - set(spec.page_indices))
    if stray:
        raise CorpusError(
            f"{spec.doc_id}: artifact covers pages {stray} the manifest does not select "
            f"{list(spec.page_indices)}"
        )
    if expected_pages is not None and tuple(artifact.physical_pages) != tuple(expected_pages):
        raise CorpusError(
            f"{spec.doc_id}: artifact pages {list(artifact.physical_pages)} do not match "
            f"requested pages {list(expected_pages)}"
        )


def _page_text(pdfium_page: pypdfium2.PdfPage) -> str:
    """The page's own text layer; empty when the page is a scan."""
    textpage = pdfium_page.get_textpage()
    try:
        return textpage.get_text_bounded()
    finally:
        textpage.close()


def _check_modality(spec: DocumentSpec, *, pages_with_text: tuple[int, ...]) -> None:
    """Fail when the text layer of the pages in scope contradicts the manifest."""
    with_text = list(pages_with_text)
    selected = list(spec.page_indices)
    if spec.modality is Modality.SCAN and with_text:
        raise CorpusError(
            f"{spec.doc_id}: declares modality 'scan' but pages {with_text} have a text layer"
        )
    if spec.modality is Modality.DIGITAL:
        without_text = [index for index in spec.page_indices if index not in set(pages_with_text)]
        if without_text:
            raise CorpusError(
                f"{spec.doc_id}: declares modality 'digital' but pages {without_text} "
                "have no text layer"
            )
    if spec.modality is Modality.MIXED and (not with_text or len(with_text) == len(selected)):
        raise CorpusError(
            f"{spec.doc_id}: declares modality 'mixed' but the selected pages {selected} "
            "are not mixed"
        )
