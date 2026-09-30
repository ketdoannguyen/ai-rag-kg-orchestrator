"""Offline tests for manifest loading, PDF validation and artifact alignment.

The PDFs here are small synthetic documents built byte by byte in the test, so
pypdfium2 gets a real file to hash, count pages in and read a text layer from,
without any book being opened and without a network call.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

from research.ocr_pipeline.corpus import (
    CorpusError,
    load_manifest,
    resolve_path,
    sha256_file,
    validate_document,
    validate_manifest,
    verify_artifact,
)
from research.ocr_pipeline.models import DocumentSpec, ExtractionArtifact

SHA = "ab" * 32


def make_pdf(page_count: int, *, text_pages: tuple[int, ...] = ()) -> bytes:
    """Return a valid PDF whose ``text_pages`` carry a text object.

    A page without a text object is what a scanned page looks like to pypdfium2:
    the text layer exists but is empty.
    """
    objects: list[bytes] = []
    font_number = 3 + page_count
    content_numbers = {index: font_number + 1 + index for index in range(page_count)}
    kids = " ".join(f"{3 + index} 0 R" for index in range(page_count))
    objects.append(b"<< /Type /Catalog /Pages 2 0 R >>")
    objects.append(f"<< /Type /Pages /Kids [{kids}] /Count {page_count} >>".encode("ascii"))
    for index in range(page_count):
        contents = f"/Contents {content_numbers[index]} 0 R " if index in text_pages else ""
        objects.append(
            f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
            f"/Resources << /Font << /F1 {font_number} 0 R >> >> {contents}>>".encode("ascii")
        )
    objects.append(b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>")
    for index in range(page_count):
        stream = b"BT /F1 24 Tf 72 700 Td (Trang mot) Tj ET" if index in text_pages else b""
        objects.append(b"<< /Length %d >>\nstream\n" % len(stream) + stream + b"\nendstream")

    document = b"%PDF-1.4\n"
    offsets: list[int] = []
    for number, body in enumerate(objects, start=1):
        offsets.append(len(document))
        document += f"{number} 0 obj\n".encode("ascii") + body + b"\nendobj\n"
    xref_offset = len(document)
    document += f"xref\n0 {len(objects) + 1}\n".encode("ascii")
    document += b"0000000000 65535 f \n"
    for offset in offsets:
        document += f"{offset:010d} 00000 n \n".encode("ascii")
    document += (
        f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\nstartxref\n{xref_offset}\n%%EOF\n"
    ).encode("ascii")
    return document


def write_pdf(
    directory: Path,
    name: str,
    page_count: int,
    *,
    text_pages: tuple[int, ...] = (),
) -> Path:
    """Write a synthetic PDF and return its path."""
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / name
    path.write_bytes(make_pdf(page_count, text_pages=text_pages))
    return path


def document(**overrides: Any) -> dict[str, Any]:
    data: dict[str, Any] = {
        "doc_id": "vat-li-12",
        "pdf_path": "vat-li-12.pdf",
        "pdf_sha256": SHA,
        "page_indices": [0, 1],
        "modality": "scan",
        "permission_note": "local research use only",
        "gold_path": None,
    }
    data.update(overrides)
    return data


def spec_for(path: Path, base_dir: Path, **overrides: Any) -> DocumentSpec:
    """Build a document spec whose checksum matches ``path``."""
    try:
        relative = path.relative_to(base_dir)
    except ValueError:
        relative = path
    return DocumentSpec.model_validate(
        document(pdf_path=str(relative), pdf_sha256=sha256_file(path), **overrides)
    )


def write_manifest(directory: Path, documents: list[dict[str, Any]]) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "manifest.json"
    path.write_text(json.dumps({"corpus_version": 1, "documents": documents}), encoding="utf-8")
    return path


def artifact_dict(**overrides: Any) -> dict[str, Any]:
    """One artifact dictionary, minimal but shaped like an engine's output."""
    data: dict[str, Any] = {
        "artifact_version": 1,
        "mapping_version": 1,
        "engine_id": "native",
        "engine_version": "pypdfium2-4.30.0",
        "status": "ok",
        "doc_id": "vat-li-12",
        "pdf_sha256": SHA,
        "source_sha256": None,
        "physical_pages": [0, 1],
        "render_dpi": None,
        "render_page_offset": None,
        "prompt_sha256": None,
        "public_config": {},
        "pages": [
            {
                "page_index": 0,
                "request_page": 1,
                "printed_folio": None,
                "running": "",
                "continues_previous": False,
                "status": "ok",
                "blocks": [
                    {
                        "order": 0,
                        "raw_kind": "Text",
                        "normalized_kind": "p",
                        "text": "Trang mot",
                        "marker": None,
                        "depth": None,
                        "group": None,
                        "bbox": None,
                        "source_id": None,
                    }
                ],
                "usage": None,
            },
            {
                "page_index": 1,
                "request_page": 2,
                "printed_folio": None,
                "running": "",
                "continues_previous": False,
                "status": "ok",
                "blocks": [
                    {
                        "order": 0,
                        "raw_kind": "Text",
                        "normalized_kind": "p",
                        "text": "Trang hai",
                        "marker": None,
                        "depth": None,
                        "group": None,
                        "bbox": None,
                        "source_id": None,
                    }
                ],
                "usage": None,
            },
        ],
        "statistics": {"blocks_by_kind": {"p": 2}, "normalization": {}},
        "notes": [],
    }
    data.update(overrides)
    return data


# --------------------------------------------------------------------------- #
# hashing and manifest loading
# --------------------------------------------------------------------------- #


def test_sha256_file_matches_hashlib_and_reads_in_chunks(tmp_path: Path) -> None:
    payload = bytes(range(256)) * 64
    path = tmp_path / "blob.bin"
    path.write_bytes(payload)

    assert sha256_file(path) == hashlib.sha256(payload).hexdigest()
    assert sha256_file(path, chunk_size=1) == hashlib.sha256(payload).hexdigest()


def test_load_manifest_reads_a_multi_document_corpus(tmp_path: Path) -> None:
    path = write_manifest(
        tmp_path,
        [document(doc_id="vat-li-12"), document(doc_id="lich-su-12", page_indices=[3])],
    )

    manifest = load_manifest(path)

    assert [entry.doc_id for entry in manifest.documents] == ["vat-li-12", "lich-su-12"]
    assert manifest.documents[0].page_indices == (0, 1)


def test_load_manifest_rejects_a_missing_file(tmp_path: Path) -> None:
    with pytest.raises(CorpusError, match="cannot read manifest"):
        load_manifest(tmp_path / "absent.json")


def test_load_manifest_rejects_malformed_json(tmp_path: Path) -> None:
    path = tmp_path / "manifest.json"
    path.write_text("{not json", encoding="utf-8")

    with pytest.raises(CorpusError, match="not valid JSON"):
        load_manifest(path)


def test_load_manifest_rejects_a_schema_violation(tmp_path: Path) -> None:
    path = write_manifest(tmp_path, [document(modality="vision")])

    with pytest.raises(CorpusError, match="does not match the schema"):
        load_manifest(path)


def test_load_manifest_schema_error_does_not_include_private_input(tmp_path: Path) -> None:
    private_text = "PRIVATE_REFERENCE_TEXT_NOT_FOR_REPORT"
    path = write_manifest(tmp_path, [document(doc_id=private_text + "?", modality="vision")])

    with pytest.raises(CorpusError) as error:
        load_manifest(path)

    assert private_text not in str(error.value)
    assert "doc_id" in str(error.value)


# --------------------------------------------------------------------------- #
# PDF validation: checksum, pages and the declared modality
# --------------------------------------------------------------------------- #


def test_validate_document_accepts_the_declared_page_subset(tmp_path: Path) -> None:
    pdf_path = write_pdf(tmp_path, "vat-li-12.pdf", page_count=3)
    spec = spec_for(pdf_path, tmp_path, page_indices=[0, 2])

    report = validate_document(spec, base_dir=tmp_path)

    assert report.doc_id == "vat-li-12"
    assert report.pdf_path == pdf_path
    assert report.page_count == 3
    assert report.page_indices == (0, 2)
    assert report.pages_with_text == ()


def test_validate_document_rejects_a_checksum_mismatch(tmp_path: Path) -> None:
    pdf_path = write_pdf(tmp_path, "vat-li-12.pdf", page_count=1)
    spec = DocumentSpec.model_validate(
        document(pdf_path=pdf_path.name, pdf_sha256="cd" * 32, page_indices=[0])
    )

    with pytest.raises(CorpusError, match="sha256 mismatch"):
        validate_document(spec, base_dir=tmp_path)


def test_validate_document_rejects_a_missing_pdf(tmp_path: Path) -> None:
    spec = DocumentSpec.model_validate(document(pdf_path="absent.pdf", page_indices=[0]))

    with pytest.raises(CorpusError, match="pdf not found"):
        validate_document(spec, base_dir=tmp_path)


def test_validate_document_rejects_a_page_beyond_the_end(tmp_path: Path) -> None:
    pdf_path = write_pdf(tmp_path, "vat-li-12.pdf", page_count=3)
    spec = spec_for(pdf_path, tmp_path, page_indices=[0, 3])

    with pytest.raises(CorpusError, match=r"page_indices \[3\] out of range"):
        validate_document(spec, base_dir=tmp_path)


def test_validate_document_rejects_bytes_that_are_not_a_pdf(tmp_path: Path) -> None:
    # The checksum matches on purpose, so the failure has to come from pypdfium2.
    path = tmp_path / "broken.pdf"
    path.write_bytes(b"this is not a PDF")
    spec = spec_for(path, tmp_path, page_indices=[0])

    with pytest.raises(CorpusError, match="cannot open"):
        validate_document(spec, base_dir=tmp_path)


def test_validate_document_reports_the_pages_that_have_a_text_layer(tmp_path: Path) -> None:
    pdf_path = write_pdf(tmp_path, "vat-li-12.pdf", page_count=3, text_pages=(0, 2))
    spec = spec_for(pdf_path, tmp_path, page_indices=[0, 1, 2], modality="mixed")

    report = validate_document(spec, base_dir=tmp_path)

    assert report.pages_with_text == (0, 2)


def test_validate_document_accepts_a_fully_digital_document(tmp_path: Path) -> None:
    pdf_path = write_pdf(tmp_path, "vat-li-12.pdf", page_count=2, text_pages=(0, 1))
    spec = spec_for(pdf_path, tmp_path, modality="digital")

    report = validate_document(spec, base_dir=tmp_path)

    assert report.pages_with_text == (0, 1)


def test_validate_document_rejects_a_scan_that_has_a_text_layer(tmp_path: Path) -> None:
    pdf_path = write_pdf(tmp_path, "vat-li-12.pdf", page_count=2, text_pages=(1,))
    spec = spec_for(pdf_path, tmp_path, modality="scan")

    with pytest.raises(CorpusError, match=r"declares modality 'scan'.*\[1\]"):
        validate_document(spec, base_dir=tmp_path)


def test_validate_document_rejects_a_digital_claim_without_a_text_layer(tmp_path: Path) -> None:
    pdf_path = write_pdf(tmp_path, "vat-li-12.pdf", page_count=2, text_pages=(0,))
    spec = spec_for(pdf_path, tmp_path, modality="digital")

    with pytest.raises(CorpusError, match=r"declares modality 'digital'.*\[1\]"):
        validate_document(spec, base_dir=tmp_path)


@pytest.mark.parametrize("text_pages", [(), (0, 1)])
def test_validate_document_rejects_a_mixed_claim_that_is_not_mixed(
    tmp_path: Path, text_pages: tuple[int, ...]
) -> None:
    pdf_path = write_pdf(tmp_path, "vat-li-12.pdf", page_count=2, text_pages=text_pages)
    spec = spec_for(pdf_path, tmp_path, modality="mixed")

    with pytest.raises(CorpusError, match="declares modality 'mixed'"):
        validate_document(spec, base_dir=tmp_path)


def test_validate_document_checks_the_modality_of_the_selected_pages_only(
    tmp_path: Path,
) -> None:
    # Page 1 has a text layer but is not in scope, so the document is a scan of
    # the pages under study.
    pdf_path = write_pdf(tmp_path, "vat-li-12.pdf", page_count=2, text_pages=(1,))
    spec = spec_for(pdf_path, tmp_path, page_indices=[0], modality="scan")

    report = validate_document(spec, base_dir=tmp_path)

    assert report.pages_with_text == ()


def test_validate_document_resolves_relative_paths_against_the_manifest_directory(
    tmp_path: Path,
) -> None:
    corpus_dir = tmp_path / "corpus"
    pdf_path = write_pdf(corpus_dir / "docs", "vat-li-12.pdf", page_count=2)
    spec = DocumentSpec.model_validate(
        document(pdf_path="docs/vat-li-12.pdf", pdf_sha256=sha256_file(pdf_path))
    )

    assert validate_document(spec, base_dir=corpus_dir).pdf_path == pdf_path


def test_validate_document_accepts_an_absolute_path(tmp_path: Path) -> None:
    pdf_path = write_pdf(tmp_path / "elsewhere", "vat-li-12.pdf", page_count=2)
    spec = DocumentSpec.model_validate(
        document(pdf_path=str(pdf_path), pdf_sha256=sha256_file(pdf_path))
    )

    assert validate_document(spec, base_dir=tmp_path / "corpus").pdf_path == pdf_path


def test_validate_manifest_reports_every_document_in_order(tmp_path: Path) -> None:
    first = write_pdf(tmp_path, "vat-li-12.pdf", page_count=2)
    second = write_pdf(tmp_path, "lich-su-12.pdf", page_count=4, text_pages=(0, 3))
    manifest = load_manifest(
        write_manifest(
            tmp_path,
            [
                document(pdf_path=first.name, pdf_sha256=sha256_file(first)),
                document(
                    doc_id="lich-su-12",
                    pdf_path=second.name,
                    pdf_sha256=sha256_file(second),
                    page_indices=[0, 3],
                    modality="digital",
                ),
            ],
        )
    )

    reports = validate_manifest(manifest, base_dir=tmp_path)

    assert [report.doc_id for report in reports] == ["vat-li-12", "lich-su-12"]
    assert [report.page_count for report in reports] == [2, 4]
    assert reports[1].pages_with_text == (0, 3)


def test_validate_manifest_fails_on_the_first_bad_document(tmp_path: Path) -> None:
    good = write_pdf(tmp_path, "vat-li-12.pdf", page_count=2)
    bad = write_pdf(tmp_path, "lich-su-12.pdf", page_count=1)
    manifest = load_manifest(
        write_manifest(
            tmp_path,
            [
                document(pdf_path=good.name, pdf_sha256=sha256_file(good)),
                document(
                    doc_id="lich-su-12",
                    pdf_path=bad.name,
                    pdf_sha256=sha256_file(bad),
                    page_indices=[5],
                ),
            ],
        )
    )

    with pytest.raises(CorpusError, match="lich-su-12"):
        validate_manifest(manifest, base_dir=tmp_path)


def test_resolve_path_joins_relative_and_keeps_absolute_paths(tmp_path: Path) -> None:
    assert resolve_path("docs/book.pdf", tmp_path) == tmp_path / "docs" / "book.pdf"

    absolute = tmp_path / "elsewhere" / "book.pdf"
    assert resolve_path(str(absolute), tmp_path) == absolute


# --------------------------------------------------------------------------- #
# artifact-to-document verification
# --------------------------------------------------------------------------- #


def test_verify_artifact_accepts_a_run_over_the_declared_pages() -> None:
    spec = DocumentSpec.model_validate(document())

    verify_artifact(ExtractionArtifact.model_validate(artifact_dict()), spec)


def test_verify_artifact_rejects_a_document_it_does_not_belong_to() -> None:
    artifact = ExtractionArtifact.model_validate(artifact_dict())
    spec = DocumentSpec.model_validate(document(doc_id="lich-su-12"))

    with pytest.raises(CorpusError, match="lich-su-12"):
        verify_artifact(artifact, spec)


def test_verify_artifact_rejects_another_pdf_revision() -> None:
    artifact = ExtractionArtifact.model_validate(artifact_dict())
    spec = DocumentSpec.model_validate(document(pdf_sha256="cd" * 32))

    with pytest.raises(CorpusError, match="pdf_sha256"):
        verify_artifact(artifact, spec)


def test_verify_artifact_rejects_a_run_that_does_not_record_the_pdf() -> None:
    spec = DocumentSpec.model_validate(document())

    with pytest.raises(CorpusError, match="records no pdf_sha256"):
        verify_artifact(ExtractionArtifact.model_validate(artifact_dict(pdf_sha256=None)), spec)


def test_verify_artifact_rejects_missing_pages_from_requested_selection() -> None:
    payload = artifact_dict()
    payload["pages"] = payload["pages"][:1]
    payload["physical_pages"] = [0]
    payload["statistics"] = {"blocks_by_kind": {"p": 1}, "normalization": {}}
    artifact = ExtractionArtifact.model_validate(payload)
    spec = DocumentSpec.model_validate(document(page_indices=[0, 1]))

    with pytest.raises(CorpusError, match="requested pages"):
        verify_artifact(artifact, spec, expected_pages=[0, 1])

    verify_artifact(artifact, spec, expected_pages=[0])


def test_verify_artifact_rejects_a_page_the_document_does_not_select() -> None:
    artifact = ExtractionArtifact.model_validate(
        artifact_dict(
            pages=[
                {
                    "page_index": 7,
                    "request_page": 8,
                    "status": "ok",
                    "blocks": [],
                }
            ],
            physical_pages=[7],
            statistics={"blocks_by_kind": {}, "normalization": {}},
        )
    )
    spec = DocumentSpec.model_validate(document(page_indices=[0, 1]))

    with pytest.raises(CorpusError, match=r"pages \[7\]"):
        verify_artifact(artifact, spec)
