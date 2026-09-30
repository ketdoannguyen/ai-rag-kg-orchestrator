"""Offline tests for the content-addressed, doc-scoped artifact store.

The store is the only writer in the research pipeline: it names a file after the
SHA-256 of the bytes inside it, so a run never overwrites another run's evidence
and two identical runs cost one file. These tests exercise the layout, the round
trip, tamper detection and the write-once guarantee, using a temporary root and
hand-built artifacts only.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

from research.ocr_pipeline.models import ExtractionArtifact, canonical_payload
from research.ocr_pipeline.store import (
    ArtifactIntegrityError,
    ArtifactNotFound,
    ArtifactStore,
    ArtifactStoreError,
    write_once,
)

SHA = "ab" * 32
ADDRESS = "0" * 64


def artifact(**overrides: Any) -> ExtractionArtifact:
    """One artifact, minimal but complete."""
    data: dict[str, Any] = {
        "artifact_version": 1,
        "mapping_version": 1,
        "engine_id": "vision",
        "engine_version": "vision-2026-09",
        "status": "ok",
        "doc_id": "vat-li-12",
        "pdf_sha256": SHA,
        "source_sha256": None,
        "physical_pages": [0],
        "render_dpi": 200,
        "render_page_offset": 4,
        "prompt_sha256": None,
        "public_config": {"model": "vision-model"},
        "pages": [
            {
                "page_index": 0,
                "request_page": 5,
                "printed_folio": "4",
                "running": "",
                "continues_previous": False,
                "status": "ok",
                "blocks": [
                    {
                        "order": 0,
                        "raw_kind": "Text",
                        "normalized_kind": "p",
                        "text": "Nội dung",
                        "marker": None,
                        "depth": None,
                        "group": None,
                        "bbox": None,
                        "source_id": None,
                    }
                ],
                "usage": None,
            }
        ],
        "statistics": {"blocks_by_kind": {"p": 1}, "normalization": {}},
        "notes": [],
    }
    data.update(overrides)
    return ExtractionArtifact.model_validate(data)


def artifacts_dir(root: Path, doc_id: str = "vat-li-12") -> Path:
    return root / doc_id / "artifacts"


# --------------------------------------------------------------------------- #
# layout and addressing
# --------------------------------------------------------------------------- #


def test_put_addresses_the_artifact_by_its_content_digest(tmp_path: Path) -> None:
    root = tmp_path / "data"
    stored = artifact()

    address = ArtifactStore(root).put(stored)

    target = artifacts_dir(root) / f"{address}.json"
    assert address == stored.content_digest
    assert len(address) == 64
    assert target.read_bytes() == canonical_payload(stored).encode("utf-8")
    assert hashlib.sha256(target.read_bytes()).hexdigest() == address


def test_put_creates_the_document_directory(tmp_path: Path) -> None:
    root = tmp_path / "data"

    address = ArtifactStore(root).put(artifact())

    assert artifacts_dir(root).is_dir()
    assert ArtifactStore(root).path_for("vat-li-12", address) == artifacts_dir(root) / (
        f"{address}.json"
    )


def test_two_documents_keep_their_artifacts_apart(tmp_path: Path) -> None:
    root = tmp_path / "data"
    store = ArtifactStore(root)

    first = store.put(artifact())
    second = store.put(artifact(doc_id="lich-su-12"))

    assert first != second  # the artifact names its document, so it is its own payload
    assert (artifacts_dir(root, "vat-li-12") / f"{first}.json").is_file()
    assert (artifacts_dir(root, "lich-su-12") / f"{second}.json").is_file()


def test_stored_bytes_carry_no_run_identity(tmp_path: Path) -> None:
    root = tmp_path / "data"

    address = ArtifactStore(root).put(artifact())

    payload = (artifacts_dir(root) / f"{address}.json").read_bytes()
    assert b"run_id" not in payload
    assert b"created_at" not in payload
    assert b"timestamp" not in payload


def test_a_repeat_run_that_read_the_same_thing_is_the_same_artifact(tmp_path: Path) -> None:
    root = tmp_path / "data"
    store = ArtifactStore(root)

    first = store.put(artifact())
    second = store.put(artifact())

    assert first == second
    assert [path.name for path in artifacts_dir(root).iterdir()] == [f"{first}.json"]


def test_an_extraction_that_read_something_else_is_a_new_address(tmp_path: Path) -> None:
    root = tmp_path / "data"
    store = ArtifactStore(root)
    other_page = {
        "page_index": 0,
        "request_page": 5,
        "status": "ok",
        "blocks": [
            {
                "order": 0,
                "raw_kind": "Text",
                "normalized_kind": "p",
                "text": "Đoạn khác",
                "marker": None,
                "depth": None,
                "group": None,
                "bbox": None,
                "source_id": None,
            }
        ],
    }
    first = store.put(artifact())
    second = store.put(artifact(pages=[other_page]))

    assert first != second
    assert sorted(path.name for path in artifacts_dir(root).iterdir()) == sorted(
        [f"{first}.json", f"{second}.json"]
    )


# --------------------------------------------------------------------------- #
# round trip
# --------------------------------------------------------------------------- #


def test_get_returns_the_artifact_that_was_stored(tmp_path: Path) -> None:
    root = tmp_path / "data"
    store = ArtifactStore(root)
    stored = artifact()

    address = store.put(stored)

    assert store.get("vat-li-12", address) == stored
    assert store.get("vat-li-12", address).content_digest == address


def test_get_reports_a_missing_artifact(tmp_path: Path) -> None:
    with pytest.raises(ArtifactNotFound, match=ADDRESS):
        ArtifactStore(tmp_path / "data").get("vat-li-12", ADDRESS)


def test_get_rejects_json_that_is_not_a_valid_artifact(tmp_path: Path) -> None:
    root = tmp_path / "data"
    target = artifacts_dir(root) / f"{ADDRESS}.json"
    target.parent.mkdir(parents=True)
    target.write_bytes(b'{"pages": []}')

    with pytest.raises(ArtifactIntegrityError, match="not a valid artifact"):
        ArtifactStore(root).get("vat-li-12", ADDRESS)


def test_get_detects_an_artifact_that_was_edited_in_place(tmp_path: Path) -> None:
    root = tmp_path / "data"
    store = ArtifactStore(root)
    address = store.put(artifact())
    target = store.path_for("vat-li-12", address)
    edited = canonical_payload(artifact()).replace("Nội dung", "Nội dung bị sửa")

    # Rewrite the file with different content under the old name: the artifact
    # no longer hashes to the address it was fetched by.
    target.write_text(edited, encoding="utf-8")

    with pytest.raises(ArtifactIntegrityError, match="not to its address"):
        store.get("vat-li-12", address)


def test_get_rejects_a_reformatted_artifact_at_a_valid_address(tmp_path: Path) -> None:
    root = tmp_path / "data"
    store = ArtifactStore(root)
    address = store.put(artifact())
    target = store.path_for("vat-li-12", address)

    # Same content, pretty-printed: the address names the bytes, not the object.
    payload = json.loads(target.read_text(encoding="utf-8"))
    target.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    with pytest.raises(ArtifactIntegrityError, match="not to its address"):
        store.get("vat-li-12", address)


def test_put_refuses_to_overwrite_bytes_at_an_existing_address(tmp_path: Path) -> None:
    root = tmp_path / "data"
    store = ArtifactStore(root)
    address = store.put(artifact())
    target = store.path_for("vat-li-12", address)
    target.write_bytes(b"someone else's bytes")

    with pytest.raises(ArtifactIntegrityError, match="refusing to overwrite"):
        store.put(artifact())

    assert target.read_bytes() == b"someone else's bytes"


def test_put_leaves_no_temporary_files_behind(tmp_path: Path) -> None:
    root = tmp_path / "data"

    address = ArtifactStore(root).put(artifact())

    assert [path.name for path in artifacts_dir(root).iterdir()] == [f"{address}.json"]


# --------------------------------------------------------------------------- #
# the store refuses addresses and doc ids it cannot trust
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "address",
    ["", "abc", "A" * 64, "z" * 64, "0x" + "a" * 62, "../" + "a" * 61, "a" * 63, "a" * 65],
)
def test_addresses_that_are_not_a_sha256_are_refused(tmp_path: Path, address: str) -> None:
    store = ArtifactStore(tmp_path / "data")

    with pytest.raises(ArtifactStoreError) as error:
        store.get("vat-li-12", address)

    assert not isinstance(error.value, ArtifactNotFound)


@pytest.mark.parametrize("doc_id", ["../etc", "..", ".", "", "vat/li", "vat li", "/abs"])
def test_doc_ids_that_could_escape_the_root_are_refused(tmp_path: Path, doc_id: str) -> None:
    store = ArtifactStore(tmp_path / "data")

    with pytest.raises(ArtifactStoreError):
        store.path_for(doc_id, ADDRESS)

    with pytest.raises(ArtifactStoreError):
        store.get(doc_id, ADDRESS)


# --------------------------------------------------------------------------- #
# the plain write-once helper
# --------------------------------------------------------------------------- #


def test_write_once_creates_the_file_and_its_parents(tmp_path: Path) -> None:
    path = tmp_path / "reports" / "run.json"

    write_once(path, b'{"measured": 1}')

    assert path.read_bytes() == b'{"measured": 1}'


def test_write_once_is_a_no_op_when_the_bytes_are_identical(tmp_path: Path) -> None:
    path = tmp_path / "report.json"
    write_once(path, b"first")

    write_once(path, b"first")

    assert path.read_bytes() == b"first"


def test_write_once_refuses_to_replace_different_bytes(tmp_path: Path) -> None:
    path = tmp_path / "report.json"
    write_once(path, b"first")

    with pytest.raises(ArtifactStoreError, match="refusing to overwrite"):
        write_once(path, b"second")

    assert path.read_bytes() == b"first"
