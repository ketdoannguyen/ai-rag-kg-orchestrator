"""Offline tests for the gold file, its coverage and gold/artifact alignment.

A gold file is the simple JSON a human hands the study:

.. code-block:: json

    {"doc_id": "vl12-sample",
     "pages": [{"page_index": 0, "text": "…", "headings": ["Bài 1"]}]}

It is taken as checked the moment it loads, so the contract under test is
strictness: the file must name its document, its headings must occur in its own
transcripts, and a transcript is never blank -- a blank one would score every
character a run read as an insertion.

No PDF is opened, no network is touched, and no book text is read.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from research.ocr_pipeline.models import ExtractionArtifact, page_text
from research.ocr_pipeline.reference import (
    GoldError,
    GoldPage,
    GoldReference,
    align_pages,
    coverage,
    load_gold,
    metrics_document,
)

SHA = "d6" + "0" * 62
DRAFT_TEXT = "CHƯƠNG I\nVẬT LÍ NHIỆT\nNội dung"


# --------------------------------------------------------------------------- #
# fixtures shaped like a hand-written gold
# --------------------------------------------------------------------------- #


def gold_page(index: int = 0, **overrides: Any) -> dict[str, Any]:
    """One gold page as a human writes it."""
    data: dict[str, Any] = {
        "page_index": index,
        "text": DRAFT_TEXT if index == 0 else f"Nội dung trang {index}",
        "headings": ["CHƯƠNG I", "VẬT LÍ NHIỆT", "Nội dung"] if index == 0 else [],
    }
    data.update(overrides)
    return data


def gold(**overrides: Any) -> dict[str, Any]:
    """One gold file."""
    pages = overrides.pop("pages", [gold_page(0)])
    data: dict[str, Any] = {"doc_id": "vl12-sample", "pages": pages}
    data.update(overrides)
    return data


def block(text: str, order: int = 0, **overrides: Any) -> dict[str, Any]:
    """One artifact block as an engine mapping writes it."""
    data: dict[str, Any] = {
        "order": order,
        "raw_kind": "Text",
        "normalized_kind": "p",
        "text": text,
        "marker": None,
        "depth": None,
        "group": None,
        "bbox": None,
        "source_id": None,
    }
    data.update(overrides)
    return data


def artifact_page(index: int = 0, **overrides: Any) -> dict[str, Any]:
    """One artifact page whose single paragraph is its page text."""
    data: dict[str, Any] = {
        "page_index": index,
        "request_page": index + 5,
        "printed_folio": str(index + 4),
        "running": "VẬT LÍ 12",
        "continues_previous": False,
        "status": "ok",
        "blocks": [block(DRAFT_TEXT if index == 0 else f"Nội dung trang {index}")],
        "usage": None,
    }
    data.update(overrides)
    return data


def artifact(**overrides: Any) -> ExtractionArtifact:
    """One extraction artifact to align the gold against."""
    pages = overrides.pop("pages", [artifact_page(0)])
    counts: dict[str, int] = {}
    for entry in pages:
        for item in entry["blocks"]:
            kind = str(item["normalized_kind"])
            counts[kind] = counts.get(kind, 0) + 1
    data: dict[str, Any] = {
        "artifact_version": 1,
        "mapping_version": 1,
        "engine_id": "vision",
        "engine_version": "vision-2026-09",
        "status": "ok",
        "doc_id": "vl12-sample",
        "pdf_sha256": SHA,
        "source_sha256": None,
        "physical_pages": [entry["page_index"] for entry in pages],
        "render_dpi": 200,
        "render_page_offset": 4,
        "prompt_sha256": None,
        "public_config": {"model": "vision-model"},
        "pages": pages,
        "statistics": {"blocks_by_kind": dict(sorted(counts.items())), "normalization": {}},
        "notes": [],
    }
    data.update(overrides)
    return ExtractionArtifact.model_validate(data)


def write_gold(directory: Path, payload: dict[str, Any]) -> Path:
    path = directory / "gold.json"
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    return path


# --------------------------------------------------------------------------- #
# the gold file contract
# --------------------------------------------------------------------------- #


def test_gold_loads_the_simple_shape_a_human_writes(tmp_path: Path) -> None:
    path = write_gold(tmp_path, gold(pages=[gold_page(0), gold_page(1)]))

    loaded = load_gold(path)

    assert loaded.doc_id == "vl12-sample"
    assert loaded.pages[0].page_index == 0
    assert loaded.pages[0].text == DRAFT_TEXT
    assert loaded.pages[0].headings == ("CHƯƠNG I", "VẬT LÍ NHIỆT", "Nội dung")
    assert loaded.pages[1].page_index == 1
    assert loaded.pages[1].headings == ()


def test_the_page_number_is_called_page_index(tmp_path: Path) -> None:
    # The one spelling of the field, so a gold cannot be written half in one
    # vocabulary and half in another. ``page`` is the pinned-away alternative.
    with pytest.raises(ValidationError):
        GoldPage.model_validate({"page": 0, "text": DRAFT_TEXT})

    path = write_gold(tmp_path, gold(pages=[{"page": 0, "text": DRAFT_TEXT}]))
    with pytest.raises(GoldError, match="does not match the schema"):
        load_gold(path)


def test_a_gold_page_rejects_undeclared_fields() -> None:
    with pytest.raises(ValidationError):
        GoldPage.model_validate(gold_page(0, review_status="human_checked"))

    with pytest.raises(ValidationError):
        GoldReference.model_validate(gold(api_key="sk-live-secret"))


def test_a_blank_transcript_is_not_a_transcript() -> None:
    # Storing a blank page would score every character the run read as an
    # insertion; a page with nothing to transcribe is left out of the file.
    for blank in ("", "  \n "):
        with pytest.raises(ValidationError, match="blank"):
            GoldPage.model_validate(gold_page(0, text=blank))


def test_a_blank_heading_is_not_a_heading() -> None:
    with pytest.raises(ValidationError, match="blank"):
        GoldPage.model_validate(gold_page(0, headings=["CHƯƠNG I", "   "]))


def test_a_negative_page_index_is_rejected() -> None:
    with pytest.raises(ValidationError):
        GoldPage.model_validate(gold_page(-1))


@pytest.mark.parametrize("doc_id", ["", "  ", "vat li", "-vat-li"])
def test_gold_rejects_a_doc_id_that_is_not_a_slug(doc_id: str) -> None:
    with pytest.raises(ValidationError, match="doc_id"):
        GoldReference.model_validate(gold(doc_id=doc_id))


def test_gold_requires_at_least_one_page_and_no_duplicates() -> None:
    with pytest.raises(ValidationError):
        GoldReference.model_validate(gold(pages=[]))

    with pytest.raises(ValidationError, match="strictly increasing"):
        GoldReference.model_validate(gold(pages=[gold_page(1), gold_page(0)]))

    with pytest.raises(ValidationError, match="strictly increasing"):
        GoldReference.model_validate(gold(pages=[gold_page(0), gold_page(0)]))


def test_gold_page_lookup_returns_only_the_pages_it_holds() -> None:
    reference = GoldReference.model_validate(gold(pages=[gold_page(2)]))

    assert reference.page(2) is not None
    assert reference.page(0) is None


# --------------------------------------------------------------------------- #
# headings: the spans the boundary metrics are scored against
# --------------------------------------------------------------------------- #


def test_heading_spans_are_located_in_reading_order() -> None:
    reference = GoldReference.model_validate(gold())

    assert reference.pages[0].heading_spans() == ((0, 8), (9, 21), (22, 30))


def test_a_heading_that_occurs_twice_is_located_twice() -> None:
    text = "Nội dung\nđoạn giữa\nNội dung"
    page = GoldPage.model_validate(gold_page(0, text=text, headings=["Nội dung", "Nội dung"]))

    assert page.heading_spans() == ((0, 8), (19, 27))


def test_a_heading_that_cannot_be_found_is_an_error_not_a_miss() -> None:
    page = GoldPage.model_validate(gold_page(0, text="CHƯƠNG I", headings=["CHƯƠNG II"]))

    with pytest.raises(GoldError, match="does not occur in page 0"):
        page.heading_spans()


def test_headings_listed_out_of_reading_order_are_an_error() -> None:
    page = GoldPage.model_validate(
        gold_page(0, text="BÀI 1\nđoạn\nBÀI 2", headings=["BÀI 2", "BÀI 1"])
    )

    with pytest.raises(GoldError, match="BÀI 1"):
        page.heading_spans()


def test_load_gold_checks_every_heading_against_its_own_page(tmp_path: Path) -> None:
    path = write_gold(tmp_path, gold(pages=[gold_page(0, headings=["CHƯƠNG II"])]))

    with pytest.raises(GoldError, match="page 0"):
        load_gold(path)


# --------------------------------------------------------------------------- #
# loading: what a bad file says about itself
# --------------------------------------------------------------------------- #


def test_load_gold_checks_the_document_it_belongs_to(tmp_path: Path) -> None:
    path = write_gold(tmp_path, gold())

    assert load_gold(path, expected_doc_id="vl12-sample").doc_id == "vl12-sample"

    with pytest.raises(GoldError, match="lich-su-12"):
        load_gold(path, expected_doc_id="lich-su-12")


def test_load_gold_rejects_a_missing_or_malformed_file(tmp_path: Path) -> None:
    with pytest.raises(GoldError, match="cannot read"):
        load_gold(tmp_path / "absent.json")

    broken = tmp_path / "broken.json"
    broken.write_text("{not json", encoding="utf-8")
    with pytest.raises(GoldError, match="not valid JSON"):
        load_gold(broken)


def test_load_gold_schema_error_does_not_echo_transcript_text(tmp_path: Path) -> None:
    secret_text = "PRIVATE_REFERENCE_TEXT_NOT_FOR_REPORT"
    path = write_gold(tmp_path, gold(pages=[secret_text]))

    with pytest.raises(GoldError, match="does not match the schema") as error:
        load_gold(path)

    assert secret_text not in str(error.value)


def test_load_gold_rejects_a_page_that_is_not_a_page(tmp_path: Path) -> None:
    path = write_gold(tmp_path, gold(pages=[gold_page(0, headings="CHƯƠNG I")]))

    with pytest.raises(GoldError, match="does not match the schema"):
        load_gold(path)


# --------------------------------------------------------------------------- #
# the digest: what a comparison depends on, and nothing else
# --------------------------------------------------------------------------- #


def test_gold_digest_follows_the_text_not_the_file_layout(tmp_path: Path) -> None:
    pretty = tmp_path / "pretty.json"
    pretty.write_text(json.dumps(gold(), ensure_ascii=False, indent=2), encoding="utf-8")
    compact = tmp_path / "compact.json"
    compact.write_text(
        json.dumps(gold(), ensure_ascii=False, separators=(",", ":")), encoding="utf-8"
    )

    assert load_gold(pretty).content_digest == load_gold(compact).content_digest
    assert len(load_gold(compact).content_digest) == 64


def test_gold_digest_changes_when_the_gold_does() -> None:
    baseline = GoldReference.model_validate(gold())

    for change in (
        {"pages": [gold_page(0, text="CHƯƠNG I\nVẬT LÍ NHIỆT\nNội dung khác")]},
        {"pages": [gold_page(0, headings=["CHƯƠNG I"])]},
        {"pages": [gold_page(0), gold_page(1)]},
        {"doc_id": "vl12-other"},
    ):
        assert (
            GoldReference.model_validate(gold(**change)).content_digest != baseline.content_digest
        )


# --------------------------------------------------------------------------- #
# coverage against the page selection
# --------------------------------------------------------------------------- #


def test_coverage_counts_the_pages_the_gold_holds() -> None:
    reference = GoldReference.model_validate(gold(pages=[gold_page(0), gold_page(1)]))

    report = coverage(reference, (0, 1, 2, 3))

    assert report.requested_pages == (0, 1, 2, 3)
    assert report.covered_pages == (0, 1)
    assert report.missing_pages == (2, 3)
    assert report.out_of_scope_pages == ()
    assert report.complete is False


def test_coverage_flags_a_gold_page_the_study_does_not_select() -> None:
    reference = GoldReference.model_validate(gold(pages=[gold_page(0), gold_page(7)]))

    report = coverage(reference, (0,))

    assert report.out_of_scope_pages == (7,)
    assert report.missing_pages == ()
    assert report.covered_pages == (0,)
    assert report.complete is True


def test_coverage_of_a_selection_the_gold_covers_whole_is_complete() -> None:
    reference = GoldReference.model_validate(gold(pages=[gold_page(0), gold_page(1)]))

    assert coverage(reference, (0, 1)).complete is True


# --------------------------------------------------------------------------- #
# alignment: gold text next to what the run produced
# --------------------------------------------------------------------------- #


def test_alignment_pairs_the_gold_text_with_the_page_transcript() -> None:
    reference = GoldReference.model_validate(gold(pages=[gold_page(0)]))
    run = artifact(pages=[artifact_page(0)])

    pages = align_pages(reference, run)

    assert len(pages) == 1
    page = pages[0]
    assert page.page_index == 0
    assert page.request_page == 5
    assert page.reference == DRAFT_TEXT
    assert page.hypothesis == page_text(run.pages[0])
    assert page.to_metrics_page() == {
        "page": 5,
        "reference": DRAFT_TEXT,
        "hypothesis": DRAFT_TEXT,
    }


def test_alignment_reports_the_transcript_without_page_furniture() -> None:
    reference = GoldReference.model_validate(gold(pages=[gold_page(0)]))
    run = artifact(
        pages=[
            artifact_page(
                0,
                running="VẬT LÍ 12",
                blocks=[
                    block("VẬT LÍ 12", raw_kind="PageHeader", normalized_kind="page_header"),
                    block("Nội dung", order=1),
                    block("14", order=2, raw_kind="PageFooter", normalized_kind="page_footer"),
                ],
            )
        ]
    )

    assert align_pages(reference, run)[0].hypothesis == "Nội dung"


def test_alignment_leaves_a_page_the_run_never_read_without_a_hypothesis() -> None:
    reference = GoldReference.model_validate(gold(pages=[gold_page(0)]))
    run = artifact(pages=[artifact_page(1)])

    pages = align_pages(reference, run)

    assert [page.page_index for page in pages] == [0, 1]
    assert pages[0].hypothesis is None
    assert pages[0].reference == DRAFT_TEXT
    assert pages[1].reference is None


def test_alignment_leaves_a_page_the_engine_could_not_read_without_a_hypothesis() -> None:
    # A page the engine answered "cannot read" for is not a page that produced
    # an empty transcript, and the alignment does not turn one into the other.
    reference = GoldReference.model_validate(gold(pages=[gold_page(0)]))
    run = artifact(pages=[artifact_page(0, status="unsupported_modality", blocks=[])])

    assert align_pages(reference, run)[0].hypothesis is None


def test_alignment_numbers_a_gold_page_the_run_never_read() -> None:
    reference = GoldReference.model_validate(gold(pages=[gold_page(3)]))
    run = artifact(pages=[artifact_page(0)])

    page = align_pages(reference, run)[1]

    assert page.page_index == 3
    assert page.request_page is None
    assert page.to_metrics_page()["page"] == 4


def test_alignment_refuses_a_run_for_another_document() -> None:
    reference = GoldReference.model_validate(gold())

    with pytest.raises(GoldError, match="lich-su-12"):
        align_pages(reference, artifact(doc_id="lich-su-12"))


# --------------------------------------------------------------------------- #
# the bridge into the accuracy metrics
# --------------------------------------------------------------------------- #


def test_metrics_document_feeds_the_accuracy_metrics() -> None:
    from research.ocr_pipeline import metrics

    reference = GoldReference.model_validate(
        gold(pages=[gold_page(0, text="Nội dung trang"), gold_page(1, text="Nội dung trang 1")])
    )
    run = artifact(
        pages=[
            artifact_page(0, blocks=[block("Nội dung trạng")]),
            artifact_page(1, blocks=[block("Nội dung trang 1")]),
            artifact_page(2),
        ]
    )

    scored = metrics.document_text_metrics(metrics_document(reference, run))

    assert scored["doc_id"] == "vl12-sample"
    assert scored["pages"] == {
        "total": 3,
        "with_gold": 2,
        "without_gold": 1,
        "empty_hypothesis": 0,
    }
    assert scored["hypothesis"] == {"present_pages": 3, "missing_pages": 0}
    assert scored["rates"]["cer"]["micro"]["status"] == "measured"
    assert scored["rates"]["cer"]["micro"]["value"] > 0
    assert scored["rates"]["measured_pages"] == 2
    # A page the gold does not cover is not a zero, it is a full deletion
    # measured against nothing: an empty reference, named as such.
    assert scored["pages_detail"][2]["cer"]["status"] == "empty_reference"


def test_metrics_document_reports_a_run_that_missed_a_gold_page() -> None:
    from research.ocr_pipeline import metrics

    reference = GoldReference.model_validate(gold(pages=[gold_page(0, text="Nội dung trang")]))

    scored = metrics.document_text_metrics(
        metrics_document(reference, artifact(pages=[artifact_page(1)]))
    )

    assert scored["pages"] == {
        "total": 2,
        "with_gold": 1,
        "without_gold": 1,
        "empty_hypothesis": 0,
    }
    assert scored["hypothesis"]["missing_pages"] == 1
    assert scored["rates"]["cer"]["micro"]["status"] == "not_measured"
    assert scored["rates"]["missing_transcript_pages"] == 1
    assert scored["pages_detail"][0]["cer"]["reason"] == "hypothesis_missing"


def test_an_empty_transcript_is_measured_and_scores_as_a_full_deletion() -> None:
    from research.ocr_pipeline import metrics

    reference = GoldReference.model_validate(gold(pages=[gold_page(0, text="Nội dung trang")]))
    run = artifact(pages=[artifact_page(0, blocks=[block("x")])])

    # The page text of an artifact with one non-empty block is that block, so an
    # empty transcript is built by hand: the engine read the page and wrote
    # nothing, which is a measured 100% error rather than a typed miss.
    document = metrics_document(reference, run)
    document["pages"][0]["hypothesis"] = ""

    scored = metrics.document_text_metrics(document)

    assert scored["pages"]["empty_hypothesis"] == 1
    assert scored["pages_detail"][0]["cer"]["status"] == "measured"
    assert scored["pages_detail"][0]["cer"]["value"] == 1.0
