"""Offline tests for heading and section-boundary alignment.

The fixtures are synthetic Vietnamese pages written for this file, never a page of
the book under study. The gold side is a transcript whose headings the file lists;
the candidate side is a run that read the same page and dropped a single colon, so
the page is deliberately longer than its marks: three headings and three section
starts, plus a body line that differs between the two sides. The third heading
therefore sits at a *different* offset on each side, which is what makes the
alignment — rather than a byte-for-byte comparison — testable.

The rest of the file proves the abstentions: a page the run never read, a page the
engine could not read and a page whose text cannot be lined up with the gold stay
counted and unscored with their reason named; a run is never scored against a
transcript identical to its own; and a run that placed nothing leaves precision
undefined while recall is a measured zero.

No PDF is opened, no network is touched, and no API is called.
"""

from __future__ import annotations

from typing import Any

import pytest

from research.ocr_pipeline.alignment import (
    DEFAULT_CONFIDENCE,
    MARK_KINDS,
    CandidateMark,
    artifact_marks,
    heading_boundary_metrics,
)
from research.ocr_pipeline.models import ExtractionArtifact, page_text
from research.ocr_pipeline.reference import GoldError, GoldReference
from research.ocr_pipeline.report import find_raw_text_keys

SHA = "d6" + "0" * 62
DOC_ID = "vl12-fixture"

HEADINGS = ("BÀI 1", "SỰ CHUYỂN THỂ CỦA CÁC CHẤT", "BÀI 2")
BODY = "Chất có thể tồn tại ở ba thể rắn, lỏng và khí."
LAST_BODY = "Sự nóng chảy là quá trình chuyển từ thể rắn sang thể lỏng."


# --------------------------------------------------------------------------- #
# synthetic pages: a gold transcript and a run that read it
# --------------------------------------------------------------------------- #


def block(text: str, order: int, kind: str = "p", depth: int | None = None) -> dict[str, Any]:
    """One artifact block as an engine mapping writes it."""
    return {
        "order": order,
        "raw_kind": "SectionHeader" if kind == "heading" else "Text",
        "normalized_kind": kind,
        "text": text,
        "marker": None,
        "depth": depth,
        "group": None,
        "bbox": None,
        "source_id": None,
    }


def run_blocks() -> list[dict[str, Any]]:
    """The run's own reading of both headings and both body lines."""
    return [
        block("BÀI 1", 0, "heading", depth=1),
        block("SỰ CHUYỂN THỂ CỦA CÁC CHẤT", 1, "heading", depth=2),
        block(BODY, 2),
        block("BÀI 2", 3, "heading", depth=1),
        block("Sự nóng chảy và sự đông đặc", 4),
        block(LAST_BODY, 5),
    ]


def joined(blocks: list[dict[str, Any]]) -> str:
    """The text ``page_text`` yields for a page holding exactly these blocks."""
    return "\n".join(entry["text"] for entry in blocks)


GOLD_TEXT = joined(run_blocks()).replace("ba thể rắn", "ba thể: rắn")
"""The gold transcript: the run's own reading with the dropped colon restored."""

OTHER_PAGE_TEXT = "Kí hiệu đại số\nSố phức được viết dưới dạng a + bi."
"""A page of another subject: nothing in it can be located in the gold above."""


def artifact_page(
    index: int = 0,
    blocks: list[dict[str, Any]] | None = None,
    *,
    status: str = "ok",
    request_page: int | None = None,
) -> dict[str, Any]:
    """One artifact page; a page the engine could not read carries no blocks."""
    return {
        "page_index": index,
        "request_page": request_page if request_page is not None else index + 1,
        "printed_folio": None,
        "running": "",
        "continues_previous": False,
        "status": status,
        "blocks": run_blocks() if blocks is None else blocks,
        "usage": None,
    }


def artifact(
    *pages: dict[str, Any], doc_id: str = DOC_ID, sha: str = SHA, offset: int | None = None
) -> ExtractionArtifact:
    """One extraction artifact to score marks from."""
    counts: dict[str, int] = {}
    for page in pages:
        for entry in page["blocks"]:
            kind = str(entry["normalized_kind"])
            counts[kind] = counts.get(kind, 0) + 1
    return ExtractionArtifact.model_validate(
        {
            "artifact_version": 1,
            "mapping_version": 1,
            "engine_id": "vision",
            "engine_version": "vision-2026-09",
            "status": "ok",
            "doc_id": doc_id,
            "pdf_sha256": sha,
            "source_sha256": None,
            "physical_pages": [page["page_index"] for page in pages],
            "render_dpi": 200,
            "render_page_offset": offset,
            "prompt_sha256": None,
            "public_config": {"model": "vision-model"},
            "pages": list(pages),
            "statistics": {"blocks_by_kind": dict(sorted(counts.items())), "normalization": {}},
            "notes": [],
        }
    )


def gold_page(
    index: int = 0, text: str = GOLD_TEXT, headings: tuple[str, ...] = HEADINGS
) -> dict[str, Any]:
    """One gold page as the gold file writes it."""
    return {"page_index": index, "text": text, "headings": list(headings)}


def reference(*pages: dict[str, Any], doc_id: str = DOC_ID) -> GoldReference:
    return GoldReference.model_validate(
        {
            "doc_id": doc_id,
            "pages": [gold_page()] if not pages else list(pages),
        }
    )


def score(run: ExtractionArtifact, **overrides: Any) -> dict[str, Any]:
    """Score one run against the gold transcript it was read from."""
    return heading_boundary_metrics(reference(), run, **overrides)


# --------------------------------------------------------------------------- #
# the marks a run claims
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("kind", MARK_KINDS)
def test_the_two_kinds_are_the_ones_gold_and_a_run_share(kind: str) -> None:
    # One vocabulary, two readings of it: a heading block marks a heading *and*
    # the section start it introduces, and both are scored against the gold.
    assert kind in MARK_KINDS
    assert CandidateMark(0, 0, kind).kind == kind


def test_every_heading_block_claims_a_heading_and_a_section_start() -> None:
    run = artifact(artifact_page())
    run_text = page_text(run.pages[0])

    marks = artifact_marks(run)

    heading_offsets = [mark.offset for mark in marks if mark.kind == "heading"]
    assert heading_offsets == [run_text.index(heading) for heading in HEADINGS]
    assert [mark.offset for mark in marks if mark.kind == "boundary"] == heading_offsets
    assert {mark.page_index for mark in marks} == {0}


def test_mark_offsets_index_the_run_page_text_the_gold_headings_sit_in() -> None:
    page = artifact_page(
        blocks=[
            block("VẬT LÍ 12", 0, "page_header"),
            block("BÀI 1", 1, "heading", depth=1),
            block("", 2, "figure"),
            block("SỰ CHUYỂN THỂ CỦA CÁC CHẤT", 3, "heading", depth=2),
            block(BODY, 4),
        ]
    )
    run = artifact(page)
    text = page_text(run.pages[0])
    offsets = [mark.offset for mark in artifact_marks(run) if mark.kind == "heading"]

    # Furniture and wordless blocks take no part in the text, so the offsets must
    # match the string the gold's own marks are located in.
    assert text == f"BÀI 1\nSỰ CHUYỂN THỂ CỦA CÁC CHẤT\n{BODY}"
    assert [
        text[offset : offset + len(heading)] for offset, heading in zip(offsets, HEADINGS)
    ] == [
        "BÀI 1",
        "SỰ CHUYỂN THỂ CỦA CÁC CHẤT",
    ]


def test_a_page_the_engine_could_not_read_claims_nothing() -> None:
    run = artifact(artifact_page(0, status="unsupported_modality", blocks=[]))

    assert artifact_marks(run) == []


def test_a_mark_must_name_a_real_page_offset_and_kind() -> None:
    for arguments in ((-1, 0, "heading"), (0, -1, "heading"), (0, 0, "paragraph")):
        with pytest.raises(ValueError, match=r"page_index|offset|kind"):
            CandidateMark(*arguments)


# --------------------------------------------------------------------------- #
# scoring placement against the gold headings
# --------------------------------------------------------------------------- #


def test_headings_the_run_placed_where_the_gold_says_score_perfectly() -> None:
    result = score(artifact(artifact_page()))

    assert result["status"] == "measured"
    assert result["reason"] is None
    assert result["doc_id"] == DOC_ID
    assert result["confidence"] == DEFAULT_CONFIDENCE
    assert result["pages"] == {"marked": 1, "considered": 1, "scored": 1, "skipped": 0}
    assert result["skipped"] == {}
    for kind in MARK_KINDS:
        block = result["kinds"][kind]
        assert block["status"] == "measured"
        assert block["gold_marks"] == len(HEADINGS)
        assert block["candidate_marks"] == len(HEADINGS)
        assert block["true_positives"] == len(HEADINGS)
        assert block["false_positives"] == 0
        assert block["false_negatives"] == 0
        assert block["precision"]["value"] == 1.0
        assert block["recall"]["value"] == 1.0
        assert block["f1"]["value"] == 1.0
    row = result["pages_detail"][0]
    assert row["status"] == "measured"
    assert row["alignment"] >= DEFAULT_CONFIDENCE


def test_a_missed_heading_costs_recall_and_not_precision() -> None:
    run = artifact(
        artifact_page(
            blocks=[
                dict(entry) if entry["text"] != "BÀI 2" else dict(entry, normalized_kind="p")
                for entry in run_blocks()
            ]
        )
    )

    block = score(run)["kinds"]["heading"]

    assert block["recall"]["value"] == pytest.approx(2 / 3)
    assert block["precision"]["value"] == 1.0
    assert block["false_negatives"] == 1


def test_an_invented_heading_costs_precision_and_not_recall() -> None:
    run = artifact(
        artifact_page(
            blocks=[
                dict(entry, normalized_kind="heading", depth=2)
                if entry["text"] == "Sự nóng chảy và sự đông đặc"
                else dict(entry)
                for entry in run_blocks()
            ]
        )
    )

    block = score(run)["kinds"]["heading"]

    assert block["precision"]["value"] == pytest.approx(3 / 4)
    assert block["recall"]["value"] == 1.0
    assert block["false_positives"] == 1


def test_a_run_that_placed_nothing_leaves_precision_undefined() -> None:
    # Recall is a measured zero — the run claimed no heading the gold holds —
    # while precision has no denominator to divide by. The page still has to
    # line up with the gold, so the run read it as paragraphs and nothing else.
    run = artifact(
        artifact_page(blocks=[dict(entry, normalized_kind="p") for entry in run_blocks()])
    )

    block = score(run)["kinds"]["heading"]

    assert block["candidate_marks"] == 0
    assert block["recall"]["status"] == "measured"
    assert block["recall"]["value"] == 0.0
    assert block["precision"]["status"] == "not_measured"
    assert block["precision"]["reason"] == "empty_denominator"
    assert block["f1"]["value"] == 0.0


def test_the_two_kinds_are_scored_over_the_same_marks() -> None:
    result = score(artifact(artifact_page()))

    heading, boundary = result["kinds"]["heading"], result["kinds"]["boundary"]

    assert heading["true_positives"] == boundary["true_positives"]
    assert heading["gold_marks"] == boundary["gold_marks"]


# --------------------------------------------------------------------------- #
# the abstentions
# --------------------------------------------------------------------------- #


def test_a_gold_page_the_run_never_read_is_skipped_and_counted() -> None:
    run = artifact(artifact_page(1, request_page=2))

    result = score(run)

    assert result["status"] == "not_measured"
    assert result["reason"] == "hypothesis_missing"
    assert result["skipped"] == {"hypothesis_missing": 1}
    assert result["pages"] == {"marked": 1, "considered": 1, "scored": 0, "skipped": 1}
    assert result["pages_detail"][0]["reason"] == "hypothesis_missing"
    assert result["kinds"]["heading"]["status"] == "not_measured"
    assert result["kinds"]["heading"]["precision"]["status"] == "not_measured"


def test_a_page_the_engine_could_not_read_is_named_as_such() -> None:
    # Not a page that failed: the engine answered that this page carries no text
    # it can read, and the report keeps the engine's own word for it.
    run = artifact(artifact_page(0, status="unsupported_modality", blocks=[]))

    result = score(run)

    assert result["skipped"] == {"unsupported_modality": 1}
    assert result["reason"] == "unsupported_modality"


def test_a_run_is_never_scored_against_its_own_transcript() -> None:
    # This page's run and gold differ by one colon and nothing else, and the
    # difference is what the score comes from; a gold written from the run's own
    # text would score itself perfectly and measure nothing.
    run = artifact(artifact_page())
    assert page_text(run.pages[0]) != GOLD_TEXT

    result = score(run)

    assert result["status"] == "measured"

    copied = heading_boundary_metrics(reference(gold_page(text=page_text(run.pages[0]))), run)

    assert copied["status"] == "not_measured"
    assert copied["skipped"] == {"self_scoring": 1}


def test_a_page_that_cannot_be_lined_up_with_the_gold_stays_unscored() -> None:
    run = artifact(
        artifact_page(
            blocks=[
                block(OTHER_PAGE_TEXT.split("\n")[0], 0, "heading", depth=1),
                block(OTHER_PAGE_TEXT.split("\n")[1], 1),
            ]
        )
    )

    result = score(run)

    assert result["skipped"] == {"alignment_below_threshold": 1}
    assert result["reason"] == "alignment_below_threshold"
    assert result["pages_detail"][0]["alignment"] is None


def test_a_gold_with_no_headings_at_all_marks_nothing() -> None:
    run = artifact(artifact_page())

    result = heading_boundary_metrics(reference(gold_page(headings=())), run)

    assert result["status"] == "not_measured"
    assert result["reason"] == "no_eligible_pages"
    assert result["pages"] == {"marked": 0, "considered": 0, "scored": 0, "skipped": 0}


def test_a_gold_page_without_headings_is_not_considered() -> None:
    run = artifact(artifact_page(0), artifact_page(1))

    result = heading_boundary_metrics(reference(gold_page(0), gold_page(1, headings=())), run)

    assert result["pages"]["considered"] == 1
    assert result["pages"]["scored"] == 1


def test_only_the_pages_the_gold_marks_are_considered() -> None:
    run = artifact(artifact_page(0), artifact_page(1), artifact_page(2))

    result = heading_boundary_metrics(reference(gold_page(0), gold_page(2)), run)

    assert result["pages"]["considered"] == 2
    assert result["pages"]["marked"] == 2
    assert result["status"] == "measured"


def test_the_dominant_reason_is_the_one_most_pages_were_skipped_for() -> None:
    run = artifact(
        artifact_page(0, status="unsupported_modality", blocks=[]),
        artifact_page(1, status="unsupported_modality", blocks=[]),
        artifact_page(2),
    )
    gold_pages = [gold_page(index) for index in (0, 1, 2)]
    gold_pages[2] = gold_page(2, text=OTHER_PAGE_TEXT, headings=(OTHER_PAGE_TEXT.split("\n")[0],))

    result = heading_boundary_metrics(reference(*gold_pages), run)

    assert result["skipped"] == {"unsupported_modality": 2, "alignment_below_threshold": 1}
    assert result["reason"] == "unsupported_modality"


# --------------------------------------------------------------------------- #
# caller-supplied marks and malformed input
# --------------------------------------------------------------------------- #


def test_passing_marks_replaces_the_runs_own_headings() -> None:
    # How the chunk family scores a chunker's boundaries: the marks are the
    # caller's, and the run's own heading blocks take no part.
    run = artifact(artifact_page())
    elsewhere = [
        CandidateMark(0, offset, kind)
        for offset in (0, len(GOLD_TEXT) // 2)
        for kind in MARK_KINDS
    ]

    own = score(run)["kinds"]["heading"]
    supplied = score(run, candidates=elsewhere)["kinds"]["heading"]

    assert own["f1"]["value"] == 1.0
    assert supplied["candidate_marks"] == 2
    assert supplied["f1"]["value"] < 1.0


def test_marks_that_name_a_page_the_run_does_not_cover_are_a_caller_bug() -> None:
    run = artifact(artifact_page())

    with pytest.raises(ValueError, match="does not cover"):
        score(run, candidates=[CandidateMark(7, 0, "heading")])

    with pytest.raises(ValueError, match="beyond page"):
        score(run, candidates=[CandidateMark(0, len(page_text(run.pages[0])) + 1, "heading")])

    with pytest.raises(ValueError, match="duplicate"):
        score(
            run,
            candidates=[CandidateMark(0, 0, "heading"), CandidateMark(0, 0, "heading")],
        )

    with pytest.raises(ValueError, match="CandidateMark"):
        score(run, candidates=[(0, 0, "heading")])  # type: ignore[list-item]


def test_confidence_must_be_a_probability() -> None:
    run = artifact(artifact_page())

    for value in (0.0, -0.5, 1.5):
        with pytest.raises(ValueError, match="confidence"):
            score(run, confidence=value)


def test_a_looser_confidence_scores_a_page_a_stricter_one_skips() -> None:
    run = artifact(artifact_page())
    reference_page = gold_page(text=GOLD_TEXT + "\n" + "\n".join([OTHER_PAGE_TEXT] * 3))

    strict = heading_boundary_metrics(reference(reference_page), run)
    loose = heading_boundary_metrics(reference(reference_page), run, confidence=0.5)

    assert strict["status"] == "not_measured"
    assert loose["status"] == "measured"


def test_a_gold_heading_that_is_not_in_its_transcript_is_an_error() -> None:
    with pytest.raises(GoldError, match="BÀI 3"):
        heading_boundary_metrics(
            reference(gold_page(headings=(*HEADINGS, "BÀI 3"))), artifact(artifact_page())
        )


def test_a_run_for_another_document_is_an_error() -> None:
    with pytest.raises(GoldError, match="lich-su-12"):
        score(artifact(artifact_page(), doc_id="lich-su-12"))


# --------------------------------------------------------------------------- #
# what the result may carry
# --------------------------------------------------------------------------- #


def test_the_result_holds_counts_and_never_transcript_text() -> None:
    result = score(artifact(artifact_page()))

    assert find_raw_text_keys(result) == []
    assert all("text" not in row for row in result["pages_detail"])
    assert all("kinds" not in row for row in result["pages_detail"] if row["status"] != "measured")
