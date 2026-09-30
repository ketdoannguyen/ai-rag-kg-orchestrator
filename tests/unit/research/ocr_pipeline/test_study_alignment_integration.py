"""Wiring heading/boundary scoring into a study: the seam, pinned by tests.

:mod:`research.ocr_pipeline.alignment` scores where a run put its headings and
section starts against the headings the gold lists; these tests pin the wire that
carries a gold file from the manifest (or a ``--gold`` flag) into a run row:

1. ``StudyRunner._gold_for`` resolves ``gold_paths[doc_id] or spec.gold_path``
   and appends a limitation naming the document whenever it abstains.
2. ``_run_row`` sets ``row["structure_metrics"]`` to
   ``heading_boundary_metrics(gold, artifact)`` whenever a gold resolved, and
   leaves the key **absent** otherwise.

The abstention rules are the point. A study that has no gold, an unreadable gold
or a run whose words cannot be lined up with the gold must report ``not_measured``
with a named reason — never a zero, which would read as "the run found no
headings" when the truth is "nobody checked". And because a report may not carry
transcription, ``structure_metrics`` holds only counts, statuses and a confidence.

Every fixture here is synthetic and local to this module: the PDF is a placeholder
the comparison path never parses (only ``validate()`` opens a PDF), no engine is
imported, no network is touched, and no page of the book under study appears.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pytest

from research.ocr_pipeline.alignment import DEFAULT_CONFIDENCE
from research.ocr_pipeline.corpus import load_manifest
from research.ocr_pipeline.models import ExtractionArtifact
from research.ocr_pipeline.reference import load_gold
from research.ocr_pipeline.report import find_raw_text_keys
from research.ocr_pipeline.runner import KIND_EXTRACT, RUN_OK, RunRef, StudyRunner

DOC_ID = "vl12-fixture"
SHA = "d6" + "0" * 62

HEADINGS = ("BÀI 1", "SỰ CHUYỂN THỂ CỦA CÁC CHẤT", "BÀI 2")
BODY = "Chất có thể tồn tại ở ba thể rắn, lỏng và khí."
LAST_BODY = "Sự nóng chảy là quá trình chuyển từ thể rắn sang thể lỏng."

PROVENANCE: dict[str, Any] = {
    "git_commit": "0" * 40,
    "git_dirty": False,
    "python": "3.11.0",
    "platform": "test",
    "label": "alignment integration tests",
}
"""A provenance stand-in: the git probe must never run inside a test."""


# --------------------------------------------------------------------------- #
# synthetic pages: a run that read a page, and the human who checked it
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
    """The run's own reading: three headings, two body lines and one paragraph."""
    return [
        block("BÀI 1", 0, "heading", depth=1),
        block("SỰ CHUYỂN THỂ CỦA CÁC CHẤT", 1, "heading", depth=2),
        block(BODY, 2),
        block("BÀI 2", 3, "heading", depth=1),
        block("Sự nóng chảy và sự đông đặc", 4),
        block(LAST_BODY, 5),
    ]


def unaligned_blocks() -> list[dict[str, Any]]:
    """A run whose words share nothing with the gold, so alignment abstains."""
    return [block("alpha beta gamma", 0, "heading", depth=1), block("delta epsilon", 1)]


GOLD_TEXT = "\n".join(entry["text"] for entry in run_blocks()).replace("ba thể rắn", "ba thể: rắn")
"""The checked transcript: the run's own reading with a dropped colon restored.

One character apart, so the page is not ``self_scoring``, and the third heading
sits at a *different* offset on each side — which is what makes the alignment,
rather than a byte-for-byte comparison, the thing under test.
"""


def artifact_page(
    index: int = 0,
    blocks: list[dict[str, Any]] | None = None,
    *,
    status: str = "ok",
) -> dict[str, Any]:
    """One artifact page; a page the engine could not read carries no blocks."""
    return {
        "page_index": index,
        "request_page": index + 1,
        "printed_folio": None,
        "running": "",
        "continues_previous": False,
        "status": status,
        "blocks": run_blocks() if blocks is None else blocks,
        "usage": None,
    }


def artifact(*pages: dict[str, Any], doc_id: str = DOC_ID, sha: str = SHA) -> ExtractionArtifact:
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
            "render_page_offset": None,
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
    """One gold page as a human writes it."""
    return {"page_index": index, "text": text, "headings": list(headings)}


def gold_payload(*pages: dict[str, Any], doc_id: str = DOC_ID) -> dict[str, Any]:
    """A gold file as it sits on disk, for ``load_gold`` to read."""
    return {"doc_id": doc_id, "pages": list(pages) or [gold_page()]}


def write_json(path: Path, payload: Any) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    return path


# --------------------------------------------------------------------------- #
# a corpus, its gold file, and a stored artifact
# --------------------------------------------------------------------------- #


def document(**overrides: Any) -> dict[str, Any]:
    """One document entry for a corpus manifest."""
    data: dict[str, Any] = {
        "doc_id": DOC_ID,
        "pdf_path": "fixture.pdf",
        "pdf_sha256": SHA,
        "page_indices": [0],
        "modality": "digital",
        "permission_note": "synthetic pages, so there is nothing to license",
        "gold_path": None,
    }
    data.update(overrides)
    return data


@pytest.fixture
def corpus_dir(tmp_path: Path) -> Path:
    """A corpus directory holding the manifest, a placeholder PDF and its gold.

    The PDF is written but never parsed: ``compare`` does not validate the corpus,
    so only ``validate()`` would open it, and these tests never call that.
    """
    (tmp_path / "fixture.pdf").write_bytes(b"%PDF-1.4\n%%EOF\n")
    write_json(tmp_path / "gold.json", gold_payload(gold_page()))
    return tmp_path


def make_manifest(corpus_dir: Path, **overrides: Any) -> Path:
    """Write a one-document manifest and return its path."""
    return write_json(
        corpus_dir / "manifest.json",
        {"corpus_version": 1, "documents": [document(**overrides)]},
    )


def make_runner(corpus_dir: Path, manifest_path: Path) -> StudyRunner:
    """A runner over the synthetic corpus, with the git probe replaced."""
    return StudyRunner(
        load_manifest(manifest_path),
        base_dir=corpus_dir,
        root=corpus_dir / "study",
        engines={},
        provenance=PROVENANCE,
    )


def store_artifact(runner: StudyRunner, run: ExtractionArtifact) -> str:
    """Put an artifact in the store and return the digest a run manifest names."""
    return runner.store.put(run)


def run_manifest(runner: StudyRunner, run_id: str, digest: str) -> dict[str, Any]:
    """A run manifest on disk, the shape ``compare`` reads back."""
    manifest = {
        "doc_id": DOC_ID,
        "run_id": run_id,
        "kind": KIND_EXTRACT,
        "status": RUN_OK,
        "artifact_digest": digest,
        "engine_id": "vision",
        "profile": None,
        "rankable": True,
        "repeat": 1,
    }
    write_json(runner.run_dir(DOC_ID, run_id) / "manifest.json", manifest)
    return manifest


def score_row(
    runner: StudyRunner,
    run: ExtractionArtifact,
    *,
    gold_paths: Mapping[str, str] | None = None,
) -> tuple[dict[str, Any], list[str]]:
    """One ``_run_row`` call, returning the row and the limitations it recorded."""
    digest = store_artifact(runner, run)
    manifest = run_manifest(runner, "vl12-fixture.vision.r1", digest)
    limitations: list[str] = []
    row, _documents, _usage = runner._run_row(
        manifest,
        gold_paths=gold_paths or {},
        limitations=limitations,
    )
    return row, limitations


def declared_runner(corpus_dir: Path) -> StudyRunner:
    """A runner whose manifest declares the gold file next to it."""
    return make_runner(corpus_dir, make_manifest(corpus_dir, gold_path="gold.json"))


# --------------------------------------------------------------------------- #
# A. resolving the gold: the explicit flag, the manifest, or an abstention
# --------------------------------------------------------------------------- #


def test_manifest_gold_path_is_used_when_no_flag_is_given(corpus_dir: Path) -> None:
    """A manifest that declares ``gold_path`` needs no flag at all."""
    runner = declared_runner(corpus_dir)
    limitations: list[str] = []

    gold = runner._gold_for(DOC_ID, {}, limitations)

    assert gold is not None
    assert [page.page_index for page in gold.pages] == [0]
    assert limitations == []


def test_an_explicit_gold_path_wins_over_the_manifest(corpus_dir: Path) -> None:
    write_json(
        corpus_dir / "other.json",
        gold_payload(gold_page(headings=("BÀI 1", "SỰ CHUYỂN THỂ CỦA CÁC CHẤT"))),
    )
    runner = declared_runner(corpus_dir)
    limitations: list[str] = []

    gold = runner._gold_for(DOC_ID, {DOC_ID: "other.json"}, limitations)

    assert gold is not None
    assert gold.pages[0].headings == ("BÀI 1", "SỰ CHUYỂN THỂ CỦA CÁC CHẤT")
    assert limitations == []


def test_no_gold_abstains_and_names_the_document(corpus_dir: Path) -> None:
    """Neither the manifest nor a flag: abstain, and say so, without raising."""
    runner = make_runner(corpus_dir, make_manifest(corpus_dir, gold_path=None))
    limitations: list[str] = []

    assert runner._gold_for(DOC_ID, {}, limitations) is None
    assert len(limitations) == 1
    assert DOC_ID in limitations[0]
    assert "no gold" in limitations[0]


@pytest.mark.parametrize(
    "payload",
    [
        pytest.param("{ not json", id="not-json"),
        pytest.param({"doc_id": DOC_ID, "pages": []}, id="no-pages"),
        pytest.param(gold_payload(gold_page(), doc_id="another-doc"), id="another-document"),
        pytest.param(
            gold_payload(gold_page(headings=("BÀI 3",))), id="heading-not-in-its-own-text"
        ),
    ],
)
def test_unusable_gold_abstains_instead_of_raising(corpus_dir: Path, payload: Any) -> None:
    """A ``GoldError`` becomes a named limitation, not a crash and not a zero."""
    path = corpus_dir / "broken.json"
    if isinstance(payload, str):
        path.write_text(payload, encoding="utf-8")
    else:
        write_json(path, payload)
    runner = declared_runner(corpus_dir)
    limitations: list[str] = []

    gold = runner._gold_for(DOC_ID, {DOC_ID: "broken.json"}, limitations)

    assert gold is None
    assert len(limitations) == 1
    assert DOC_ID in limitations[0]

    # The manifest's own gold still scores, so the flag named the broken file
    # and nothing else did.
    assert runner._gold_for(DOC_ID, {}, []) is not None


def test_the_gold_digest_is_the_one_the_file_loads_to(corpus_dir: Path) -> None:
    row, _limitations = score_row(declared_runner(corpus_dir), artifact(artifact_page()))

    assert row["gold_digest"] == load_gold(corpus_dir / "gold.json").content_digest


# --------------------------------------------------------------------------- #
# B. the row a study scores: gold marks against the run's, or an abstention
# --------------------------------------------------------------------------- #


def test_a_checked_page_feeds_heading_and_boundary_metrics(corpus_dir: Path) -> None:
    row, limitations = score_row(declared_runner(corpus_dir), artifact(artifact_page()))

    assert limitations == []
    metrics = row["structure_metrics"]
    assert metrics["status"] == "measured"
    assert metrics["reason"] is None
    assert metrics["confidence"] == DEFAULT_CONFIDENCE
    assert metrics["pages"] == {"marked": 1, "considered": 1, "scored": 1, "skipped": 0}
    for kind in ("heading", "boundary"):
        block_metrics = metrics["kinds"][kind]
        assert block_metrics["status"] == "measured"
        assert block_metrics["gold_marks"] == len(HEADINGS)
        assert block_metrics["true_positives"] == len(HEADINGS)
        assert block_metrics["false_positives"] == 0
        assert block_metrics["false_negatives"] == 0
        assert block_metrics["f1"]["value"] == 1.0


def test_no_gold_leaves_the_row_without_a_number(corpus_dir: Path) -> None:
    """The key is absent, so nothing downstream can read a zero that was never measured."""
    runner = make_runner(corpus_dir, make_manifest(corpus_dir, gold_path=None))

    row, limitations = score_row(runner, artifact(artifact_page()))

    assert "structure_metrics" not in row
    assert any("no gold" in note for note in limitations)
    assert "structure_metrics" not in json.dumps(row)


def test_a_gold_that_marks_no_heading_measures_nothing(corpus_dir: Path) -> None:
    """Measuring the absence of a claim is not measuring the placement of one."""
    write_json(corpus_dir / "headless.json", gold_payload(gold_page(headings=())))
    runner = declared_runner(corpus_dir)

    row, _limitations = score_row(
        runner, artifact(artifact_page()), gold_paths={DOC_ID: "headless.json"}
    )

    metrics = row["structure_metrics"]
    assert metrics["status"] == "not_measured"
    assert metrics["reason"] == "no_eligible_pages"
    assert metrics["pages"]["marked"] == 0
    assert metrics["kinds"]["heading"]["f1"]["value"] is None


def test_a_run_that_cannot_be_aligned_abstains(corpus_dir: Path) -> None:
    """Words that cannot be located in the gold cannot have boundaries located there."""
    row, _limitations = score_row(
        declared_runner(corpus_dir), artifact(artifact_page(blocks=unaligned_blocks()))
    )

    metrics = row["structure_metrics"]
    assert metrics["status"] == "not_measured"
    assert metrics["reason"] == "alignment_below_threshold"
    assert metrics["skipped"] == {"alignment_below_threshold": 1}
    assert metrics["kinds"]["heading"]["f1"]["value"] is None


def test_a_self_scoring_page_abstains(corpus_dir: Path) -> None:
    """A run that hands back the gold verbatim would score perfectly and say nothing."""
    write_json(
        corpus_dir / "gold.json",
        gold_payload(gold_page(text="\n".join(entry["text"] for entry in run_blocks()))),
    )

    row, _limitations = score_row(declared_runner(corpus_dir), artifact(artifact_page()))

    metrics = row["structure_metrics"]
    assert metrics["status"] == "not_measured"
    assert metrics["reason"] == "self_scoring"


def test_a_page_the_engine_could_not_read_abstains(corpus_dir: Path) -> None:
    """The engine's own word for it, not a blanket failure name."""
    row, _limitations = score_row(
        declared_runner(corpus_dir),
        artifact(artifact_page(status="unsupported_modality", blocks=[])),
    )

    metrics = row["structure_metrics"]
    assert metrics["status"] == "not_measured"
    assert metrics["reason"] == "unsupported_modality"
    assert metrics["skipped"] == {"unsupported_modality": 1}


def test_a_gold_page_the_run_never_read_is_counted_and_skipped(corpus_dir: Path) -> None:
    """The page the run did read still scores; the one it never read is named."""
    write_json(corpus_dir / "two-pages.json", gold_payload(gold_page(0), gold_page(1)))
    runner = make_runner(
        corpus_dir,
        make_manifest(corpus_dir, page_indices=[0, 1], gold_path="two-pages.json"),
    )

    row, _limitations = score_row(runner, artifact(artifact_page(1)))

    metrics = row["structure_metrics"]
    assert metrics["status"] == "measured"
    assert metrics["skipped"] == {"hypothesis_missing": 1}
    assert metrics["pages"] == {"marked": 2, "considered": 2, "scored": 1, "skipped": 1}
    assert metrics["pages_detail"][0]["reason"] == "hypothesis_missing"


def test_the_report_carries_structure_metrics_and_no_page_text(corpus_dir: Path) -> None:
    """The whole public path: compare writes a report, and the report holds no prose."""
    runner = declared_runner(corpus_dir)
    digest = store_artifact(runner, artifact(artifact_page()))
    run_manifest(runner, "vl12-fixture.vision.r1", digest)

    result = runner.compare("ocr", [RunRef(DOC_ID, "vl12-fixture.vision.r1")], write=False)

    report = result["report"]
    row = report["metadata"]["run_rows"][0]
    assert row["structure_metrics"]["status"] == "measured"
    assert row["structure_metrics"]["kinds"]["heading"]["f1"]["value"] == 1.0
    assert row["gold_digest"] == load_gold(corpus_dir / "gold.json").content_digest
    assert find_raw_text_keys(report) == []
    serialized = json.dumps(report, ensure_ascii=False)
    for prose in (*HEADINGS, BODY, LAST_BODY):
        assert prose not in serialized
