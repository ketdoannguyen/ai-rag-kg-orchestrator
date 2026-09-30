"""Synthetic tests for research.ocr_pipeline.report.

The three renderers must agree on counts, and no page text may leak into any
format. Everything is built from dicts created in this file; the usage sections
come from the accounting module itself, so the report is fed what a run really
produces — attempts and input/output tokens, and no money anywhere.
"""

from __future__ import annotations

import csv
import io
import json
from collections.abc import Callable

import pytest

from research.ocr_pipeline.metrics import chunk_metrics, corpus_text_metrics
from research.ocr_pipeline.report import (
    MAX_STRING_LENGTH,
    build_report,
    find_raw_text_keys,
    to_csv,
    to_json,
    to_markdown,
)
from research.ocr_pipeline.usage import paired_usage, run_usage

GOLD = "con mèo ngồi trên bàn"
OCR = "con mèo ngồi trên bàn"

USED = {"attempts": 2, "prompt_tokens": 1000, "completion_tokens": 500}


def _text_metrics() -> dict:
    return corpus_text_metrics(
        [
            {
                "doc_id": "d1",
                "pages": [
                    {"page": 0, "reference": GOLD, "hypothesis": OCR},
                    {"page": 1, "reference": "trên bàn", "hypothesis": "tren ban"},
                ],
            }
        ]
    )


def _chunk_metrics() -> dict:
    return chunk_metrics(
        [
            {"chunk_id": "c1", "doc_id": "d1", "text": "con mèo", "pages": [0]},
            {"chunk_id": "c2", "doc_id": "d1", "text": "trên bàn", "pages": [1]},
        ],
        max_chunk_size=1024,
        page_totals={"d1": 2},
    )


def _stages() -> dict:
    return {
        "extract": {
            "status": "measured",
            "reason": None,
            "value": 12.5,
            "unit": "seconds",
            "attempts": 2,
            "retries": 1,
        },
        "merge": {"status": "not_measured", "reason": "failed", "value": None, "unit": "seconds"},
    }


def _resources() -> dict:
    return {
        "extract": {
            "wall": {
                "status": "measured",
                "reason": None,
                "value": 12.5,
                "unit": "seconds",
                "estimate": False,
            },
            "rss_peak": {
                "status": "measured",
                "reason": None,
                "value": 1048576,
                "unit": "bytes",
                "estimate": True,
            },
            "gpu_vram": {
                "status": "not_available",
                "reason": "no_drm_vram_counter",
                "value": None,
                "unit": "bytes",
                "estimate": False,
            },
            "sampling": {"interval_seconds": 0.05, "samples": 12, "tree_peak": 3},
        },
        "merge": {
            "wall": {
                "status": "not_available",
                "reason": "hosted_backend",
                "value": None,
                "unit": "seconds",
                "estimate": False,
            }
        },
    }


def _fidelity() -> dict:
    return {
        "vision_groups": {
            "omitted_chars": {
                "status": "measured",
                "reason": None,
                "value": 4,
                "unit": "characters",
            },
            "added_chars": {
                "status": "measured",
                "reason": None,
                "value": 0,
                "unit": "characters",
            },
        },
        "paired": {
            "vision_groups_vs_atomic": {
                "chunk_count_delta": {
                    "status": "measured",
                    "reason": None,
                    "value": -1,
                    "unit": "chunks",
                }
            }
        },
    }


def _run(run_id: str, prompt_tokens: int) -> dict:
    """One run's documents, as the usage accounting reads them."""
    document = {"doc_id": "d1", "model": "m", "usage": {**USED, "prompt_tokens": prompt_tokens}}
    return {"run_id": run_id, "documents": [document]}


def _usage() -> dict:
    return run_usage(_run("vision", 1000)["documents"])


def _paired_usage() -> dict:
    return paired_usage(_run("vision", 1000), _run("baseline", 2000))


def _report() -> dict:
    return build_report(
        {"run_id": "vision", "engine": "vision", "model": "m"},
        metadata={"study_id": "s5-synthetic", "family": "ocr", "repeat": 1, "cold": True},
        stages=_stages(),
        resources=_resources(),
        fidelity=_fidelity(),
        limitations=["single_document", "one_page_selection"],
        text_metrics=_text_metrics(),
        chunk_metrics=_chunk_metrics(),
        usage=_usage(),
        paired_usage=_paired_usage(),
    )


def _csv_counts(text: str) -> dict[str, int | None]:
    rows = csv.DictReader(io.StringIO(text))
    return {
        row["metric"]: int(row["value"]) if row["value"] else None
        for row in rows
        if row["section"] == "counts"
    }


def _markdown_counts(text: str) -> dict[str, int | None]:
    counts: dict[str, int | None] = {}
    section = None
    for line in text.splitlines():
        if line.startswith("## "):
            section = line[3:].strip()
            continue
        if section != "counts" or not line.startswith("|"):
            continue
        cells = [cell.strip().replace("\\|", "|") for cell in line.strip("|").split("|")]
        if cells[0] in ("metric", "---"):
            continue
        counts[cells[0]] = int(cells[2]) if cells[2] else None
    return counts


class TestBuildReport:
    """Counts are derived from the metric sections, metadata is whitelisted."""

    def test_counts_are_derived(self) -> None:
        assert _report()["counts"] == {
            "documents": 1,
            "documents_with_gold": 1,
            "pages_total": 2,
            "pages_with_gold": 2,
            "pages_without_gold": 0,
            "pages_missing_hypothesis": 0,
            "pages_empty_hypothesis": 0,
            "chunks": 2,
            "chunks_with_page_metadata": 2,
            "documents_with_usage": 1,
            "attempts": 2,
            "prompt_tokens": 1000,
            "completion_tokens": 500,
            "paired_documents": 1,
            "union_documents": 1,
            "stages_total": 2,
            "stages_measured": 1,
            "resources_stages": 2,
            "fidelity_profiles": 2,
            "limitations": 2,
        }

    def test_run_metadata_is_whitelisted(self) -> None:
        report = build_report({"run_id": "r1", "engine": "e", "model": "m", "secret": "/tmp/x"})
        assert report["run"] == {"run_id": "r1", "engine": "e", "model": "m"}

    def test_missing_sections_are_typed_misses(self) -> None:
        report = build_report()
        assert report["text_metrics"] == {"status": "not_measured", "reason": "not_provided"}
        assert report["counts"]["documents"] is None

    def test_rejects_raw_transcription(self) -> None:
        with pytest.raises(ValueError, match="raw transcription"):
            build_report(text_metrics={"hypothesis": "con mèo"})
        with pytest.raises(ValueError, match="raw transcription"):
            build_report(chunk_metrics={"char_length": {"text": "con mèo"}})
        with pytest.raises(ValueError, match="raw transcription"):
            build_report({"run_id": "x" * (MAX_STRING_LENGTH + 1)})


class TestFindRawTextKeys:
    """Detection of text-like keys and over-long strings."""

    def test_flags_nested_keys_and_long_strings(self) -> None:
        payload = {"a": [{"hypothesis": "abc"}, {"notes": "x" * 600}]}
        assert find_raw_text_keys(payload) == ["a.0.hypothesis", "a.1.notes"]

    def test_accepts_short_metadata(self) -> None:
        payload = {"run_id": "r1", "counts": {"pages": 3}, "label": "vision baseline"}
        assert find_raw_text_keys(payload) == []

    def test_rejects_raw_text_in_tuples(self) -> None:
        assert find_raw_text_keys({"metadata": ({"title": "private page title"},)}) == [
            "metadata.0.title"
        ]
        with pytest.raises(ValueError, match="raw transcription"):
            build_report(metadata={"items": ({"text": "private page text"},)})


class TestRenderers:
    """JSON, CSV and Markdown stay in sync and stay text-free."""

    def test_counts_are_identical_across_formats(self) -> None:
        report = _report()
        assert _csv_counts(to_csv(report)) == report["counts"]
        assert _markdown_counts(to_markdown(report)) == report["counts"]
        assert json.loads(to_json(report))["counts"] == report["counts"]

    def test_no_transcription_reaches_any_format(self) -> None:
        report = _report()
        for rendered in (to_json(report), to_csv(report), to_markdown(report)):
            assert GOLD not in rendered
            assert "tren ban" not in rendered
        # The rendered JSON must also survive the raw-text guard unchanged.
        assert find_raw_text_keys(json.loads(to_json(report))) == []

    @pytest.mark.parametrize("render", [to_json, to_csv, to_markdown])
    def test_renderers_reject_raw_text(self, render) -> None:
        with pytest.raises(ValueError, match="raw transcription"):
            render({"text_metrics": {"hypothesis": "con mèo"}})

    def test_csv_carries_every_section(self) -> None:
        rows = list(csv.DictReader(io.StringIO(to_csv(_report()))))
        sections = {row["section"] for row in rows}
        assert {
            "run",
            "counts",
            "text_metrics",
            "chunk_metrics",
            "usage",
            "paired_usage",
        } <= sections
        usage_rows = {row["metric"]: row for row in rows if row["section"] == "usage"}
        assert usage_rows["prompt_tokens"]["value"] == "1000"
        assert usage_rows["completion_tokens"]["value"] == "500"
        paired_rows = {row["metric"]: row for row in rows if row["section"] == "paired_usage"}
        assert paired_rows["delta.prompt_tokens"]["value"] == "1000"

    def test_markdown_escapes_pipes(self) -> None:
        report = build_report({"run_id": "a|b"})
        assert "a\\|b" in to_markdown(report)
        assert "## counts" in to_markdown(report)

    def test_empty_report_still_renders(self) -> None:
        report = build_report()
        assert json.loads(to_json(report)) == report
        counts = _csv_counts(to_csv(report))
        assert counts == _markdown_counts(to_markdown(report))
        # A section that was not provided counts as missing, not as zero; only
        # the limitation list has a length even when nothing was measured.
        assert counts == {**dict.fromkeys(report["counts"]), "limitations": 0}

    def test_json_is_stable_and_newline_terminated(self) -> None:
        report = _report()
        assert to_json(report) == to_json(report)
        assert to_json(report).endswith("\n")


class TestTitleLeakage:
    """A document title is source text: it may not enter a report, in any key."""

    def test_a_title_may_not_enter_a_report(self) -> None:
        with pytest.raises(ValueError, match="raw transcription"):
            build_report({"run_id": "r1", "title": "SGK Vật lí 12"})
        with pytest.raises(ValueError, match="raw transcription"):
            build_report(metadata={"title": "SGK Vật lí 12"})
        with pytest.raises(ValueError, match="raw transcription"):
            build_report(text_metrics={"title": "SGK Vật lí 12"})

    @pytest.mark.parametrize("render", [to_json, to_csv, to_markdown])
    def test_renderers_reject_a_title(self, render: Callable[[dict], str]) -> None:
        with pytest.raises(ValueError, match="raw transcription"):
            render({"metadata": {"title": "SGK Vật lí 12"}})

    def test_an_over_long_key_is_not_a_metric_name(self) -> None:
        long_key = "x" * (MAX_STRING_LENGTH + 1)
        assert find_raw_text_keys({"stages": {long_key: 1}}) == ["stages." + long_key]

    def test_the_rendered_heading_is_never_derived_from_payload_text(self) -> None:
        report = _report()
        markdown = to_markdown(report)
        assert markdown.startswith("# OCR run report")
        for rendered in (to_json(report), to_csv(report), markdown):
            assert "title" not in rendered.lower()


class TestStudySections:
    """Stage, resource and fidelity metadata travel with the study report."""

    def test_new_sections_render_in_every_format(self) -> None:
        report = _report()
        assert {"metadata", "stages", "resources", "fidelity", "limitations"} <= set(report)
        assert json.loads(to_json(report))["counts"] == report["counts"]
        assert _csv_counts(to_csv(report)) == report["counts"]
        assert _markdown_counts(to_markdown(report)) == report["counts"]

    def test_absent_sections_are_typed_misses(self) -> None:
        report = build_report()
        for section in ("metadata", "stages", "resources", "fidelity"):
            assert report[section] == {"status": "not_measured", "reason": "not_provided"}
        assert report["limitations"] == []
        assert report["counts"]["stages_total"] is None
        assert report["counts"]["limitations"] == 0
        # A missing section still renders its status in text formats.
        assert "not_measured" in to_markdown(report)

    def test_stage_statuses_and_resources_reach_the_csv(self) -> None:
        rows = list(csv.DictReader(io.StringIO(to_csv(_report()))))
        stages = {row["metric"]: row for row in rows if row["section"] == "stages"}
        assert stages["extract"]["status"] == "measured"
        assert stages["extract"]["value"] == "12.5"
        assert stages["extract"]["reason"] == ""
        assert stages["merge"]["status"] == "not_measured"
        assert stages["merge"]["reason"] == "failed"
        resources = {row["metric"]: row for row in rows if row["section"] == "resources"}
        assert resources["extract.rss_peak"]["value"] == "1048576"
        assert resources["extract.gpu_vram"]["status"] == "not_available"
        assert resources["merge.wall"]["reason"] == "hosted_backend"

    def test_markdown_shows_why_a_metric_is_missing(self) -> None:
        markdown = to_markdown(_report())
        merge_row = next(line for line in markdown.splitlines() if "| merge |" in line)
        assert "not_measured" in merge_row
        assert "failed" in merge_row

    def test_paired_fidelity_metadata_renders(self) -> None:
        rows = list(csv.DictReader(io.StringIO(to_csv(_report()))))
        fidelity = {row["metric"]: row for row in rows if row["section"] == "fidelity"}
        assert fidelity["paired.vision_groups_vs_atomic.chunk_count_delta"]["value"] == "-1"
        assert fidelity["vision_groups.omitted_chars"]["value"] == "4"

    def test_unread_and_empty_pages_are_counted_apart(self) -> None:
        report = build_report(
            text_metrics=corpus_text_metrics(
                [
                    {
                        "doc_id": "d1",
                        "pages": [
                            {"page": 0, "reference": GOLD, "hypothesis": None},
                            {"page": 1, "reference": GOLD, "hypothesis": ""},
                        ],
                    }
                ]
            )
        )
        assert report["counts"]["pages_missing_hypothesis"] == 1
        assert report["counts"]["pages_empty_hypothesis"] == 1
        assert report["counts"]["pages_with_gold"] == 2

    def test_a_half_reported_run_has_no_token_total(self) -> None:
        # A page that reported nothing is not a page that spent nothing: the
        # subtotal is published, the total stays unknown, and the CSV shows it.
        report = build_report(
            usage=run_usage(
                [{"doc_id": "d1", "model": "m", "usage": USED}, {"doc_id": "d2", "model": "m"}]
            )
        )
        assert report["usage"]["status"] == "partial"
        assert report["usage"]["reason"] == "usage_missing"
        assert report["counts"]["prompt_tokens"] is None
        assert report["usage"]["subtotal"]["prompt_tokens"] == 1000
        rows = {
            row["metric"]: row
            for row in csv.DictReader(io.StringIO(to_csv(report)))
            if row["section"] == "usage"
        }
        assert rows["prompt_tokens"]["value"] == ""
        assert rows["subtotal.prompt_tokens"]["value"] == "1000"

    def test_no_money_reaches_a_report(self) -> None:
        report = _report()
        for rendered in (to_json(report), to_csv(report), to_markdown(report)):
            for word in ("cost", "price", "usd"):
                assert word not in rendered.lower()

    def test_no_study_text_reaches_the_sections(self) -> None:
        report = _report()
        for rendered in (to_json(report), to_csv(report), to_markdown(report)):
            assert GOLD not in rendered
            assert "tren ban" not in rendered
        assert find_raw_text_keys(json.loads(to_json(report))) == []
