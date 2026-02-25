"""Tests for human_review.py — export/import review pipeline."""

from __future__ import annotations

import csv
import json

from workflows.benchmark.human_review import (
    ReviewLabel,
    apply_review_labels,
    export_for_review_csv,
    export_for_review_json,
    import_review_labels_csv,
)
from workflows.benchmark.models import BenchmarkQuery, ConfidenceScore, FactNode


def _make_query(query_id: str, direction: str = "TD") -> BenchmarkQuery:
    return BenchmarkQuery(
        query_id=query_id,
        query_text="Test question?",
        subgraph_id="sg_test",
        fact_nodes=[
            FactNode(node_id="content_nodes/n1", grade=2, parent_section="sec/a"),
            FactNode(node_id="content_nodes/n2", grade=0, parent_section="sec/b"),
        ],
        relations_used=[],
        confidence=ConfidenceScore(
            question_clarity=8,
            fact_node_completeness=8,
            fact_node_precision=8,
            grade_certainty=8,
            overall=80,
        ),
        confidence_concerns="",
        reasoning="Test",
        direction=direction,
    )


class TestExportCSV:
    def test_creates_file(self, tmp_path):
        queries = [_make_query("q1"), _make_query("q2")]
        out = tmp_path / "review.csv"
        result = export_for_review_csv(queries, str(out))
        assert out.exists()
        assert result == str(out)

    def test_correct_columns(self, tmp_path):
        queries = [_make_query("q1")]
        out = tmp_path / "review.csv"
        export_for_review_csv(queries, str(out))
        with open(out, encoding="utf-8") as f:
            reader = csv.DictReader(f)
            assert "query_id" in reader.fieldnames
            assert "query_text" in reader.fieldnames
            assert "decision" in reader.fieldnames
            assert "notes" in reader.fieldnames

    def test_data_integrity(self, tmp_path):
        queries = [_make_query("abc123", direction="HZ")]
        out = tmp_path / "review.csv"
        export_for_review_csv(queries, str(out))
        with open(out, encoding="utf-8") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        assert len(rows) == 1
        assert rows[0]["query_id"] == "abc123"
        assert rows[0]["direction"] == "HZ"
        assert rows[0]["decision"] == ""  # Empty for reviewer to fill

    def test_creates_parent_dirs(self, tmp_path):
        queries = [_make_query("q1")]
        out = tmp_path / "sub" / "dir" / "review.csv"
        export_for_review_csv(queries, str(out))
        assert out.exists()


class TestExportJSON:
    def test_creates_file(self, tmp_path):
        queries = [_make_query("q1"), _make_query("q2")]
        out = tmp_path / "review.json"
        result = export_for_review_json(queries, str(out))
        assert out.exists()
        assert result == str(out)

    def test_correct_format(self, tmp_path):
        queries = [_make_query("q1")]
        out = tmp_path / "review.json"
        export_for_review_json(queries, str(out))
        with open(out, encoding="utf-8") as f:
            data = json.load(f)
        assert len(data) == 1
        assert data[0]["query_id"] == "q1"
        assert data[0]["decision"] is None
        assert "fact_nodes" in data[0]

    def test_fact_nodes_structure(self, tmp_path):
        queries = [_make_query("q1")]
        out = tmp_path / "review.json"
        export_for_review_json(queries, str(out))
        with open(out, encoding="utf-8") as f:
            data = json.load(f)
        fn = data[0]["fact_nodes"][0]
        assert "node_id" in fn
        assert "grade" in fn
        assert "parent_section" in fn


class TestImportLabelsCSV:
    def test_import_basic(self, tmp_path):
        csv_path = tmp_path / "labels.csv"
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["query_id", "decision", "notes"])
            writer.writerow(["q1", "approved", "looks good"])
            writer.writerow(["q2", "rejected", ""])
        labels = import_review_labels_csv(str(csv_path))
        assert len(labels) == 2
        assert labels[0].query_id == "q1"
        assert labels[0].decision == "approved"
        assert labels[1].decision == "rejected"

    def test_skips_empty_decision(self, tmp_path):
        csv_path = tmp_path / "labels.csv"
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["query_id", "decision", "notes"])
            writer.writerow(["q1", "", ""])
            writer.writerow(["q2", "approved", "ok"])
        labels = import_review_labels_csv(str(csv_path))
        assert len(labels) == 1
        assert labels[0].query_id == "q2"

    def test_parse_grade_corrections(self, tmp_path):
        csv_path = tmp_path / "labels.csv"
        corrections = json.dumps({"content_nodes/n1": 1})
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["query_id", "decision", "notes", "grade_corrections"])
            writer.writerow(["q1", "fix", "", corrections])
        labels = import_review_labels_csv(str(csv_path))
        assert labels[0].grade_corrections == {"content_nodes/n1": 1}

    def test_invalid_json_corrections_skipped(self, tmp_path):
        csv_path = tmp_path / "labels.csv"
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["query_id", "decision", "notes", "grade_corrections"])
            writer.writerow(["q1", "fix", "", "not valid json"])
        labels = import_review_labels_csv(str(csv_path))
        assert labels[0].grade_corrections is None


class TestApplyLabels:
    def test_approved_sets_status(self):
        queries = [_make_query("q1")]
        labels = [ReviewLabel(query_id="q1", decision="approved")]
        result, summary = apply_review_labels(queries, labels)
        assert len(result) == 1
        assert result[0].human_review_status == "approved"
        assert summary.approved == 1
        assert summary.rejected == 0

    def test_rejected_removes_query(self):
        queries = [_make_query("q1"), _make_query("q2")]
        labels = [ReviewLabel(query_id="q1", decision="rejected")]
        result, summary = apply_review_labels(queries, labels)
        assert len(result) == 1
        assert result[0].query_id == "q2"
        assert summary.rejected == 1

    def test_fix_applies_grade_corrections(self):
        queries = [_make_query("q1")]
        labels = [
            ReviewLabel(
                query_id="q1",
                decision="fix",
                grade_corrections={"content_nodes/n1": 1, "content_nodes/n2": 2},
            )
        ]
        result, summary = apply_review_labels(queries, labels)
        assert summary.fixed == 1
        assert result[0].human_review_status == "approved"
        grades = {fn.node_id: fn.grade for fn in result[0].fact_nodes}
        assert grades["content_nodes/n1"] == 1
        assert grades["content_nodes/n2"] == 2

    def test_unknown_decision_keeps_query(self):
        queries = [_make_query("q1")]
        labels = [ReviewLabel(query_id="q1", decision="maybe")]
        result, summary = apply_review_labels(queries, labels)
        assert len(result) == 1

    def test_unlabeled_query_kept_as_is(self):
        queries = [_make_query("q1"), _make_query("q2")]
        labels = [ReviewLabel(query_id="q1", decision="approved")]
        result, summary = apply_review_labels(queries, labels)
        assert len(result) == 2  # q2 kept without label

    def test_not_found_counted(self):
        queries = [_make_query("q1")]
        labels = [
            ReviewLabel(query_id="q1", decision="approved"),
            ReviewLabel(query_id="GHOST", decision="approved"),
        ]
        _, summary = apply_review_labels(queries, labels)
        assert summary.not_found == 1

    def test_empty_queries(self):
        labels = [ReviewLabel(query_id="q1", decision="approved")]
        result, summary = apply_review_labels([], labels)
        assert result == []
        assert summary.not_found == 1
