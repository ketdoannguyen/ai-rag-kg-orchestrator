"""Tests for runner.py — distractor injection, validation, progress tracking."""

from __future__ import annotations

import random

from workflows.benchmark.models import (
    BenchmarkQuery,
    ConfidenceScore,
    FactNode,
    Subgraph,
    SubgraphNode,
    SubgraphRelation,
)
from workflows.benchmark.runner import (
    _get_completed_subgraphs,
    _load_progress,
    _save_progress,
    inject_distractors,
    inject_distractors_batch,
    validate_queries,
)

# ---- Fixtures ---- #


def _make_subgraph(n_nodes: int = 6) -> Subgraph:
    """Create a test subgraph with n_nodes content nodes across 2 sections."""
    nodes = []
    for i in range(n_nodes):
        sec_key = "sections/sec_A" if i < n_nodes // 2 else "sections/sec_B"
        sec_title = "Section A" if i < n_nodes // 2 else "Section B"
        nodes.append(
            SubgraphNode(
                node_id=f"content_nodes/n{i}",
                text=f"Node {i} text",
                parent_section_key=sec_key,
                parent_section_title=sec_title,
            )
        )

    return Subgraph(
        subgraph_id="sg_test",
        anchor_key="sections/sec_A",
        anchor_title="Test Section",
        breadcrumb="Ch1 > Test",
        nodes=nodes,
        core_node_ids=[n.node_id for n in nodes[:3]],
        enriched_node_ids=[n.node_id for n in nodes[3:]],
        relations=[
            SubgraphRelation(
                from_id="content_nodes/n0",
                to_id="content_nodes/n1",
                relation_type="CAUSED_BY",
            )
        ],
        content_count=n_nodes,
        relation_count=1,
        tier="medium",
        query_allocation=5,
    )


def _make_query(
    query_id: str = "q1",
    fact_node_ids: list[str] | None = None,
    grades: list[int] | None = None,
) -> BenchmarkQuery:
    """Create a test query with specified fact nodes."""
    if fact_node_ids is None:
        fact_node_ids = ["content_nodes/n0", "content_nodes/n1"]
    if grades is None:
        grades = [2, 1]

    fact_nodes = [
        FactNode(node_id=nid, grade=g, parent_section="Section A")
        for nid, g in zip(fact_node_ids, grades)
    ]

    return BenchmarkQuery(
        query_id=query_id,
        query_text="Test query?",
        subgraph_id="sg_test",
        fact_nodes=fact_nodes,
        relations_used=["CAUSED_BY"],
        confidence=ConfidenceScore(
            question_clarity=8,
            fact_node_completeness=9,
            fact_node_precision=8,
            grade_certainty=7,
            overall=82,
        ),
        confidence_concerns="",
        reasoning="Test reasoning",
    )


# ---- Tests: Distractor Injection ---- #


class TestInjectDistractors:
    def test_adds_grade0_nodes(self):
        sg = _make_subgraph(6)
        q = _make_query(fact_node_ids=["content_nodes/n0", "content_nodes/n1"])
        result = inject_distractors(q, sg, count=2, rng=random.Random(42))

        assert len(result.fact_nodes) == 4
        new_nodes = result.fact_nodes[2:]
        assert all(fn.grade == 0 for fn in new_nodes)

    def test_no_duplicate_node_ids(self):
        sg = _make_subgraph(6)
        q = _make_query(fact_node_ids=["content_nodes/n0", "content_nodes/n1"])
        result = inject_distractors(q, sg, count=3, rng=random.Random(42))

        all_ids = [fn.node_id for fn in result.fact_nodes]
        assert len(all_ids) == len(set(all_ids))

    def test_prefers_cross_section(self):
        sg = _make_subgraph(6)
        # Grade-2 nodes in sec_A → distractors should prefer sec_B
        q = _make_query(
            fact_node_ids=["content_nodes/n0", "content_nodes/n1"],
            grades=[2, 2],
        )
        result = inject_distractors(q, sg, count=2, rng=random.Random(42))
        new_nodes = result.fact_nodes[2:]
        for fn in new_nodes:
            # New distractor nodes should be from sec_B (n3, n4, n5)
            assert fn.node_id in ("content_nodes/n3", "content_nodes/n4", "content_nodes/n5")

    def test_handles_insufficient_candidates(self):
        sg = _make_subgraph(3)
        q = _make_query(
            fact_node_ids=["content_nodes/n0", "content_nodes/n1"],
            grades=[2, 1],
        )
        result = inject_distractors(q, sg, count=5, rng=random.Random(42))
        # Only 1 candidate left (n2) — should add just that
        assert len(result.fact_nodes) == 3

    def test_empty_subgraph_no_crash(self):
        sg = _make_subgraph(2)
        q = _make_query(
            fact_node_ids=["content_nodes/n0", "content_nodes/n1"],
            grades=[2, 1],
        )
        result = inject_distractors(q, sg, count=2, rng=random.Random(42))
        # All nodes used up, no distractors added
        assert len(result.fact_nodes) == 2


class TestInjectDistractorsBatch:
    def test_deterministic_with_seed(self):
        sg = _make_subgraph(8)
        queries = [_make_query(f"q{i}") for i in range(3)]

        r1 = inject_distractors_batch(queries, sg, count=2, seed=42)
        r2 = inject_distractors_batch(queries, sg, count=2, seed=42)

        for a, b in zip(r1, r2):
            assert [fn.node_id for fn in a.fact_nodes] == [fn.node_id for fn in b.fact_nodes]


# ---- Tests: Validation ---- #


class TestValidateQueries:
    def test_valid_query_passes(self):
        sg = _make_subgraph(6)
        q = _make_query()
        # Add a distractor
        augmented = inject_distractors(q, sg, count=1, rng=random.Random(42))
        valid, issues = validate_queries([augmented], sg)
        assert len(valid) == 1
        assert len(issues) == 0

    def test_invalid_node_id_removed(self):
        sg = _make_subgraph(4)
        q = _make_query(
            fact_node_ids=["content_nodes/n0", "content_nodes/INVALID"],
            grades=[2, 0],
        )
        valid, issues = validate_queries([q], sg)
        assert len(valid) == 0
        assert len(issues) == 1
        assert "invalid_node_ids" in issues[0]["problems"][0]

    def test_no_grade2_flagged(self):
        sg = _make_subgraph(4)
        q = _make_query(
            fact_node_ids=["content_nodes/n0", "content_nodes/n1"],
            grades=[1, 0],
        )
        valid, issues = validate_queries([q], sg)
        assert len(valid) == 0
        assert any("no_grade_2" in p for p in issues[0]["problems"])


# ---- Tests: Progress Tracking ---- #


class TestProgressTracking:
    def test_load_empty_progress(self, tmp_path):
        progress = _load_progress(tmp_path)
        assert "completed" in progress
        assert progress["completed"] == []

    def test_save_and_load_progress(self, tmp_path):
        progress = {"completed": ["sg1", "sg2"], "started_at": "2024-01-01"}
        _save_progress(tmp_path, progress)

        loaded = _load_progress(tmp_path)
        assert loaded["completed"] == ["sg1", "sg2"]
        assert "updated_at" in loaded

    def test_get_completed_subgraphs(self, tmp_path):
        # Create some checkpoint files
        (tmp_path / "sg1.json").write_text("{}")
        (tmp_path / "sg2.json").write_text("{}")
        (tmp_path / "_progress.json").write_text("{}")

        completed = _get_completed_subgraphs(tmp_path)
        assert completed == {"sg1", "sg2"}
        assert "_progress" not in completed
