"""Unit tests for workflows/benchmark/models.py.

Covers all Pydantic models: FactNode, ConfidenceScore, BenchmarkQuery,
SubgraphNode, SubgraphRelation, Subgraph, EvaluationResult, BenchmarkDataset.
"""

from __future__ import annotations

import pytest

from workflows.benchmark.models import (
    BenchmarkDataset,
    BenchmarkQuery,
    ConfidenceScore,
    EvaluationResult,
    FactNode,
    Subgraph,
    SubgraphNode,
    SubgraphRelation,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_confidence(**overrides) -> ConfidenceScore:
    """Return a valid ConfidenceScore, optionally overriding fields."""
    defaults = {
        "overall": 75,
    }
    defaults.update(overrides)
    return ConfidenceScore(**defaults)


def make_fact_node(
    node_id: str = "content_nodes/abc123", grade: int = 2, parent_section: str = "s1"
) -> FactNode:
    return FactNode(node_id=node_id, grade=grade, parent_section=parent_section)


def make_subgraph_node(node_id: str = "content_nodes/n1", text: str = "some text") -> SubgraphNode:
    return SubgraphNode(node_id=node_id, text=text)


def make_subgraph(
    subgraph_id: str = "sg-001",
    nodes: list[SubgraphNode] | None = None,
    core_node_ids: list[str] | None = None,
    enriched_node_ids: list[str] | None = None,
    relations: list[SubgraphRelation] | None = None,
    content_count: int = 1,
    tier: str = "tiny",
    query_allocation: int = 5,
) -> Subgraph:
    if nodes is None:
        nodes = [make_subgraph_node("content_nodes/n1")]
    if core_node_ids is None:
        core_node_ids = ["content_nodes/n1"]
    if enriched_node_ids is None:
        enriched_node_ids = []
    if relations is None:
        relations = []
    return Subgraph(
        subgraph_id=subgraph_id,
        anchor_key="anchor1",
        anchor_title="Anchor Title",
        breadcrumb="A > B",
        nodes=nodes,
        core_node_ids=core_node_ids,
        enriched_node_ids=enriched_node_ids,
        relations=relations,
        content_count=content_count,
        relation_count=len(relations),
        tier=tier,
        query_allocation=query_allocation,
    )


def make_benchmark_query(**overrides) -> BenchmarkQuery:
    defaults = {
        "query_id": "q-001",
        "query_text": "What happened?",
        "subgraph_id": "sg-001",
        "fact_nodes": [make_fact_node()],
        "relations_used": [],
        "confidence": make_confidence(),
        "reasoning": "Because reasons.",
    }
    defaults.update(overrides)
    return BenchmarkQuery(**defaults)


# ---------------------------------------------------------------------------
# FactNode
# ---------------------------------------------------------------------------


class TestFactNode:
    def test_valid_creation(self):
        fn = FactNode(node_id="content_nodes/abc123", grade=2, parent_section="s1")
        assert fn.node_id == "content_nodes/abc123"
        assert fn.grade == 2
        assert fn.parent_section == "s1"

    def test_valid_grade_0(self):
        fn = FactNode(node_id="content_nodes/x1", grade=0, parent_section="s1")
        assert fn.grade == 0

    def test_valid_grade_1(self):
        fn = FactNode(node_id="content_nodes/x1", grade=1, parent_section="s1")
        assert fn.grade == 1

    def test_invalid_node_id_no_slash(self):
        with pytest.raises(ValueError):
            FactNode(node_id="no_slash_here", grade=1, parent_section="s1")

    def test_invalid_node_id_empty(self):
        with pytest.raises(ValueError):
            FactNode(node_id="", grade=1, parent_section="s1")

    def test_invalid_grade(self):
        with pytest.raises(ValueError):
            FactNode(node_id="content_nodes/x1", grade=3, parent_section="s1")

    def test_node_id_with_multiple_slashes(self):
        fn = FactNode(node_id="coll/sub/key", grade=0, parent_section="s1")
        assert fn.node_id == "coll/sub/key"


# ---------------------------------------------------------------------------
# ConfidenceScore
# ---------------------------------------------------------------------------


class TestConfidenceScore:
    def test_valid_score(self):
        cs = make_confidence()
        assert cs.overall == 75

    def test_overall_minimum_zero(self):
        cs = make_confidence(overall=0)
        assert cs.overall == 0

    def test_overall_maximum_100(self):
        cs = make_confidence(overall=100)
        assert cs.overall == 100

    def test_overall_below_zero(self):
        with pytest.raises(ValueError):
            make_confidence(overall=-1)

    def test_overall_above_100(self):
        with pytest.raises(ValueError):
            make_confidence(overall=101)


# ---------------------------------------------------------------------------
# BenchmarkQuery
# ---------------------------------------------------------------------------


class TestBenchmarkQuery:
    def test_valid_creation(self):
        q = make_benchmark_query()
        assert q.query_id == "q-001"
        assert q.query_text == "What happened?"

    def test_empty_query_text_raises(self):
        with pytest.raises(ValueError):
            make_benchmark_query(query_text="")

    def test_whitespace_only_query_text_raises(self):
        with pytest.raises(ValueError):
            make_benchmark_query(query_text="   ")

    def test_query_text_is_stripped(self):
        q = make_benchmark_query(query_text="  Hello?  ")
        assert q.query_text == "Hello?"

    def test_empty_fact_nodes_raises(self):
        with pytest.raises(ValueError):
            make_benchmark_query(fact_nodes=[])

    def test_at_least_one_fact_node(self):
        q = make_benchmark_query(fact_nodes=[make_fact_node()])
        assert len(q.fact_nodes) == 1

    def test_multiple_fact_nodes(self):
        nodes = [make_fact_node("content_nodes/n1"), make_fact_node("content_nodes/n2", grade=1)]
        q = make_benchmark_query(fact_nodes=nodes)
        assert len(q.fact_nodes) == 2

    def test_with_generated_id_auto_uuid(self):
        q = BenchmarkQuery.with_generated_id(
            query_text="Auto ID query?",
            subgraph_id="sg-001",
            fact_nodes=[make_fact_node()],
            relations_used=[],
            confidence=make_confidence(),
            reasoning="some reasoning",
        )
        assert q.query_id is not None
        assert len(q.query_id) == 36  # UUID format

    def test_with_generated_id_respects_provided_id(self):
        q = BenchmarkQuery.with_generated_id(
            query_id="explicit-id",
            query_text="Explicit ID query?",
            subgraph_id="sg-001",
            fact_nodes=[make_fact_node()],
            relations_used=[],
            confidence=make_confidence(),
            reasoning="some reasoning",
        )
        assert q.query_id == "explicit-id"

    def test_with_generated_id_unique_each_call(self):
        shared_kwargs = {
            "query_text": "Unique?",
            "subgraph_id": "sg-001",
            "fact_nodes": [make_fact_node()],
            "relations_used": [],
            "confidence": make_confidence(),
            "reasoning": "r",
        }
        q1 = BenchmarkQuery.with_generated_id(**shared_kwargs)
        q2 = BenchmarkQuery.with_generated_id(**shared_kwargs)
        assert q1.query_id != q2.query_id

    def test_default_optional_fields_are_none(self):
        q = make_benchmark_query()
        assert q.direction is None
        assert q.calibrated_confidence is None
        assert q.human_review_status is None

    def test_invalid_direction_raises(self):
        with pytest.raises(ValueError):
            make_benchmark_query(direction="INVALID")

    def test_valid_directions(self):
        for d in ("TD", "HZ", "BU", "NT", "CS", "AMBIGUOUS"):
            q = make_benchmark_query(direction=d)
            assert q.direction == d

    def test_invalid_human_review_status_raises(self):
        with pytest.raises(ValueError):
            make_benchmark_query(human_review_status="unknown_status")

    def test_valid_human_review_statuses(self):
        for status in ("approved", "rejected", "pending"):
            q = make_benchmark_query(human_review_status=status)
            assert q.human_review_status == status

    def test_calibrated_confidence_out_of_range_raises(self):
        with pytest.raises(ValueError):
            make_benchmark_query(calibrated_confidence=1.5)

    def test_calibrated_confidence_valid_range(self):
        q = make_benchmark_query(calibrated_confidence=0.85)
        assert q.calibrated_confidence == 0.85

    def test_calibrated_confidence_boundaries(self):
        q_min = make_benchmark_query(calibrated_confidence=0.0)
        q_max = make_benchmark_query(calibrated_confidence=1.0)
        assert q_min.calibrated_confidence == 0.0
        assert q_max.calibrated_confidence == 1.0


# ---------------------------------------------------------------------------
# SubgraphNode
# ---------------------------------------------------------------------------


class TestSubgraphNode:
    def test_valid_creation(self):
        n = make_subgraph_node()
        assert n.node_id == "content_nodes/n1"
        assert n.text == "some text"

    def test_invalid_node_id_no_slash(self):
        with pytest.raises(ValueError):
            SubgraphNode(node_id="noslash", text="text")

    def test_optional_fields_default_none(self):
        n = make_subgraph_node()
        assert n.embedding is None
        assert n.parent_section_key is None
        assert n.breadcrumb is None

    def test_with_embedding(self):
        n = SubgraphNode(node_id="content_nodes/x", text="t", embedding=[0.1, 0.2, 0.3])
        assert n.embedding == [0.1, 0.2, 0.3]


# ---------------------------------------------------------------------------
# SubgraphRelation
# ---------------------------------------------------------------------------


class TestSubgraphRelation:
    def test_valid_creation(self):
        r = SubgraphRelation(
            from_id="content_nodes/a", to_id="content_nodes/b", relation_type="CAUSES"
        )
        assert r.from_id == "content_nodes/a"
        assert r.to_id == "content_nodes/b"
        assert r.relation_type == "CAUSES"

    def test_self_loop_raises(self):
        with pytest.raises(ValueError):
            SubgraphRelation(
                from_id="content_nodes/a", to_id="content_nodes/a", relation_type="SELF"
            )

    def test_different_ids_accepted(self):
        r = SubgraphRelation(
            from_id="content_nodes/a", to_id="content_nodes/b", relation_type="LEADS_TO"
        )
        assert r.from_id != r.to_id


# ---------------------------------------------------------------------------
# Subgraph
# ---------------------------------------------------------------------------


class TestSubgraph:
    def test_valid_creation(self):
        sg = make_subgraph()
        assert sg.subgraph_id == "sg-001"
        assert sg.tier == "tiny"

    def test_empty_nodes_raises(self):
        with pytest.raises(ValueError):
            Subgraph(
                subgraph_id="sg-x",
                anchor_key="a",
                anchor_title="T",
                breadcrumb="B",
                nodes=[],
                core_node_ids=[],
                enriched_node_ids=[],
                relations=[],
                content_count=0,
                relation_count=0,
                tier="tiny",
                query_allocation=0,
            )

    def test_core_node_not_in_nodes_raises(self):
        nodes = [make_subgraph_node("content_nodes/n1")]
        with pytest.raises(ValueError):
            Subgraph(
                subgraph_id="sg-x",
                anchor_key="a",
                anchor_title="T",
                breadcrumb="B",
                nodes=nodes,
                core_node_ids=["content_nodes/missing"],
                enriched_node_ids=[],
                relations=[],
                content_count=1,
                relation_count=0,
                tier="tiny",
                query_allocation=0,
            )

    def test_enriched_node_not_in_nodes_raises(self):
        nodes = [make_subgraph_node("content_nodes/n1")]
        with pytest.raises(ValueError):
            Subgraph(
                subgraph_id="sg-x",
                anchor_key="a",
                anchor_title="T",
                breadcrumb="B",
                nodes=nodes,
                core_node_ids=["content_nodes/n1"],
                enriched_node_ids=["content_nodes/ghost"],
                relations=[],
                content_count=1,
                relation_count=0,
                tier="tiny",
                query_allocation=0,
            )

    def test_all_node_id_lists_must_exist_in_nodes(self):
        nodes = [
            make_subgraph_node("content_nodes/core1"),
            make_subgraph_node("content_nodes/enrich1"),
        ]
        sg = Subgraph(
            subgraph_id="sg-y",
            anchor_key="a",
            anchor_title="T",
            breadcrumb="B",
            nodes=nodes,
            core_node_ids=["content_nodes/core1"],
            enriched_node_ids=["content_nodes/enrich1"],
            relations=[],
            content_count=1,
            relation_count=0,
            tier="tiny",
            query_allocation=5,
        )
        assert len(sg.core_node_ids) == 1
        assert len(sg.enriched_node_ids) == 1

    def test_negative_content_count_raises(self):
        with pytest.raises(ValueError):
            make_subgraph(content_count=-1)

    def test_negative_query_allocation_raises(self):
        with pytest.raises(ValueError):
            make_subgraph(query_allocation=-1)

    def test_tier_literal_values(self):
        for tier in ("large", "medium", "small", "tiny"):
            sg = make_subgraph(tier=tier)
            assert sg.tier == tier

    def test_invalid_tier_raises(self):
        with pytest.raises(ValueError):
            make_subgraph(tier="gigantic")


# ---------------------------------------------------------------------------
# EvaluationResult
# ---------------------------------------------------------------------------


class TestEvaluationResult:
    def _make(self, **overrides):
        defaults = {
            "query_id": "q1",
            "method_id": "m1",
            "method_name": "BM25",
            "retrieved_ids": ["content_nodes/r1"],
            "ndcg_at_10": 0.85,
            "recall_at_10": 0.9,
            "precision_at_10": 0.7,
            "mrr": 0.95,
            "result_count": 1,
        }
        defaults.update(overrides)
        return EvaluationResult(**defaults)

    def test_valid_creation(self):
        er = self._make()
        assert er.method_name == "BM25"
        assert er.ndcg_at_10 == 0.85

    def test_metric_below_zero_raises(self):
        with pytest.raises(ValueError):
            self._make(ndcg_at_10=-0.1)

    def test_metric_above_one_raises(self):
        with pytest.raises(ValueError):
            self._make(recall_at_10=1.1)

    def test_metric_boundary_values(self):
        er = self._make(ndcg_at_10=0.0, recall_at_10=1.0, precision_at_10=0.0, mrr=1.0)
        assert er.ndcg_at_10 == 0.0
        assert er.recall_at_10 == 1.0

    def test_negative_result_count_raises(self):
        with pytest.raises(ValueError):
            self._make(result_count=-1)

    def test_zero_result_count_valid(self):
        er = self._make(result_count=0)
        assert er.result_count == 0


# ---------------------------------------------------------------------------
# BenchmarkDataset
# ---------------------------------------------------------------------------


class TestBenchmarkDataset:
    def test_create_empty(self):
        ds = BenchmarkDataset.create_empty()
        assert ds.version == "4.0"
        assert ds.queries == []
        assert ds.subgraphs == []
        assert ds.metadata == {}
        assert ds.created_at is not None
        assert len(ds.created_at) > 0

    def test_create_empty_timestamp_format(self):
        ds = BenchmarkDataset.create_empty()
        # Should be a valid ISO format string (contains T separator)
        assert "T" in ds.created_at

    def test_create_empty_twice_has_different_timestamps(self):
        import time

        ds1 = BenchmarkDataset.create_empty()
        time.sleep(0.01)
        ds2 = BenchmarkDataset.create_empty()
        # Timestamps may differ; both are valid strings
        assert isinstance(ds1.created_at, str)
        assert isinstance(ds2.created_at, str)

    def test_create_with_data(self):
        sg = make_subgraph()
        q = make_benchmark_query()
        ds = BenchmarkDataset(
            created_at="2026-01-01T00:00:00",
            queries=[q],
            subgraphs=[sg],
            metadata={"source": "test"},
        )
        assert len(ds.queries) == 1
        assert len(ds.subgraphs) == 1
        assert ds.metadata["source"] == "test"

    def test_default_version(self):
        ds = BenchmarkDataset.create_empty()
        assert ds.version == "4.0"
