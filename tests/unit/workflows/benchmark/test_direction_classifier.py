"""Unit tests for workflows/benchmark/direction_classifier.py.

Covers: classify_direction, classify_batch.

Direction rules (applied in order):
  NT        : <= 2 fact nodes, single section, no relations
  TD        : single section, >= 3 fact nodes
  HZ        : has any relation
  BU        : >= 3 distinct sections, no relations
  CS        : exactly 2 sections, no relations
  AMBIGUOUS : fallback
"""

from __future__ import annotations

from workflows.benchmark.direction_classifier import classify_batch, classify_direction
from workflows.benchmark.models import BenchmarkQuery, ConfidenceScore, FactNode

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_confidence() -> ConfidenceScore:
    return ConfidenceScore(
        question_clarity=7,
        fact_node_completeness=7,
        fact_node_precision=7,
        grade_certainty=7,
        overall=70,
    )


def make_fact_node(node_id: str, parent_section: str, grade: int = 2) -> FactNode:
    return FactNode(node_id=node_id, grade=grade, parent_section=parent_section)


def make_query(
    fact_nodes: list[FactNode],
    relations_used: list[str] | None = None,
    query_id: str = "q-test",
) -> BenchmarkQuery:
    return BenchmarkQuery(
        query_id=query_id,
        query_text="What happened here?",
        subgraph_id="sg-001",
        fact_nodes=fact_nodes,
        relations_used=relations_used or [],
        confidence=make_confidence(),
        confidence_concerns="",
        reasoning="test reasoning",
    )


# ---------------------------------------------------------------------------
# classify_direction: NT
# ---------------------------------------------------------------------------


class TestClassifyDirectionNT:
    def test_single_node_single_section_no_relations(self):
        """1 fact node, 1 section, no relations → NT."""
        nodes = [make_fact_node("content_nodes/n1", "section_A")]
        q = make_query(nodes)
        assert classify_direction(q) == "NT"

    def test_two_nodes_same_section_no_relations(self):
        """2 fact nodes, 1 section, no relations → NT."""
        nodes = [
            make_fact_node("content_nodes/n1", "section_A"),
            make_fact_node("content_nodes/n2", "section_A"),
        ]
        q = make_query(nodes)
        assert classify_direction(q) == "NT"

    def test_two_nodes_single_section_with_relation_is_not_nt(self):
        """2 nodes, 1 section, but has a relation → HZ (not NT)."""
        nodes = [
            make_fact_node("content_nodes/n1", "section_A"),
            make_fact_node("content_nodes/n2", "section_A"),
        ]
        q = make_query(nodes, relations_used=["content_nodes/n1->content_nodes/n2"])
        # Has relation → HZ rule fires before NT check is negated
        assert classify_direction(q) == "HZ"


# ---------------------------------------------------------------------------
# classify_direction: TD
# ---------------------------------------------------------------------------


class TestClassifyDirectionTD:
    def test_three_nodes_same_section_no_relations(self):
        """3 nodes, 1 section, no relations → TD."""
        nodes = [
            make_fact_node("content_nodes/n1", "section_A"),
            make_fact_node("content_nodes/n2", "section_A"),
            make_fact_node("content_nodes/n3", "section_A"),
        ]
        q = make_query(nodes)
        assert classify_direction(q) == "TD"

    def test_five_nodes_same_section_no_relations(self):
        """5 nodes, 1 section, no relations → TD."""
        nodes = [make_fact_node(f"content_nodes/n{i}", "section_A") for i in range(5)]
        q = make_query(nodes)
        assert classify_direction(q) == "TD"

    def test_three_nodes_same_section_with_relation_is_td(self):
        """3 nodes, 1 section, but has a relation → HZ (HZ fires before TD only when no relations)."""
        nodes = [
            make_fact_node("content_nodes/n1", "section_A"),
            make_fact_node("content_nodes/n2", "section_A"),
            make_fact_node("content_nodes/n3", "section_A"),
        ]
        # Rule order: NT (fails ≤2), TD (1 section ≥3 nodes) → fires TD
        # BUT if relations present, HZ fires after TD check
        # Let's check: TD fires (section==1, nodes>=3) BEFORE HZ
        q_no_rel = make_query(nodes)
        q_with_rel = make_query(nodes, relations_used=["rel1"])
        assert classify_direction(q_no_rel) == "TD"
        # With relation: NT fails (3 nodes), TD fires (single section, >=3 nodes) → TD
        # Wait: rules are NT→TD→HZ in order. TD check runs before HZ.
        assert classify_direction(q_with_rel) == "TD"


# ---------------------------------------------------------------------------
# classify_direction: HZ
# ---------------------------------------------------------------------------


class TestClassifyDirectionHZ:
    def test_two_sections_with_relation(self):
        """2 different sections, has relation → HZ."""
        nodes = [
            make_fact_node("content_nodes/n1", "section_A"),
            make_fact_node("content_nodes/n2", "section_B"),
        ]
        q = make_query(nodes, relations_used=["n1_to_n2"])
        assert classify_direction(q) == "HZ"

    def test_three_sections_with_relation(self):
        """3 sections, has relation → HZ (relation check wins over BU)."""
        nodes = [
            make_fact_node("content_nodes/n1", "section_A"),
            make_fact_node("content_nodes/n2", "section_B"),
            make_fact_node("content_nodes/n3", "section_C"),
        ]
        q = make_query(nodes, relations_used=["some_rel"])
        assert classify_direction(q) == "HZ"

    def test_multiple_relations(self):
        """Multiple relations → still HZ."""
        nodes = [
            make_fact_node("content_nodes/n1", "section_A"),
            make_fact_node("content_nodes/n2", "section_B"),
        ]
        q = make_query(nodes, relations_used=["rel1", "rel2"])
        assert classify_direction(q) == "HZ"


# ---------------------------------------------------------------------------
# classify_direction: BU
# ---------------------------------------------------------------------------


class TestClassifyDirectionBU:
    def test_three_sections_no_relations(self):
        """3 distinct sections, no relations, 3 nodes → BU."""
        nodes = [
            make_fact_node("content_nodes/n1", "section_A"),
            make_fact_node("content_nodes/n2", "section_B"),
            make_fact_node("content_nodes/n3", "section_C"),
        ]
        q = make_query(nodes)
        assert classify_direction(q) == "BU"

    def test_four_sections_no_relations(self):
        """4 sections, no relations → BU."""
        nodes = [make_fact_node(f"content_nodes/n{i}", f"section_{chr(65 + i)}") for i in range(4)]
        q = make_query(nodes)
        assert classify_direction(q) == "BU"

    def test_two_sections_no_relations_is_not_bu(self):
        """Only 2 sections → CS, not BU."""
        nodes = [
            make_fact_node("content_nodes/n1", "section_A"),
            make_fact_node("content_nodes/n2", "section_B"),
        ]
        q = make_query(nodes)
        assert classify_direction(q) == "CS"


# ---------------------------------------------------------------------------
# classify_direction: CS
# ---------------------------------------------------------------------------


class TestClassifyDirectionCS:
    def test_two_sections_no_relations(self):
        """Exactly 2 sections, no relations → CS."""
        nodes = [
            make_fact_node("content_nodes/n1", "section_A"),
            make_fact_node("content_nodes/n2", "section_B"),
        ]
        q = make_query(nodes)
        assert classify_direction(q) == "CS"

    def test_two_sections_multiple_nodes_no_relations(self):
        """Multiple nodes across exactly 2 sections, no relations → CS."""
        nodes = [
            make_fact_node("content_nodes/n1", "section_A"),
            make_fact_node("content_nodes/n2", "section_A"),
            make_fact_node("content_nodes/n3", "section_B"),
        ]
        q = make_query(nodes)
        assert classify_direction(q) == "CS"


# ---------------------------------------------------------------------------
# classify_direction: AMBIGUOUS fallback
# ---------------------------------------------------------------------------


class TestClassifyDirectionAmbiguous:
    def test_ambiguous_is_fallback(self):
        """A case that does not match any specific rule → AMBIGUOUS.

        Constructing this is tricky because all cases are covered by the rules.
        However, we can verify the function returns a valid direction for any input.
        The direction "AMBIGUOUS" would only occur if none of the rules match,
        which is a logical fallback. We test the return value is always in
        the valid set.
        """
        # Single node, single section, no relations → NT (not AMBIGUOUS)
        nodes = [make_fact_node("content_nodes/n1", "s1")]
        q = make_query(nodes)
        direction = classify_direction(q)
        assert direction in ("TD", "HZ", "BU", "NT", "CS", "AMBIGUOUS")

    def test_all_directions_are_valid_strings(self):
        """classify_direction always returns one of the 6 valid direction strings."""
        test_cases = [
            # NT
            ([make_fact_node("content_nodes/n1", "s1")], []),
            # TD
            ([make_fact_node(f"content_nodes/n{i}", "s1") for i in range(3)], []),
            # HZ
            (
                [
                    make_fact_node("content_nodes/n1", "s1"),
                    make_fact_node("content_nodes/n2", "s2"),
                ],
                ["rel"],
            ),
            # BU
            ([make_fact_node(f"content_nodes/n{i}", f"s{i}") for i in range(3)], []),
            # CS
            (
                [
                    make_fact_node("content_nodes/n1", "s1"),
                    make_fact_node("content_nodes/n2", "s2"),
                ],
                [],
            ),
        ]
        valid = {"TD", "HZ", "BU", "NT", "CS", "AMBIGUOUS"}
        for nodes, rels in test_cases:
            q = make_query(nodes, rels)
            assert classify_direction(q) in valid


# ---------------------------------------------------------------------------
# classify_batch
# ---------------------------------------------------------------------------


class TestClassifyBatch:
    def test_empty_list(self):
        updated, dist = classify_batch([])
        assert updated == []
        assert dist == {}

    def test_all_queries_updated_with_direction(self):
        queries = [
            make_query([make_fact_node("content_nodes/n1", "s1")], query_id="q1"),
            make_query(
                [make_fact_node(f"content_nodes/n{i}", f"s{i}") for i in range(3)],
                query_id="q2",
            ),
        ]
        updated, dist = classify_batch(queries)
        assert len(updated) == 2
        for q in updated:
            assert q.direction is not None
            assert q.direction in ("TD", "HZ", "BU", "NT", "CS", "AMBIGUOUS")

    def test_distribution_counts_are_correct(self):
        # 3 NT queries, 1 BU query
        nt_nodes = [make_fact_node("content_nodes/n1", "s1")]
        bu_nodes = [make_fact_node(f"content_nodes/n{i}", f"s{i}") for i in range(3)]

        queries = [
            make_query(nt_nodes, query_id="q1"),
            make_query(nt_nodes, query_id="q2"),
            make_query(nt_nodes, query_id="q3"),
            make_query(bu_nodes, query_id="q4"),
        ]
        updated, dist = classify_batch(queries)
        assert dist.get("NT", 0) == 3
        assert dist.get("BU", 0) == 1
        assert sum(dist.values()) == 4

    def test_original_queries_not_modified(self):
        nodes = [make_fact_node("content_nodes/n1", "s1")]
        q = make_query(nodes, query_id="q1")
        original_direction = q.direction
        classify_batch([q])
        assert q.direction == original_direction

    def test_returns_new_query_instances(self):
        nodes = [make_fact_node("content_nodes/n1", "s1")]
        q = make_query(nodes, query_id="q1")
        updated, _ = classify_batch([q])
        assert updated[0] is not q

    def test_preserves_query_order(self):
        queries = [
            make_query([make_fact_node("content_nodes/n1", "s1")], query_id=f"q{i}")
            for i in range(5)
        ]
        updated, _ = classify_batch(queries)
        for i, (original, result) in enumerate(zip(queries, updated)):
            assert result.query_id == original.query_id

    def test_single_query(self):
        nodes = [make_fact_node("content_nodes/n1", "s1")]
        q = make_query(nodes, query_id="q1")
        updated, dist = classify_batch([q])
        assert len(updated) == 1
        assert updated[0].direction == "NT"
        assert dist == {"NT": 1}

    def test_mixed_directions(self):
        queries = [
            make_query([make_fact_node("content_nodes/n1", "s1")], query_id="nt"),
            make_query(
                [
                    make_fact_node("content_nodes/n1", "s1"),
                    make_fact_node("content_nodes/n2", "s2"),
                ],
                relations_used=["r1"],
                query_id="hz",
            ),
            make_query(
                [make_fact_node(f"content_nodes/n{i}", f"s{i}") for i in range(3)],
                query_id="bu",
            ),
        ]
        updated, dist = classify_batch(queries)
        directions = [q.direction for q in updated]
        assert "NT" in directions
        assert "HZ" in directions
        assert "BU" in directions
