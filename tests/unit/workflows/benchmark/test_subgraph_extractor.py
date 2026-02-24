"""Unit tests for workflows/benchmark/subgraph_extractor.py.

Covers: SEED depth, ENRICH cross-L5 filter, BOUND cap, coverage metric,
structure_nodes collection.
Uses mocked AQL client to avoid database dependency.
"""

from __future__ import annotations

from workflows.benchmark.models import (
    Subgraph,
    SubgraphNode,
    SubgraphRelation,
)
from workflows.benchmark.subgraph_extractor import SubgraphExtractor

# ---------------------------------------------------------------------------
# Mock client helper
# ---------------------------------------------------------------------------


class MockArangoClient:
    """Mock ArangoDB client that returns pre-configured AQL results."""

    def __init__(self, responses: list[list[dict]] | None = None) -> None:
        self._responses = responses or []
        self._call_idx = 0
        self.queries: list[str] = []
        self.bind_vars_log: list[dict] = []

    def execute_aql(self, query: str, bind_vars: dict | None = None) -> list[dict]:
        self.queries.append(query)
        self.bind_vars_log.append(bind_vars or {})
        if self._call_idx < len(self._responses):
            result = self._responses[self._call_idx]
            self._call_idx += 1
            return result
        return []


# ---------------------------------------------------------------------------
# SEED depth tests
# ---------------------------------------------------------------------------


class TestSeedDepth:
    def test_seed_query_uses_depth_2(self):
        """SEED AQL should traverse 1..2 to cover L5→L6→content."""
        client = MockArangoClient(responses=[[]])
        extractor = SubgraphExtractor(client)
        result = extractor.extract(
            anchor_key="sec1",
            anchor_title="Section 1",
            anchor_summary="Summary",
            breadcrumb="A > B",
        )
        assert result is None  # no content nodes
        assert len(client.queries) == 1
        assert "1..2" in client.queries[0]

    def test_seed_includes_has_section_edge_type(self):
        """SEED AQL should filter both HAS_SECTION and HAS_CONTENT."""
        client = MockArangoClient(responses=[[]])
        extractor = SubgraphExtractor(client)
        extractor.extract(
            anchor_key="sec1",
            anchor_title="Section 1",
            anchor_summary=None,
            breadcrumb="A",
        )
        assert "HAS_SECTION" in client.queries[0]
        assert "HAS_CONTENT" in client.queries[0]

    def test_seed_returns_content_nodes(self):
        """SEED should return content nodes found under anchor."""
        seed_nodes = [
            {
                "node_id": "content_nodes/c1",
                "text": "Para 1",
                "embedding": [0.1],
                "parent_section_key": "sec1",
                "parent_section_title": "Section 1",
                "parent_section_level": 5,
            },
            {
                "node_id": "content_nodes/c2",
                "text": "Para 2",
                "embedding": [0.2],
                "parent_section_key": "sec1",
                "parent_section_title": "Section 1",
                "parent_section_level": 5,
            },
        ]
        # Responses: seed, structure_nodes, enrich, collect_relations
        client = MockArangoClient(responses=[seed_nodes, [], [], []])
        extractor = SubgraphExtractor(client)
        result = extractor.extract(
            anchor_key="sec1",
            anchor_title="Section 1",
            anchor_summary="Summary",
            breadcrumb="A > B",
        )
        assert result is not None
        assert len(result.core_node_ids) == 2
        assert result.core_node_ids == ["content_nodes/c1", "content_nodes/c2"]

    def test_seed_returns_actual_parent_key(self):
        """SEED should return the actual direct parent key, not anchor key."""
        seed_nodes = [
            {
                "node_id": "content_nodes/c1",
                "text": "Para under L6",
                "embedding": None,
                "parent_section_key": "l6_sub",
                "parent_section_title": "Sub Section",
                "parent_section_level": 6,
            },
        ]
        # Responses: seed, structure, enrich, relations
        client = MockArangoClient(responses=[seed_nodes, [], [], []])
        extractor = SubgraphExtractor(client)
        result = extractor.extract(
            anchor_key="sec1",
            anchor_title="S1",
            anchor_summary=None,
            breadcrumb="X",
        )
        assert result is not None
        # The node should have the actual parent key "l6_sub", not "sec1"
        core_node = result.nodes[0]
        assert core_node.parent_section_key == "l6_sub"
        assert core_node.parent_section_title == "Sub Section"
        assert core_node.parent_section_level == 6


# ---------------------------------------------------------------------------
# ENRICH cross-L5 filter tests
# ---------------------------------------------------------------------------


class TestEnrichCrossL5:
    def test_enrich_passes_anchor_key(self):
        """ENRICH AQL should receive anchor_key as bind variable for cross-L5 filter."""
        seed_nodes = [
            {
                "node_id": "content_nodes/c1",
                "text": "P1",
                "embedding": None,
                "parent_section_key": "sec1",
                "parent_section_title": "S1",
                "parent_section_level": 5,
            },
        ]
        enrich_result = []
        # Responses: seed, structure, enrich, relations
        client = MockArangoClient(responses=[seed_nodes, [], enrich_result, []])
        extractor = SubgraphExtractor(client)
        extractor.extract(
            anchor_key="sec1",
            anchor_title="S1",
            anchor_summary=None,
            breadcrumb="X",
        )
        # Third call is enrich (0=seed, 1=structure, 2=enrich)
        enrich_vars = client.bind_vars_log[2]
        assert enrich_vars["anchor_key"] == "sec1"

    def test_enrich_query_filters_by_anchor_key(self):
        """ENRICH AQL should contain v_parent != @anchor_key filter."""
        seed_nodes = [
            {
                "node_id": "content_nodes/c1",
                "text": "P1",
                "embedding": None,
                "parent_section_key": "sec1",
                "parent_section_title": "S1",
                "parent_section_level": 5,
            },
        ]
        # Responses: seed, structure, enrich, relations
        client = MockArangoClient(responses=[seed_nodes, [], [], []])
        extractor = SubgraphExtractor(client)
        extractor.extract(
            anchor_key="sec1",
            anchor_title="S1",
            anchor_summary=None,
            breadcrumb="X",
        )
        enrich_query = client.queries[2]
        assert "anchor_key" in enrich_query
        assert "v_parent" in enrich_query

    def test_enrich_aggregates_connections(self):
        """ENRICH should aggregate multiple connections for the same node."""
        seed_nodes = [
            {
                "node_id": "content_nodes/c1",
                "text": "P1",
                "embedding": None,
                "parent_section_key": "sec1",
                "parent_section_title": "S1",
                "parent_section_level": 5,
            },
        ]
        enrich_rows = [
            {
                "node_id": "content_nodes/e1",
                "text": "Enriched",
                "embedding": None,
                "parent_section_key": "sec2",
                "parent_section_title": "S2",
                "parent_section_level": 5,
                "relation_type": "SUPPORTS",
                "connected_to": "content_nodes/c1",
            },
            {
                "node_id": "content_nodes/e1",
                "text": "Enriched",
                "embedding": None,
                "parent_section_key": "sec2",
                "parent_section_title": "S2",
                "parent_section_level": 5,
                "relation_type": "CONTRASTS",
                "connected_to": "content_nodes/c1",
            },
        ]
        # Responses: seed, structure, enrich, relations
        client = MockArangoClient(responses=[seed_nodes, [], enrich_rows, []])
        extractor = SubgraphExtractor(client)
        result = extractor.extract(
            anchor_key="sec1",
            anchor_title="S1",
            anchor_summary=None,
            breadcrumb="X",
        )
        assert result is not None
        assert len(result.enriched_node_ids) == 1
        assert result.enriched_node_ids[0] == "content_nodes/e1"


# ---------------------------------------------------------------------------
# Structure nodes tests
# ---------------------------------------------------------------------------


class TestStructureNodes:
    def test_structure_nodes_populated(self):
        """extract() should populate structure_nodes with anchor + L6."""
        seed_nodes = [
            {
                "node_id": "content_nodes/c1",
                "text": "P1",
                "embedding": None,
                "parent_section_key": "sec1",
                "parent_section_title": "S1",
                "parent_section_level": 5,
            },
        ]
        l6_nodes = [
            {"key": "l6a", "title": "Sub A", "summary": "Sum A", "level": 6},
        ]
        # Responses: seed, structure, enrich, relations
        client = MockArangoClient(responses=[seed_nodes, l6_nodes, [], []])
        extractor = SubgraphExtractor(client)
        result = extractor.extract(
            anchor_key="sec1",
            anchor_title="S1",
            anchor_summary="Summary",
            breadcrumb="X",
        )
        assert result is not None
        # Should have anchor L5 + l6a
        assert len(result.structure_nodes) == 2
        keys = {sn.section_key for sn in result.structure_nodes}
        assert "sec1" in keys
        assert "l6a" in keys

    def test_l6_nodes_collected(self):
        """_collect_structure_nodes() AQL should query L6 children."""
        seed_nodes = [
            {
                "node_id": "content_nodes/c1",
                "text": "P1",
                "embedding": None,
                "parent_section_key": "sec1",
                "parent_section_title": "S1",
                "parent_section_level": 5,
            },
        ]
        # Responses: seed, structure, enrich, relations
        client = MockArangoClient(responses=[seed_nodes, [], [], []])
        extractor = SubgraphExtractor(client)
        extractor.extract(
            anchor_key="sec1",
            anchor_title="S1",
            anchor_summary=None,
            breadcrumb="X",
        )
        # Second query should be _collect_structure_nodes
        structure_query = client.queries[1]
        assert "HAS_SECTION" in structure_query
        assert "level == 6" in structure_query

    def test_enriched_parents_added_to_structure(self):
        """Enriched node parents should be added to structure_nodes."""
        seed_nodes = [
            {
                "node_id": "content_nodes/c1",
                "text": "P1",
                "embedding": None,
                "parent_section_key": "sec1",
                "parent_section_title": "S1",
                "parent_section_level": 5,
            },
        ]
        enrich_rows = [
            {
                "node_id": "content_nodes/e1",
                "text": "Enriched",
                "embedding": None,
                "parent_section_key": "sec2",
                "parent_section_title": "Section 2",
                "parent_section_level": 5,
                "relation_type": "SUPPORTS",
                "connected_to": "content_nodes/c1",
            },
        ]
        # Responses: seed, structure, enrich, relations
        client = MockArangoClient(responses=[seed_nodes, [], enrich_rows, []])
        extractor = SubgraphExtractor(client)
        result = extractor.extract(
            anchor_key="sec1",
            anchor_title="S1",
            anchor_summary=None,
            breadcrumb="X",
        )
        assert result is not None
        keys = {sn.section_key for sn in result.structure_nodes}
        assert "sec2" in keys
        sec2_sn = next(sn for sn in result.structure_nodes if sn.section_key == "sec2")
        assert sec2_sn.title == "Section 2"
        assert sec2_sn.level == 5


# ---------------------------------------------------------------------------
# BOUND cap tests
# ---------------------------------------------------------------------------


class TestBound:
    def test_bound_caps_enriched_nodes(self):
        """BOUND should cap total nodes at max_size."""
        seed_nodes = [
            {
                "node_id": f"content_nodes/c{i}",
                "text": f"P{i}",
                "embedding": None,
                "parent_section_key": "sec1",
                "parent_section_title": "S1",
                "parent_section_level": 5,
            }
            for i in range(15)
        ]
        enrich_rows = [
            {
                "node_id": f"content_nodes/e{i}",
                "text": f"E{i}",
                "embedding": None,
                "parent_section_key": f"sec{i + 2}",
                "parent_section_title": f"S{i + 2}",
                "parent_section_level": 5,
                "relation_type": "SUPPORTS",
                "connected_to": "content_nodes/c1",
            }
            for i in range(10)
        ]
        # Responses: seed, structure, enrich, relations
        client = MockArangoClient(responses=[seed_nodes, [], enrich_rows, []])
        extractor = SubgraphExtractor(client, max_size=20)
        result = extractor.extract(
            anchor_key="sec1",
            anchor_title="S1",
            anchor_summary=None,
            breadcrumb="X",
        )
        assert result is not None
        total_nodes = len(result.core_node_ids) + len(result.enriched_node_ids)
        assert total_nodes <= 20

    def test_bound_returns_all_when_under_limit(self):
        """BOUND should keep all enriched when total is under max_size."""
        seed_nodes = [
            {
                "node_id": "content_nodes/c1",
                "text": "P1",
                "embedding": None,
                "parent_section_key": "sec1",
                "parent_section_title": "S1",
                "parent_section_level": 5,
            },
        ]
        enrich_rows = [
            {
                "node_id": "content_nodes/e1",
                "text": "E1",
                "embedding": None,
                "parent_section_key": "sec2",
                "parent_section_title": "S2",
                "parent_section_level": 5,
                "relation_type": "SUPPORTS",
                "connected_to": "content_nodes/c1",
            },
        ]
        # Responses: seed, structure, enrich, relations
        client = MockArangoClient(responses=[seed_nodes, [], enrich_rows, []])
        extractor = SubgraphExtractor(client, max_size=20)
        result = extractor.extract(
            anchor_key="sec1",
            anchor_title="S1",
            anchor_summary=None,
            breadcrumb="X",
        )
        assert result is not None
        assert len(result.enriched_node_ids) == 1


# ---------------------------------------------------------------------------
# Coverage metric tests
# ---------------------------------------------------------------------------


class TestCoverageMetric:
    def test_coverage_empty_subgraph_returns_zero(self):
        """Coverage of subgraph with no nodes should be 0.

        Note: Subgraph validator requires >=1 node, so we test
        compute_relation_coverage's early-return by mocking .nodes to empty.
        """
        client = MockArangoClient()
        extractor = SubgraphExtractor(client)
        sg = Subgraph(
            subgraph_id="sg1",
            anchor_key="sec1",
            anchor_title="S1",
            breadcrumb="X",
            nodes=[SubgraphNode(node_id="content_nodes/dummy", text="placeholder")],
            core_node_ids=["content_nodes/dummy"],
            enriched_node_ids=[],
            relations=[],
            content_count=1,
            relation_count=0,
            tier="small",
            query_allocation=0,
        )
        # Override nodes to empty to test the early-return path
        sg.__dict__["nodes"] = []
        assert extractor.compute_relation_coverage(sg) == 0.0

    def test_coverage_no_external_relations(self):
        """Coverage should be 0 when total external relations are 0."""
        client = MockArangoClient(responses=[[]])  # empty total
        extractor = SubgraphExtractor(client)
        sg = Subgraph(
            subgraph_id="sg1",
            anchor_key="sec1",
            anchor_title="S1",
            breadcrumb="X",
            nodes=[SubgraphNode(node_id="content_nodes/c1", text="P1")],
            core_node_ids=["content_nodes/c1"],
            enriched_node_ids=[],
            relations=[],
            content_count=1,
            relation_count=0,
            tier="small",
            query_allocation=0,
        )
        assert extractor.compute_relation_coverage(sg) == 0.0

    def test_coverage_ratio_calculated_correctly(self):
        """Coverage = internal/total for relations involving subgraph nodes."""
        # Return 4 total relations from AQL
        total_relations = [1, 1, 1, 1]
        client = MockArangoClient(responses=[total_relations])
        extractor = SubgraphExtractor(client)
        sg = Subgraph(
            subgraph_id="sg1",
            anchor_key="sec1",
            anchor_title="S1",
            breadcrumb="X",
            nodes=[
                SubgraphNode(node_id="content_nodes/c1", text="P1"),
                SubgraphNode(node_id="content_nodes/c2", text="P2"),
            ],
            core_node_ids=["content_nodes/c1", "content_nodes/c2"],
            enriched_node_ids=[],
            relations=[
                SubgraphRelation(
                    from_id="content_nodes/c1", to_id="content_nodes/c2", relation_type="SUPPORTS"
                ),
                SubgraphRelation(
                    from_id="content_nodes/c2", to_id="content_nodes/c1", relation_type="CONTRASTS"
                ),
            ],
            content_count=2,
            relation_count=2,
            tier="small",
            query_allocation=0,
        )
        coverage = extractor.compute_relation_coverage(sg)
        assert coverage == 2 / 4  # 2 internal / 4 total
