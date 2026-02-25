"""Unit tests for workflows/benchmark/export.py.

Covers: format_subgraph_for_llm, export_subgraph_json, export_all_subgraphs.
"""

from __future__ import annotations

import json
from pathlib import Path

from workflows.benchmark.export import (
    build_id_alias_map,
    export_all_subgraphs,
    export_subgraph_json,
    format_subgraph_for_llm,
)
from workflows.benchmark.models import (
    Subgraph,
    SubgraphNode,
    SubgraphRelation,
    SubgraphStructureNode,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_subgraph_node(
    node_id: str,
    text: str = "sample text",
    breadcrumb: str | None = None,
    parent_section_key: str | None = None,
    parent_section_title: str | None = None,
    parent_section_level: int | None = None,
) -> SubgraphNode:
    return SubgraphNode(
        node_id=node_id,
        text=text,
        breadcrumb=breadcrumb,
        parent_section_key=parent_section_key,
        parent_section_title=parent_section_title,
        parent_section_level=parent_section_level,
    )


def make_subgraph(
    subgraph_id: str = "sg-abcdef12",
    anchor_key: str = "anchor_key_1",
    anchor_title: str = "Test Title",
    anchor_summary: str | None = None,
    breadcrumb: str = "Root > Section > Sub",
    nodes: list[SubgraphNode] | None = None,
    core_node_ids: list[str] | None = None,
    enriched_node_ids: list[str] | None = None,
    relations: list[SubgraphRelation] | None = None,
    structure_nodes: list[SubgraphStructureNode] | None = None,
    tier: str = "small",
    content_count: int = 2,
    query_allocation: int = 5,
) -> Subgraph:
    if nodes is None:
        nodes = [
            make_subgraph_node(
                "content_nodes/n1",
                "First content node.",
                parent_section_key=anchor_key,
            ),
            make_subgraph_node(
                "content_nodes/n2",
                "Second content node.",
                parent_section_key="other_sec",
            ),
        ]
    if core_node_ids is None:
        core_node_ids = ["content_nodes/n1"]
    if enriched_node_ids is None:
        enriched_node_ids = ["content_nodes/n2"]
    if relations is None:
        relations = []
    if structure_nodes is None:
        structure_nodes = [
            SubgraphStructureNode(
                section_key=anchor_key,
                title=anchor_title,
                summary=anchor_summary,
                level=5,
            ),
        ]
    return Subgraph(
        subgraph_id=subgraph_id,
        anchor_key=anchor_key,
        anchor_title=anchor_title,
        anchor_summary=anchor_summary,
        breadcrumb=breadcrumb,
        nodes=nodes,
        core_node_ids=core_node_ids,
        enriched_node_ids=enriched_node_ids,
        relations=relations,
        structure_nodes=structure_nodes,
        content_count=content_count,
        relation_count=len(relations),
        tier=tier,
        query_allocation=query_allocation,
    )


# ---------------------------------------------------------------------------
# format_subgraph_for_llm
# ---------------------------------------------------------------------------


class TestFormatSubgraphForLlm:
    def test_output_is_string(self):
        sg = make_subgraph()
        result, alias_map = format_subgraph_for_llm(sg)
        assert isinstance(result, str)

    def test_returns_tuple_with_alias_map(self):
        sg = make_subgraph()
        result = format_subgraph_for_llm(sg)
        assert isinstance(result, tuple)
        assert len(result) == 2
        text, alias_map = result
        assert isinstance(text, str)
        assert isinstance(alias_map, dict)

    def test_header_contains_title(self):
        sg = make_subgraph(anchor_title="My Special Title")
        result, _ = format_subgraph_for_llm(sg)
        assert "My Special Title" in result

    def test_header_contains_subgraph_id_prefix(self):
        sg = make_subgraph(subgraph_id="sg-abcdef12")
        result, _ = format_subgraph_for_llm(sg)
        assert "SG-sg-abcd" in result

    def test_l5_section_header_present(self):
        sg = make_subgraph()
        result, _ = format_subgraph_for_llm(sg)
        assert "### L5:" in result

    def test_relations_section_header_present(self):
        sg = make_subgraph()
        result, _ = format_subgraph_for_llm(sg)
        assert "### Relations" in result

    def test_relations_count_in_header(self):
        sg = make_subgraph()
        result, _ = format_subgraph_for_llm(sg)
        assert f"### Relations ({len(sg.relations)}):" in result

    def test_breadcrumb_section_present(self):
        sg = make_subgraph(breadcrumb="Root > Topic > Sub")
        result, _ = format_subgraph_for_llm(sg)
        assert "### Breadcrumb:" in result
        assert "Root > Topic > Sub" in result

    def test_node_alias_used_not_uuid(self):
        sg = make_subgraph(
            nodes=[
                make_subgraph_node(
                    "content_nodes/eb131378-62ae-4394-a92d-405198cc355b",
                    "text",
                    parent_section_key="anchor_key_1",
                )
            ],
            core_node_ids=["content_nodes/eb131378-62ae-4394-a92d-405198cc355b"],
            enriched_node_ids=[],
        )
        result, alias_map = format_subgraph_for_llm(sg)
        assert "eb131378-62ae-4394-a92d-405198cc355b" not in result
        assert "[n1]" in result

    def test_text_truncated_at_200_chars(self):
        long_text = "A" * 300
        sg = make_subgraph(
            nodes=[
                make_subgraph_node(
                    "content_nodes/n1",
                    long_text,
                    parent_section_key="anchor_key_1",
                )
            ],
            core_node_ids=["content_nodes/n1"],
            enriched_node_ids=[],
        )
        result, _ = format_subgraph_for_llm(sg)
        assert "..." in result

    def test_short_text_not_truncated(self):
        short_text = "Short text."
        sg = make_subgraph(
            nodes=[
                make_subgraph_node(
                    "content_nodes/n1",
                    short_text,
                    parent_section_key="anchor_key_1",
                )
            ],
            core_node_ids=["content_nodes/n1"],
            enriched_node_ids=[],
        )
        result, _ = format_subgraph_for_llm(sg)
        assert short_text in result

    def test_relation_formatted_with_aliases(self):
        nodes = [
            make_subgraph_node("content_nodes/n1", "text1", parent_section_key="anchor_key_1"),
            make_subgraph_node("content_nodes/n2", "text2", parent_section_key="other_sec"),
        ]
        relation = SubgraphRelation(
            from_id="content_nodes/n1",
            to_id="content_nodes/n2",
            relation_type="LEADS_TO",
        )
        sg = make_subgraph(
            nodes=nodes,
            core_node_ids=["content_nodes/n1"],
            enriched_node_ids=["content_nodes/n2"],
            relations=[relation],
        )
        result, alias_map = format_subgraph_for_llm(sg)
        assert "--LEADS_TO-->" in result
        # Aliases must appear; no raw UUIDs/keys in relations
        from_alias = alias_map["content_nodes/n1"]
        to_alias = alias_map["content_nodes/n2"]
        assert f"[{from_alias}] --LEADS_TO--> [{to_alias}]" in result

    def test_per_node_breadcrumbs_included_when_different(self):
        nodes = [
            make_subgraph_node(
                "content_nodes/n1",
                "text1",
                breadcrumb="Root > A",
                parent_section_key="anchor_key_1",
            ),
            make_subgraph_node(
                "content_nodes/n2",
                "text2",
                breadcrumb="Root > B",
                parent_section_key="other_sec",
            ),
        ]
        sg = make_subgraph(
            nodes=nodes,
            core_node_ids=["content_nodes/n1"],
            enriched_node_ids=["content_nodes/n2"],
            breadcrumb="Root",
        )
        result, _ = format_subgraph_for_llm(sg)
        assert "Root > A" in result
        assert "Root > B" in result

    def test_per_node_breadcrumbs_not_duplicated_when_same_as_subgraph(self):
        main_breadcrumb = "Root > Section"
        nodes = [
            make_subgraph_node(
                "content_nodes/n1",
                "text",
                breadcrumb=main_breadcrumb,
                parent_section_key="anchor_key_1",
            ),
        ]
        sg = make_subgraph(
            nodes=nodes,
            core_node_ids=["content_nodes/n1"],
            enriched_node_ids=[],
            breadcrumb=main_breadcrumb,
        )
        result, _ = format_subgraph_for_llm(sg)
        assert result.count(main_breadcrumb) == 1

    def test_numbered_node_list(self):
        sg = make_subgraph()
        result, _ = format_subgraph_for_llm(sg)
        assert "1." in result

    def test_anchor_summary_in_header(self):
        sg = make_subgraph(anchor_summary="This is the anchor summary")
        result, _ = format_subgraph_for_llm(sg)
        assert "Summary: This is the anchor summary" in result

    def test_scenario_a_flat_content(self):
        """Scenario A: L5 with only content, no L6 subsections."""
        nodes = [
            make_subgraph_node("content_nodes/c1", "Content 1", parent_section_key="sec5"),
            make_subgraph_node("content_nodes/c2", "Content 2", parent_section_key="sec5"),
        ]
        structure = [
            SubgraphStructureNode(
                section_key="sec5",
                title="Main Section",
                summary="Main summary",
                level=5,
            ),
        ]
        sg = make_subgraph(
            anchor_key="sec5",
            anchor_title="Main Section",
            anchor_summary="Main summary",
            nodes=nodes,
            core_node_ids=["content_nodes/c1", "content_nodes/c2"],
            enriched_node_ids=[],
            structure_nodes=structure,
        )
        result, alias_map = format_subgraph_for_llm(sg)
        assert '### L5: "Main Section"' in result
        assert "Summary: Main summary" in result
        # Aliases should appear, not raw keys
        assert f"[{alias_map['content_nodes/c1']}]" in result
        assert f"[{alias_map['content_nodes/c2']}]" in result
        # No L6 headers
        assert "#### L6:" not in result

    def test_scenario_b_l6_subsections(self):
        """Scenario B: L5 with L6 children."""
        nodes = [
            make_subgraph_node("content_nodes/d1", "Direct L5 content", parent_section_key="sec5"),
            make_subgraph_node("content_nodes/c1", "L6a content", parent_section_key="l6a"),
            make_subgraph_node("content_nodes/c2", "L6b content", parent_section_key="l6b"),
        ]
        structure = [
            SubgraphStructureNode(
                section_key="sec5",
                title="Parent",
                summary="Parent summary",
                level=5,
            ),
            SubgraphStructureNode(
                section_key="l6a",
                title="Sub A",
                summary="Sub A summary",
                level=6,
                parent_key="sec5",
            ),
            SubgraphStructureNode(
                section_key="l6b",
                title="Sub B",
                summary=None,
                level=6,
                parent_key="sec5",
            ),
        ]
        sg = make_subgraph(
            anchor_key="sec5",
            anchor_title="Parent",
            anchor_summary="Parent summary",
            nodes=nodes,
            core_node_ids=[
                "content_nodes/d1",
                "content_nodes/c1",
                "content_nodes/c2",
            ],
            enriched_node_ids=[],
            structure_nodes=structure,
            content_count=3,
        )
        result, alias_map = format_subgraph_for_llm(sg)
        assert "Direct content:" in result
        assert '#### L6: "Sub A" — Summary: Sub A summary' in result
        assert '#### L6: "Sub B"' in result
        assert f"[{alias_map['content_nodes/d1']}]" in result
        assert f"[{alias_map['content_nodes/c1']}]" in result
        assert f"[{alias_map['content_nodes/c2']}]" in result

    def test_enriched_grouped_as_related_l5(self):
        """Enriched nodes should be grouped under Related L5 headers."""
        nodes = [
            make_subgraph_node(
                "content_nodes/c1",
                "Core text",
                parent_section_key="anchor_key_1",
            ),
            make_subgraph_node(
                "content_nodes/e1",
                "Enriched text",
                parent_section_key="related_sec",
                parent_section_title="Related Section",
            ),
        ]
        structure = [
            SubgraphStructureNode(
                section_key="anchor_key_1",
                title="Anchor",
                level=5,
            ),
            SubgraphStructureNode(
                section_key="related_sec",
                title="Related Section",
                level=5,
            ),
        ]
        sg = make_subgraph(
            nodes=nodes,
            core_node_ids=["content_nodes/c1"],
            enriched_node_ids=["content_nodes/e1"],
            structure_nodes=structure,
        )
        result, alias_map = format_subgraph_for_llm(sg)
        assert '### Related L5: "Related Section"' in result
        assert f"[{alias_map['content_nodes/e1']}]" in result

    def test_fallback_when_no_structure_nodes(self):
        """When structure_nodes is empty, should still produce valid output."""
        sg = make_subgraph(structure_nodes=[])
        result, _ = format_subgraph_for_llm(sg)
        assert '## Subgraph: "Test Title"' in result
        assert "### L5:" in result
        assert "### Relations" in result

    def test_related_section_fallback_key(self):
        """When enriched parent has no structure node, show key as fallback."""
        nodes = [
            make_subgraph_node(
                "content_nodes/c1",
                "Core text",
                parent_section_key="anchor_key_1",
            ),
            make_subgraph_node(
                "content_nodes/e1",
                "Enriched text",
                parent_section_key="unknown_sec",
            ),
        ]
        sg = make_subgraph(
            nodes=nodes,
            core_node_ids=["content_nodes/c1"],
            enriched_node_ids=["content_nodes/e1"],
            structure_nodes=[
                SubgraphStructureNode(
                    section_key="anchor_key_1",
                    title="Anchor",
                    level=5,
                ),
            ],
        )
        result, _ = format_subgraph_for_llm(sg)
        assert "### Related: [unknown_sec]" in result


# ---------------------------------------------------------------------------
# build_id_alias_map
# ---------------------------------------------------------------------------


class TestBuildIdAliasMap:
    def test_all_nodes_covered(self):
        sg = make_subgraph()
        alias_map = build_id_alias_map(sg)
        for node in sg.nodes:
            assert node.node_id in alias_map

    def test_aliases_are_sequential(self):
        nodes = [make_subgraph_node(f"content_nodes/node{i}", f"text {i}") for i in range(5)]
        sg = make_subgraph(
            nodes=nodes,
            core_node_ids=[nodes[0].node_id],
            enriched_node_ids=[n.node_id for n in nodes[1:]],
        )
        alias_map = build_id_alias_map(sg)
        aliases = list(alias_map.values())
        assert aliases == [f"n{i}" for i in range(1, 6)]

    def test_deterministic_across_calls(self):
        sg = make_subgraph()
        map1 = build_id_alias_map(sg)
        map2 = build_id_alias_map(sg)
        assert map1 == map2

    def test_relation_ids_outside_nodes_not_in_alias_map(self):
        """Nodes referenced only in relations but not in subgraph.nodes must NOT get an alias."""
        nodes = [make_subgraph_node("content_nodes/n1", "text")]
        from workflows.benchmark.models import SubgraphRelation

        relations = [
            SubgraphRelation(
                from_id="content_nodes/n1",
                to_id="content_nodes/extra",
                relation_type="LEADS_TO",
            )
        ]
        sg = make_subgraph(
            nodes=nodes,
            core_node_ids=["content_nodes/n1"],
            enriched_node_ids=[],
            relations=relations,
        )
        alias_map = build_id_alias_map(sg)
        assert "content_nodes/extra" not in alias_map

    def test_relation_outside_subgraph_not_rendered(self):
        """A relation where one endpoint is outside subgraph.nodes must not appear in output."""
        nodes = [make_subgraph_node("content_nodes/n1", "text", parent_section_key="anchor_key_1")]
        from workflows.benchmark.models import SubgraphRelation

        relations = [
            SubgraphRelation(
                from_id="content_nodes/n1",
                to_id="content_nodes/external_node",
                relation_type="LEADS_TO",
            )
        ]
        sg = make_subgraph(
            nodes=nodes,
            core_node_ids=["content_nodes/n1"],
            enriched_node_ids=[],
            relations=relations,
        )
        result, alias_map = format_subgraph_for_llm(sg)
        assert "LEADS_TO" not in result
        assert "external_node" not in result

    def test_only_intra_subgraph_relations_rendered(self):
        """Relations between two in-subgraph nodes are rendered; external ones are filtered."""
        nodes = [
            make_subgraph_node("content_nodes/n1", "text1", parent_section_key="anchor_key_1"),
            make_subgraph_node("content_nodes/n2", "text2", parent_section_key="anchor_key_1"),
        ]
        from workflows.benchmark.models import SubgraphRelation

        relations = [
            SubgraphRelation(from_id="content_nodes/n1", to_id="content_nodes/n2", relation_type="LEADS_TO"),
            SubgraphRelation(from_id="content_nodes/n1", to_id="content_nodes/outside", relation_type="CAUSES"),
        ]
        sg = make_subgraph(
            nodes=nodes,
            core_node_ids=["content_nodes/n1", "content_nodes/n2"],
            enriched_node_ids=[],
            relations=relations,
            content_count=2,
        )
        result, alias_map = format_subgraph_for_llm(sg)
        assert "LEADS_TO" in result
        assert "CAUSES" not in result
        assert "outside" not in result

    def test_no_duplicate_aliases(self):
        sg = make_subgraph()
        alias_map = build_id_alias_map(sg)
        aliases = list(alias_map.values())
        assert len(aliases) == len(set(aliases))

    def test_format_uses_aliases_not_raw_ids(self):
        nodes = [
            make_subgraph_node(
                "content_nodes/eb131378-62ae-4394-a92d-405198cc355b",
                "text",
                parent_section_key="anchor_key_1",
            )
        ]
        sg = make_subgraph(
            nodes=nodes,
            core_node_ids=["content_nodes/eb131378-62ae-4394-a92d-405198cc355b"],
            enriched_node_ids=[],
        )
        text, alias_map = format_subgraph_for_llm(sg)
        assert "eb131378-62ae-4394-a92d-405198cc355b" not in text
        assert alias_map["content_nodes/eb131378-62ae-4394-a92d-405198cc355b"] == "n1"
        assert "[n1]" in text


# ---------------------------------------------------------------------------
# export_subgraph_json
# ---------------------------------------------------------------------------


class TestExportSubgraphJson:
    def test_creates_json_file(self, tmp_path):
        sg = make_subgraph(subgraph_id="sg-test001")
        output_dir = str(tmp_path / "subgraphs")
        filepath = export_subgraph_json(sg, output_dir=output_dir)
        assert Path(filepath).exists()

    def test_returns_correct_path(self, tmp_path):
        sg = make_subgraph(subgraph_id="sg-test001")
        output_dir = str(tmp_path / "subgraphs")
        filepath = export_subgraph_json(sg, output_dir=output_dir)
        assert "sg-test001.json" in filepath

    def test_json_is_valid_and_parseable(self, tmp_path):
        sg = make_subgraph(subgraph_id="sg-test002")
        output_dir = str(tmp_path / "subgraphs")
        filepath = export_subgraph_json(sg, output_dir=output_dir)
        with open(filepath) as f:
            data = json.load(f)
        assert data["subgraph_id"] == "sg-test002"

    def test_llm_format_included_in_json(self, tmp_path):
        sg = make_subgraph(subgraph_id="sg-test003")
        output_dir = str(tmp_path / "subgraphs")
        filepath = export_subgraph_json(sg, output_dir=output_dir)
        with open(filepath) as f:
            data = json.load(f)
        assert "llm_format" in data
        assert isinstance(data["llm_format"], str)
        assert len(data["llm_format"]) > 0

    def test_embeddings_excluded_from_json(self, tmp_path):
        node_with_embedding = SubgraphNode(
            node_id="content_nodes/n1",
            text="text",
            embedding=[0.1, 0.2, 0.3],
        )
        sg = make_subgraph(
            subgraph_id="sg-emb",
            nodes=[node_with_embedding],
            core_node_ids=["content_nodes/n1"],
            enriched_node_ids=[],
        )
        output_dir = str(tmp_path / "subgraphs")
        filepath = export_subgraph_json(sg, output_dir=output_dir)
        with open(filepath) as f:
            data = json.load(f)
        for node in data["nodes"]:
            assert "embedding" not in node

    def test_creates_output_directory_if_missing(self, tmp_path):
        sg = make_subgraph(subgraph_id="sg-newdir")
        output_dir = str(tmp_path / "new" / "nested" / "dir")
        filepath = export_subgraph_json(sg, output_dir=output_dir)
        assert Path(filepath).exists()

    def test_json_contains_anchor_title(self, tmp_path):
        sg = make_subgraph(subgraph_id="sg-title", anchor_title="My Title")
        output_dir = str(tmp_path / "subgraphs")
        filepath = export_subgraph_json(sg, output_dir=output_dir)
        with open(filepath) as f:
            data = json.load(f)
        assert data["anchor_title"] == "My Title"


# ---------------------------------------------------------------------------
# export_all_subgraphs
# ---------------------------------------------------------------------------


class TestExportAllSubgraphs:
    def test_creates_individual_files(self, tmp_path):
        subgraphs = [make_subgraph(subgraph_id=f"sg-{i:04d}", content_count=2) for i in range(3)]
        output_dir = str(tmp_path / "all_subgraphs")
        result = export_all_subgraphs(subgraphs, output_dir=output_dir)
        for sg in subgraphs:
            expected_file = Path(output_dir) / f"{sg.subgraph_id}.json"
            assert expected_file.exists()

    def test_creates_index_json(self, tmp_path):
        subgraphs = [make_subgraph(subgraph_id=f"sg-{i:04d}") for i in range(2)]
        output_dir = str(tmp_path / "all_subgraphs")
        result = export_all_subgraphs(subgraphs, output_dir=output_dir)
        index_path = Path(output_dir) / "index.json"
        assert index_path.exists()

    def test_returns_export_stats(self, tmp_path):
        subgraphs = [make_subgraph(subgraph_id=f"sg-{i:04d}") for i in range(3)]
        output_dir = str(tmp_path / "all_subgraphs")
        result = export_all_subgraphs(subgraphs, output_dir=output_dir)
        assert result["exported_count"] == 3
        assert "index_path" in result
        assert "output_dir" in result

    def test_index_json_has_correct_total_count(self, tmp_path):
        subgraphs = [make_subgraph(subgraph_id=f"sg-{i:04d}") for i in range(4)]
        output_dir = str(tmp_path / "all_subgraphs")
        export_all_subgraphs(subgraphs, output_dir=output_dir)
        with open(Path(output_dir) / "index.json") as f:
            index = json.load(f)
        assert index["total_subgraphs"] == 4

    def test_index_json_contains_all_subgraph_entries(self, tmp_path):
        subgraphs = [make_subgraph(subgraph_id=f"sg-{i:04d}") for i in range(3)]
        output_dir = str(tmp_path / "all_subgraphs")
        export_all_subgraphs(subgraphs, output_dir=output_dir)
        with open(Path(output_dir) / "index.json") as f:
            index = json.load(f)
        entry_ids = {e["subgraph_id"] for e in index["subgraphs"]}
        for sg in subgraphs:
            assert sg.subgraph_id in entry_ids

    def test_index_json_total_queries_allocated(self, tmp_path):
        subgraphs = [
            make_subgraph(subgraph_id="sg-0001", query_allocation=10),
            make_subgraph(subgraph_id="sg-0002", query_allocation=20),
        ]
        output_dir = str(tmp_path / "all_subgraphs")
        result = export_all_subgraphs(subgraphs, output_dir=output_dir)
        assert result["total_queries_allocated"] == 30
        with open(Path(output_dir) / "index.json") as f:
            index = json.load(f)
        assert index["total_queries_allocated"] == 30

    def test_index_entry_has_required_fields(self, tmp_path):
        subgraphs = [make_subgraph(subgraph_id="sg-fields")]
        output_dir = str(tmp_path / "all_subgraphs")
        export_all_subgraphs(subgraphs, output_dir=output_dir)
        with open(Path(output_dir) / "index.json") as f:
            index = json.load(f)
        entry = index["subgraphs"][0]
        required_fields = {
            "subgraph_id",
            "anchor_key",
            "anchor_title",
            "tier",
            "content_count",
            "relation_count",
            "total_nodes",
            "core_nodes",
            "enriched_nodes",
            "query_allocation",
            "file",
        }
        for field in required_fields:
            assert field in entry, f"Missing field: {field}"

    def test_empty_list_creates_empty_index(self, tmp_path):
        output_dir = str(tmp_path / "empty_export")
        result = export_all_subgraphs([], output_dir=output_dir)
        assert result["exported_count"] == 0
        index_path = Path(output_dir) / "index.json"
        assert index_path.exists()
        with open(index_path) as f:
            index = json.load(f)
        assert index["total_subgraphs"] == 0
        assert index["subgraphs"] == []

    def test_tier_distribution_in_index(self, tmp_path):
        subgraphs = [
            make_subgraph(subgraph_id="sg-s1", tier="small"),
            make_subgraph(subgraph_id="sg-s2", tier="small"),
            make_subgraph(subgraph_id="sg-t1", tier="tiny"),
        ]
        output_dir = str(tmp_path / "tier_dist")
        export_all_subgraphs(subgraphs, output_dir=output_dir)
        with open(Path(output_dir) / "index.json") as f:
            index = json.load(f)
        dist = index["tier_distribution"]
        assert dist.get("small", 0) == 2
        assert dist.get("tiny", 0) == 1

    def test_file_reference_in_index_entry(self, tmp_path):
        subgraphs = [make_subgraph(subgraph_id="sg-fileref")]
        output_dir = str(tmp_path / "fileref_test")
        export_all_subgraphs(subgraphs, output_dir=output_dir)
        with open(Path(output_dir) / "index.json") as f:
            index = json.load(f)
        assert index["subgraphs"][0]["file"] == "sg-fileref.json"
