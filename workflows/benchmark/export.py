"""Export Subgraphs to JSON for Benchmark v4.0 (Task 12).

Exports extracted subgraphs as individual JSON files + index + LLM-readable
markdown format for prompt insertion.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

from workflows.benchmark.models import Subgraph, SubgraphNode, SubgraphStructureNode


def build_id_alias_map(subgraph: Subgraph) -> dict[str, str]:
    """Build a deterministic mapping from full node_id → short alias (n1, n2, ...).

    Only nodes present in ``subgraph.nodes`` receive an alias.  Relation
    endpoints that point outside the subgraph are intentionally excluded.
    Returns ``{full_id: "n1", ...}``.
    """
    alias_map: dict[str, str] = {}
    counter = 1

    for node in subgraph.nodes:
        if node.node_id not in alias_map:
            alias_map[node.node_id] = f"n{counter}"
            counter += 1

    return alias_map


def _format_node_line(
    idx: int,
    node: SubgraphNode,
    alias_map: dict[str, str] | None = None,
) -> str:
    """Format a single content node as a numbered line.

    When ``alias_map`` is provided uses the short alias; otherwise falls back
    to the bare UUID key after the ``/``.
    """
    if alias_map is not None:
        label = alias_map.get(node.node_id, node.node_id)
    else:
        label = node.node_id.split("/")[-1] if "/" in node.node_id else node.node_id
    text_preview = node.text[:200] + "..." if len(node.text) > 200 else node.text
    return f"{idx}. [{label}] {text_preview}"


def format_subgraph_for_llm(
    subgraph: Subgraph,
    context_mode: Literal["full", "node_only", "relation", "breadth"] = "full",
) -> tuple[str, dict[str, str]]:
    """Format a Subgraph into hierarchy-aware LLM-readable markdown.

    Node IDs are replaced with short aliases (``n1``, ``n2``, …) to reduce token
    usage.  The alias map is returned alongside the text so callers can resolve
    aliases back to full node IDs after parsing the LLM response.

    Args:
        subgraph: The subgraph to format.
        context_mode: Controls how much context to include:
            - ``"full"``: Full format (backward-compatible default).
            - ``"node_only"``: Anchor summary + section titles + core nodes + breadcrumb.
              Best for NT/TD queries where relations add noise.
            - ``"relation"``: Compact section titles + all nodes + full relation list.
              Best for HZ queries that need to follow relation edges.
            - ``"breadth"``: All sections + summaries + node excerpts + relation types only.
              Best for BU/CS queries requiring multi-section synthesis.

    Returns:
        ``(formatted_text, alias_map)`` where ``alias_map`` maps full node ID →
        short alias string.

    Scenario A (no L6 children): L5 summary → content nodes → related sections
    Scenario B (has L6 children): L5 summary → direct L5 content → L6 blocks → related
    """
    alias_map = build_id_alias_map(subgraph)
    if context_mode == "node_only":
        return _format_node_only(subgraph, alias_map), alias_map
    if context_mode == "relation":
        return _format_relation(subgraph, alias_map), alias_map
    if context_mode == "breadth":
        return _format_breadth(subgraph, alias_map), alias_map
    return _format_full(subgraph, alias_map), alias_map


def _format_full(subgraph: Subgraph, alias_map: dict[str, str]) -> str:
    """Original full format (backward-compatible)."""
    sg_id_short = subgraph.subgraph_id[:8]
    anchor_key = subgraph.anchor_key

    # Build structure map and group nodes by parent_section_key
    struct_map: dict[str, SubgraphStructureNode] = {
        sn.section_key: sn for sn in subgraph.structure_nodes
    }
    nodes_by_parent: dict[str | None, list[SubgraphNode]] = {}
    for node in subgraph.nodes:
        pkey = node.parent_section_key
        nodes_by_parent.setdefault(pkey, []).append(node)

    # Identify L6 sections under anchor
    l6_keys = [
        sn.section_key
        for sn in subgraph.structure_nodes
        if sn.level == 6 and sn.parent_key == anchor_key
    ]

    lines: list[str] = [
        f'## Subgraph: "{subgraph.anchor_title}" (SG-{sg_id_short})',
        "",
    ]

    # ----- Anchor block -----
    anchor_sn = struct_map.get(anchor_key)
    summary_text = (
        anchor_sn.summary if anchor_sn and anchor_sn.summary else subgraph.anchor_summary
    )
    summary_suffix = f" — Summary: {summary_text}" if summary_text else ""
    lines.append(f'### L5: "{subgraph.anchor_title}"{summary_suffix}')

    if not l6_keys:
        # Scenario A: flat — all anchor content
        anchor_nodes = nodes_by_parent.get(anchor_key, [])
        counter = 1
        for node in anchor_nodes:
            if node.node_id in subgraph.core_node_ids:
                lines.append(_format_node_line(counter, node, alias_map))
                counter += 1
    else:
        # Scenario B: L5 direct content first, then L6 blocks
        direct_nodes = [
            n for n in nodes_by_parent.get(anchor_key, []) if n.node_id in subgraph.core_node_ids
        ]
        if direct_nodes:
            lines.append("Direct content:")
            counter = 1
            for node in direct_nodes:
                lines.append(_format_node_line(counter, node, alias_map))
                counter += 1

        for l6_key in l6_keys:
            l6_sn = struct_map.get(l6_key)
            l6_title = l6_sn.title if l6_sn and l6_sn.title else l6_key
            l6_summary = l6_sn.summary if l6_sn else None
            l6_suffix = f" — Summary: {l6_summary}" if l6_summary else ""
            lines.append("")
            lines.append(f'#### L6: "{l6_title}"{l6_suffix}')
            l6_content = [
                n for n in nodes_by_parent.get(l6_key, []) if n.node_id in subgraph.core_node_ids
            ]
            if l6_content:
                lines.append("Content nodes:")
                counter = 1
                for node in l6_content:
                    lines.append(_format_node_line(counter, node, alias_map))
                    counter += 1

    # ----- Enriched / Related sections -----
    enriched_parent_keys: list[str] = []
    for node in subgraph.nodes:
        if node.node_id in subgraph.enriched_node_ids:
            pkey = node.parent_section_key
            if pkey and pkey not in enriched_parent_keys:
                enriched_parent_keys.append(pkey)

    for pkey in enriched_parent_keys:
        sn = struct_map.get(pkey)
        enrich_nodes = [
            n for n in nodes_by_parent.get(pkey, []) if n.node_id in subgraph.enriched_node_ids
        ]
        if not enrich_nodes:
            continue

        lines.append("")
        if sn and sn.title:
            s_suffix = f" — Summary: {sn.summary}" if sn.summary else ""
            lines.append(f'### Related L5: "{sn.title}"{s_suffix}')
        else:
            lines.append(f"### Related: [{pkey}]")

        lines.append("Content nodes:")
        counter = 1
        for node in enrich_nodes:
            lines.append(_format_node_line(counter, node, alias_map))
            counter += 1

    # ----- Relations -----
    node_ids = {n.node_id for n in subgraph.nodes}
    intra_relations = [r for r in subgraph.relations if r.from_id in node_ids and r.to_id in node_ids]
    lines.append("")
    lines.append(f"### Relations ({len(intra_relations)}):")
    for rel in intra_relations:
        from_alias = alias_map.get(rel.from_id, rel.from_id)
        to_alias = alias_map.get(rel.to_id, rel.to_id)
        lines.append(f"- [{from_alias}] --{rel.relation_type}--> [{to_alias}]")

    # ----- Breadcrumb -----
    lines.append("")
    lines.append("### Breadcrumb:")
    lines.append(f"- {subgraph.breadcrumb}")

    seen_breadcrumbs: set[str] = set()
    for node in subgraph.nodes:
        if node.breadcrumb and node.breadcrumb != subgraph.breadcrumb:
            if node.breadcrumb not in seen_breadcrumbs:
                seen_breadcrumbs.add(node.breadcrumb)
                lines.append(f"- {node.breadcrumb}")

    return "\n".join(lines)


def _format_node_only(subgraph: Subgraph, alias_map: dict[str, str]) -> str:
    """Compact format for NT/TD: anchor summary + section titles + core nodes + breadcrumb.

    Omits full relation list and enriched sections to reduce noise.
    """
    sg_id_short = subgraph.subgraph_id[:8]
    anchor_key = subgraph.anchor_key

    struct_map: dict[str, SubgraphStructureNode] = {
        sn.section_key: sn for sn in subgraph.structure_nodes
    }
    nodes_by_parent: dict[str | None, list[SubgraphNode]] = {}
    for node in subgraph.nodes:
        nodes_by_parent.setdefault(node.parent_section_key, []).append(node)

    anchor_sn = struct_map.get(anchor_key)
    summary_text = (
        anchor_sn.summary if anchor_sn and anchor_sn.summary else subgraph.anchor_summary
    )
    summary_suffix = f" — Summary: {summary_text}" if summary_text else ""

    lines: list[str] = [
        f'## Subgraph: "{subgraph.anchor_title}" (SG-{sg_id_short})',
        "",
        f'### L5: "{subgraph.anchor_title}"{summary_suffix}',
    ]

    # List all section titles present in subgraph
    section_titles = [sn.title for sn in subgraph.structure_nodes if sn.title]
    if section_titles:
        lines.append("Sections: " + " | ".join(section_titles))
    lines.append("")

    # Core nodes under anchor
    counter = 1
    for node in subgraph.nodes:
        if node.node_id in subgraph.core_node_ids:
            lines.append(_format_node_line(counter, node, alias_map))
            counter += 1

    # Breadcrumb
    lines.append("")
    lines.append("### Breadcrumb:")
    lines.append(f"- {subgraph.breadcrumb}")

    return "\n".join(lines)


def _format_relation(subgraph: Subgraph, alias_map: dict[str, str]) -> str:
    """Relation-focused format for HZ: compact section titles + all nodes + full relation list.

    Keeps section context but emphasises relation edges.
    """
    sg_id_short = subgraph.subgraph_id[:8]

    lines: list[str] = [
        f'## Subgraph: "{subgraph.anchor_title}" (SG-{sg_id_short})',
        "",
    ]

    # Compact section titles (no full summaries)
    section_titles = [sn.title for sn in subgraph.structure_nodes if sn.title]
    if section_titles:
        lines.append("Sections: " + " | ".join(section_titles))
    lines.append("")

    # All nodes (core + enriched)
    lines.append("### Nodes:")
    counter = 1
    for node in subgraph.nodes:
        lines.append(_format_node_line(counter, node, alias_map))
        counter += 1

    # Full relation list — only intra-subgraph edges, this is the key signal for HZ
    node_ids = {n.node_id for n in subgraph.nodes}
    intra_relations = [r for r in subgraph.relations if r.from_id in node_ids and r.to_id in node_ids]
    lines.append("")
    lines.append(f"### Relations ({len(intra_relations)}):")
    for rel in intra_relations:
        from_alias = alias_map.get(rel.from_id, rel.from_id)
        to_alias = alias_map.get(rel.to_id, rel.to_id)
        lines.append(f"- [{from_alias}] --{rel.relation_type}--> [{to_alias}]")

    return "\n".join(lines)


def _format_breadth(subgraph: Subgraph, alias_map: dict[str, str]) -> str:
    """Multi-section breadth format for BU/CS: all sections + summaries + node excerpts.

    Includes relation type inventory (not individual pairs) to avoid HZ bias.
    """
    sg_id_short = subgraph.subgraph_id[:8]

    nodes_by_parent: dict[str | None, list[SubgraphNode]] = {}
    for node in subgraph.nodes:
        nodes_by_parent.setdefault(node.parent_section_key, []).append(node)

    lines: list[str] = [
        f'## Subgraph: "{subgraph.anchor_title}" (SG-{sg_id_short})',
        "",
    ]

    # All sections with summaries and concrete node excerpts
    for sn in subgraph.structure_nodes:
        summary_suffix = f" — Summary: {sn.summary}" if sn.summary else ""
        lines.append(f'### Section: "{sn.title}"{summary_suffix}')
        section_nodes = [
            n
            for n in nodes_by_parent.get(sn.section_key, [])
            if n.node_id in subgraph.core_node_ids or n.node_id in subgraph.enriched_node_ids
        ]
        if section_nodes:
            counter = 1
            for node in section_nodes:
                lines.append(_format_node_line(counter, node, alias_map))
                counter += 1
        lines.append("")

    # Relation types only — suppress individual pairs to avoid HZ bias
    if subgraph.relations:
        rel_types = sorted({r.relation_type for r in subgraph.relations})
        lines.append(f"### Relation types present: {', '.join(rel_types)}")

    return "\n".join(lines)


def export_subgraph_json(
    subgraph: Subgraph,
    output_dir: str = "data/benchmark/subgraphs",
) -> str:
    """Export a single subgraph to a JSON file.

    Returns path to the written file.
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    filename = f"{subgraph.subgraph_id}.json"
    filepath = str(Path(output_dir) / filename)

    data = subgraph.model_dump(mode="json", exclude={"nodes": {"__all__": {"embedding"}}})
    data["llm_format"] = format_subgraph_for_llm(subgraph)[0]

    with open(filepath, "w") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    return filepath


def export_all_subgraphs(
    subgraphs: list[Subgraph],
    output_dir: str = "data/benchmark/subgraphs",
) -> dict:
    """Export all subgraphs to individual JSON files + index.

    Returns:
        dict with export stats and paths.
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    exported_paths = []
    index_entries = []

    for sg in subgraphs:
        path = export_subgraph_json(sg, output_dir)
        exported_paths.append(path)

        index_entries.append(
            {
                "subgraph_id": sg.subgraph_id,
                "anchor_key": sg.anchor_key,
                "anchor_title": sg.anchor_title,
                "tier": sg.tier,
                "content_count": sg.content_count,
                "relation_count": sg.relation_count,
                "total_nodes": len(sg.nodes),
                "core_nodes": len(sg.core_node_ids),
                "enriched_nodes": len(sg.enriched_node_ids),
                "query_allocation": sg.query_allocation,
                "file": f"{sg.subgraph_id}.json",
            }
        )

    # Write index
    index_path = str(Path(output_dir) / "index.json")
    index_data = {
        "total_subgraphs": len(subgraphs),
        "total_queries_allocated": sum(sg.query_allocation for sg in subgraphs),
        "tier_distribution": _tier_distribution(subgraphs),
        "subgraphs": index_entries,
    }
    with open(index_path, "w") as f:
        json.dump(index_data, f, indent=2, ensure_ascii=False)

    return {
        "exported_count": len(exported_paths),
        "index_path": index_path,
        "output_dir": output_dir,
        "total_queries_allocated": index_data["total_queries_allocated"],
    }


def _tier_distribution(subgraphs: list[Subgraph]) -> dict[str, int]:
    """Count subgraphs per tier."""
    dist: dict[str, int] = {}
    for sg in subgraphs:
        dist[sg.tier] = dist.get(sg.tier, 0) + 1
    return dist
