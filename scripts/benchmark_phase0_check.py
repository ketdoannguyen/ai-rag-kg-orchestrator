"""Phase 0: Prerequisites Check for Benchmark v4.0 — Subgraph-First.

Runs T0.1-T0.5 against live ArangoDB and prints GO/NO-GO decision.
v4.0: Added cross-L5 relation analysis (ST1.4) and L5+content check (ST1.3 update).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from workflows.benchmark.models import TIER2_TYPES
from workflows.dependencies import init_arango_client


def t01_count_tier2_edges(client) -> dict:
    """T0.1: Count Tier 2 edges using explicit type list (v4.0)."""
    query = """
    LET tier2_types = @tier2_types
    RETURN {
      total_tier2: LENGTH(FOR e IN knowledge_edges FILTER e.type IN tier2_types RETURN 1),
      by_type: (FOR e IN knowledge_edges FILTER e.type IN tier2_types
                COLLECT type = e.type WITH COUNT INTO cnt
                SORT cnt DESC RETURN {type, cnt})
    }
    """
    results = client.execute_aql(query, bind_vars={"tier2_types": TIER2_TYPES})
    row = results[0] if results else {"total_tier2": 0, "by_type": []}
    return {"tier2_edges": row["by_type"], "total_tier2": row["total_tier2"]}


def t02_verify_embeddings(client) -> dict:
    """T0.2: Verify embedding dimensions and coverage."""
    dim_query = """
    FOR doc IN content_nodes
      FILTER doc.embedding != null
      LIMIT 1
      RETURN LENGTH(doc.embedding)
    """
    dim_results = client.execute_aql(dim_query)
    dimension = dim_results[0] if dim_results else None

    coverage_query = """
    LET total = (FOR doc IN content_nodes RETURN 1)
    LET with_emb = (FOR doc IN content_nodes FILTER doc.embedding != null RETURN 1)
    RETURN {
      total: LENGTH(total),
      with_embedding: LENGTH(with_emb),
      coverage_pct: LENGTH(total) > 0
        ? ROUND(LENGTH(with_emb) * 100 / LENGTH(total))
        : 0
    }
    """
    coverage = client.execute_aql(coverage_query)
    return {"dimension": dimension, "coverage": coverage[0] if coverage else {}}


def t03_kg_stats(client) -> dict:
    """T0.3: Check KG size, coverage stats, and L5 sections with content."""
    query = """
    LET structure = (FOR doc IN structure_nodes RETURN 1)
    LET content = (FOR doc IN content_nodes RETURN 1)
    LET edges = (FOR doc IN knowledge_edges RETURN 1)

    LET levels = (
      FOR doc IN structure_nodes
        COLLECT level = doc.level WITH COUNT INTO cnt
        SORT level
        RETURN {level, cnt}
    )

    LET with_summary = (
      FOR doc IN structure_nodes
        FILTER doc.summary != null AND doc.summary != ""
        RETURN 1
    )

    LET tier2_cross = (
      FOR e IN knowledge_edges
        FILTER e.type NOT IN ["HAS_SECTION", "HAS_CONTENT", "NEXT_SIBLING"]
        LET from_parent = FIRST(
          FOR p IN knowledge_edges
            FILTER p._to == e._from AND p.type == "HAS_CONTENT"
            RETURN p._from
        )
        LET to_parent = FIRST(
          FOR p IN knowledge_edges
            FILTER p._to == e._to AND p.type == "HAS_CONTENT"
            RETURN p._from
        )
        FILTER from_parent != null AND to_parent != null
        COLLECT same = (from_parent == to_parent) WITH COUNT INTO cnt
        RETURN {same_section: same, count: cnt}
    )

    LET l5_with_content = LENGTH(
      FOR s IN structure_nodes
        FILTER s.level == 5
        LET children = LENGTH(
          FOR v, e IN 1..1 OUTBOUND s knowledge_edges
            FILTER e.type == "HAS_CONTENT"
            RETURN 1
        )
        FILTER children >= 1
        RETURN 1
    )

    RETURN {
      structure_nodes: LENGTH(structure),
      content_nodes: LENGTH(content),
      total_edges: LENGTH(edges),
      levels: levels,
      summary_coverage: {
        total: LENGTH(structure),
        with_summary: LENGTH(with_summary),
        pct: LENGTH(structure) > 0
          ? ROUND(LENGTH(with_summary) * 100 / LENGTH(structure))
          : 0
      },
      cross_section_relations: tier2_cross,
      l5_with_content: l5_with_content
    }
    """
    results = client.execute_aql(query)
    return results[0] if results else {}


def t04_cross_l5_analysis(client) -> dict:
    """T0.5 (ST1.4): Classify Tier 2 relations as same-L5 vs cross-L5.

    Maps content nodes to their L5 ancestor, then classifies each Tier 2
    relation as same-L5 (both nodes share the same L5 ancestor) or cross-L5.
    """
    query = """
    LET tier2_types = @tier2_types

    // Map each content node to its nearest L5 ancestor
    LET node_l5_map = (
      FOR c IN content_nodes
        LET l5_ancestors = (
          FOR v, e IN 1..10 INBOUND c knowledge_edges
            FILTER e.type IN ["HAS_CONTENT", "HAS_SECTION"]
            FILTER IS_SAME_COLLECTION("structure_nodes", v)
            FILTER v.level == 5
            LIMIT 1
            RETURN v._key
        )
        RETURN {node_id: c._id, l5_key: LENGTH(l5_ancestors) > 0 ? l5_ancestors[0] : null}
    )

    LET mapped = (FOR m IN node_l5_map FILTER m.l5_key != null RETURN m)

    LET relations_classified = (
      FOR e IN knowledge_edges
        FILTER e.type IN tier2_types
        LET from_l5 = FIRST(FOR m IN mapped FILTER m.node_id == e._from RETURN m.l5_key)
        LET to_l5 = FIRST(FOR m IN mapped FILTER m.node_id == e._to RETURN m.l5_key)
        FILTER from_l5 != null AND to_l5 != null
        RETURN {same_l5: (from_l5 == to_l5)}
    )

    LET same_count = LENGTH(FOR r IN relations_classified FILTER r.same_l5 RETURN 1)
    LET cross_count = LENGTH(FOR r IN relations_classified FILTER !r.same_l5 RETURN 1)
    LET total = same_count + cross_count

    RETURN {
      same_l5: same_count,
      cross_l5: cross_count,
      total_mapped: total,
      cross_l5_pct: total > 0 ? ROUND(cross_count * 100 / total) : 0,
      content_with_l5: LENGTH(mapped)
    }
    """
    results = client.execute_aql(query, bind_vars={"tier2_types": TIER2_TYPES})
    return results[0] if results else {}


def evaluate_go_nogo(t01, t02, t03, t04) -> dict:
    """T0.4: Evaluate GO/NO-GO based on collected metrics (v4.0)."""
    blockers = []
    warnings = []
    decision = "GO"

    # T0.1: Tier 2 edges
    total_t2 = t01["total_tier2"]
    if total_t2 < 100:
        blockers.append(
            f"NO-GO: Tier 2 edges = {total_t2} (< 100). Must run extract_relations first "
            "(scripts/run_extraction.py --mode relations)."
        )
        decision = "NO-GO"
    elif total_t2 < 200:
        warnings.append(f"Only {total_t2} Tier 2 edges. Scale down to 500 queries.")
        if decision == "GO":
            decision = "PARTIAL GO"

    # T0.2: Embeddings
    dim = t02["dimension"]
    coverage = t02["coverage"]
    if dim is None:
        blockers.append("NO-GO: No embeddings found. Must run save_embeddings_to_arango.")
        decision = "NO-GO"
    elif coverage.get("coverage_pct", 0) < 90:
        warnings.append(
            f"Embedding coverage = {coverage.get('coverage_pct', 0)}% (< 90%). "
            "Consider re-embedding."
        )
        if decision == "GO":
            decision = "PARTIAL GO"

    # T0.3: KG size
    content_nodes = t03.get("content_nodes", 0)
    structure_nodes = t03.get("structure_nodes", 0)
    summary_pct = t03.get("summary_coverage", {}).get("pct", 0)

    if content_nodes < 100:
        warnings.append(f"Content nodes = {content_nodes} (< 100). KG may be too small.")
        if decision == "GO":
            decision = "PARTIAL GO"
    if structure_nodes < 50:
        warnings.append(f"Structure nodes = {structure_nodes} (< 50). KG may be too small.")
        if decision == "GO":
            decision = "PARTIAL GO"
    if summary_pct < 60:
        warnings.append(
            f"Summary coverage = {summary_pct}% (< 60%). "
            "Run enrich_knowledge_tree for missing summaries."
        )
        if decision == "GO":
            decision = "PARTIAL GO"

    # Cross-section relations
    cross_section = t03.get("cross_section_relations", [])
    cross_diff = sum(r["count"] for r in cross_section if r.get("same_section") is False)
    if cross_diff < 50:
        warnings.append(
            f"Cross-section relations = {cross_diff} (< 50). "
            "May limit horizontal retrieval testing."
        )

    # L5 sections
    levels = t03.get("levels", [])
    l5_count = next((lv["cnt"] for lv in levels if lv.get("level") == 5), 0)
    if l5_count < 20:
        warnings.append(f"L5 sections = {l5_count} (< 20). Fewer anchor candidates.")

    # L5 sections with content (ST1.3 v4.0 check)
    l5_with_content = t03.get("l5_with_content", 0)
    if l5_with_content < 50:
        blockers.append(
            f"NO-GO: L5 sections with content = {l5_with_content} (< 50). "
            "Insufficient anchors for subgraph extraction."
        )
        decision = "NO-GO"

    # Cross-L5 coverage analysis (ST1.4 v4.0)
    cross_l5_pct = t04.get("cross_l5_pct", 0)
    if cross_l5_pct < 50:
        warnings.append(
            f"Cross-L5 relations = {cross_l5_pct}% (expected ~89% for v4.0). "
            "ENRICH step may have limited benefit."
        )

    return {
        "decision": decision,
        "blockers": blockers,
        "warnings": warnings,
    }


def main():
    print("=" * 70)
    print("BENCHMARK v4.0 — Phase 0: Prerequisites Check")
    print("=" * 70)

    client = init_arango_client()

    # T0.1
    print("\n--- T0.1: Count Tier 2 Edges ---")
    t01 = t01_count_tier2_edges(client)
    print(f"Total Tier 2 edges: {t01['total_tier2']}")
    for edge in t01["tier2_edges"]:
        print(f"  {edge['type']}: {edge['cnt']}")

    # T0.2
    print("\n--- T0.2: Verify Embeddings ---")
    t02 = t02_verify_embeddings(client)
    print(f"Embedding dimension: {t02['dimension']}")
    cov = t02["coverage"]
    print(
        f"Coverage: {cov.get('with_embedding', 0)}/{cov.get('total', 0)} "
        f"({cov.get('coverage_pct', 0)}%)"
    )

    # T0.3
    print("\n--- T0.3: KG Size & Coverage ---")
    t03 = t03_kg_stats(client)
    print(f"Structure nodes: {t03.get('structure_nodes', 0)}")
    print(f"Content nodes: {t03.get('content_nodes', 0)}")
    print(f"Total edges: {t03.get('total_edges', 0)}")
    print("Levels:")
    for lv in t03.get("levels", []):
        print(f"  L{lv['level']}: {lv['cnt']}")
    sc = t03.get("summary_coverage", {})
    print(
        f"Summary coverage: {sc.get('with_summary', 0)}/{sc.get('total', 0)} ({sc.get('pct', 0)}%)"
    )
    print(f"L5 sections with content: {t03.get('l5_with_content', 0)} (min: 50)")
    print("Cross-section relations:")
    for cs in t03.get("cross_section_relations", []):
        label = "same-section" if cs.get("same_section") else "cross-section"
        print(f"  {label}: {cs['count']}")

    # T0.4 (ST1.4): Cross-L5 analysis
    print("\n--- T0.4 (v4.0): Cross-L5 Relation Coverage ---")
    t04 = t04_cross_l5_analysis(client)
    print(f"Content nodes mapped to L5: {t04.get('content_with_l5', 0)}")
    print(f"Same-L5 relations: {t04.get('same_l5', 0)}")
    print(f"Cross-L5 relations: {t04.get('cross_l5', 0)} ({t04.get('cross_l5_pct', 0)}%)")

    # T0.5: Decision
    print("\n" + "═" * 70)
    print("BENCHMARK PREREQUISITES CHECK")
    print("═" * 70)
    result = evaluate_go_nogo(t01, t02, t03, t04)

    # Summary table
    total_t2 = t01["total_tier2"]
    dim = t02.get("dimension")
    cov_pct = cov.get("coverage_pct", 0)
    l5_with_content = t03.get("l5_with_content", 0)
    cross_l5_pct = t04.get("cross_l5_pct", 0)

    ok = "[✓]"
    nok = "[✗]"
    warn = "[!]"

    print(f"{ok if total_t2 >= 100 else nok} Tier 2 edges: {total_t2:,} (min: 100)")
    print(f"{ok if dim is not None else nok} Embedding dims: {dim} (expected: 3072)")
    print(
        f"{ok if cov_pct >= 90 else warn} Embedding coverage: {cov_pct}%"
        f" ({t03.get('content_nodes', 0)} content nodes)"
    )
    print(
        f"{ok if l5_with_content >= 50 else nok} L5 sections with content: {l5_with_content} (min: 50)"
    )
    print(
        f"{ok if cross_l5_pct >= 50 else warn} Cross-L5 relations: {cross_l5_pct}% (expected ~89% for v4.0)"
    )
    print("═" * 70)
    print(f"DECISION: {result['decision']}")
    if result["blockers"]:
        print("\nBLOCKERS:")
        for b in result["blockers"]:
            print(f"  [X] {b}")
    if result["warnings"]:
        print("\nWARNINGS:")
        for w in result["warnings"]:
            print(f"  [!] {w}")
    print("═" * 70)

    # Output JSON for session logging
    full_report = {
        "t01_tier2_edges": t01,
        "t02_embeddings": t02,
        "t03_kg_stats": t03,
        "t04_cross_l5": t04,
        "decision": result,
    }
    print("\n--- Full Report (JSON) ---")
    print(json.dumps(full_report, indent=2, default=str))

    # Save to file
    out_dir = Path("data/benchmark")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "phase0_results.json"
    with open(out_path, "w") as f:
        json.dump(full_report, f, indent=2, default=str)
    print(f"\nReport saved to {out_path}")

    return 0 if result["decision"] in ("GO", "PARTIAL GO") else 1


if __name__ == "__main__":
    sys.exit(main())
