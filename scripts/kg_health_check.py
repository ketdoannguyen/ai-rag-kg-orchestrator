"""KG Health Check — verify graph density for E2 Horizontal Retrieval.

Standalone script that checks Knowledge Graph health metrics to determine
if E2 Horizontal Retrieval is feasible. Outputs a colored console report
and saves results to JSON.

Decision thresholds:
    Green (OK):      Tier 2 coverage > 50%, avg degree > 3.0, edges > 500
    Yellow (Caution): Tier 2 coverage 30-50%, avg degree 1.5-3.0, edges 200-500
    Red (Block):     Tier 2 coverage < 30%, avg degree < 1.5, edges < 200

Usage:
    python scripts/kg_health_check.py
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

from arango import ArangoClient
from loguru import logger

# =============================================================================
# Tier 2 + Shared relation types
# =============================================================================

TIER2_RELATION_TYPES = [
    "ELABORATES",
    "SUMMARIZES",
    "EXEMPLIFIES",
    "DEFINES",
    "SUPPORTS",
    "CONTRADICTS",
    "CONTRASTS",
    "CONTINUES",
    "INTRODUCES",
    "CONCLUDES",
    "CONTEXTUALIZES",
    "VISUALIZES",
    "PREREQUISITE_OF",
    "BUILDS_UPON",
    "CAUSED_BY",
    "LED_TO",
    "RESULTED_IN",
    "CONSEQUENCE_OF",
    "PRECEDED_BY",
    "FOLLOWED_BY",
    "DURING",
    "CONTEMPORARY_WITH",
    "LED_BY",
    "PARTICIPATED_IN",
    "CONTRIBUTED_TO",
    "OPPOSED_BY",
    "IS_A",
    "PART_OF",
    "SIMILAR_TO",
    "REFERS_TO",
    "DUPLICATE_OF",
    "SUPERSEDES",
]

# =============================================================================
# Threshold classification
# =============================================================================

THRESHOLDS = {
    "tier2_coverage_pct": {"green": 50.0, "yellow": 30.0},
    "avg_degree": {"green": 3.0, "yellow": 1.5},
    "total_tier2_edges": {"green": 500, "yellow": 200},
    "embedding_coverage_pct": {"green": 90.0, "yellow": 70.0},
}

STATUS_ICONS = {
    "green": "\033[92m●\033[0m",
    "yellow": "\033[93m●\033[0m",
    "red": "\033[91m●\033[0m",
}


def classify_metric(metric_name: str, value: float) -> str:
    """Classify a metric value as green/yellow/red."""
    thresholds = THRESHOLDS.get(metric_name)
    if not thresholds:
        return "green"
    if value >= thresholds["green"]:
        return "green"
    if value >= thresholds["yellow"]:
        return "yellow"
    return "red"


# =============================================================================
# Health check queries
# =============================================================================


def check_totals(db) -> dict:
    """CHECK 1: Total nodes and edges."""
    cursor = db.aql.execute("""
        LET total_content = LENGTH(content_nodes)
        LET total_structure = LENGTH(structure_nodes) + LENGTH(root_nodes)
        LET total_edges = LENGTH(knowledge_edges)
        RETURN { total_content, total_structure, total_edges }
    """)
    return next(cursor)


def check_edge_types(db) -> list[dict]:
    """CHECK 2: Edge count by relation type."""
    cursor = db.aql.execute("""
        FOR e IN knowledge_edges
            COLLECT type = e.type WITH COUNT INTO cnt
            SORT cnt DESC
            RETURN { type, cnt }
    """)
    return list(cursor)


def check_tier2_coverage(db) -> dict:
    """CHECK 3: Tier 2 relation coverage (% content nodes with at least 1 Tier 2 edge)."""
    cursor = db.aql.execute(
        """
        LET tier2_types = @tier2_types
        LET content_with_relations = LENGTH(
            FOR c IN content_nodes
                LET has_tier2 = LENGTH(
                    FOR v, e IN 1..1 ANY c knowledge_edges
                        FILTER e.type IN tier2_types
                        LIMIT 1
                        RETURN 1
                )
                FILTER has_tier2 > 0
                RETURN 1
        )
        LET total_content = LENGTH(content_nodes)
        RETURN {
            content_with_tier2: content_with_relations,
            total_content: total_content,
            coverage_pct: total_content > 0
                ? ROUND(content_with_relations * 1000.0 / total_content) / 10
                : 0
        }
    """,
        bind_vars={"tier2_types": TIER2_RELATION_TYPES},
    )
    return next(cursor)


def check_degree_stats(db) -> dict:
    """CHECK 4: Tier 2 edge degree statistics per content node."""
    cursor = db.aql.execute(
        """
        LET tier2_types = @tier2_types
        LET degrees = (
            FOR c IN content_nodes
                LET degree = LENGTH(
                    FOR v, e IN 1..1 ANY c knowledge_edges
                        FILTER e.type IN tier2_types
                        RETURN 1
                )
                RETURN degree
        )
        RETURN {
            avg_degree: LENGTH(degrees) > 0 ? AVERAGE(degrees) : 0,
            max_degree: LENGTH(degrees) > 0 ? MAX(degrees) : 0,
            min_degree: LENGTH(degrees) > 0 ? MIN(degrees) : 0,
            median_degree: LENGTH(degrees) > 0 ? MEDIAN(degrees) : 0,
            nodes_with_0_edges: LENGTH(FOR d IN degrees FILTER d == 0 RETURN 1)
        }
    """,
        bind_vars={"tier2_types": TIER2_RELATION_TYPES},
    )
    return next(cursor)


def check_embedding_coverage(db) -> dict:
    """CHECK 5: Content node embedding coverage."""
    cursor = db.aql.execute("""
        LET with_emb = LENGTH(
            FOR c IN content_nodes
                FILTER c.embedding != null AND LENGTH(c.embedding) > 0
                RETURN 1
        )
        LET total = LENGTH(content_nodes)
        RETURN {
            with_embedding: with_emb,
            total: total,
            coverage_pct: total > 0 ? ROUND(with_emb * 1000.0 / total) / 10 : 0
        }
    """)
    return next(cursor)


# =============================================================================
# Report
# =============================================================================


def count_tier2_edges(edge_types: list[dict]) -> int:
    """Count total Tier 2 edges from edge type breakdown."""
    return sum(et["cnt"] for et in edge_types if et["type"] in TIER2_RELATION_TYPES)


def run_all_checks(db) -> dict:
    """Run all health checks and return combined results."""
    logger.info("Running KG health checks...")

    t0 = time.perf_counter()
    totals = check_totals(db)
    logger.info("CHECK 1 done: totals", **totals)

    edge_types = check_edge_types(db)
    logger.info("CHECK 2 done: edge types", count=len(edge_types))

    coverage = check_tier2_coverage(db)
    logger.info("CHECK 3 done: Tier 2 coverage", **coverage)

    degree_stats = check_degree_stats(db)
    logger.info("CHECK 4 done: degree stats", **degree_stats)

    emb_coverage = check_embedding_coverage(db)
    logger.info("CHECK 5 done: embedding coverage", **emb_coverage)

    total_tier2_edges = count_tier2_edges(edge_types)
    elapsed_ms = (time.perf_counter() - t0) * 1000

    return {
        "totals": totals,
        "edge_types": edge_types,
        "tier2_coverage": coverage,
        "degree_stats": degree_stats,
        "embedding_coverage": emb_coverage,
        "total_tier2_edges": total_tier2_edges,
        "elapsed_ms": round(elapsed_ms, 1),
    }


def classify_and_report(results: dict) -> str:
    """Classify metrics and print colored report. Returns overall verdict."""
    coverage_pct = results["tier2_coverage"]["coverage_pct"]
    avg_degree = results["degree_stats"]["avg_degree"]
    total_tier2 = results["total_tier2_edges"]
    emb_pct = results["embedding_coverage"]["coverage_pct"]

    classifications = {
        "tier2_coverage_pct": classify_metric("tier2_coverage_pct", coverage_pct),
        "avg_degree": classify_metric("avg_degree", avg_degree),
        "total_tier2_edges": classify_metric("total_tier2_edges", total_tier2),
        "embedding_coverage_pct": classify_metric("embedding_coverage_pct", emb_pct),
    }

    # Overall verdict: Red if any red, else Green
    has_red = any(v == "red" for v in classifications.values())
    verdict = "RED — E2 not feasible, enrich KG first" if has_red else "GREEN — proceed with E2"

    # Print report
    print("\n" + "=" * 60)
    print("  KG HEALTH CHECK — E2 Horizontal Retrieval Readiness")
    print("=" * 60)

    totals = results["totals"]
    print(f"\n  Content nodes:   {totals['total_content']:,}")
    print(f"  Structure nodes: {totals['total_structure']:,}")
    print(f"  Total edges:     {totals['total_edges']:,}")
    print(f"  Tier 2 edges:    {total_tier2:,}")

    print(f"\n  {'Metric':<30} {'Value':>10}  Status")
    print("  " + "-" * 55)

    rows = [
        ("Tier 2 coverage", f"{coverage_pct:.1f}%", classifications["tier2_coverage_pct"]),
        ("Avg degree (Tier 2)", f"{avg_degree:.2f}", classifications["avg_degree"]),
        ("Total Tier 2 edges", f"{total_tier2:,}", classifications["total_tier2_edges"]),
        ("Embedding coverage", f"{emb_pct:.1f}%", classifications["embedding_coverage_pct"]),
    ]

    for label, value, status in rows:
        icon = STATUS_ICONS[status]
        print(f"  {label:<30} {value:>10}  {icon} {status.upper()}")

    # Edge type breakdown (top 10)
    print(f"\n  {'Relation Type':<30} {'Count':>8}")
    print("  " + "-" * 40)
    tier2_edges = [et for et in results["edge_types"] if et["type"] in TIER2_RELATION_TYPES]
    for et in tier2_edges[:15]:
        print(f"  {et['type']:<30} {et['cnt']:>8,}")

    print(f"\n  Elapsed: {results['elapsed_ms']:.0f}ms")
    print(f"\n  VERDICT: {verdict}")
    print("=" * 60 + "\n")

    return "red" if has_red else "green"


# =============================================================================
# Main
# =============================================================================


def main() -> None:
    """Run KG health check."""
    from config.settings import get_settings

    settings = get_settings()

    client = ArangoClient(hosts=settings.ARANGO_URL)
    db = client.db(
        settings.ARANGO_DATABASE,
        username=settings.ARANGO_USER,
        password=settings.ARANGO_PASSWORD,
    )

    results = run_all_checks(db)

    verdict = classify_and_report(results)

    # Save JSON
    output_dir = Path("data")
    output_dir.mkdir(exist_ok=True)
    output_path = output_dir / "kg_health_check_results.json"
    results["verdict"] = verdict
    results["classifications"] = {
        "tier2_coverage_pct": classify_metric(
            "tier2_coverage_pct", results["tier2_coverage"]["coverage_pct"]
        ),
        "avg_degree": classify_metric("avg_degree", results["degree_stats"]["avg_degree"]),
        "total_tier2_edges": classify_metric("total_tier2_edges", results["total_tier2_edges"]),
        "embedding_coverage_pct": classify_metric(
            "embedding_coverage_pct", results["embedding_coverage"]["coverage_pct"]
        ),
    }
    output_path.write_text(json.dumps(results, indent=2, default=str))
    logger.info("Results saved", path=str(output_path))

    if verdict == "red":
        sys.exit(1)


if __name__ == "__main__":
    main()
