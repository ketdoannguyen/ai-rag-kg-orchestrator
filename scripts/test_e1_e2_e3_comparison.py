#!/usr/bin/env python
"""E1 vs E2 vs E3 Comparison — Side-by-side benchmark of all retrieval strategies.

Compares:
  E1 Top-Down:    structure → content descent
  E2 Horizontal:  content → content via Tier 2 relations
  E3 Bottom-Up:   content → parent summaries aggregation

Shows:
  - Result overlap (Venn diagram style: all three, pairs, unique)
  - Score distributions per strategy
  - Latency comparison
  - Node type distributions
  - Unique contributions from each strategy
  - Metrics specific to each strategy

Usage:
    python scripts/test_e1_e2_e3_comparison.py
    python scripts/test_e1_e2_e3_comparison.py "Nguyên nhân chiến tranh thế giới thứ 2?"
    python scripts/test_e1_e2_e3_comparison.py --query "ASEAN là gì?" --top-k 10
"""

from __future__ import annotations

import argparse
import asyncio
import sys
import time
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.resolve()))

from loguru import logger

logger.remove()
logger.add(
    sys.stderr,
    level="INFO",
    format=("<green>{time:HH:mm:ss.SSS}</green> | <level>{level: <8}</level> | {message}"),
)


# =============================================================================
# Helpers
# =============================================================================


def header(title: str) -> None:
    """Print a bold section header."""
    width = 80
    print(f"\n{'=' * width}")
    print(f"  {title}")
    print(f"{'=' * width}")


def subheader(title: str) -> None:
    """Print a sub-section header."""
    print(f"\n--- {title} {'─' * max(0, 64 - len(title))}")


def table_row(label: str, value, width: int = 36) -> None:
    """Print a key-value row."""
    print(f"  {label:<{width}} {value}")


def truncate(text: str | None, max_len: int = 100) -> str:
    """Truncate text for display."""
    if not text:
        return "(empty)"
    text = text.replace("\n", " ").strip()
    return text[:max_len] + "..." if len(text) > max_len else text


def color_score(score: float) -> str:
    """Color-code score for terminal."""
    if score >= 0.7:
        return f"\033[92m{score:.4f}\033[0m"  # green
    elif score >= 0.4:
        return f"\033[93m{score:.4f}\033[0m"  # yellow
    else:
        return f"\033[91m{score:.4f}\033[0m"  # red


def color_text(text: str, color: str) -> str:
    """Apply ANSI color to text."""
    colors = {
        "red": "\033[91m",
        "green": "\033[92m",
        "yellow": "\033[93m",
        "blue": "\033[94m",
        "magenta": "\033[95m",
        "cyan": "\033[96m",
        "white": "\033[97m",
        "reset": "\033[0m",
    }
    return f"{colors.get(color, '')}{text}{colors['reset']}"


# =============================================================================
# Strategy Results Container
# =============================================================================


@dataclass
class StrategyResult:
    """Container for a single strategy's execution results."""

    name: str
    short_name: str  # E1, E2, E3
    results: list
    latency_ms: float
    metrics: dict


# =============================================================================
# Run E1 Top-Down
# =============================================================================


async def run_e1(
    db, query_embedding: list[float], top_k: int, scoring_method: str = "cosine"
) -> StrategyResult:
    """Run E1 Top-Down strategy."""
    from workflows.query_retrieval.strategies.scoring import E1ScoringMethod
    from workflows.query_retrieval.strategies.top_down import TopDownRetrieval

    strategy = TopDownRetrieval(db=db)

    # Convert string to enum
    method_enum = E1ScoringMethod(scoring_method)

    t0 = time.perf_counter()
    results = await strategy.execute(
        query_embedding=query_embedding,
        top_k=top_k,
        scoring_method=method_enum,
    )
    elapsed_ms = (time.perf_counter() - t0) * 1000

    return StrategyResult(
        name="E1 Top-Down",
        short_name="E1",
        results=results,
        latency_ms=elapsed_ms,
        metrics=strategy.metrics.to_dict() if hasattr(strategy, "metrics") else {},
    )


# =============================================================================
# Run E2 Horizontal
# =============================================================================


async def run_e2(
    db,
    query_embedding: list[float],
    query_type: str,
    top_k: int,
    scoring_method: str = "rrf_cosine_seed",
) -> StrategyResult:
    """Run E2 Horizontal strategy."""
    from workflows.query_retrieval.strategies.horizontal import HorizontalRetrieval
    from workflows.query_retrieval.strategies.scoring import E2ScoringMethod

    strategy = HorizontalRetrieval(db=db, graph_name="knowledge_graph")

    # Convert string to enum
    method_enum = E2ScoringMethod(scoring_method)

    t0 = time.perf_counter()
    results = await strategy.execute(
        query="",
        query_embedding=query_embedding,
        query_type=query_type,
        top_k=top_k,
        scoring_method=method_enum,
    )
    elapsed_ms = (time.perf_counter() - t0) * 1000

    return StrategyResult(
        name="E2 Horizontal",
        short_name="E2",
        results=results,
        latency_ms=elapsed_ms,
        metrics=strategy.metrics.to_dict(),
    )


# =============================================================================
# Run E3 Bottom-Up
# =============================================================================


async def run_e3(
    db, query_embedding: list[float], top_k: int, scoring_method: str = "avg_log"
) -> StrategyResult:
    """Run E3 Bottom-Up strategy."""
    from workflows.query_retrieval.strategies.bottom_up import BottomUpRetrieval
    from workflows.query_retrieval.strategies.scoring import E3ScoringMethod

    strategy = BottomUpRetrieval(db=db, graph_name="knowledge_graph")

    # Convert string to enum
    method_enum = E3ScoringMethod(scoring_method)

    t0 = time.perf_counter()
    results = await strategy.execute(
        query_embedding=query_embedding,
        top_k=top_k,
        scoring_method=method_enum,
    )
    elapsed_ms = (time.perf_counter() - t0) * 1000

    return StrategyResult(
        name="E3 Bottom-Up",
        short_name="E3",
        results=results,
        latency_ms=elapsed_ms,
        metrics=strategy.metrics.to_dict(),
    )


# =============================================================================
# Venn Diagram Analysis
# =============================================================================


def compute_venn_sections(
    e1_ids: set[str], e2_ids: set[str], e3_ids: set[str]
) -> dict[str, set[str]]:
    """Compute all 7 sections of a 3-set Venn diagram.

    Returns:
        Dict with keys: 'all_three', 'e1_e2_only', 'e1_e3_only', 'e2_e3_only',
                        'e1_only', 'e2_only', 'e3_only'
    """
    all_three = e1_ids & e2_ids & e3_ids
    e1_e2_only = (e1_ids & e2_ids) - e3_ids
    e1_e3_only = (e1_ids & e3_ids) - e2_ids
    e2_e3_only = (e2_ids & e3_ids) - e1_ids
    e1_only = e1_ids - e2_ids - e3_ids
    e2_only = e2_ids - e1_ids - e3_ids
    e3_only = e3_ids - e1_ids - e2_ids

    return {
        "all_three": all_three,
        "e1_e2_only": e1_e2_only,
        "e1_e3_only": e1_e3_only,
        "e2_e3_only": e2_e3_only,
        "e1_only": e1_only,
        "e2_only": e2_only,
        "e3_only": e3_only,
    }


def print_venn_summary(venn: dict[str, set[str]]) -> None:
    """Print a visual summary of Venn diagram sections."""
    subheader("Venn Diagram Summary")

    # Simplified Venn representation
    all_three = len(venn["all_three"])
    e1_e2 = len(venn["e1_e2_only"])
    e1_e3 = len(venn["e1_e3_only"])
    e2_e3 = len(venn["e2_e3_only"])
    e1_only = len(venn["e1_only"])
    e2_only = len(venn["e2_only"])
    e3_only = len(venn["e3_only"])

    print(
        f"""
         ╭──────────────────────────╮
         │       E1 Top-Down        │
         │     (unique: {e1_only:>2} nodes)   │
         │                          │
         │      ╭───────────────────┼────────────────────╮
         │      │  E1∩E2: {e1_e2:>2} nodes │                    │
         │      │                   │   E2 Horizontal    │
         │      │    ╭──────────────┼──╮ (unique: {e2_only:>2})   │
         │      │    │ALL THREE: {all_three:>2} │  │                  │
         │      │    ╰──────────────┼──╯                  │
         │      │                   │    E2∩E3: {e2_e3:>2}       │
         │      ╰───────────────────┼────────────────────╯
         │  E1∩E3: {e1_e3:>2}             │
         │                          │   ╭────────────────╮
         ╰──────────────────────────╯   │  E3 Bottom-Up  │
                                        │ (unique: {e3_only:>2})   │
                                        ╰────────────────╯
    """
    )

    # Summary table
    print(f"  {'Section':<24} {'Count':>8}  {'Description'}")
    print(f"  {'-' * 24} {'-' * 8}  {'-' * 40}")
    table_row("E1 ∩ E2 ∩ E3 (all three)", len(venn["all_three"]))
    table_row("E1 ∩ E2 only", len(venn["e1_e2_only"]))
    table_row("E1 ∩ E3 only", len(venn["e1_e3_only"]))
    table_row("E2 ∩ E3 only", len(venn["e2_e3_only"]))
    table_row("E1 only (unique)", len(venn["e1_only"]))
    table_row("E2 only (unique)", len(venn["e2_only"]))
    table_row("E3 only (unique)", len(venn["e3_only"]))


# =============================================================================
# Comparison Logic
# =============================================================================


def compare_results(
    e1: StrategyResult,
    e2: StrategyResult,
    e3: StrategyResult,
    top_k: int,
) -> None:
    """Compare E1, E2, and E3 results comprehensively."""

    # Extract node IDs
    e1_ids = {r.node_id for r in e1.results}
    e2_ids = {r.node_id for r in e2.results}
    e3_ids = {r.node_id for r in e3.results}

    # Compute Venn sections
    venn = compute_venn_sections(e1_ids, e2_ids, e3_ids)

    # ══════════════════════════════════════════════════════════════════════════
    # Summary Section
    # ══════════════════════════════════════════════════════════════════════════
    header("COMPARISON SUMMARY")

    subheader("Result Counts")
    table_row("E1 Top-Down results", len(e1.results))
    table_row("E2 Horizontal results", len(e2.results))
    table_row("E3 Bottom-Up results", len(e3.results))
    table_row("Total unique nodes (union)", len(e1_ids | e2_ids | e3_ids))

    # Pairwise Jaccard
    subheader("Pairwise Jaccard Similarity")

    def jaccard(a: set, b: set) -> float:
        return len(a & b) / len(a | b) if (a | b) else 0.0

    table_row("E1 ↔ E2", f"{jaccard(e1_ids, e2_ids):.2%}")
    table_row("E1 ↔ E3", f"{jaccard(e1_ids, e3_ids):.2%}")
    table_row("E2 ↔ E3", f"{jaccard(e2_ids, e3_ids):.2%}")

    # Three-way overlap
    three_way_jaccard = (
        len(venn["all_three"]) / len(e1_ids | e2_ids | e3_ids)
        if (e1_ids | e2_ids | e3_ids)
        else 0.0
    )
    table_row("Three-way overlap ratio", f"{three_way_jaccard:.2%}")

    # Venn diagram
    print_venn_summary(venn)

    # ══════════════════════════════════════════════════════════════════════════
    # Latency Comparison
    # ══════════════════════════════════════════════════════════════════════════
    subheader("Latency Comparison")

    latencies = [
        ("E1 Top-Down", e1.latency_ms),
        ("E2 Horizontal", e2.latency_ms),
        ("E3 Bottom-Up", e3.latency_ms),
    ]
    latencies_sorted = sorted(latencies, key=lambda x: x[1])

    for name, lat in latencies_sorted:
        bar_len = int(lat / max(e1.latency_ms, e2.latency_ms, e3.latency_ms) * 30)
        bar = "█" * bar_len
        print(f"  {name:<16} {lat:>7.0f}ms  {bar}")

    fastest = latencies_sorted[0][0]
    slowest = latencies_sorted[-1][0]
    speedup = latencies_sorted[-1][1] / latencies_sorted[0][1]
    print(f"\n  Fastest: {fastest} ({speedup:.1f}x faster than {slowest})")

    # ══════════════════════════════════════════════════════════════════════════
    # Score Distribution
    # ══════════════════════════════════════════════════════════════════════════
    subheader("Score Distribution")

    strategies_data = [
        ("E1", e1.results),
        ("E2", e2.results),
        ("E3", e3.results),
    ]

    print(f"\n  {'Strategy':<12} {'Min':>10} {'Max':>10} {'Mean':>10} {'Median':>10}")
    print(f"  {'-' * 12} {'-' * 10} {'-' * 10} {'-' * 10} {'-' * 10}")

    for name, results in strategies_data:
        if results:
            scores = sorted([r.score for r in results])
            min_s = min(scores)
            max_s = max(scores)
            mean_s = sum(scores) / len(scores)
            median_s = scores[len(scores) // 2]
            print(f"  {name:<12} {min_s:>10.4f} {max_s:>10.4f} {mean_s:>10.4f} {median_s:>10.4f}")
        else:
            print(f"  {name:<12} {'—':>10} {'—':>10} {'—':>10} {'—':>10}")

    # ══════════════════════════════════════════════════════════════════════════
    # Node Type Distribution
    # ══════════════════════════════════════════════════════════════════════════
    subheader("Node Type Distribution")

    for name, results in strategies_data:
        type_counts: dict[str, int] = {}
        for r in results:
            t = r.node_type or "unknown"
            type_counts[t] = type_counts.get(t, 0) + 1
        types_str = ", ".join(f"{t}:{c}" for t, c in sorted(type_counts.items()))
        table_row(name, types_str or "(no results)")

    # ══════════════════════════════════════════════════════════════════════════
    # Strategy-Specific Metrics
    # ══════════════════════════════════════════════════════════════════════════
    header("STRATEGY-SPECIFIC METRICS")

    # E1 Metrics
    if e1.metrics:
        subheader("E1 Top-Down Metrics")
        for key, val in e1.metrics.items():
            table_row(key, val)

    # E2 Metrics
    if e2.metrics:
        subheader("E2 Horizontal Metrics")
        m = e2.metrics
        table_row("Seeds used", m.get("seed_count", "?"))
        table_row("Relations scored", m.get("relations_scored", "?"))
        table_row("Relations selected", m.get("relations_selected", "?"))
        table_row("Selected relation types", m.get("selected_relation_types", []))
        table_row("Traversed (raw)", m.get("traversed_nodes_raw", "?"))
        table_row("Traversed (unique)", m.get("traversed_nodes_unique", "?"))
        table_row("Unique from traversal", m.get("unique_nodes_from_traversal", "?"))

    # E3 Metrics
    if e3.metrics:
        subheader("E3 Bottom-Up Metrics")
        m = e3.metrics
        table_row("Content nodes retrieved", m.get("content_nodes_retrieved", "?"))
        table_row("Unique parents found", m.get("unique_parents_found", "?"))
        table_row("Parents returned", m.get("parents_returned", "?"))
        table_row("Avg children/parent", m.get("avg_children_per_parent", "?"))
        table_row("Max children/parent", m.get("max_children_per_parent", "?"))
        table_row("Parents with summary", m.get("parents_with_summary", "?"))
        table_row("Parents with title fallback", m.get("parents_with_title_fallback", "?"))
        table_row("Level distribution", m.get("parent_levels_distribution", {}))

    # ══════════════════════════════════════════════════════════════════════════
    # Side-by-Side Top Results
    # ══════════════════════════════════════════════════════════════════════════
    header("SIDE-BY-SIDE: TOP RESULTS")

    max_show = min(top_k, 10)

    # Header row
    print(
        f"\n  {'#':<3} {'E1 Score':<12} {'E1 Node':<22} │ "
        f"{'E2 Score':<12} {'E2 Node':<22} │ "
        f"{'E3 Score':<12} {'E3 Node':<22}"
    )
    print(f"  {'─' * 3} {'─' * 11} {'─' * 21} │ {'─' * 11} {'─' * 21} │ {'─' * 11} {'─' * 21}")

    for i in range(max_show):
        # E1
        if i < len(e1.results):
            e1_s = color_score(e1.results[i].score)
            e1_id = e1.results[i].node_id[:20]
            e1_mark = "●" if e1.results[i].node_id in (e2_ids | e3_ids) else " "
        else:
            e1_s, e1_id, e1_mark = "—", "—", " "

        # E2
        if i < len(e2.results):
            e2_s = color_score(e2.results[i].score)
            e2_id = e2.results[i].node_id[:20]
            e2_mark = "●" if e2.results[i].node_id in (e1_ids | e3_ids) else " "
        else:
            e2_s, e2_id, e2_mark = "—", "—", " "

        # E3
        if i < len(e3.results):
            e3_s = color_score(e3.results[i].score)
            e3_id = e3.results[i].node_id[:20]
            e3_mark = "●" if e3.results[i].node_id in (e1_ids | e2_ids) else " "
        else:
            e3_s, e3_id, e3_mark = "—", "—", " "

        print(
            f"  {i + 1:<3} {e1_s:<20} {e1_id:<20}{e1_mark}│ "
            f"{e2_s:<20} {e2_id:<20}{e2_mark}│ "
            f"{e3_s:<20} {e3_id:<20}{e3_mark}"
        )

    print("\n  ● = also appears in at least one other strategy's results")

    # ══════════════════════════════════════════════════════════════════════════
    # Unique Nodes Details
    # ══════════════════════════════════════════════════════════════════════════
    header("UNIQUE CONTRIBUTIONS")

    # E1 unique
    if venn["e1_only"]:
        subheader(f"NODES UNIQUE TO E1 ({len(venn['e1_only'])} nodes)")
        print("  Found via structure descent — not in E2 or E3.\n")
        e1_unique_results = [r for r in e1.results if r.node_id in venn["e1_only"]]
        for i, r in enumerate(e1_unique_results[:5]):
            print(f"  [{i + 1}] score={r.score:.4f} | type={r.node_type}")
            print(f"      {truncate(r.text, 85)}")
            print()

    # E2 unique
    if venn["e2_only"]:
        subheader(f"NODES UNIQUE TO E2 ({len(venn['e2_only'])} nodes)")
        print("  Found via Tier 2 relations — not in E1 or E3.\n")
        e2_unique_results = [r for r in e2.results if r.node_id in venn["e2_only"]]
        for i, r in enumerate(e2_unique_results[:5]):
            relation = r.found_via_relation or "seed"
            print(f"  [{i + 1}] score={r.score:.4f} | via={relation}")
            print(f"      {truncate(r.text, 85)}")
            print()

    # E3 unique
    if venn["e3_only"]:
        subheader(f"NODES UNIQUE TO E3 ({len(venn['e3_only'])} nodes)")
        print("  Found via parent aggregation — returns parent summaries, not content nodes.\n")
        e3_unique_results = [r for r in e3.results if r.node_id in venn["e3_only"]]
        for i, r in enumerate(e3_unique_results[:5]):
            print(f"  [{i + 1}] score={r.score:.4f} | type={r.node_type}")
            print(f"      {truncate(r.text, 85)}")
            print()

    # ══════════════════════════════════════════════════════════════════════════
    # All Three Overlap
    # ══════════════════════════════════════════════════════════════════════════
    if venn["all_three"]:
        subheader(f"NODES FOUND BY ALL THREE STRATEGIES ({len(venn['all_three'])} nodes)")
        print("  These nodes are highly relevant — found regardless of approach.\n")

        # Build lookups
        e1_lookup = {r.node_id: r for r in e1.results}
        e2_lookup = {r.node_id: r for r in e2.results}
        e3_lookup = {r.node_id: r for r in e3.results}

        for nid in list(venn["all_three"])[:5]:
            e1_r = e1_lookup.get(nid)
            e2_r = e2_lookup.get(nid)
            e3_r = e3_lookup.get(nid)

            print(f"  Node: {nid}")
            if e1_r:
                print(f"    E1 score: {e1_r.score:.4f}")
            if e2_r:
                print(f"    E2 score: {e2_r.score:.4f} (via: {e2_r.found_via_relation or 'seed'})")
            if e3_r:
                print(f"    E3 score: {e3_r.score:.4f}")
            if e1_r:
                print(f"    Text: {truncate(e1_r.text, 75)}")
            print()


# =============================================================================
# Main
# =============================================================================


async def main():
    parser = argparse.ArgumentParser(description="E1 vs E2 vs E3 Comparison")
    parser.add_argument("query", nargs="?", default=None, help="Query text")
    parser.add_argument(
        "--query", "-q", dest="query_flag", default=None, help="Query text (flag form)"
    )
    parser.add_argument(
        "--top-k", "-k", type=int, default=10, help="Number of results (default: 10)"
    )
    parser.add_argument(
        "--query-type",
        "-t",
        default="causal",
        choices=["factual", "causal", "comparison", "concept"],
        help="Query type for E2 relation selection (default: causal)",
    )
    parser.add_argument(
        "--skip-llm", action="store_true", help="Use borrowed embedding (skip LLM call)"
    )
    # Scoring method arguments
    parser.add_argument(
        "--e1-scoring",
        choices=["cosine", "rrf_content_seed"],
        default="cosine",
        help="E1 scoring method (default: cosine)",
    )
    parser.add_argument(
        "--e2-scoring",
        choices=["cosine", "rrf_cosine_seed", "rrf_cosine_relation", "rrf_cosine_seed_relation"],
        default="rrf_cosine_seed",
        help="E2 scoring method (default: rrf_cosine_seed)",
    )
    parser.add_argument(
        "--e3-scoring",
        choices=["parent_cosine", "avg_log", "rrf_parent_avglog"],
        default="avg_log",
        help="E3 scoring method (default: avg_log)",
    )
    args = parser.parse_args()

    query = args.query_flag or args.query or "Nguyên nhân dẫn đến sự sụp đổ của Liên Xô?"

    print(f"\n{'#' * 80}")
    print("#  E1 vs E2 vs E3 COMPARISON — Full Strategy Benchmark")
    print(f"#  Query: {query}")
    print(f"#  top_k={args.top_k}  query_type={args.query_type}")
    print(f"#  E1 Scoring: {args.e1_scoring}")
    print(f"#  E2 Scoring: {args.e2_scoring}")
    print(f"#  E3 Scoring: {args.e3_scoring}")
    print(f"{'#' * 80}")

    # ══════════════════════════════════════════════════════════════════════════
    # Connect to DB
    # ══════════════════════════════════════════════════════════════════════════
    from workflows.dependencies import init_arango_client, init_llm_client

    try:
        client = init_arango_client()
        db = client._db
    except Exception as e:
        logger.error(f"Failed to connect to ArangoDB: {e}")
        sys.exit(1)

    # ══════════════════════════════════════════════════════════════════════════
    # Embed Query
    # ══════════════════════════════════════════════════════════════════════════
    query_embedding = None

    if not args.skip_llm:
        try:
            llm_client = init_llm_client()
            from llm_gateway_client.models.embedding import EmbeddingRequest

            header("EMBEDDING QUERY")
            table_row("Query", query)

            t0 = time.perf_counter()
            request = EmbeddingRequest(input=[query])
            response = await llm_client.embed(request)
            query_embedding = response.embeddings[0]
            elapsed = (time.perf_counter() - t0) * 1000

            table_row("Dimensions", len(query_embedding))
            table_row("Latency", f"{elapsed:.0f}ms")
        except Exception as e:
            logger.warning(f"LLM embedding failed: {e}. Using borrowed embedding.")

    if query_embedding is None:
        header("BORROWING EMBEDDING FROM DB")
        cursor = db.aql.execute(
            """
            FOR c IN content_nodes
                FILTER c.embedding != null AND LENGTH(c.embedding) > 0
                LIMIT 1
                RETURN { embedding: c.embedding, text: c.text, _key: c._key }
        """
        )
        docs = list(cursor)
        if not docs:
            logger.error("No content nodes with embeddings. Cannot proceed.")
            sys.exit(1)
        doc = docs[0]
        query_embedding = doc["embedding"]
        table_row("Borrowed from", f"content_nodes/{doc['_key']}")
        table_row("Text preview", truncate(doc.get("text"), 70))
        logger.warning("Using BORROWED embedding — results reflect similarity to that node")

    # ══════════════════════════════════════════════════════════════════════════
    # Run All Strategies
    # ══════════════════════════════════════════════════════════════════════════
    header("RUNNING ALL STRATEGIES")

    strategies_to_run = [
        ("E1 Top-Down", run_e1(db, query_embedding, args.top_k, args.e1_scoring)),
        (
            "E2 Horizontal",
            run_e2(db, query_embedding, args.query_type, args.top_k, args.e2_scoring),
        ),
        ("E3 Bottom-Up", run_e3(db, query_embedding, args.top_k, args.e3_scoring)),
    ]

    print("\n  Running E1, E2, E3 concurrently...")
    t0 = time.perf_counter()

    try:
        results = await asyncio.gather(
            *[task for _, task in strategies_to_run],
            return_exceptions=True,
        )
    except Exception as e:
        logger.error(f"Strategy execution failed: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)

    total_time = (time.perf_counter() - t0) * 1000

    # Unpack results
    e1_result, e2_result, e3_result = results

    # Handle exceptions
    for name, result in [("E1", e1_result), ("E2", e2_result), ("E3", e3_result)]:
        if isinstance(result, Exception):
            logger.error(f"{name} failed: {result}")
            import traceback

            traceback.print_exception(type(result), result, result.__traceback__)
            sys.exit(1)

    # Print summary
    subheader("Execution Summary")
    table_row("E1 Top-Down", f"{len(e1_result.results)} results in {e1_result.latency_ms:.0f}ms")
    table_row("E2 Horizontal", f"{len(e2_result.results)} results in {e2_result.latency_ms:.0f}ms")
    table_row("E3 Bottom-Up", f"{len(e3_result.results)} results in {e3_result.latency_ms:.0f}ms")
    table_row("Total wall-clock time", f"{total_time:.0f}ms (concurrent)")

    # ══════════════════════════════════════════════════════════════════════════
    # Compare
    # ══════════════════════════════════════════════════════════════════════════
    compare_results(e1_result, e2_result, e3_result, args.top_k)

    # ══════════════════════════════════════════════════════════════════════════
    # Final Summary
    # ══════════════════════════════════════════════════════════════════════════
    header("FINAL SUMMARY")

    print(f"  Query: {query}\n")

    e1_ids = {r.node_id for r in e1_result.results}
    e2_ids = {r.node_id for r in e2_result.results}
    e3_ids = {r.node_id for r in e3_result.results}
    venn = compute_venn_sections(e1_ids, e2_ids, e3_ids)

    print("  Strategy Results:")
    print(
        f"    E1 Top-Down:   {len(e1_result.results):>3} nodes in {e1_result.latency_ms:>6.0f}ms"
    )
    print(
        f"    E2 Horizontal: {len(e2_result.results):>3} nodes in {e2_result.latency_ms:>6.0f}ms"
    )
    print(
        f"    E3 Bottom-Up:  {len(e3_result.results):>3} nodes in {e3_result.latency_ms:>6.0f}ms"
    )

    print("\n  Unique Contributions:")
    print(f"    E1-only: {len(venn['e1_only']):>3} (structure descent)")
    print(f"    E2-only: {len(venn['e2_only']):>3} (Tier 2 relations)")
    print(f"    E3-only: {len(venn['e3_only']):>3} (parent summaries)")

    print("\n  Overlap:")
    print(f"    All three: {len(venn['all_three']):>3} (consensus nodes)")
    print(f"    E1∩E2:     {len(venn['e1_e2_only']):>3}")
    print(f"    E1∩E3:     {len(venn['e1_e3_only']):>3}")
    print(f"    E2∩E3:     {len(venn['e2_e3_only']):>3}")

    # Key insights
    print("\n  Key Insights:")

    if len(venn["e2_only"]) > 0:
        print(f"    ✓ E2 discovered {len(venn['e2_only'])} nodes via relations (E1/E3 missed)")

    if len(venn["e3_only"]) > 0:
        print(f"    ✓ E3 returns {len(venn['e3_only'])} parent summaries (higher-level view)")

    if len(venn["all_three"]) > len(e1_result.results) // 2:
        print(f"    ✓ Strong consensus: {len(venn['all_three'])} nodes found by all strategies")

    # Note about E3's different nature
    if e3_result.results:
        structure_count = sum(1 for r in e3_result.results if r.node_type == "structure")
        if structure_count > 0:
            print(
                f"    [i] E3 returns structure nodes ({structure_count}/{len(e3_result.results)}) - "
                "summaries, not content details"
            )

    print()


if __name__ == "__main__":
    asyncio.run(main())
