#!/usr/bin/env python
"""E1 vs E2 Comparison — Side-by-side benchmark of retrieval strategies.

Compares:
  E1 Top-Down:   structure → content descent
  E2 Horizontal: content → content via Tier 2 relations

Shows:
  - Result overlap (how many nodes appear in both)
  - Unique nodes found by each strategy
  - Score distributions
  - Latency comparison
  - Relation contribution (E2 only)

Usage:
    python scripts/test_e1_vs_e2_comparison.py
    python scripts/test_e1_vs_e2_comparison.py "Nguyên nhân chiến tranh thế giới thứ 2?"
    python scripts/test_e1_vs_e2_comparison.py --query "ASEAN là gì?" --top-k 10
"""

from __future__ import annotations

import argparse
import asyncio
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.resolve()))

from loguru import logger

logger.remove()
logger.add(
    sys.stderr,
    level="INFO",
    format=("<green>{time:HH:mm:ss.SSS}</green> | <level>{level: <8}</level> | {message}"),
)

# ─── Helpers ─────────────────────────────────────────────────────────────────


def header(title: str) -> None:
    """Print a bold section header."""
    width = 76
    print(f"\n{'=' * width}")
    print(f"  {title}")
    print(f"{'=' * width}")


def subheader(title: str) -> None:
    """Print a sub-section header."""
    print(f"\n--- {title} {'─' * max(0, 60 - len(title))}")


def table_row(label: str, value, width: int = 32) -> None:
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


# ─── Run E1 ──────────────────────────────────────────────────────────────────


async def run_e1(db, query_embedding: list[float], top_k: int) -> dict:
    """Run E1 Top-Down strategy."""
    from workflows.query_retrieval.strategies.top_down import TopDownRetrieval

    strategy = TopDownRetrieval(db=db)

    t0 = time.perf_counter()
    results = await strategy.execute(
        query_embedding=query_embedding,
        top_k=top_k,
    )
    elapsed_ms = (time.perf_counter() - t0) * 1000

    return {
        "name": "E1 Top-Down",
        "results": results,
        "latency_ms": elapsed_ms,
        "metrics": strategy.metrics.to_dict() if hasattr(strategy, "metrics") else {},
    }


# ─── Run E2 ──────────────────────────────────────────────────────────────────


async def run_e2(db, query_embedding: list[float], query_type: str, top_k: int) -> dict:
    """Run E2 Horizontal strategy."""
    from workflows.query_retrieval.strategies.horizontal import HorizontalRetrieval

    strategy = HorizontalRetrieval(db=db, graph_name="knowledge_graph")

    t0 = time.perf_counter()
    results = await strategy.execute(
        query="",
        query_embedding=query_embedding,
        query_type=query_type,
        top_k=top_k,
    )
    elapsed_ms = (time.perf_counter() - t0) * 1000

    return {
        "name": "E2 Horizontal",
        "results": results,
        "latency_ms": elapsed_ms,
        "metrics": strategy.metrics.to_dict(),
    }


# ─── Compare Results ─────────────────────────────────────────────────────────


def compare_results(e1_data: dict, e2_data: dict, top_k: int) -> None:
    """Compare E1 and E2 results side by side."""

    e1_results = e1_data["results"]
    e2_results = e2_data["results"]

    # Extract node IDs
    e1_ids = {r.node_id for r in e1_results}
    e2_ids = {r.node_id for r in e2_results}

    overlap = e1_ids & e2_ids
    e1_unique = e1_ids - e2_ids
    e2_unique = e2_ids - e1_ids

    # ── Summary ──
    header("COMPARISON SUMMARY")

    subheader("Result Counts")
    table_row("E1 Top-Down results", len(e1_results))
    table_row("E2 Horizontal results", len(e2_results))
    table_row("Overlap (same nodes)", len(overlap))
    table_row("E1-only (unique to E1)", len(e1_unique))
    table_row("E2-only (unique to E2)", len(e2_unique))

    # Jaccard similarity
    if e1_ids or e2_ids:
        jaccard = len(overlap) / len(e1_ids | e2_ids)
        table_row("Jaccard similarity", f"{jaccard:.2%}")

    # ── Latency ──
    subheader("Latency Comparison")
    table_row("E1 latency", f"{e1_data['latency_ms']:.0f}ms")
    table_row("E2 latency", f"{e2_data['latency_ms']:.0f}ms")
    diff = e2_data["latency_ms"] - e1_data["latency_ms"]
    table_row("Difference (E2 - E1)", f"{diff:+.0f}ms")

    # ── Score Distributions ──
    subheader("Score Distribution")

    e1_scores = [r.score for r in e1_results]
    e2_scores = [r.score for r in e2_results]

    if e1_scores:
        table_row("E1 max score", f"{max(e1_scores):.4f}")
        table_row("E1 min score", f"{min(e1_scores):.4f}")
        table_row("E1 mean score", f"{sum(e1_scores) / len(e1_scores):.4f}")

    if e2_scores:
        table_row("E2 max score", f"{max(e2_scores):.4f}")
        table_row("E2 min score", f"{min(e2_scores):.4f}")
        table_row("E2 mean score", f"{sum(e2_scores) / len(e2_scores):.4f}")

    # ── E2 Metrics ──
    if e2_data["metrics"]:
        subheader("E2 Horizontal Metrics")
        m = e2_data["metrics"]
        table_row("Seeds used", m.get("seed_count", "?"))
        table_row("Relations scored", m.get("relations_scored", "?"))
        table_row("Relations selected", m.get("relations_selected", "?"))
        table_row("Selected types", m.get("selected_relation_types", []))
        table_row("Traversed (raw)", m.get("traversed_nodes_raw", "?"))
        table_row("Traversed (unique)", m.get("traversed_nodes_unique", "?"))
        table_row("From traversal in final", m.get("unique_nodes_from_traversal", "?"))

    # ── Side by Side Top Results ──
    header("SIDE-BY-SIDE: TOP RESULTS")

    max_show = min(top_k, 10)

    print(
        f"\n  {'#':<3} {'E1 Score':<12} {'E1 Node ID':<30} │ {'E2 Score':<12} {'E2 Node ID':<30}"
    )
    print(f"  {'─' * 3} {'─' * 11} {'─' * 29} │ {'─' * 11} {'─' * 29}")

    for i in range(max_show):
        e1_score = color_score(e1_results[i].score) if i < len(e1_results) else "—"
        e1_id = e1_results[i].node_id[:29] if i < len(e1_results) else "—"
        e2_score = color_score(e2_results[i].score) if i < len(e2_results) else "—"
        e2_id = e2_results[i].node_id[:29] if i < len(e2_results) else "—"

        # Mark overlap
        overlap_e1 = "●" if i < len(e1_results) and e1_results[i].node_id in e2_ids else " "
        overlap_e2 = "●" if i < len(e2_results) and e2_results[i].node_id in e1_ids else " "

        print(
            f"  {i + 1:<3} {e1_score:<20} {e1_id:<28}{overlap_e1}│ {e2_score:<20} {e2_id:<28}{overlap_e2}"
        )

    print("\n  ● = also appears in other strategy's results")

    # ── Unique to E2 (the interesting part) ──
    if e2_unique:
        subheader(f"NODES UNIQUE TO E2 ({len(e2_unique)} nodes)")
        print("  These nodes were found via Tier 2 relations, not by vector similarity alone.\n")

        e2_unique_results = [r for r in e2_results if r.node_id in e2_unique]
        for i, r in enumerate(e2_unique_results[:5]):
            relation = r.found_via_relation or "seed"
            print(f"  [{i + 1}] score={r.score:.4f} | via={relation}")
            print(f"      {truncate(r.text, 90)}")
            print()

    # ── Unique to E1 ──
    if e1_unique:
        subheader(f"NODES UNIQUE TO E1 ({len(e1_unique)} nodes)")
        print("  These nodes were found via structure descent, not in E2 results.\n")

        e1_unique_results = [r for r in e1_results if r.node_id in e1_unique]
        for i, r in enumerate(e1_unique_results[:5]):
            print(f"  [{i + 1}] score={r.score:.4f}")
            print(f"      {truncate(r.text, 90)}")
            print()

    # ── Overlap Details ──
    if overlap:
        subheader(f"OVERLAPPING NODES ({len(overlap)} nodes)")
        print("  These nodes appear in both E1 and E2 results.\n")

        # Build lookup
        e1_lookup = {r.node_id: r for r in e1_results}
        e2_lookup = {r.node_id: r for r in e2_results}

        overlap_list = list(overlap)[:5]
        for nid in overlap_list:
            e1_r = e1_lookup[nid]
            e2_r = e2_lookup[nid]
            print(f"  Node: {nid}")
            print(f"    E1 score: {e1_r.score:.4f}")
            print(f"    E2 score: {e2_r.score:.4f} (via: {e2_r.found_via_relation or 'seed'})")
            print(f"    Text: {truncate(e1_r.text, 80)}")
            print()


# ─── Main ────────────────────────────────────────────────────────────────────


async def main():
    parser = argparse.ArgumentParser(description="E1 vs E2 Comparison")
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
    args = parser.parse_args()

    query = args.query_flag or args.query or "Nguyên nhân dẫn đến sự sụp đổ của Liên Xô?"

    print(f"\n{'#' * 76}")
    print("#  E1 vs E2 COMPARISON — Side-by-Side Benchmark")
    print(f"#  Query: {query}")
    print(f"#  top_k={args.top_k}  query_type={args.query_type}")
    print(f"{'#' * 76}")

    # ── Connect ──
    from workflows.dependencies import init_arango_client, init_llm_client

    try:
        client = init_arango_client()
        db = client._db
    except Exception as e:
        logger.error(f"Failed to connect to ArangoDB: {e}")
        sys.exit(1)

    # ── Embed Query ──
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
        cursor = db.aql.execute("""
            FOR c IN content_nodes
                FILTER c.embedding != null AND LENGTH(c.embedding) > 0
                LIMIT 1
                RETURN { embedding: c.embedding, text: c.text, _key: c._key }
        """)
        docs = list(cursor)
        if not docs:
            logger.error("No content nodes with embeddings. Cannot proceed.")
            sys.exit(1)
        doc = docs[0]
        query_embedding = doc["embedding"]
        table_row("Borrowed from", f"content_nodes/{doc['_key']}")
        table_row("Text preview", truncate(doc.get("text"), 70))
        logger.warning("Using BORROWED embedding — results reflect similarity to that node")

    # ── Run E1 ──
    header("RUNNING E1 TOP-DOWN")
    try:
        e1_data = await run_e1(db, query_embedding, args.top_k)
        table_row("Results", len(e1_data["results"]))
        table_row("Latency", f"{e1_data['latency_ms']:.0f}ms")
    except Exception as e:
        logger.error(f"E1 failed: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)

    # ── Run E2 ──
    header("RUNNING E2 HORIZONTAL")
    try:
        e2_data = await run_e2(db, query_embedding, args.query_type, args.top_k)
        table_row("Results", len(e2_data["results"]))
        table_row("Latency", f"{e2_data['latency_ms']:.0f}ms")

        # Show E2 metrics summary
        m = e2_data["metrics"]
        table_row(
            "Seeds → Traversed → Final",
            f"{m.get('seed_count', '?')} → {m.get('traversed_nodes_unique', '?')} → {len(e2_data['results'])}",
        )
        table_row("Relations used", m.get("selected_relation_types", []))
    except Exception as e:
        logger.error(f"E2 failed: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)

    # ── Compare ──
    compare_results(e1_data, e2_data, args.top_k)

    # ── Done ──
    header("DONE")
    print(f"  Query: {query}")
    print(f"  E1: {len(e1_data['results'])} results in {e1_data['latency_ms']:.0f}ms")
    print(f"  E2: {len(e2_data['results'])} results in {e2_data['latency_ms']:.0f}ms")

    e1_ids = {r.node_id for r in e1_data["results"]}
    e2_ids = {r.node_id for r in e2_data["results"]}
    unique_e2 = len(e2_ids - e1_ids)

    if unique_e2 > 0:
        print(f"\n  ✓ E2 found {unique_e2} nodes that E1 missed (via Tier 2 relations)")
    else:
        print("\n  E2 did not find additional nodes beyond E1 (may need more Tier 2 edges)")

    print()


if __name__ == "__main__":
    asyncio.run(main())
