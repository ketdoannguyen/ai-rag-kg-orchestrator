#!/usr/bin/env python
"""E1 Top-Down Retrieval — step-by-step debug script.

Shows every stage's input/output so you can verify correctness:
  Step 0: Database health check (collections, edges, embeddings)
  Step 1: Embed query via LLM API
  Step 2: Build content-rich pool (structure nodes)
  Step 3: Structure cosine search (find relevant sections)
  Step 4: Content harvest (descend to paragraphs)
  Step 5: Re-rank harvested content (final cosine sort)
  Step 6: Full workflow pipeline (embed → route → retrieve)

Usage:
    python scripts/test_e1_top_down_debug.py
    python scripts/test_e1_top_down_debug.py "Tại sao nhà Trần sụp đổ?"
    python scripts/test_e1_top_down_debug.py --query "ASEAN là gì?" --top-k 5
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
import time
import traceback

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from loguru import logger

# ─── Logger setup ────────────────────────────────────────────────────────────

logger.remove()
logger.add(
    sys.stderr,
    level="DEBUG",
    format=(
        "<green>{time:HH:mm:ss.SSS}</green> | "
        "<level>{level: <8}</level> | "
        "<cyan>{extra[step]}</cyan> | "
        "{message}"
    ),
    filter=lambda record: "step" in record["extra"],
)
# Keep a plain handler for logs from other modules (workflow, router, etc.)
logger.add(
    sys.stderr,
    level="INFO",
    format=(
        "<green>{time:HH:mm:ss.SSS}</green> | "
        "<level>{level: <8}</level> | "
        "<dim>{name}</dim> | "
        "{message}"
    ),
    filter=lambda record: "step" not in record["extra"],
)
log = logger.bind(step="main")


# ─── Helpers ─────────────────────────────────────────────────────────────────


def header(title: str) -> None:
    """Print a bold section header."""
    width = 72
    print(f"\n{'=' * width}")
    print(f"  {title}")
    print(f"{'=' * width}")


def subheader(title: str) -> None:
    """Print a sub-section header."""
    print(f"\n--- {title} {'─' * max(0, 56 - len(title))}")


def table_row(label: str, value, width: int = 30) -> None:
    """Print a key-value row."""
    print(f"  {label:<{width}} {value}")


def truncate(text: str | None, max_len: int = 120) -> str:
    """Truncate text for display."""
    if not text:
        return "(empty)"
    text = text.replace("\n", " ").strip()
    return text[:max_len] + "..." if len(text) > max_len else text


# ─── Step 0: Database Health ─────────────────────────────────────────────────


def step0_database_health(db) -> dict:
    """Check database collections, edges, and embedding coverage."""
    step = logger.bind(step="step-0")
    header("STEP 0: Database Health Check")

    # Version
    version = db.version()
    table_row("ArangoDB version", version)
    table_row("Database name", db.name)

    # Collection counts
    subheader("Collection Counts")
    collections = [
        "content_nodes",
        "structure_nodes",
        "root_nodes",
        "entity_nodes",
        "knowledge_edges",
        "ontology_relations",
    ]
    counts = {}
    for name in collections:
        if db.has_collection(name):
            c = db.collection(name).count()
            counts[name] = c
            table_row(name, c)
        else:
            counts[name] = 0
            table_row(name, "NOT FOUND")

    # Edge type breakdown
    subheader("Edge Types")
    cursor = db.aql.execute("""
        FOR e IN knowledge_edges
            COLLECT type = e.type WITH COUNT INTO cnt
            SORT cnt DESC
            RETURN { type, cnt }
    """)
    for doc in cursor:
        table_row(doc["type"], doc["cnt"])

    # Embedding coverage
    subheader("Embedding Coverage")
    for coll in ["structure_nodes", "content_nodes"]:
        if not db.has_collection(coll) or counts.get(coll, 0) == 0:
            continue
        cursor = db.aql.execute(f"""
            LET total = LENGTH({coll})
            LET with_emb = LENGTH(
                FOR d IN {coll}
                    FILTER d.embedding != null AND LENGTH(d.embedding) > 0
                    RETURN 1
            )
            RETURN {{ total: total, with_emb: with_emb }}
        """)
        stats = list(cursor)[0]
        pct = (stats["with_emb"] / stats["total"] * 100) if stats["total"] > 0 else 0
        table_row(coll, f"{stats['with_emb']}/{stats['total']} ({pct:.0f}%)")

    # Sample embedding dimension
    cursor = db.aql.execute("""
        FOR s IN structure_nodes
            FILTER s.embedding != null AND LENGTH(s.embedding) > 0
            LIMIT 1
            RETURN LENGTH(s.embedding)
    """)
    dims = list(cursor)
    if dims:
        table_row("Embedding dimensions", dims[0])

    step.info(
        f"Health check done — {counts.get('content_nodes', 0)} content, "
        f"{counts.get('structure_nodes', 0)} structure, "
        f"{counts.get('knowledge_edges', 0)} edges"
    )

    return counts


# ─── Step 1: Embed Query ─────────────────────────────────────────────────────


async def step1_embed_query(llm_client, query: str) -> list[float]:
    """Embed query text using LLM API."""
    step = logger.bind(step="step-1")
    header("STEP 1: Embed Query")

    table_row("Query", query)

    from llm_gateway_client.models.embedding import EmbeddingRequest

    t0 = time.perf_counter()
    request = EmbeddingRequest(input=[query])
    response = await llm_client.embed(request)
    elapsed = (time.perf_counter() - t0) * 1000

    embedding = response.embeddings[0]
    table_row("Dimensions", len(embedding))
    table_row("Latency", f"{elapsed:.0f}ms")
    table_row("First 5 values", [round(v, 6) for v in embedding[:5]])
    table_row("Norm", f"{sum(v**2 for v in embedding) ** 0.5:.4f}")

    step.info(f"Embedded query in {elapsed:.0f}ms → {len(embedding)}-dim vector")
    return embedding


# ─── Step 2: Content-Rich Pool ───────────────────────────────────────────────


async def step2_build_pool(strategy, source_id: str | None) -> list[dict]:
    """Build content-rich pool of structure nodes."""
    step = logger.bind(step="step-2")
    header("STEP 2: Build Content-Rich Pool")

    table_row("Filter", f"source_id={source_id}" if source_id else "ALL sources")
    table_row("Criteria", "HAS_CONTENT count >= HAS_SECTION count, HAS_CONTENT > 0")

    t0 = time.perf_counter()
    pool = await strategy._build_content_rich_pool(source_id)
    elapsed = (time.perf_counter() - t0) * 1000

    table_row("Pool size", len(pool))
    table_row("Latency", f"{elapsed:.0f}ms")

    with_emb = sum(1 for n in pool if n.get("embedding"))
    table_row("With embeddings", f"{with_emb}/{len(pool)}")

    # Show pool nodes
    if pool:
        subheader("Pool Nodes (all)")
        print(f"  {'#':<4} {'Level':<6} {'HAS_CONTENT':<12} {'Has Emb':<8} {'Summary'}")
        print(f"  {'─' * 4} {'─' * 5} {'─' * 11} {'─' * 7} {'─' * 40}")
        for i, n in enumerate(pool):
            has_emb = "yes" if n.get("embedding") else "NO"
            print(
                f"  {i + 1:<4} L{n.get('level', '?'):<5} "
                f"{n.get('has_content_count', '?'):<12} "
                f"{has_emb:<8} "
                f"{truncate(n.get('summary'), 50)}"
            )

    step.info(f"Pool built: {len(pool)} nodes ({with_emb} with embeddings) in {elapsed:.0f}ms")
    return pool


# ─── Step 3: Structure Cosine Search ─────────────────────────────────────────


def step3_structure_search(
    strategy,
    query_embedding: list[float],
    pool: list[dict],
    k_structure: int,
) -> list[dict]:
    """Search pool for most relevant structure nodes."""
    step = logger.bind(step="step-3")
    header("STEP 3: Structure Cosine Search")

    table_row("k_structure", k_structure)
    table_row("Pool size", len(pool))

    t0 = time.perf_counter()
    results = strategy._structure_cosine_search(query_embedding, pool, k_structure)
    elapsed = (time.perf_counter() - t0) * 1000

    table_row("Matched", len(results))
    table_row("Latency", f"{elapsed:.1f}ms")

    if results:
        subheader("Matched Structure Nodes (ranked by cosine)")
        print(f"  {'#':<4} {'Score':<10} {'Level':<6} {'_key':<40} {'Summary'}")
        print(f"  {'─' * 4} {'─' * 9} {'─' * 5} {'─' * 39} {'─' * 40}")
        for i, n in enumerate(results):
            print(
                f"  {i + 1:<4} {n['structure_score']:<10.4f} "
                f"L{n.get('level', '?'):<5} "
                f"{n.get('_key', '')[:39]:<40} "
                f"{truncate(n.get('summary'), 45)}"
            )

        # Show score distribution
        scores = [n["structure_score"] for n in results]
        subheader("Score Distribution")
        table_row("Max", f"{scores[0]:.4f}")
        table_row("Min", f"{scores[-1]:.4f}")
        table_row("Mean", f"{sum(scores) / len(scores):.4f}")
        if len(scores) > 1:
            gaps = [scores[i] - scores[i + 1] for i in range(len(scores) - 1)]
            table_row(
                "Largest gap",
                f"{max(gaps):.4f} (between rank {gaps.index(max(gaps)) + 1} and {gaps.index(max(gaps)) + 2})",
            )

    step.info(f"Found {len(results)} structures in {elapsed:.1f}ms")
    return results


# ─── Step 4: Content Harvest ─────────────────────────────────────────────────


async def step4_content_harvest(strategy, top_structures: list[dict]) -> list[dict]:
    """Harvest content nodes from matched structure nodes."""
    step = logger.bind(step="step-4")
    header("STEP 4: Content Harvest")

    structure_ids = [s["_id"] for s in top_structures]
    table_row("Input structures", len(structure_ids))
    for sid in structure_ids:
        table_row("  →", sid)

    t0 = time.perf_counter()
    harvested = await strategy._content_harvest(structure_ids)
    elapsed = (time.perf_counter() - t0) * 1000

    table_row("Harvested nodes", len(harvested))
    table_row("Latency", f"{elapsed:.0f}ms")

    if harvested:
        # Depth distribution
        depths = {}
        for h in harvested:
            d = h.get("depth", 0)
            depths[d] = depths.get(d, 0) + 1

        subheader("Depth Distribution")
        for d in sorted(depths):
            table_row(f"depth={d}", f"{depths[d]} nodes")

        # Show embedding coverage
        with_emb = sum(1 for h in harvested if h.get("embedding"))
        with_text = sum(1 for h in harvested if h.get("text"))
        table_row("With embeddings", f"{with_emb}/{len(harvested)}")
        table_row("With text", f"{with_text}/{len(harvested)}")

        # Show all harvested nodes
        subheader("Harvested Content Nodes")
        print(
            f"  {'#':<4} {'Depth':<6} {'Type':<12} {'Has Emb':<8} {'Source Structure':<45} {'Text'}"
        )
        print(f"  {'─' * 4} {'─' * 5} {'─' * 11} {'─' * 7} {'─' * 44} {'─' * 40}")
        for i, h in enumerate(harvested):
            has_emb = "yes" if h.get("embedding") else "NO"
            print(
                f"  {i + 1:<4} {h.get('depth', '?'):<6} "
                f"{h.get('node_type', '?'):<12} "
                f"{has_emb:<8} "
                f"{str(h.get('source', ''))[:44]:<45} "
                f"{truncate(h.get('text'), 50)}"
            )

    step.info(f"Harvested {len(harvested)} content nodes in {elapsed:.0f}ms")
    return harvested


# ─── Step 5: Re-rank ─────────────────────────────────────────────────────────


def step5_rerank(
    strategy,
    query_embedding: list[float],
    harvested: list[dict],
    top_k: int,
) -> list:
    """Re-rank harvested content by cosine similarity."""
    step = logger.bind(step="step-5")
    header("STEP 5: Re-rank Content (Final Cosine Sort)")

    table_row("Input nodes", len(harvested))
    table_row("top_k", top_k)

    t0 = time.perf_counter()
    results = strategy._rerank_cosine(query_embedding, harvested, top_k)
    elapsed = (time.perf_counter() - t0) * 1000

    table_row("Returned", len(results))
    table_row("Latency", f"{elapsed:.1f}ms")

    if results:
        subheader("Final Results (ranked by relevance)")
        print(f"  {'#':<4} {'Score':<10} {'Type':<12} {'Node ID':<40} {'Text'}")
        print(f"  {'─' * 4} {'─' * 9} {'─' * 11} {'─' * 39} {'─' * 50}")
        for i, r in enumerate(results):
            print(
                f"  {i + 1:<4} {r.score:<10.4f} "
                f"{r.node_type:<12} "
                f"{r.node_id[:39]:<40} "
                f"{truncate(r.text, 55)}"
            )

        # Show full text of top 3
        subheader("Top 3 — Full Text")
        for i, r in enumerate(results[:3]):
            print(f"\n  [{i + 1}] Score: {r.score:.4f} | Type: {r.node_type} | ID: {r.node_id}")
            print(f"  {'─' * 68}")
            # Wrap text at ~70 chars for readability
            text = r.text or "(empty)"
            for line in text.split("\n"):
                while len(line) > 70:
                    print(f"    {line[:70]}")
                    line = line[70:]
                print(f"    {line}")

    step.info(f"Re-ranked → {len(results)} results in {elapsed:.1f}ms")
    return results


# ─── Step 6: Full Workflow ───────────────────────────────────────────────────


async def step6_full_workflow(db, llm_client, query: str, top_k: int) -> None:
    """Run the complete query_retrieval workflow and show all outputs."""
    step = logger.bind(step="step-6")
    header("STEP 6: Full Workflow Pipeline")

    from workflows.query_retrieval.models import QueryRetrievalInput
    from workflows.query_retrieval.workflow import QueryRetrievalWorkflow

    table_row("Query", query)
    table_row("top_k", top_k)

    workflow = QueryRetrievalWorkflow(db=db, llm_client=llm_client)

    t0 = time.perf_counter()
    result = await workflow.run(QueryRetrievalInput(query=query, top_k=top_k))
    elapsed = (time.perf_counter() - t0) * 1000

    # Classification
    subheader("Query Classification")
    c = result.classification
    table_row("query_type", c.query_type.value)
    table_row("confidence", f"{c.confidence:.2f}")
    table_row("strategy", c.strategy.value)
    table_row("use_relations", c.use_relations)
    table_row("selected_relations", c.selected_relations or "(none)")
    table_row("reasoning", truncate(c.reasoning, 80))

    # Timing
    subheader("Timing Breakdown")
    table_row("Embedding", f"{result.embedding_time_ms:.0f}ms")
    table_row("Relation filtering", f"{result.relation_filter_time_ms:.0f}ms")
    table_row("LLM routing", f"{result.llm_routing_time_ms:.0f}ms")
    table_row("Retrieval", f"{result.retrieval_time_ms:.0f}ms")
    table_row("TOTAL", f"{result.total_time_ms:.0f}ms")

    # Top-down metrics (if used)
    if result.top_down_metrics:
        subheader("E1 Top-Down Metrics")
        for k, v in result.top_down_metrics.items():
            table_row(k, v)

    # Content results
    subheader(f"Content Results ({len(result.content_results)})")
    for i, r in enumerate(result.content_results[:10]):
        print(f"  [{i + 1}] score={r.score:.4f} | type={r.node_type} | {truncate(r.text, 70)}")

    # Related by relation
    if result.related_by_relation:
        subheader("Related Nodes by Relation")
        for rel, nodes in result.related_by_relation.items():
            print(f"  {rel}: {len(nodes)} nodes")
            for n in nodes[:3]:
                print(f"    → {truncate(n.text, 60)}")

    # Summary
    subheader("Summary")
    table_row("Total results", result.total_results)
    table_row("Content results", len(result.content_results))
    table_row("Related results", sum(len(v) for v in result.related_by_relation.values()))

    step.info(f"Full workflow done in {elapsed:.0f}ms — {result.total_results} results")


# ─── Main ────────────────────────────────────────────────────────────────────


async def main():
    parser = argparse.ArgumentParser(description="E1 Top-Down Retrieval debug script")
    parser.add_argument("query", nargs="?", default=None, help="Query text")
    parser.add_argument(
        "--query", "-q", dest="query_flag", default=None, help="Query text (flag form)"
    )
    parser.add_argument(
        "--top-k", "-k", type=int, default=5, help="Number of results (default: 10)"
    )
    parser.add_argument(
        "--k-structure", type=int, default=10, help="Structure nodes to match (default: 7)"
    )
    parser.add_argument("--source-id", default=None, help="Filter by source ID")
    parser.add_argument(
        "--skip-llm", action="store_true", help="Skip LLM calls (use borrowed embedding)"
    )
    parser.add_argument("--step", type=int, default=None, help="Run only a specific step (0-6)")
    args = parser.parse_args()

    query = (
        args.query_flag
        or args.query
        or "Ý tưởng xây dựng Cộng đồng ASEAN được hình thành như thế nào?"
    )

    print(f"\n{'#' * 72}")
    print("#  E1 TOP-DOWN RETRIEVAL — DEBUG SCRIPT")
    print(f"#  Query: {query}")
    print(f"#  top_k={args.top_k}  k_structure={args.k_structure}  source_id={args.source_id}")
    print(f"{'#' * 72}")

    # Connect to DB
    from workflows.dependencies import init_arango_client, init_llm_client

    try:
        client = init_arango_client()
        db = client._db
    except Exception as e:
        log.error(f"Failed to connect to ArangoDB: {e}")
        traceback.print_exc()
        sys.exit(1)

    # ── Step 0: Health check ──
    if args.step is None or args.step == 0:
        try:
            step0_database_health(db)
        except Exception as e:
            log.error(f"Step 0 failed: {e}")
            traceback.print_exc()

    # ── Step 1: Embed query ──
    query_embedding = None
    llm_client = None

    if not args.skip_llm:
        try:
            llm_client = init_llm_client()
        except Exception as e:
            log.warning(f"Cannot init LLM client: {e}. Will use borrowed embedding.")
            args.skip_llm = True

    if not args.skip_llm and (args.step is None or args.step == 1):
        try:
            query_embedding = await step1_embed_query(llm_client, query)
        except Exception as e:
            log.error(f"Step 1 failed: {e}")
            traceback.print_exc()
            log.warning("Falling back to borrowed embedding")
            args.skip_llm = True

    # Borrow embedding if needed
    if query_embedding is None:
        header("FALLBACK: Borrowing Embedding from DB")
        cursor = db.aql.execute("""
            FOR s IN structure_nodes
                FILTER s.embedding != null AND LENGTH(s.embedding) > 0
                LIMIT 1
                RETURN { embedding: s.embedding, summary: s.summary, _key: s._key }
        """)
        docs = list(cursor)
        if not docs:
            log.error("No structure nodes with embeddings. Cannot proceed.")
            sys.exit(1)
        doc = docs[0]
        query_embedding = doc["embedding"]
        table_row("Borrowed from", f"structure_nodes/{doc['_key']}")
        table_row("Dimensions", len(query_embedding))
        table_row("Summary", truncate(doc.get("summary"), 80))
        log.warning("Using BORROWED embedding — results will match the source node, not the query")

    # ── Steps 2-5: TopDownRetrieval internals ──
    from workflows.query_retrieval.strategies.top_down import TopDownRetrieval

    strategy = TopDownRetrieval(db=db)

    pool = None
    top_structures = None
    harvested = None

    if args.step is None or args.step == 2:
        try:
            pool = await step2_build_pool(strategy, args.source_id)
        except Exception as e:
            log.error(f"Step 2 failed: {e}")
            traceback.print_exc()

    if pool and (args.step is None or args.step == 3):
        try:
            top_structures = step3_structure_search(
                strategy,
                query_embedding,
                pool,
                args.k_structure,
            )
        except Exception as e:
            log.error(f"Step 3 failed: {e}")
            traceback.print_exc()

    if top_structures and (args.step is None or args.step == 4):
        try:
            harvested = await step4_content_harvest(strategy, top_structures)
        except Exception as e:
            log.error(f"Step 4 failed: {e}")
            traceback.print_exc()

    if harvested and (args.step is None or args.step == 5):
        try:
            step5_rerank(strategy, query_embedding, harvested, args.top_k)
        except Exception as e:
            log.error(f"Step 5 failed: {e}")
            traceback.print_exc()

    # ── Step 6: Full workflow ──
    if not args.skip_llm and llm_client and (args.step is None or args.step == 6):
        try:
            await step6_full_workflow(db, llm_client, query, args.top_k)
        except Exception as e:
            log.error(f"Step 6 failed: {e}")
            traceback.print_exc()
    elif args.step is None or args.step == 6:
        header("STEP 6: Full Workflow Pipeline — SKIPPED (no LLM)")
        print("  Use without --skip-llm to run the full workflow with embedding + routing")

    # ── Done ──
    header("DONE")
    print(f"  Query: {query}")
    print("  All steps completed. Review output above for correctness.\n")


if __name__ == "__main__":
    asyncio.run(main())
