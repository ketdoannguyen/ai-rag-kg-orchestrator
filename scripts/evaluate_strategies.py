#!/usr/bin/env python3
"""Clean 6-method evaluation script (B1, B2, B3, E1a, E2a, E3a).

Reads a benchmark dataset JSON, runs real retrieval for 6 cosine-only
methods, and outputs a results table plus a JSON results file.

Usage:
    uv run python scripts/evaluate_strategies.py
    uv run python scripts/evaluate_strategies.py --dataset data/benchmark/dataset_50q/benchmark_dataset_50q.json
    uv run python scripts/evaluate_strategies.py --k 10 --output data/benchmark/results
    uv run python scripts/evaluate_strategies.py --concurrency 4
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(name)s | %(message)s")
logger = logging.getLogger("evaluate_strategies")

# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────
DEFAULT_DATASET = "data/benchmark/dataset_50q/benchmark_dataset_50q.json"
DEFAULT_OUTPUT_DIR = "data/benchmark/strategy_results"
DEFAULT_K = 10
DEFAULT_CONCURRENCY = 4

# AQL for B1: vector-only cosine search
AQL_VECTOR_ONLY = """
FOR c IN knowledge_vector_view
    SEARCH ANALYZER(c.node_type IN ["paragraph", "image", "table"], "identity")
    LET score = COSINE_SIMILARITY(c.embedding, @query_embedding)
    FILTER score > 0
    SORT score DESC
    LIMIT @top_k
    RETURN { _id: c._id, score: score }
"""


# ─────────────────────────────────────────────────────────────────────────────
# Dataset loading
# ─────────────────────────────────────────────────────────────────────────────


def load_dataset(path: str):
    from workflows.benchmark.models import BenchmarkDataset

    with open(path) as f:
        data = json.load(f)
    dataset = BenchmarkDataset(**data)
    logger.info(
        "Loaded %d queries, %d subgraphs from %s",
        len(dataset.queries),
        len(dataset.subgraphs),
        path,
    )
    return dataset


def build_subgraphs_index(dataset) -> dict:
    """Build {subgraph_id: Subgraph} index for E3 grading."""
    return {sg.subgraph_id: sg for sg in dataset.subgraphs}


# ─────────────────────────────────────────────────────────────────────────────
# Query embedding
# ─────────────────────────────────────────────────────────────────────────────


async def embed_queries(queries, llm_client) -> dict[str, list[float]]:
    """Embed all queries and return {query_id: embedding}."""
    from llm_gateway_client.models.embedding import EmbeddingRequest

    logger.info("Embedding %d queries...", len(queries))
    texts = [q.query_text for q in queries]
    batch_size = 20
    embeddings: list[list[float]] = []

    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        resp = await llm_client.embed(EmbeddingRequest(input=batch))
        embeddings.extend(resp.embeddings)
        logger.info("  Embedded %d/%d", min(i + batch_size, len(texts)), len(texts))

    result = {q.query_id: emb for q, emb in zip(queries, embeddings)}
    logger.info("All queries embedded (dim=%d)", len(embeddings[0]))
    return result


# ─────────────────────────────────────────────────────────────────────────────
# Retrieval functions (6 methods)
# ─────────────────────────────────────────────────────────────────────────────


async def retrieve_b1(db, query_embedding: list[float], top_k: int) -> list[str]:
    """B1: Vector-Only cosine search."""
    import asyncio
    from concurrent.futures import ThreadPoolExecutor

    loop = asyncio.get_event_loop()
    cursor = await loop.run_in_executor(
        ThreadPoolExecutor(max_workers=2),
        lambda: db.aql.execute(
            AQL_VECTOR_ONLY,
            bind_vars={"query_embedding": query_embedding, "top_k": top_k},
        ),
    )
    return [r["_id"] for r in list(cursor)]


async def retrieve_b2(db, query_text: str, top_k: int) -> list[str]:
    """B2: BM25 full-text search (identity analyzer for Vietnamese)."""
    from workflows.query_retrieval.strategies.baselines import BM25Retrieval

    b2 = BM25Retrieval(db=db)
    results = await b2.execute(query_text=query_text, top_k=top_k)
    return [
        f"content_nodes/{r.node_id}" if not r.node_id.startswith("content_nodes/") else r.node_id
        for r in results
    ]


async def retrieve_b3(db, query_text: str, query_embedding: list[float], top_k: int) -> list[str]:
    """B3: Hybrid RRF(Vector+BM25)."""
    from workflows.query_retrieval.strategies.baselines import HybridRetrieval

    b3 = HybridRetrieval(db=db)
    results = await b3.execute(query_text=query_text, query_embedding=query_embedding, top_k=top_k)
    return [
        f"content_nodes/{r.node_id}" if not r.node_id.startswith("content_nodes/") else r.node_id
        for r in results
    ]


async def retrieve_e1a(db, query_embedding: list[float], top_k: int) -> list[str]:
    """E1a: Top-Down COSINE."""
    from workflows.query_retrieval.strategies.scoring import E1ScoringMethod
    from workflows.query_retrieval.strategies.top_down import TopDownRetrieval

    e1 = TopDownRetrieval(db=db)
    results = await e1.execute(
        query_embedding=query_embedding,
        top_k=top_k,
        scoring_method=E1ScoringMethod.COSINE,
    )
    return [
        f"content_nodes/{r.node_id}" if not r.node_id.startswith("content_nodes/") else r.node_id
        for r in results
    ]


async def retrieve_e2a(db, query_text: str, query_embedding: list[float], top_k: int) -> list[str]:
    """E2a: Horizontal COSINE."""
    from workflows.query_retrieval.strategies.horizontal import HorizontalRetrieval
    from workflows.query_retrieval.strategies.scoring import E2ScoringMethod

    e2 = HorizontalRetrieval(db=db)
    results = await e2.execute(
        query=query_text,
        query_embedding=query_embedding,
        query_type="factual",
        top_k=top_k,
        scoring_method=E2ScoringMethod.COSINE,
    )
    return [
        f"content_nodes/{r.node_id}" if not r.node_id.startswith("content_nodes/") else r.node_id
        for r in results
    ]


async def retrieve_e3a(db, query_embedding: list[float], top_k: int) -> list[str]:
    """E3a: Bottom-Up PARENT_COSINE — returns structure_nodes."""
    from workflows.query_retrieval.strategies.bottom_up import BottomUpRetrieval
    from workflows.query_retrieval.strategies.scoring import E3ScoringMethod

    e3 = BottomUpRetrieval(db=db)
    results = await e3.execute(
        query_embedding=query_embedding,
        top_k=top_k,
        scoring_method=E3ScoringMethod.PARENT_COSINE,
    )
    return [f"structure_nodes/{r.node_id}" if "/" not in r.node_id else r.node_id for r in results]


# ─────────────────────────────────────────────────────────────────────────────
# Method dispatch
# ─────────────────────────────────────────────────────────────────────────────


async def run_method_on_query(
    method_id: str,
    query_text: str,
    query_embedding: list[float],
    db,
    top_k: int,
) -> list[str]:
    """Dispatch to correct retrieval strategy."""
    try:
        if method_id == "B1":
            return await retrieve_b1(db, query_embedding, top_k)
        elif method_id == "B2":
            return await retrieve_b2(db, query_text, top_k)
        elif method_id == "B3":
            return await retrieve_b3(db, query_text, query_embedding, top_k)
        elif method_id == "E1a":
            return await retrieve_e1a(db, query_embedding, top_k)
        elif method_id == "E2a":
            return await retrieve_e2a(db, query_text, query_embedding, top_k)
        elif method_id == "E3a":
            return await retrieve_e3a(db, query_embedding, top_k)
        else:
            logger.warning("Unknown method %s", method_id)
            return []
    except Exception as e:
        logger.error("Method %s failed: %s", method_id, str(e)[:120])
        return []


# ─────────────────────────────────────────────────────────────────────────────
# Run all 6 methods
# ─────────────────────────────────────────────────────────────────────────────

METHODS = [
    {"id": "B1", "name": "Vector-Only"},
    {"id": "B2", "name": "BM25 (identity)"},
    {"id": "B3", "name": "Hybrid RRF(B1,B2)"},
    {"id": "E1a", "name": "TopDown COSINE"},
    {"id": "E2a", "name": "Horizontal COSINE"},
    {"id": "E3a", "name": "BottomUp PARENT_COSINE"},
]


async def run_all_methods(
    queries,
    embeddings: dict[str, list[float]],
    db,
    top_k: int,
    concurrency: int,
) -> dict[str, dict[str, list[str]]]:
    """Run all 6 methods on all queries. Returns {method_id: {query_id: [ids]}}."""
    all_results: dict[str, dict[str, list[str]]] = {m["id"]: {} for m in METHODS}
    total = len(METHODS) * len(queries)
    done = 0
    sem = asyncio.Semaphore(concurrency)

    async def run_one(method, query):
        nonlocal done
        async with sem:
            qid = query.query_id
            retrieved = await run_method_on_query(
                method_id=method["id"],
                query_text=query.query_text,
                query_embedding=embeddings[qid],
                db=db,
                top_k=top_k,
            )
            all_results[method["id"]][qid] = retrieved
            done += 1
            if done % 20 == 0 or done == total:
                logger.info("  Progress: %d/%d", done, total)

    tasks = [run_one(m, q) for m in METHODS for q in queries]
    logger.info(
        "Running %d retrieval calls (%d methods x %d queries)...",
        total,
        len(METHODS),
        len(queries),
    )
    t0 = time.perf_counter()
    await asyncio.gather(*tasks)
    elapsed = time.perf_counter() - t0
    logger.info("All retrievals done in %.1fs (%.1f calls/s)", elapsed, total / elapsed)
    return all_results


# ─────────────────────────────────────────────────────────────────────────────
# Evaluation + display
# ─────────────────────────────────────────────────────────────────────────────


def run_evaluation(queries, all_results: dict, subgraphs_by_id: dict, top_k: int):
    from workflows.benchmark.run_evaluation import (
        COSINE_ONLY_CONFIGS,
        evaluate_all_methods_offline,
    )

    logger.info("=== Evaluating IR metrics (k=%d) ===", top_k)
    summary = evaluate_all_methods_offline(
        queries,
        all_results,
        k=top_k,
        method_configs=COSINE_ONLY_CONFIGS,
        subgraphs_by_id=subgraphs_by_id,
    )
    return summary


def print_results_table(summary):
    """Print results as a clean terminal table."""
    header = (
        f"{'ID':<6} {'Name':<30}  {'NDCG@10':>8}  {'Recall@10':>10}  {'Prec@10':>8}  {'MRR':>8}"
    )
    sep = "-" * len(header)
    print()
    print("=" * len(header))
    print("  EVALUATION RESULTS")
    print("=" * len(header))
    print(header)
    print(sep)
    for mr in summary.method_results:
        print(
            f"  {mr.method_id:<4} {mr.method_name:<30}  {mr.avg_ndcg:>8.4f}  {mr.avg_recall:>10.4f}  "
            f"{mr.avg_precision:>8.4f}  {mr.avg_mrr:>8.4f}"
        )
    print(sep)
    best = max(summary.method_results, key=lambda x: x.avg_ndcg)
    print(f"  Best: {best.method_id} ({best.method_name}) — NDCG@10={best.avg_ndcg:.4f}")
    print()


def save_results(output_dir: Path, dataset, all_results: dict, summary, dataset_path: str, k: int):
    output_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")

    results_path = output_dir / f"eval_strategies_{ts}.json"
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "generated_at": datetime.now(UTC).isoformat(),
                "dataset_path": dataset_path,
                "query_count": len(dataset.queries),
                "k": k,
                "methods": summary.to_dict(),
                "retrieval_results": {
                    mid: {qid: ids[:k] for qid, ids in qmap.items()}
                    for mid, qmap in all_results.items()
                },
            },
            f,
            indent=2,
            ensure_ascii=False,
        )
    logger.info("Saved results → %s", results_path)
    return results_path


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────


async def async_main(args):
    # Load dataset
    dataset = load_dataset(args.dataset)
    queries = dataset.queries
    if not queries:
        logger.error("No queries in dataset!")
        sys.exit(1)

    subgraphs_by_id = build_subgraphs_index(dataset)

    # Connect
    logger.info("=== Connecting to ArangoDB ===")
    from workflows.dependencies import init_arango_client, init_llm_client

    client = init_arango_client()
    llm_client = init_llm_client()
    db = client.db
    logger.info(
        "Connected: %s (%d content nodes)", db.name, db.collection("content_nodes").count()
    )

    # Embed queries
    logger.info("=== Embedding queries ===")
    embeddings = await embed_queries(queries, llm_client)

    # Run retrieval
    logger.info("=== Running retrieval for 6 methods ===")
    all_results = await run_all_methods(
        queries=queries,
        embeddings=embeddings,
        db=db,
        top_k=args.k,
        concurrency=args.concurrency,
    )

    # Evaluate
    summary = run_evaluation(queries, all_results, subgraphs_by_id, args.k)

    # Display
    print_results_table(summary)

    # Save
    output_dir = Path(args.output_dir)
    results_path = save_results(output_dir, dataset, all_results, summary, args.dataset, args.k)

    logger.info("Done. Results → %s", results_path)
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate B1/B2/B3/E1a/E2a/E3a retrieval strategies"
    )
    parser.add_argument(
        "--dataset",
        default=DEFAULT_DATASET,
        help=f"Path to benchmark dataset JSON (default: {DEFAULT_DATASET})",
    )
    parser.add_argument(
        "--output-dir",
        default=DEFAULT_OUTPUT_DIR,
        help=f"Output directory (default: {DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=DEFAULT_K,
        help=f"Retrieval cutoff (default: {DEFAULT_K})",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=DEFAULT_CONCURRENCY,
        help=f"Max concurrent retrieval calls (default: {DEFAULT_CONCURRENCY})",
    )
    args = parser.parse_args()
    success = asyncio.run(async_main(args))
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
