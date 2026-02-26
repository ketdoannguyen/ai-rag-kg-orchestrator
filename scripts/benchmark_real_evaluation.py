#!/usr/bin/env python3
"""Real Benchmark Evaluation — No Fake Retrieval.

Loads the 48-query LLM-generated dataset and runs ACTUAL retrieval for 6
methods (B1/B2/B3/E1a/E2a/E3a) using real embeddings + ArangoDB
+ graph traversal. Zero simulated recall rates.

Usage:
    uv run python scripts/benchmark_real_evaluation.py
    uv run python scripts/benchmark_real_evaluation.py --dataset data/benchmark/dataset_50q/benchmark_dataset_50q.json
    uv run python scripts/benchmark_real_evaluation.py --concurrency 3 --top-k 10
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
logger = logging.getLogger("real_eval")

# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────
DEFAULT_DATASET = "data/benchmark/dataset_100q/benchmark_dataset_100q.json"
DEFAULT_OUTPUT_DIR = "data/benchmark/real_evaluation"
K = 10

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
# Step 1: Load dataset
# ─────────────────────────────────────────────────────────────────────────────


def load_dataset(path: str):
    from workflows.benchmark.models import BenchmarkDataset

    with open(path) as f:
        data = json.load(f)
    dataset = BenchmarkDataset(**data)
    logger.info("Loaded dataset: %d queries from %s", len(dataset.queries), path)
    return dataset


# ─────────────────────────────────────────────────────────────────────────────
# Step 2: Embed all queries (batch)
# ─────────────────────────────────────────────────────────────────────────────


async def embed_queries(queries, llm_client) -> dict[str, list[float]]:
    """Embed all queries and return {query_id: embedding}."""
    from llm_gateway_client.models.embedding import EmbeddingRequest

    logger.info("Embedding %d queries...", len(queries))
    texts = [q.query_text for q in queries]

    # Batch in groups of 20 to avoid API limits
    batch_size = 20
    embeddings: list[list[float]] = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        resp = await llm_client.embed(EmbeddingRequest(input=batch))
        embeddings.extend(resp.embeddings)
        logger.info("  Embedded %d/%d", min(i + batch_size, len(texts)), len(texts))

    result = {q.query_id: emb for q, emb in zip(queries, embeddings)}
    logger.info("All queries embedded. Dim=%d", len(embeddings[0]))
    return result


# ─────────────────────────────────────────────────────────────────────────────
# Step 3: Real retrieval for each method
# ─────────────────────────────────────────────────────────────────────────────


async def retrieve_b1(db, query_embedding: list[float], top_k: int) -> list[str]:
    """B1: Vector-Only — direct cosine AQL query."""
    import asyncio
    from concurrent.futures import ThreadPoolExecutor

    loop = asyncio.get_event_loop()
    executor = ThreadPoolExecutor(max_workers=2)

    cursor = await loop.run_in_executor(
        executor,
        lambda: db.aql.execute(
            AQL_VECTOR_ONLY,
            bind_vars={"query_embedding": query_embedding, "top_k": top_k},
        ),
    )
    results = list(cursor)
    return [r["_id"] for r in results]


async def retrieve_b2(db, query_text: str, top_k: int) -> list[str]:
    """B2: BM25 full-text search."""
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


async def retrieve_e1(db, query_embedding: list[float], top_k: int, scoring: str) -> list[str]:
    """E1: Top-Down retrieval with specified scoring method."""
    from workflows.query_retrieval.strategies.top_down import TopDownRetrieval
    from workflows.query_retrieval.strategies.scoring import E1ScoringMethod

    scoring_map = {
        "COSINE": E1ScoringMethod.COSINE,
        "RRF_CONTENT_SEED": E1ScoringMethod.RRF_CONTENT_SEED,
    }
    scoring_method = scoring_map.get(scoring, E1ScoringMethod.COSINE)

    e1 = TopDownRetrieval(db=db)
    results = await e1.execute(
        query_embedding=query_embedding,
        top_k=top_k,
        scoring_method=scoring_method,
    )
    return [
        f"content_nodes/{r.node_id}" if not r.node_id.startswith("content_nodes/") else r.node_id
        for r in results
    ]


async def retrieve_e2(
    db, query: str, query_embedding: list[float], top_k: int, scoring: str
) -> list[str]:
    """E2: Horizontal retrieval with specified scoring method."""
    from workflows.query_retrieval.strategies.horizontal import HorizontalRetrieval
    from workflows.query_retrieval.strategies.scoring import E2ScoringMethod

    scoring_map = {
        "COSINE": E2ScoringMethod.COSINE,
        "RRF_COSINE_SEED": E2ScoringMethod.RRF_COSINE_SEED,
        "RRF_COSINE_RELATION": E2ScoringMethod.RRF_COSINE_RELATION,
        "RRF_COSINE_SEED_RELATION": E2ScoringMethod.RRF_COSINE_SEED_RELATION,
    }
    scoring_method = scoring_map.get(scoring, E2ScoringMethod.COSINE)

    e2 = HorizontalRetrieval(db=db)
    results = await e2.execute(
        query=query,
        query_embedding=query_embedding,
        query_type="factual",
        top_k=top_k,
        scoring_method=scoring_method,
    )
    return [
        f"content_nodes/{r.node_id}" if not r.node_id.startswith("content_nodes/") else r.node_id
        for r in results
    ]


async def retrieve_e3(db, query_embedding: list[float], top_k: int, scoring: str) -> list[str]:
    """E3: Bottom-Up retrieval with specified scoring method."""
    from workflows.query_retrieval.strategies.bottom_up import BottomUpRetrieval
    from workflows.query_retrieval.strategies.scoring import E3ScoringMethod

    scoring_map = {
        "PARENT_COSINE": E3ScoringMethod.PARENT_COSINE,
        "AVG_LOG": E3ScoringMethod.AVG_LOG,
        "RRF_PARENT_AVGLOG": E3ScoringMethod.RRF_PARENT_AVGLOG,
        "FILTERED_AVG_LOG": E3ScoringMethod.FILTERED_AVG_LOG,
        "RRF_PARENT_FILTERED": E3ScoringMethod.RRF_PARENT_FILTERED,
    }
    scoring_method = scoring_map.get(scoring, E3ScoringMethod.PARENT_COSINE)

    e3 = BottomUpRetrieval(db=db)
    results = await e3.execute(
        query_embedding=query_embedding,
        top_k=top_k,
        scoring_method=scoring_method,
    )
    # NOTE: E3 returns structure_nodes (parent section summaries), NOT content_nodes.
    # The benchmark dataset only has content_nodes in fact_nodes, so E3 will score 0.
    # This is a real finding: E3 operates at a different granularity level.
    return [f"structure_nodes/{r.node_id}" if "/" not in r.node_id else r.node_id for r in results]


# ─────────────────────────────────────────────────────────────────────────────
# Method dispatch table
# ─────────────────────────────────────────────────────────────────────────────


async def run_method_on_query(
    method_config: dict,
    query_text: str,
    query_embedding: list[float],
    db,
    top_k: int,
) -> list[str]:
    """Dispatch to the correct retrieval strategy and return list of node IDs."""
    mid = method_config["id"]
    family = method_config.get("family", "")
    scoring = method_config.get("scoring", "")
    strategy = method_config.get("strategy", "")

    try:
        if mid == "B1":
            return await retrieve_b1(db, query_embedding, top_k)
        elif mid == "B2":
            return await retrieve_b2(db, query_text, top_k)
        elif mid == "B3":
            return await retrieve_b3(db, query_text, query_embedding, top_k)
        elif family == "E1":
            return await retrieve_e1(db, query_embedding, top_k, scoring)
        elif family == "E2":
            return await retrieve_e2(db, query_text, query_embedding, top_k, scoring)
        elif family == "E3":
            return await retrieve_e3(db, query_embedding, top_k, scoring)
        else:
            logger.warning("Unknown method %s, returning empty", mid)
            return []
    except Exception as e:
        logger.error("Method %s failed on query: %s", mid, str(e)[:100])
        return []


# ─────────────────────────────────────────────────────────────────────────────
# Step 4: Run all methods on all queries
# ─────────────────────────────────────────────────────────────────────────────


async def run_all_methods(
    queries,
    embeddings: dict[str, list[float]],
    db,
    top_k: int,
    concurrency: int,
) -> dict[str, dict[str, list[str]]]:
    """Run 6 cosine-only methods on all queries with controlled concurrency.

    Returns: {method_id: {query_id: [retrieved_node_ids]}}
    """
    from workflows.benchmark.run_evaluation import COSINE_ONLY_CONFIGS

    # all_results[method_id][query_id] = [node_ids]
    all_results: dict[str, dict[str, list[str]]] = {cfg["id"]: {} for cfg in COSINE_ONLY_CONFIGS}

    total = len(COSINE_ONLY_CONFIGS) * len(queries)
    done = 0
    sem = asyncio.Semaphore(concurrency)

    async def run_one(method_config, query):
        nonlocal done
        async with sem:
            qid = query.query_id
            qemb = embeddings[qid]
            retrieved = await run_method_on_query(
                method_config=method_config,
                query_text=query.query_text,
                query_embedding=qemb,
                db=db,
                top_k=top_k,
            )
            all_results[method_config["id"]][qid] = retrieved
            done += 1
            if done % 20 == 0 or done == total:
                logger.info("  Progress: %d/%d", done, total)

    tasks = []
    for cfg in COSINE_ONLY_CONFIGS:
        for q in queries:
            tasks.append(run_one(cfg, q))

    logger.info("Running %d retrieval calls (6 methods × %d queries)...", total, len(queries))
    t0 = time.perf_counter()
    await asyncio.gather(*tasks)
    elapsed = time.perf_counter() - t0
    logger.info("All retrievals done in %.1fs (%.1f calls/s)", elapsed, total / elapsed)

    return all_results


# ─────────────────────────────────────────────────────────────────────────────
# Step 5: Evaluate
# ─────────────────────────────────────────────────────────────────────────────


def run_evaluation(queries, all_results: dict, top_k: int, subgraphs_by_id: dict | None = None):
    from workflows.benchmark.run_evaluation import evaluate_all_methods_offline

    logger.info("=== Evaluating IR metrics ===")
    summary = evaluate_all_methods_offline(
        queries, all_results, k=top_k, subgraphs_by_id=subgraphs_by_id
    )

    logger.info("")
    logger.info("  %-6s %-40s  NDCG@10  Recall@10  Prec@10   MRR", "ID", "Name")
    logger.info("  " + "-" * 86)
    for mr in summary.method_results:
        logger.info(
            "  %-6s %-40s  %.4f   %.4f     %.4f   %.4f",
            mr.method_id,
            mr.method_name,
            mr.avg_ndcg,
            mr.avg_recall,
            mr.avg_precision,
            mr.avg_mrr,
        )
    return summary


# ─────────────────────────────────────────────────────────────────────────────
# Step 6: Statistics
# ─────────────────────────────────────────────────────────────────────────────


def run_statistics(queries, all_results: dict, summary, top_k: int):
    from dataclasses import dataclass

    from workflows.benchmark.statistics import (
        run_friedman_test,
        run_pairwise_wilcoxon,
        test_hypotheses,
    )

    logger.info("=== Statistical Analysis ===")

    # run_friedman_test / run_pairwise_wilcoxon take list[MethodRunResult]
    method_results = summary.method_results

    stat, p = run_friedman_test(method_results)
    significant = p < 0.05

    @dataclass
    class FriedmanResult:
        statistic: float
        p_value: float
        significant: bool

    friedman = FriedmanResult(statistic=stat, p_value=p, significant=significant)
    logger.info(
        "  Friedman χ²=%.4f, p=%.2e, significant=%s",
        friedman.statistic,
        friedman.p_value,
        friedman.significant,
    )

    pairwise = run_pairwise_wilcoxon(method_results)
    sig_count = sum(1 for r in pairwise if r.significant)
    logger.info("  Pairwise Wilcoxon: %d/%d significant", sig_count, len(pairwise))

    # test_hypotheses(pairwise_results, method_results)
    hypotheses = test_hypotheses(pairwise, method_results)
    for h in hypotheses:
        status = "✅ SUPPORTED" if h.supported else "❌ NOT supported"
        logger.info("  %s: %s — %s", h.hypothesis_id, status, h.description)

    return friedman, pairwise, hypotheses


# ─────────────────────────────────────────────────────────────────────────────
# Step 7: Save results + generate report
# ─────────────────────────────────────────────────────────────────────────────


def save_results(
    output_dir: Path,
    dataset,
    all_results: dict,
    summary,
    friedman,
    pairwise,
    hypotheses,
    dataset_path: str,
):
    output_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")

    # Save raw results
    results_path = output_dir / f"real_eval_results_{ts}.json"
    with open(results_path, "w") as f:
        json.dump(
            {
                "generated_at": datetime.now(UTC).isoformat(),
                "dataset_path": dataset_path,
                "query_count": len(dataset.queries),
                "methods": summary.to_dict(),
                "retrieval_results": {
                    mid: {qid: ids[:K] for qid, ids in qmap.items()}
                    for mid, qmap in all_results.items()
                },
            },
            f,
            indent=2,
            ensure_ascii=False,
        )
    logger.info("Saved raw results → %s", results_path)

    # Generate markdown report
    report_path = output_dir / f"real_eval_report_{ts}.md"
    _write_report(report_path, dataset, summary, friedman, pairwise, hypotheses, dataset_path)
    logger.info("Saved report → %s", report_path)

    return results_path, report_path


def _write_report(
    path: Path,
    dataset,
    summary,
    friedman,
    pairwise,
    hypotheses,
    dataset_path: str,
):
    from workflows.benchmark.run_evaluation import COSINE_ONLY_CONFIGS

    lines = []
    lines.append("# Real Benchmark Evaluation Report")
    lines.append("")
    lines.append(f"**Generated:** {datetime.now(UTC).strftime('%Y-%m-%d %H:%M UTC')}")
    lines.append(f"**Dataset:** {dataset_path}")
    lines.append(f"**Queries:** {len(dataset.queries)}")
    lines.append(f"**Retrieval:** REAL (no simulation)")
    lines.append(f"**Evaluation:** K={K}")
    lines.append("")

    # Direction distribution
    from collections import Counter

    dir_counts = Counter(q.direction for q in dataset.queries)
    lines.append("## Query Direction Distribution")
    lines.append("")
    lines.append("| Direction | Count | % |")
    lines.append("|-----------|-------|---|")
    for d, c in sorted(dir_counts.items()):
        lines.append(f"| {d or 'N/A'} | {c} | {c / len(dataset.queries) * 100:.1f}% |")
    lines.append("")

    # Method results table
    lines.append("## Performance — 6 Methods (Real Retrieval)")
    lines.append("")
    lines.append("| Rank | ID | Name | NDCG@10 | Recall@10 | Prec@10 | MRR |")
    lines.append("|------|----|----|---------|-----------|---------|-----|")

    ranked = sorted(summary.method_results, key=lambda x: x.avg_ndcg, reverse=True)
    for rank, mr in enumerate(ranked, 1):
        lines.append(
            f"| {rank} | **{mr.method_id}** | {mr.method_name} | **{mr.avg_ndcg:.4f}** | "
            f"{mr.avg_recall:.4f} | {mr.avg_precision:.4f} | {mr.avg_mrr:.4f} |"
        )
    lines.append("")

    # Strategy family summary
    lines.append("## Strategy Family Summary")
    lines.append("")
    families = {"baseline": [], "E1": [], "E2": [], "E3": []}
    cfg_map = {cfg["id"]: cfg for cfg in COSINE_ONLY_CONFIGS}
    for mr in summary.method_results:
        cfg = cfg_map.get(mr.method_id, {})
        fam = cfg.get("family", "baseline")
        families[fam].append(mr)

    lines.append("| Family | Best Method | NDCG@10 | vs Best Baseline |")
    lines.append("|--------|------------|---------|-----------------|")
    baseline_best = max(families["baseline"], key=lambda x: x.avg_ndcg)
    for fam, mrs in families.items():
        if not mrs:
            continue
        best = max(mrs, key=lambda x: x.avg_ndcg)
        if fam == "baseline":
            lines.append(
                f"| **Baseline** | {best.method_id} ({best.method_name}) | {best.avg_ndcg:.4f} | — |"
            )
        else:
            delta = (best.avg_ndcg - baseline_best.avg_ndcg) / baseline_best.avg_ndcg * 100
            lines.append(
                f"| **{fam}** | {best.method_id} ({best.method_name}) | {best.avg_ndcg:.4f} | "
                f"{delta:+.1f}% vs {baseline_best.method_id} |"
            )
    lines.append("")

    # Statistics
    lines.append("## Statistical Analysis")
    lines.append("")
    lines.append("### Friedman Test")
    lines.append("")
    lines.append("| Metric | Value |")
    lines.append("|--------|-------|")
    lines.append(f"| χ² | {friedman.statistic:.4f} |")
    lines.append(f"| p-value | {friedman.p_value:.2e} |")
    lines.append(f"| Significant | {'**YES**' if friedman.significant else 'NO'} (p < 0.05) |")
    lines.append("")

    sig_pairs = [r for r in pairwise if r.significant]
    lines.append("### Pairwise Wilcoxon (BH-corrected)")
    lines.append("")
    lines.append(f"- **Total pairs:** {len(pairwise)} (C(14,2))")
    lines.append(f"- **Significant pairs:** {len(sig_pairs)}/{len(pairwise)}")
    lines.append("")

    if sig_pairs:
        lines.append("| Comparison | p-adj | Cliff's δ | Effect |")
        lines.append("|-----------|-------|-----------|--------|")
        for r in sorted(sig_pairs, key=lambda x: x.p_adjusted)[:15]:
            lines.append(
                f"| {r.method_a} vs {r.method_b} | {r.p_adjusted:.4f} | "
                f"{r.cliff_delta:.3f} | {r.effect_size} |"
            )
        lines.append("")

    # Hypotheses
    lines.append("### Hypothesis Testing")
    lines.append("")
    lines.append("| Hypothesis | Description | Result |")
    lines.append("|-----------|-------------|--------|")
    for h in hypotheses:
        status = "✅ **SUPPORTED**" if h.supported else "❌ NOT supported"
        lines.append(f"| **{h.hypothesis_id}** | {h.description} | {status} |")
    lines.append("")
    supported = sum(1 for h in hypotheses if h.supported)
    lines.append(f"**Summary:** {supported}/{len(hypotheses)} hypotheses supported.")
    lines.append("")

    # Key findings
    lines.append("## Key Findings")
    lines.append("")
    best_method = ranked[0]
    baseline_ndcg = baseline_best.avg_ndcg
    improvement = (
        (best_method.avg_ndcg - baseline_ndcg) / baseline_ndcg * 100 if baseline_ndcg > 0 else 0
    )
    lines.append(
        f"1. **Best method**: {best_method.method_id} ({best_method.method_name}) — NDCG@10 = {best_method.avg_ndcg:.4f}"
    )
    lines.append(
        f"2. **Best baseline**: {baseline_best.method_id} — NDCG@10 = {baseline_best.avg_ndcg:.4f}"
    )
    lines.append(f"3. **Best improvement**: {improvement:+.1f}% over best baseline")
    lines.append(f"4. **Statistical significance**: Friedman p={friedman.p_value:.2e}")
    lines.append(f"5. **Evaluation type**: REAL retrieval (no simulation)")
    lines.append("")

    lines.append("---")
    lines.append("*Generated by `scripts/benchmark_real_evaluation.py` — Claude Code*")

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────


async def async_main(args):
    # Step 1: Load dataset
    dataset = load_dataset(args.dataset)
    queries = dataset.queries
    if not queries:
        logger.error("No queries in dataset!")
        sys.exit(1)
    logger.info("Dataset: %d queries", len(queries))

    subgraphs_by_id = {sg.subgraph_id: sg for sg in dataset.subgraphs}

    # Step 2: Connect
    logger.info("=== Connecting to ArangoDB ===")
    from workflows.dependencies import init_arango_client, init_llm_client

    client = init_arango_client()
    llm_client = init_llm_client()
    db = client.db
    logger.info(
        "Connected: %s (%d content nodes)", db.name, db.collection("content_nodes").count()
    )

    # Step 3: Embed all queries
    logger.info("=== Embedding queries ===")
    embeddings = await embed_queries(queries, llm_client)

    # Step 4: Run 6 cosine-only methods (real retrieval)
    logger.info("=== Running real retrieval for 6 methods ===")
    all_results = await run_all_methods(
        queries=queries,
        embeddings=embeddings,
        db=db,
        top_k=args.top_k,
        concurrency=args.concurrency,
    )

    # Step 5: Evaluate (with subgraphs for E3 grading)
    summary = run_evaluation(queries, all_results, args.top_k, subgraphs_by_id=subgraphs_by_id)

    # Step 6: Statistics
    friedman, pairwise, hypotheses = run_statistics(queries, all_results, summary, args.top_k)

    # Step 7: Save results + report
    logger.info("=== Saving results ===")
    output_dir = Path(args.output_dir)
    results_path, report_path = save_results(
        output_dir=output_dir,
        dataset=dataset,
        all_results=all_results,
        summary=summary,
        friedman=friedman,
        pairwise=pairwise,
        hypotheses=hypotheses,
        dataset_path=args.dataset,
    )

    # Summary
    logger.info("")
    logger.info("=" * 70)
    logger.info("REAL EVALUATION COMPLETE")
    logger.info("=" * 70)
    logger.info("Queries:   %d", len(queries))
    logger.info("Methods:   %d", summary.total_methods)
    ranked = sorted(summary.method_results, key=lambda x: x.avg_ndcg, reverse=True)
    logger.info("Best:      %s — NDCG@10=%.4f", ranked[0].method_id, ranked[0].avg_ndcg)
    logger.info("Report:    %s", report_path)
    logger.info("Raw data:  %s", results_path)

    return True


def main():
    parser = argparse.ArgumentParser(description="Real benchmark evaluation (no fake retrieval)")
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
        "--top-k",
        type=int,
        default=K,
        help=f"Number of results to retrieve per method (default: {K})",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=4,
        help="Max concurrent retrieval calls (default: 4)",
    )
    args = parser.parse_args()

    success = asyncio.run(async_main(args))
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
