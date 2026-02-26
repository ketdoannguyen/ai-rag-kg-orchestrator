"""14-Method Evaluation Runner for Benchmark v4.0 (Task 25).

Runs all 14 retrieval method variants on the full query set,
collects results, and exports for statistical analysis.
Supports checkpointing per method-query batch.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from workflows.benchmark.evaluation import evaluate_batch
from workflows.benchmark.models import BenchmarkQuery, EvaluationResult, Subgraph

logger = logging.getLogger(__name__)

# Method configuration: each entry describes one retrieval variant
# strategy_class and scoring are resolved at runtime to avoid
# importing heavy DB-dependent modules at module level.

METHOD_CONFIGS: list[dict[str, Any]] = [
    {"id": "B1", "name": "Vector-Only", "family": "baseline", "strategy": "VECTOR_ONLY"},
    {"id": "B2", "name": "BM25", "family": "baseline", "strategy": "BM25"},
    {"id": "B3", "name": "Hybrid RRF(B1,B2)", "family": "baseline", "strategy": "HYBRID"},
    {"id": "E1a", "name": "TopDown COSINE", "family": "E1", "scoring": "COSINE"},
    {
        "id": "E1b",
        "name": "TopDown RRF_CONTENT_SEED",
        "family": "E1",
        "scoring": "RRF_CONTENT_SEED",
    },
    {"id": "E2a", "name": "Horizontal COSINE", "family": "E2", "scoring": "COSINE"},
    {
        "id": "E2b",
        "name": "Horizontal RRF_COSINE_SEED",
        "family": "E2",
        "scoring": "RRF_COSINE_SEED",
    },
    {
        "id": "E2c",
        "name": "Horizontal RRF_COSINE_RELATION",
        "family": "E2",
        "scoring": "RRF_COSINE_RELATION",
    },
    {
        "id": "E2d",
        "name": "Horizontal RRF_COSINE_SEED_RELATION",
        "family": "E2",
        "scoring": "RRF_COSINE_SEED_RELATION",
    },
    {"id": "E3a", "name": "BottomUp PARENT_COSINE", "family": "E3", "scoring": "PARENT_COSINE"},
    {"id": "E3b", "name": "BottomUp AVG_LOG", "family": "E3", "scoring": "AVG_LOG"},
    {
        "id": "E3c",
        "name": "BottomUp RRF_PARENT_AVGLOG",
        "family": "E3",
        "scoring": "RRF_PARENT_AVGLOG",
    },
    {
        "id": "E3d",
        "name": "BottomUp FILTERED_AVG_LOG",
        "family": "E3",
        "scoring": "FILTERED_AVG_LOG",
    },
    {
        "id": "E3e",
        "name": "BottomUp RRF_PARENT_FILTERED",
        "family": "E3",
        "scoring": "RRF_PARENT_FILTERED",
    },
]

# Subset: only cosine-scoring variants + baselines (for quick evaluation runs)
COSINE_ONLY_CONFIGS: list[dict[str, Any]] = [
    cfg for cfg in METHOD_CONFIGS if cfg["id"] in {"B1", "B2", "B3", "E1a", "E2a", "E3a"}
]


@dataclass
class MethodRunResult:
    """Result of running one method across all queries."""

    method_id: str
    method_name: str
    evaluations: list[EvaluationResult]
    avg_ndcg: float
    avg_recall: float
    avg_precision: float
    avg_mrr: float
    query_count: int


@dataclass
class EvaluationRunSummary:
    """Summary of the complete evaluation run."""

    total_methods: int
    total_queries: int
    total_evaluations: int
    method_results: list[MethodRunResult] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "total_methods": self.total_methods,
            "total_queries": self.total_queries,
            "total_evaluations": self.total_evaluations,
            "methods": [
                {
                    "method_id": mr.method_id,
                    "method_name": mr.method_name,
                    "query_count": mr.query_count,
                    "avg_ndcg_at_10": round(mr.avg_ndcg, 4),
                    "avg_recall_at_10": round(mr.avg_recall, 4),
                    "avg_precision_at_10": round(mr.avg_precision, 4),
                    "avg_mrr": round(mr.avg_mrr, 4),
                }
                for mr in self.method_results
            ],
        }


def aggregate_method_results(
    method_id: str,
    method_name: str,
    evaluations: list[EvaluationResult],
) -> MethodRunResult:
    """Compute average metrics for a single method across queries."""
    n = len(evaluations)
    if n == 0:
        return MethodRunResult(
            method_id=method_id,
            method_name=method_name,
            evaluations=[],
            avg_ndcg=0.0,
            avg_recall=0.0,
            avg_precision=0.0,
            avg_mrr=0.0,
            query_count=0,
        )

    return MethodRunResult(
        method_id=method_id,
        method_name=method_name,
        evaluations=evaluations,
        avg_ndcg=sum(e.ndcg_at_10 for e in evaluations) / n,
        avg_recall=sum(e.recall_at_10 for e in evaluations) / n,
        avg_precision=sum(e.precision_at_10 for e in evaluations) / n,
        avg_mrr=sum(e.mrr for e in evaluations) / n,
        query_count=n,
    )


def evaluate_method_offline(
    method_id: str,
    method_name: str,
    queries: list[BenchmarkQuery],
    results_map: dict[str, list[str]],
    k: int = 10,
    subgraphs_by_id: dict[str, Subgraph] | None = None,
) -> MethodRunResult:
    """Evaluate one method using pre-computed retrieval results.

    Args:
        method_id: Short ID (e.g. "B1").
        method_name: Human-readable name.
        queries: Full query set.
        results_map: {query_id: [retrieved_node_ids]}.
        k: Cutoff rank.
        subgraphs_by_id: Optional {subgraph_id: Subgraph} for E3 grading.

    Returns:
        MethodRunResult with per-query evaluations and averages.
    """
    evaluations = evaluate_batch(queries, method_id, method_name, results_map, k, subgraphs_by_id)

    return aggregate_method_results(method_id, method_name, evaluations)


def evaluate_all_methods_offline(
    queries: list[BenchmarkQuery],
    all_results: dict[str, dict[str, list[str]]],
    k: int = 10,
    method_configs: list[dict[str, Any]] | None = None,
    subgraphs_by_id: dict[str, Subgraph] | None = None,
) -> EvaluationRunSummary:
    """Evaluate all methods using pre-computed retrieval results.

    Args:
        queries: Full query set.
        all_results: {method_id: {query_id: [retrieved_ids]}}.
        k: Cutoff rank.
        method_configs: Optional subset of METHOD_CONFIGS (defaults to all 14).
        subgraphs_by_id: Optional {subgraph_id: Subgraph} for E3 grading.

    Returns:
        Complete evaluation summary.
    """
    configs = method_configs if method_configs is not None else METHOD_CONFIGS
    method_results = []
    total_evaluations = 0

    for config in configs:
        mid = config["id"]
        mname = config["name"]
        results_map = all_results.get(mid, {})

        mr = evaluate_method_offline(mid, mname, queries, results_map, k, subgraphs_by_id)
        method_results.append(mr)
        total_evaluations += mr.query_count

        logger.info(
            "Method %s: NDCG=%.4f, Recall=%.4f, Precision=%.4f, MRR=%.4f (%d queries)",
            mid,
            mr.avg_ndcg,
            mr.avg_recall,
            mr.avg_precision,
            mr.avg_mrr,
            mr.query_count,
        )

    return EvaluationRunSummary(
        total_methods=len(method_results),
        total_queries=len(queries),
        total_evaluations=total_evaluations,
        method_results=method_results,
    )
