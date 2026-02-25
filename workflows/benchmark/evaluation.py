"""Evaluation Metrics for Benchmark v4.0.

Implements grade-aware IR metrics: NDCG@k, Recall@k, Precision@k, MRR.
Relevance grades: 0 (irrelevant), 1 (partial), 2 (highly relevant).
"""

from __future__ import annotations

import math

from workflows.benchmark.models import BenchmarkQuery, EvaluationResult


def ndcg_at_k(
    retrieved_ids: list[str],
    relevance: dict[str, int],
    k: int = 10,
) -> float:
    """Compute Normalized Discounted Cumulative Gain at rank k.

    Uses graded relevance (0/1/2). DCG = sum(grade_i / log2(i+1)) for i=1..k.
    """
    top_k = retrieved_ids[:k]

    # DCG
    dcg = 0.0
    for rank, doc_id in enumerate(top_k, start=1):
        grade = relevance.get(doc_id, 0)
        dcg += grade / math.log2(rank + 1)

    # IDCG: ideal ranking (all relevant grades sorted desc)
    ideal_grades = sorted(relevance.values(), reverse=True)[:k]
    idcg = 0.0
    for rank, grade in enumerate(ideal_grades, start=1):
        idcg += grade / math.log2(rank + 1)

    if idcg == 0.0:
        return 0.0
    return dcg / idcg


def recall_at_k(
    retrieved_ids: list[str],
    relevance: dict[str, int],
    k: int = 10,
) -> float:
    """Compute Recall at rank k.

    Fraction of relevant nodes (grade > 0) found in top-k retrieved.
    """
    relevant_ids = {doc_id for doc_id, grade in relevance.items() if grade > 0}
    if not relevant_ids:
        return 0.0

    top_k = set(retrieved_ids[:k])
    found = relevant_ids & top_k
    return len(found) / len(relevant_ids)


def precision_at_k(
    retrieved_ids: list[str],
    relevance: dict[str, int],
    k: int = 10,
) -> float:
    """Compute Precision at rank k.

    Fraction of top-k retrieved that are relevant (grade > 0).
    """
    if k == 0:
        return 0.0

    top_k = retrieved_ids[:k]
    if not top_k:
        return 0.0

    relevant_count = sum(1 for doc_id in top_k if relevance.get(doc_id, 0) > 0)
    return relevant_count / len(top_k)


def mrr(
    retrieved_ids: list[str],
    relevance: dict[str, int],
) -> float:
    """Compute Mean Reciprocal Rank.

    Returns 1/rank of the first relevant result (grade > 0), or 0.0 if none.
    """
    for rank, doc_id in enumerate(retrieved_ids, start=1):
        if relevance.get(doc_id, 0) > 0:
            return 1.0 / rank
    return 0.0


def build_relevance_map(query: BenchmarkQuery) -> dict[str, int]:
    """Convert query fact_nodes into a {node_id: grade} mapping."""
    return {fn.node_id: fn.grade for fn in query.fact_nodes}


def evaluate_single(
    query: BenchmarkQuery,
    method_id: str,
    method_name: str,
    retrieved_ids: list[str],
    k: int = 10,
) -> EvaluationResult:
    """Evaluate a single query against one retrieval method's results."""
    rel = build_relevance_map(query)
    return EvaluationResult(
        query_id=query.query_id,
        method_id=method_id,
        method_name=method_name,
        retrieved_ids=retrieved_ids,
        ndcg_at_10=ndcg_at_k(retrieved_ids, rel, k),
        recall_at_10=recall_at_k(retrieved_ids, rel, k),
        precision_at_10=precision_at_k(retrieved_ids, rel, k),
        mrr=mrr(retrieved_ids, rel),
        result_count=len(retrieved_ids),
    )


def evaluate_batch(
    queries: list[BenchmarkQuery],
    method_id: str,
    method_name: str,
    results_map: dict[str, list[str]],
    k: int = 10,
) -> list[EvaluationResult]:
    """Evaluate a batch of queries against one method's results.

    Args:
        queries: List of benchmark queries.
        method_id: Short identifier (e.g. "B1", "E3a").
        method_name: Human-readable name.
        results_map: Mapping of query_id -> list of retrieved document IDs.
        k: Cutoff rank for metrics.

    Returns:
        List of EvaluationResult, one per query found in results_map.
    """
    results = []
    for query in queries:
        retrieved = results_map.get(query.query_id)
        if retrieved is None:
            continue
        results.append(evaluate_single(query, method_id, method_name, retrieved, k))
    return results
