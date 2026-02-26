"""Evaluation Metrics for Benchmark v4.0.

Implements grade-aware IR metrics: NDCG@k, Recall@k, Precision@k, MRR.
Relevance grades: 0 (irrelevant), 1 (partial), 2 (highly relevant).

E3 grading note:
    E3 returns structure_nodes (parent sections), not content_nodes.
    To grade E3 results, we build an extended relevance map that includes
    structure node IDs derived from the subgraph's parent_section_key fields.
    A structure node gets the maximum grade of its child fact_nodes.
"""

from __future__ import annotations

import math

from workflows.benchmark.models import BenchmarkQuery, EvaluationResult, Subgraph


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


def build_relevance_map_with_e3(
    query: BenchmarkQuery,
    subgraph: Subgraph | None,
) -> dict[str, int]:
    """Build relevance map including structure node IDs for E3 grading.

    E3 returns structure_nodes (parent sections). This function extends the
    standard content-node relevance map with entries for their parent structure
    nodes. A structure node gets the maximum grade of its child fact_nodes.

    Args:
        query: Benchmark query with fact_nodes (content node IDs + grades).
        subgraph: Subgraph for this query, used to resolve parent_section_keys.
                  If None, falls back to content-node-only map.

    Returns:
        {doc_id: grade} for content nodes AND structure nodes.
    """
    rel: dict[str, int] = {fn.node_id: fn.grade for fn in query.fact_nodes}

    if subgraph is None:
        return rel

    # Build content_key → parent_section_key mapping from subgraph nodes
    content_key_to_parent: dict[str, str] = {}
    for node in subgraph.nodes:
        if node.parent_section_key:
            # node_id is like "content_nodes/UUID"
            content_key_to_parent[node.node_id] = node.parent_section_key

    # For each fact_node, find its parent section key and assign max grade
    structure_grades: dict[str, int] = {}
    for fn in query.fact_nodes:
        parent_key = content_key_to_parent.get(fn.node_id)
        if parent_key:
            structure_id = f"structure_nodes/{parent_key}"
            existing = structure_grades.get(structure_id, 0)
            structure_grades[structure_id] = max(existing, fn.grade)

    rel.update(structure_grades)
    return rel


def evaluate_single(
    query: BenchmarkQuery,
    method_id: str,
    method_name: str,
    retrieved_ids: list[str],
    k: int = 10,
    subgraph: Subgraph | None = None,
) -> EvaluationResult:
    """Evaluate a single query against one retrieval method's results.

    Args:
        query: Benchmark query with fact_nodes.
        method_id: Short identifier (e.g. "B1", "E3a").
        method_name: Human-readable name.
        retrieved_ids: List of retrieved document IDs.
        k: Cutoff rank for metrics.
        subgraph: Optional subgraph for E3 structure-node grading.
    """
    rel = build_relevance_map_with_e3(query, subgraph)
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
    subgraphs_by_id: dict[str, Subgraph] | None = None,
) -> list[EvaluationResult]:
    """Evaluate a batch of queries against one method's results.

    Args:
        queries: List of benchmark queries.
        method_id: Short identifier (e.g. "B1", "E3a").
        method_name: Human-readable name.
        results_map: Mapping of query_id -> list of retrieved document IDs.
        k: Cutoff rank for metrics.
        subgraphs_by_id: Optional {subgraph_id: Subgraph} for E3 grading.

    Returns:
        List of EvaluationResult, one per query found in results_map.
    """
    results = []
    for query in queries:
        retrieved = results_map.get(query.query_id)
        if retrieved is None:
            continue
        subgraph = None
        if subgraphs_by_id is not None:
            subgraph = subgraphs_by_id.get(query.subgraph_id)
        results.append(evaluate_single(query, method_id, method_name, retrieved, k, subgraph))
    return results
