"""Retrieval strategy implementations.

This package contains pluggable retrieval strategies for query_retrieval workflow.
Each strategy implements a different approach to finding relevant content nodes.
"""

from __future__ import annotations

from workflows.query_retrieval.strategies.base import (
    compute_ranks_descending,
    compute_ranks_with_ties,
    cosine_similarity_batch,
    get_executor,
    rrf_fusion_2way,
    rrf_fusion_3way,
    to_content_with_context,
)
from workflows.query_retrieval.strategies.baselines import (
    BaselineMetrics,
    BM25Retrieval,
    HybridRetrieval,
)
from workflows.query_retrieval.strategies.bottom_up import BottomUpRetrieval
from workflows.query_retrieval.strategies.horizontal import HorizontalRetrieval
from workflows.query_retrieval.strategies.scoring import (
    RRF_K,
    E1ScoringMethod,
    E2ScoringMethod,
    E3ScoringMethod,
)
from workflows.query_retrieval.strategies.top_down import TopDownRetrieval

__all__ = [
    # Strategy classes
    "TopDownRetrieval",
    "HorizontalRetrieval",
    "BottomUpRetrieval",
    "BM25Retrieval",
    "HybridRetrieval",
    "BaselineMetrics",
    # Scoring enums
    "E1ScoringMethod",
    "E2ScoringMethod",
    "E3ScoringMethod",
    "RRF_K",
    # Utility functions
    "cosine_similarity_batch",
    "to_content_with_context",
    "get_executor",
    "compute_ranks_descending",
    "compute_ranks_with_ties",
    "rrf_fusion_2way",
    "rrf_fusion_3way",
]
