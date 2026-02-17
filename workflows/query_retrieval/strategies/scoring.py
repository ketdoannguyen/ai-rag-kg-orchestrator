"""Scoring method enums and constants for retrieval strategies.

This module defines the available scoring methods for each strategy:
- E1 Top-Down: cosine, rrf_content_seed
- E2 Horizontal: cosine, rrf_cosine_seed, rrf_cosine_relation, rrf_cosine_seed_relation
- E3 Bottom-Up: parent_cosine, avg_log, rrf_parent_avglog

References:
    Cormack, G. V., Clarke, C. L., & Buettcher, S. (2009).
    "Reciprocal rank fusion outperforms condorcet and individual
    rank learning methods." SIGIR '09.
"""

from __future__ import annotations

from enum import Enum

# =============================================================================
# Global RRF Constant
# =============================================================================

RRF_K: int = 60  # Literature standard (Cormack et al. SIGIR 2009)


# =============================================================================
# E1 Top-Down Scoring Methods
# =============================================================================


class E1ScoringMethod(str, Enum):
    """Scoring methods for E1 Top-Down Retrieval.

    Attributes:
        COSINE: Pure cosine similarity between query and content node.
                Formula: cosine(query_embedding, content_embedding)

        RRF_CONTENT_SEED: RRF fusion of content cosine and seed (structure) cosine.
                          Formula: 1/(k+rank_content) + 1/(k+rank_seed)
    """

    COSINE = "cosine"
    RRF_CONTENT_SEED = "rrf_content_seed"


# =============================================================================
# E2 Horizontal Scoring Methods
# =============================================================================


class E2ScoringMethod(str, Enum):
    """Scoring methods for E2 Horizontal Retrieval.

    Attributes:
        COSINE: Pure cosine similarity (no RRF).
                Formula: cosine(query_embedding, content_embedding)

        RRF_COSINE_SEED: RRF of cosine and seed scores.
                         Formula: 1/(k+rank_cosine) + 1/(k+rank_seed)

        RRF_COSINE_RELATION: RRF of cosine and relation scores.
                             Formula: 1/(k+rank_cosine) + 1/(k+rank_relation)

        RRF_COSINE_SEED_RELATION: 3-way RRF fusion.
                                  Formula: 1/(k+rank_cosine) + 1/(k+rank_seed) + 1/(k+rank_relation)
    """

    COSINE = "cosine"
    RRF_COSINE_SEED = "rrf_cosine_seed"
    RRF_COSINE_RELATION = "rrf_cosine_relation"
    RRF_COSINE_SEED_RELATION = "rrf_cosine_seed_relation"


# =============================================================================
# E3 Bottom-Up Scoring Methods
# =============================================================================


class E3ScoringMethod(str, Enum):
    """Scoring methods for E3 Bottom-Up Retrieval.

    Attributes:
        PARENT_COSINE: Direct cosine similarity of parent with query.
                       Formula: cosine(query_embedding, parent_embedding)

        AVG_LOG: Quality x Coverage aggregation.
                 Formula: AVG(children_cosine) x ln(1 + children_count)

        RRF_PARENT_AVGLOG: RRF fusion of parent cosine and avg_log scores.
                           Formula: 1/(k+rank_parent_cosine) + 1/(k+rank_avg_log)
    """

    PARENT_COSINE = "parent_cosine"
    AVG_LOG = "avg_log"
    RRF_PARENT_AVGLOG = "rrf_parent_avglog"
