"""Base utilities for retrieval strategies.

Shared code between E1 (TopDownRetrieval) and E2 (HorizontalRetrieval) strategies:
- Vectorized cosine similarity computation
- ContentWithContext conversion
- Async executor for database operations
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

import numpy as np

from workflows.query_retrieval.models import ContentWithContext

# Shared executor for async database operations
_EXECUTOR = ThreadPoolExecutor(max_workers=4)


def get_executor() -> ThreadPoolExecutor:
    """Get the shared ThreadPoolExecutor for async DB operations."""
    return _EXECUTOR


def cosine_similarity_batch(
    query_embedding: list[float],
    embeddings: list[list[float]],
) -> np.ndarray:
    """Compute cosine similarity between query and a batch of embeddings.

    Args:
        query_embedding: Query embedding vector
        embeddings: List of embedding vectors to compare against

    Returns:
        1-D numpy array of cosine scores clipped to [0, 1]
    """
    query_vec = np.array(query_embedding, dtype=np.float32)
    query_norm = np.linalg.norm(query_vec)

    if query_norm == 0 or not embeddings:
        return np.zeros(len(embeddings), dtype=np.float32)

    emb_matrix = np.array(embeddings, dtype=np.float32)
    row_norms = np.linalg.norm(emb_matrix, axis=1)

    scores = np.zeros(len(embeddings), dtype=np.float32)
    valid_mask = row_norms > 0

    if valid_mask.any():
        dot_products = emb_matrix @ query_vec
        scores[valid_mask] = dot_products[valid_mask] / (row_norms[valid_mask] * query_norm)
        scores = np.clip(scores, 0, 1)

    return scores


def to_content_with_context(
    node_data: dict,
    score: float,
    found_via_relation: str | None = None,
) -> ContentWithContext:
    """Convert a node dict to ContentWithContext.

    Args:
        node_data: Node document from ArangoDB
        score: Final relevance score
        found_via_relation: Relation type used to find this node (optional)

    Returns:
        ContentWithContext instance
    """
    return ContentWithContext(
        node_id=node_data.get("_key", node_data.get("_id", "")),
        text=node_data.get("text", ""),
        score=score,
        node_type=node_data.get("node_type", "paragraph"),
        found_via_relation=found_via_relation,
    )
