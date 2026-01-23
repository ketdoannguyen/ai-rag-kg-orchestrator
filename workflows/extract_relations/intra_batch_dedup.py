"""Intra-batch deduplication using Union-Find algorithm.

This module handles Phase 1 of the Two-Phase extraction:
1. Compute pairwise similarities within batch (using numpy for performance)
2. Build connected components using Union-Find (O(alpha(n)) ~ O(1) per operation)
3. Identify merge operations and relation candidates

Key benefits:
- Catches intra-batch duplicates that would be missed with reviewed-only comparison
- No race conditions (all computation done before any writes)
- Deterministic merge direction (older node or lower ID becomes target)
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np
from loguru import logger

if TYPE_CHECKING:
    from arango_client.models import ContentNode, StructureNode


@dataclass
class MergeOperation:
    """A merge operation to be executed.

    Attributes:
        source_id: ID of node to delete (merge source)
        target_id: ID of node to keep (merge target)
        similarity: Cosine similarity score between nodes
    """

    source_id: str
    target_id: str
    similarity: float


@dataclass
class RelationCandidate:
    """A pair of nodes that should be connected with a relation.

    Attributes:
        node_a_idx: Index of first node in batch
        node_b_idx: Index of second node in batch
        similarity: Cosine similarity score
    """

    node_a_idx: int
    node_b_idx: int
    similarity: float


@dataclass
class IntraBatchResult:
    """Result of intra-batch deduplication.

    Attributes:
        leaders: Nodes that survived deduplication (targets of merges + standalone)
        merge_operations: List of merge operations to execute
        relation_candidates: Pairs for relation extraction (0.40-0.72 similarity)
        high_similarity_candidates: Pairs with similarity >= merge_threshold when allow_merge=False
        similarity_matrix: Full pairwise similarity matrix (for debugging)
        leader_indices: Indices of leader nodes in original batch
    """

    leaders: list[ContentNode | StructureNode] = field(default_factory=list)
    merge_operations: list[MergeOperation] = field(default_factory=list)
    relation_candidates: list[RelationCandidate] = field(default_factory=list)
    high_similarity_candidates: list[RelationCandidate] = field(default_factory=list)
    similarity_matrix: np.ndarray | None = None
    leader_indices: list[int] = field(default_factory=list)


class UnionFind:
    """Union-Find (Disjoint Set Union) with path compression and union by rank.

    Time Complexity:
    - find(): O(alpha(n)) ~ O(1) amortized
    - union(): O(alpha(n)) ~ O(1) amortized
    - get_components(): O(n)

    Where alpha(n) is the inverse Ackermann function, effectively constant for all practical n.
    """

    def __init__(self, n: int) -> None:
        """Initialize n disjoint sets, each element is its own parent.

        Args:
            n: Number of elements
        """
        self.parent = list(range(n))
        self.rank = [0] * n

    def find(self, x: int) -> int:
        """Find the root/leader of the set containing x.

        Uses path compression: all nodes on path point directly to root.

        Args:
            x: Element to find root for

        Returns:
            Root element of the set containing x
        """
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])  # Path compression
        return self.parent[x]

    def union(self, x: int, y: int) -> bool:
        """Merge the sets containing x and y.

        Uses union by rank: smaller tree is attached to root of larger tree.

        Args:
            x: First element
            y: Second element

        Returns:
            True if sets were merged (different sets), False if already same set
        """
        px, py = self.find(x), self.find(y)

        if px == py:
            return False  # Already in same set

        # Union by rank: attach smaller tree to larger tree
        if self.rank[px] < self.rank[py]:
            px, py = py, px

        self.parent[py] = px

        if self.rank[px] == self.rank[py]:
            self.rank[px] += 1

        return True

    def get_components(self) -> dict[int, list[int]]:
        """Get all connected components.

        Returns:
            Dict mapping root/leader index to list of member indices
        """
        components: dict[int, list[int]] = defaultdict(list)
        for i in range(len(self.parent)):
            root = self.find(i)
            components[root].append(i)
        return dict(components)


def compute_similarity_matrix(embeddings: list[list[float]]) -> np.ndarray:
    """Compute cosine similarity matrix using numpy vectorized operations.

    Args:
        embeddings: List of embedding vectors

    Returns:
        n x n similarity matrix where similarity[i,j] is cosine similarity
        between embeddings[i] and embeddings[j]
    """
    if not embeddings:
        return np.array([])

    arr = np.array(embeddings, dtype=np.float32)

    # Compute L2 norms
    norms = np.linalg.norm(arr, axis=1, keepdims=True)

    # Handle zero vectors (avoid division by zero)
    norms[norms == 0] = 1.0

    # Normalize embeddings
    normalized = arr / norms

    # Compute cosine similarity matrix: (n x n) = (n x d) @ (d x n)
    similarity = np.dot(normalized, normalized.T)

    return similarity


def find_intra_batch_components(
    nodes: list[ContentNode | StructureNode],
    merge_threshold: float = 0.72,
    relation_threshold: float = 0.40,
    allow_merge: bool = True,
) -> IntraBatchResult:
    """Find connected components for merging within a batch.

    This is Phase 1 of Two-Phase extraction:
    1. Compute pairwise similarity matrix (O(n^2 * d), vectorized)
    2. Use Union-Find to find merge components (>= merge_threshold) if allow_merge=True
    3. Identify relation candidates (relation_threshold <= sim < merge_threshold)
    4. If allow_merge=False, treat high-similarity pairs as high_similarity_candidates

    Args:
        nodes: List of nodes in the batch (all unreviewed)
        merge_threshold: Similarity threshold for merging (default 0.72)
        relation_threshold: Minimum similarity for relations (default 0.40)
        allow_merge: If True, merge high-similarity pairs. If False, add to
                     high_similarity_candidates instead (for Structure tier).

    Returns:
        IntraBatchResult with leaders, merge operations, and relation candidates
    """
    log = logger.bind(module="intra_batch_dedup")
    n = len(nodes)

    # Handle edge cases
    if n == 0:
        return IntraBatchResult()

    if n == 1:
        return IntraBatchResult(
            leaders=[nodes[0]],
            leader_indices=[0],
            similarity_matrix=np.array([[1.0]]),
        )

    # Step 1: Extract embeddings and track valid indices
    embeddings: list[list[float]] = []
    valid_indices: list[int] = []

    for i, node in enumerate(nodes):
        embedding = getattr(node, "embedding", None)
        if embedding and len(embedding) > 0:
            embeddings.append(embedding)
            valid_indices.append(i)

    if len(embeddings) < 2:
        # Not enough embeddings for comparison
        log.debug(
            "Not enough nodes with embeddings for dedup", nodes_with_embeddings=len(embeddings)
        )
        return IntraBatchResult(
            leaders=nodes,
            leader_indices=list(range(n)),
        )

    # Step 2: Compute similarity matrix
    similarity_matrix = compute_similarity_matrix(embeddings)

    log.debug(
        "Computed similarity matrix",
        size=f"{len(embeddings)}x{len(embeddings)}",
        nodes_total=n,
    )

    # Step 3: Build Union-Find for high-similarity pairs (only if allow_merge=True)
    uf = UnionFind(len(embeddings))
    relation_candidates: list[RelationCandidate] = []
    high_similarity_candidates: list[RelationCandidate] = []

    for i in range(len(embeddings)):
        for j in range(i + 1, len(embeddings)):
            sim = float(similarity_matrix[i, j])

            if sim >= merge_threshold:
                if allow_merge:
                    # High similarity: merge these nodes
                    uf.union(i, j)
                else:
                    # No merge allowed: add to high_similarity_candidates
                    high_similarity_candidates.append(
                        RelationCandidate(
                            node_a_idx=valid_indices[i],
                            node_b_idx=valid_indices[j],
                            similarity=sim,
                        )
                    )
            elif sim >= relation_threshold:
                # Medium similarity: relation candidate
                relation_candidates.append(
                    RelationCandidate(
                        node_a_idx=valid_indices[i],
                        node_b_idx=valid_indices[j],
                        similarity=sim,
                    )
                )

    # Step 4: Get components and determine leaders
    components = uf.get_components()

    # Create reverse lookup for O(1) access: valid_index -> embedding_index
    idx_to_emb_idx = {node_idx: emb_idx for emb_idx, node_idx in enumerate(valid_indices)}

    leaders: list[ContentNode | StructureNode] = []
    leader_indices: list[int] = []
    merge_operations: list[MergeOperation] = []

    for _, member_emb_indices in components.items():
        # Map embedding indices back to original node indices
        member_node_indices = [valid_indices[i] for i in member_emb_indices]
        member_data = [(idx, nodes[idx]) for idx in member_node_indices]

        # Determine leader: prefer node with earlier created_at, or lower ID
        member_data.sort(
            key=lambda x: (
                # First by created_at if available
                getattr(x[1], "created_at", None) or "",
                # Then by ID string
                str(getattr(x[1], "id", "") or ""),
            )
        )

        leader_idx, leader_node = member_data[0]
        leaders.append(leader_node)
        leader_indices.append(leader_idx)

        # All other nodes merge into leader
        emb_leader = idx_to_emb_idx[leader_idx]
        for idx, node in member_data[1:]:
            emb_i = idx_to_emb_idx[idx]
            sim = float(similarity_matrix[emb_i, emb_leader])

            merge_operations.append(
                MergeOperation(
                    source_id=str(getattr(node, "id", "")),
                    target_id=str(getattr(leader_node, "id", "")),
                    similarity=sim,
                )
            )

    # Step 5: Add nodes without embeddings as standalone leaders
    nodes_with_embeddings = set(valid_indices)
    for i, node in enumerate(nodes):
        if i not in nodes_with_embeddings:
            leaders.append(node)
            leader_indices.append(i)

    log.info(
        "Intra-batch dedup complete",
        total_nodes=n,
        nodes_with_embeddings=len(embeddings),
        components=len(components),
        leaders=len(leaders),
        merges=len(merge_operations),
        relation_candidates=len(relation_candidates),
        high_similarity_candidates=len(high_similarity_candidates),
        allow_merge=allow_merge,
    )

    return IntraBatchResult(
        leaders=leaders,
        merge_operations=merge_operations,
        relation_candidates=relation_candidates,
        high_similarity_candidates=high_similarity_candidates,
        similarity_matrix=similarity_matrix,
        leader_indices=leader_indices,
    )
