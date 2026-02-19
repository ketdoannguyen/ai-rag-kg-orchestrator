"""Handler for ontology relation operations.

Provides CRUD operations and specialized methods for managing
ontology relation types with embeddings.
"""

from __future__ import annotations

from datetime import UTC, datetime

import numpy as np
from arango.database import StandardDatabase

from ..models.ontology import OntologyRelation, RelationStatus, RelationTier
from ..utils import normalize_tier, normalize_vector
from .base_handler import BaseHandler

# Collection name
ONTOLOGY_RELATIONS_COLLECTION = "ontology_relations"


class OntologyHandler(BaseHandler[OntologyRelation]):
    """Handler for ontology relation operations.

    Provides:
    - CRUD operations for relation types
    - Tier-based filtering
    - Embedding-based relation filtering using Centroid method
    - Bulk operations for seeding ontology
    - Relation proposal and validation workflow
    """

    def __init__(self, db: StandardDatabase):
        """Initialize handler with database connection."""
        super().__init__(db, OntologyRelation, ONTOLOGY_RELATIONS_COLLECTION)

    def _ensure_collection(self) -> None:
        """Ensure collection exists."""
        if not self.db.has_collection(self.collection_name):
            self.db.create_collection(self.collection_name)

    def get_all_active(self) -> list[OntologyRelation]:
        """Get all active relations."""
        return self.get_by_filter({"status": RelationStatus.ACTIVE.value})

    def get_all_with_embeddings(self) -> list[OntologyRelation]:
        """Get all active relations that have embeddings computed."""
        query = """
        FOR doc IN @@collection
            FILTER doc.status == @status
            FILTER doc.embedding != null
            FILTER LENGTH(doc.embedding) > 0
            RETURN doc
        """
        cursor = self.db.aql.execute(
            query,
            bind_vars={
                "@collection": self.collection_name,
                "status": RelationStatus.ACTIVE.value,
            },
        )
        return [OntologyRelation.from_document(doc) for doc in cursor]

    def get_by_tier(self, tier: RelationTier | str) -> list[OntologyRelation]:
        """Get all active relations that apply to a specific tier.

        Args:
            tier: RelationTier enum or string value

        Returns:
            List of active relations that include this tier in their tiers list
        """
        tier_value = normalize_tier(tier)
        query = """
        FOR doc IN @@collection
            FILTER doc.status == @status
            FILTER @tier IN doc.tiers
            RETURN doc
        """
        cursor = self.db.aql.execute(
            query,
            bind_vars={
                "@collection": self.collection_name,
                "status": RelationStatus.ACTIVE.value,
                "tier": tier_value,
            },
        )
        return [OntologyRelation.from_document(doc) for doc in cursor]

    def get_by_tiers(self, tiers: list[RelationTier | str]) -> list[OntologyRelation]:
        """Get all active relations that apply to ANY of the given tiers.

        Args:
            tiers: List of RelationTier enums or string values

        Returns:
            List of active relations that include at least one of the tiers
        """
        tier_values = [normalize_tier(t) for t in tiers]
        query = """
        FOR doc IN @@collection
            FILTER doc.status == @status
            FILTER LENGTH(INTERSECTION(doc.tiers, @tiers)) > 0
            RETURN doc
        """
        cursor = self.db.aql.execute(
            query,
            bind_vars={
                "@collection": self.collection_name,
                "status": RelationStatus.ACTIVE.value,
                "tiers": tier_values,
            },
        )
        return [OntologyRelation.from_document(doc) for doc in cursor]

    def get_by_tier_with_embeddings(self, tier: RelationTier | str) -> list[OntologyRelation]:
        """Get all active relations with embeddings that apply to a specific tier.

        Args:
            tier: RelationTier enum or string value

        Returns:
            List of active relations with embeddings that include this tier
        """
        tier_value = normalize_tier(tier)
        query = """
        FOR doc IN @@collection
            FILTER doc.status == @status
            FILTER @tier IN doc.tiers
            FILTER doc.embedding != null
            FILTER LENGTH(doc.embedding) > 0
            RETURN doc
        """
        cursor = self.db.aql.execute(
            query,
            bind_vars={
                "@collection": self.collection_name,
                "status": RelationStatus.ACTIVE.value,
                "tier": tier_value,
            },
        )
        return [OntologyRelation.from_document(doc) for doc in cursor]

    def get_by_tiers_with_embeddings(
        self, tiers: list[RelationTier | str]
    ) -> list[OntologyRelation]:
        """Get all active relations with embeddings that apply to ANY of the given tiers.

        Args:
            tiers: List of RelationTier enums or string values

        Returns:
            List of active relations with embeddings that include at least one tier
        """
        tier_values = [normalize_tier(t) for t in tiers]
        query = """
        FOR doc IN @@collection
            FILTER doc.status == @status
            FILTER LENGTH(INTERSECTION(doc.tiers, @tiers)) > 0
            FILTER doc.embedding != null
            FILTER LENGTH(doc.embedding) > 0
            RETURN doc
        """
        cursor = self.db.aql.execute(
            query,
            bind_vars={
                "@collection": self.collection_name,
                "status": RelationStatus.ACTIVE.value,
                "tiers": tier_values,
            },
        )
        return [OntologyRelation.from_document(doc) for doc in cursor]

    def update_embedding(
        self, relation_id: str, embedding: list[float]
    ) -> OntologyRelation | None:
        """Update the embedding for a relation."""
        return self.update(
            relation_id,
            {
                "embedding": embedding,
                "updated_at": datetime.now(UTC).isoformat(),
            },
        )

    def increment_usage(self, relation_id: str) -> bool:
        """Atomically increment the usage count for a relation.

        Uses AQL update to ensure consistency under concurrent access.
        """
        query = """
        FOR doc IN @@collection
            FILTER doc._key == @relation_id
            UPDATE doc WITH {
                usage_count: (doc.usage_count || 0) + 1,
                updated_at: @now
            } IN @@collection
            RETURN NEW
        """
        cursor = self.db.aql.execute(
            query,
            bind_vars={
                "@collection": self.collection_name,
                "relation_id": relation_id,
                "now": datetime.now(UTC).isoformat(),
            },
        )
        return len(list(cursor)) > 0

    def add_tier_if_missing(
        self,
        relation_id: str,
        tier: RelationTier | str,
    ) -> bool:
        """Add a tier to relation's tiers list if not already present.

        Uses atomic AQL update to ensure consistency under concurrent access.
        This allows relations discovered in one tier to be extended when used
        in another tier context.

        Args:
            relation_id: Relation ID to update
            tier: Tier to add (RelationTier enum or string)

        Returns:
            True if tier was added, False if already present or relation not found
        """
        tier_value = normalize_tier(tier)

        # Atomic check-and-update: only updates if tier not in list
        # Uses (doc.tiers || []) to handle null/missing tiers field
        query = """
        FOR doc IN @@collection
            FILTER doc._key == @relation_id
            FILTER @tier NOT IN (doc.tiers || [])
            UPDATE doc WITH {
                tiers: APPEND(doc.tiers || [], @tier),
                updated_at: @now
            } IN @@collection
            RETURN NEW
        """
        cursor = self.db.aql.execute(
            query,
            bind_vars={
                "@collection": self.collection_name,
                "relation_id": relation_id,
                "tier": tier_value,
                "now": datetime.now(UTC).isoformat(),
            },
        )
        updated = list(cursor)
        return len(updated) > 0

    def propose_relation(
        self,
        relation_id: str,
        name: str,
        description: str,
        embedding: list[float] | None = None,
        source_node_types: list[str] | None = None,
        target_node_types: list[str] | None = None,
        tiers: list[RelationTier | str] | None = None,
    ) -> OntologyRelation:
        """Create a new relation from LLM extraction.

        This creates a relation in ACTIVE status because it has already been
        validated by LLM Strong Verify before reaching this point.

        Args:
            relation_id: Unique identifier for the relation
            name: Human-readable name
            description: Description in Vietnamese
            embedding: Optional pre-computed embedding
            source_node_types: Valid source node types
            target_node_types: Valid target node types
            tiers: List of tiers this relation applies to (defaults to [TIER_2_CONTENT])
        """
        # Handle tiers - convert strings to enums if needed
        if tiers is None:
            tier_values = [RelationTier.TIER_2_CONTENT]
        else:
            tier_values = [normalize_tier(t) for t in tiers]

        relation = OntologyRelation(
            id=relation_id,
            name=name,
            description=description,
            embedding=embedding,
            source_node_types=source_node_types or [],
            target_node_types=target_node_types or [],
            is_builtin=False,
            status=RelationStatus.ACTIVE,
            tiers=tier_values,
        )
        return self.create(relation)

    def approve_relation(self, relation_id: str) -> OntologyRelation | None:
        """Approve a proposed relation, making it active."""
        return self.update(
            relation_id,
            {
                "status": RelationStatus.ACTIVE.value,
                "updated_at": datetime.now(UTC).isoformat(),
            },
        )

    def reject_relation(self, relation_id: str) -> OntologyRelation | None:
        """Reject a proposed relation."""
        return self.update(
            relation_id,
            {
                "status": RelationStatus.REJECTED.value,
                "updated_at": datetime.now(UTC).isoformat(),
            },
        )

    def deprecate_relation(self, relation_id: str) -> OntologyRelation | None:
        """Deprecate an active relation."""
        return self.update(
            relation_id,
            {
                "status": RelationStatus.DEPRECATED.value,
                "updated_at": datetime.now(UTC).isoformat(),
            },
        )

    def filter_relations_by_centroid(
        self,
        source_embedding: list[float],
        target_embedding: list[float],
        top_k: int = 20,
        tier: RelationTier | str | None = None,
        exclude_relations: set[str] | None = None,
    ) -> list[tuple[str, float]]:
        """Filter relations using Centroid method.

        Uses the average of source and target embeddings to find
        the most relevant relations.

        Args:
            source_embedding: Embedding of source paragraph
            target_embedding: Embedding of target paragraph
            top_k: Number of top relations to return
            tier: Optional tier filter (tier_1_structure, tier_2_content, tier_3_entity)
            exclude_relations: Set of relation IDs to exclude from results
                (e.g., {"SIMILAR_TO"} to force LLM to think deeper)

        Returns:
            List of (relation_id, score) tuples sorted by score descending
        """
        exclude_relations = exclude_relations or set()

        # Compute centroid
        source_arr = np.array(source_embedding)
        target_arr = np.array(target_embedding)
        centroid = (source_arr + target_arr) / 2

        # Normalize centroid
        centroid_norm = normalize_vector(centroid)

        # Get relations with embeddings (optionally filtered by tier)
        if tier:
            relations = self.get_by_tier_with_embeddings(tier)
        else:
            relations = self.get_all_with_embeddings()

        # Compute cosine similarity with each relation (excluding specified relations)
        scores = []
        for rel in relations:
            # Skip excluded relations
            if rel.id in exclude_relations:
                continue
            if rel.embedding:
                rel_norm = normalize_vector(rel.embedding)
                score = float(np.dot(centroid_norm, rel_norm))
                scores.append((rel.id, score))

        # Sort by score descending and return top_k
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_k]

    def filter_relations_by_max_pooling(
        self,
        pair_embeddings: list[tuple[list[float], list[float]]],
        top_k: int = 20,
        tier: RelationTier | str | None = None,
        exclude_relations: set[str] | None = None,
    ) -> list[tuple[str, float]]:
        """Filter relations using Max-Pooling across multiple pairs.

        For each relation, computes cosine similarity with every pair's centroid
        and takes the maximum score. This ensures no pair's important relation
        is "drowned out" by averaging.

        Args:
            pair_embeddings: List of (source_embedding, target_embedding) tuples
            top_k: Number of top relations to return
            tier: Optional tier filter
            exclude_relations: Set of relation IDs to exclude

        Returns:
            List of (relation_id, max_score) tuples sorted by score descending
        """
        if not pair_embeddings:
            return []

        exclude_relations = exclude_relations or set()

        # Compute and normalize centroids for all pairs
        centroids = []
        for source_emb, target_emb in pair_embeddings:
            source_arr = np.array(source_emb)
            target_arr = np.array(target_emb)
            centroid = (source_arr + target_arr) / 2
            centroids.append(normalize_vector(centroid))

        # Get relations with embeddings (optionally filtered by tier)
        if tier:
            relations = self.get_by_tier_with_embeddings(tier)
        else:
            relations = self.get_all_with_embeddings()

        # For each relation: max score across all pair centroids
        scores: list[tuple[str, float]] = []
        for rel in relations:
            if rel.id in exclude_relations:
                continue
            if not rel.embedding:
                continue

            rel_norm = normalize_vector(rel.embedding)
            max_score = max(float(np.dot(c, rel_norm)) for c in centroids)
            scores.append((rel.id, max_score))

        # Sort by score descending and return top_k
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_k]

    def get_relation_ids_by_centroid(
        self,
        source_embedding: list[float],
        target_embedding: list[float],
        top_k: int = 20,
        tier: RelationTier | str | None = None,
    ) -> list[str]:
        """Get top-k relation IDs using Centroid method.

        Convenience method that returns only the relation IDs.
        """
        scored = self.filter_relations_by_centroid(source_embedding, target_embedding, top_k, tier)
        return [rel_id for rel_id, _ in scored]

    def seed_from_ontology(
        self,
        relations: list[dict],
        overwrite: bool = False,
    ) -> int:
        """Seed relations from ontology definitions.

        Args:
            relations: List of relation dicts with keys:
                - id: Relation type name
                - name: Human-readable name
                - summary: Short description (10-20 words)
                - description: Vietnamese description (full, for embedding)
                - source_node_types: List of valid source types
                - target_node_types: List of valid target types
                - is_symmetric: Optional, default False
                - is_transitive: Optional, default False
                - embedding: Optional, pre-computed embedding
                - tiers: Optional, list of tier classifications (defaults to [tier_2_content])
                - families: Optional, list of family names (causal, temporal, etc.)
                - keywords: Optional, list of Vietnamese keywords for query matching
            overwrite: If True, overwrite existing relations

        Returns:
            Number of relations created/updated
        """
        self._ensure_collection()
        count = 0

        for rel_data in relations:
            relation_id = rel_data["id"]

            # Check if exists
            existing = self.get_by_id(relation_id)
            if existing and not overwrite:
                continue

            # Handle tiers - convert strings to enums if needed
            tiers_raw = rel_data.get("tiers", [RelationTier.TIER_2_CONTENT])
            tier_values = [normalize_tier(t) for t in tiers_raw]

            relation = OntologyRelation(
                id=relation_id,
                name=rel_data.get("name", relation_id),
                summary=rel_data.get("summary", ""),
                description=rel_data.get("description", ""),
                embedding=rel_data.get("embedding"),
                source_node_types=rel_data.get("source_node_types", []),
                target_node_types=rel_data.get("target_node_types", []),
                is_symmetric=rel_data.get("is_symmetric", False),
                is_transitive=rel_data.get("is_transitive", False),
                is_builtin=True,
                status=RelationStatus.ACTIVE,
                tiers=tier_values,
                families=rel_data.get("families", []),
                keywords=rel_data.get("keywords", []),
            )

            if existing:
                # Update existing
                self.update_by_model(relation)
            else:
                # Create new
                self.create(relation)

            count += 1

        return count

    def get_relation_embedding_map(self) -> dict[str, np.ndarray]:
        """Get a map of relation ID to embedding array.

        Useful for batch processing and caching.
        """
        relations = self.get_all_with_embeddings()
        return {rel.id: np.array(rel.embedding) for rel in relations if rel.embedding}

    def find_similar_excluding(
        self,
        embedding: list[float],
        exclude_ids: list[str],
        threshold: float = 0.8,
    ) -> tuple[str, float] | None:
        """Find most similar relation excluding certain IDs.

        Used for validating new relations - checks if a similar relation
        already exists in the database.

        Args:
            embedding: Embedding of new relation to check
            exclude_ids: Relation IDs to exclude (e.g., top-k already considered)
            threshold: Minimum similarity threshold

        Returns:
            (relation_id, score) if found above threshold, None otherwise
        """
        relations = self.get_all_with_embeddings()

        emb_norm = normalize_vector(embedding)

        best_match = None
        best_score = 0.0

        for rel in relations:
            if rel.id in exclude_ids:
                continue
            if not rel.embedding:
                continue

            rel_norm = normalize_vector(rel.embedding)
            score = float(np.dot(emb_norm, rel_norm))

            if score >= threshold and score > best_score:
                best_match = rel.id
                best_score = score

        if best_match:
            return (best_match, best_score)
        return None

    def get_all_excluding(self, exclude_ids: list[str]) -> list[OntologyRelation]:
        """Get all active relations excluding certain IDs.

        Args:
            exclude_ids: Relation IDs to exclude

        Returns:
            List of active relations not in exclude_ids
        """
        if not exclude_ids:
            return self.get_all_active()

        query = """
        FOR doc IN @@collection
            FILTER doc.status == @status
            FILTER doc._key NOT IN @exclude_ids
            RETURN doc
        """
        cursor = self.db.aql.execute(
            query,
            bind_vars={
                "@collection": self.collection_name,
                "status": RelationStatus.ACTIVE.value,
                "exclude_ids": exclude_ids,
            },
        )
        return [OntologyRelation.from_document(doc) for doc in cursor]
