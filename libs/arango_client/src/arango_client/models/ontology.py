"""Ontology relation model for storing relation types with embeddings.

This module provides the OntologyRelation model for storing and managing
relation types in the knowledge graph, including their embeddings for
efficient similarity-based relation filtering.
"""

from __future__ import annotations

from datetime import UTC, datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class RelationStatus(str, Enum):
    """Status of a relation type."""

    ACTIVE = "active"  # Currently in use
    PROPOSED = "proposed"  # LLM-proposed, pending validation
    DEPRECATED = "deprecated"  # No longer used
    REJECTED = "rejected"  # Proposed but rejected


class RelationTier(str, Enum):
    """Tier classification for relations.

    Used to filter relations based on the processing context:
    - TIER_1_STRUCTURE: Relations between structure nodes (HAS_SECTION, HAS_CONTENT, etc.)
    - TIER_2_CONTENT: Relations between content nodes (SIMILAR_TO, ELABORATES, etc.)
    - TIER_3_ENTITY: Relations between entity nodes (IS_A, DEPENDS_ON, etc.)
    """

    TIER_1_STRUCTURE = "tier_1_structure"
    TIER_2_CONTENT = "tier_2_content"
    TIER_3_ENTITY = "tier_3_entity"


class OntologyRelation(BaseModel):
    """Model for storing ontology relation types with embeddings.

    Attributes:
        id: Unique identifier (relation type name, e.g., "PREREQUISITE_OF")
        name: Human-readable name
        description: Detailed description in Vietnamese
        embedding: Vector embedding computed from name + description
        source_node_types: Valid source node types for this relation
        target_node_types: Valid target node types for this relation
        is_symmetric: If True, A->B implies B->A
        is_transitive: If True, A->B and B->C implies A->C
        is_builtin: True for ontology-defined relations, False for LLM-proposed
        status: Current status of the relation
        usage_count: Number of times this relation has been used
        tiers: List of processing tiers this relation applies to
        created_at: Creation timestamp
        updated_at: Last update timestamp
    """

    id: str = Field(..., description="Unique identifier (relation type name)")
    name: str = Field(..., description="Human-readable name")
    description: str = Field(..., description="Detailed description in Vietnamese")
    summary: str = Field(default="", description="Short description (~10-20 words)")
    embedding: list[float] | None = Field(
        default=None, description="Vector embedding (3072-dim for text-embedding-3-large)"
    )
    source_node_types: list[str] = Field(
        default_factory=list, description="Valid source node types"
    )
    target_node_types: list[str] = Field(
        default_factory=list, description="Valid target node types"
    )
    is_symmetric: bool = Field(default=False, description="A->B implies B->A")
    is_transitive: bool = Field(default=False, description="A->B, B->C implies A->C")
    is_builtin: bool = Field(
        default=True, description="True for ontology-defined, False for LLM-proposed"
    )
    status: RelationStatus = Field(default=RelationStatus.ACTIVE, description="Current status")
    usage_count: int = Field(default=0, description="Number of times used")
    tiers: list[RelationTier] = Field(
        default_factory=lambda: [RelationTier.TIER_2_CONTENT],
        description="List of processing tiers this relation applies to",
    )
    families: list[str] = Field(
        default_factory=list,
        description="List of family names this relation belongs to (e.g., ['causal', 'temporal'])",
    )
    keywords: list[str] = Field(
        default_factory=list, description="Keywords for query matching (Vietnamese)"
    )
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(UTC), description="Creation timestamp"
    )
    updated_at: datetime = Field(
        default_factory=lambda: datetime.now(UTC), description="Last update timestamp"
    )

    def applies_to_tier(self, tier: RelationTier | str) -> bool:
        """Check if relation applies to a specific tier.

        Args:
            tier: RelationTier enum or string value

        Returns:
            True if relation applies to the tier
        """
        if isinstance(tier, str):
            tier = RelationTier(tier)
        return tier in self.tiers

    def applies_to_any_tier(self, tiers: list[RelationTier | str]) -> bool:
        """Check if relation applies to any of the given tiers.

        Args:
            tiers: List of RelationTier enums or string values

        Returns:
            True if relation applies to at least one of the tiers
        """
        normalized_tiers = [RelationTier(t) if isinstance(t, str) else t for t in tiers]
        return any(t in self.tiers for t in normalized_tiers)

    def to_document(self) -> dict[str, Any]:
        """Convert to ArangoDB document format."""
        doc = self.model_dump(mode="json")
        doc["_key"] = doc.pop("id")
        return doc

    @classmethod
    def from_document(cls, doc: dict[str, Any]) -> OntologyRelation:
        """Create from ArangoDB document."""
        data = doc.copy()
        data["id"] = data.pop("_key", data.get("id"))
        # Remove ArangoDB system fields
        data.pop("_id", None)
        data.pop("_rev", None)
        return cls(**data)

    def get_embedding_text(self) -> str:
        """Get text used for computing embedding."""
        return f"{self.name}: {self.description}"
