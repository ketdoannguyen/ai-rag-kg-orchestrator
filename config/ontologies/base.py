"""Base ontology classes with universal relationship types.

This module provides:
- RelationType: Enum of 16 domain-agnostic relationship types
- RelationConstraint: Schema for relationship constraints
- BaseOntology: Base class for domain-specific ontologies

The design follows the same pattern as config/domains for extensibility.
"""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field


class RelationType(str, Enum):
    """Base relationship types (domain-agnostic).

    16 universal relation types organized by category:
    - Structural (3): HAS_SECTION, HAS_CONTENT, NEXT_SIBLING
    - Semantic (1): SIMILAR_TO
    - Elaboration (3): ELABORATES, SUMMARIZES, EXEMPLIFIES
    - Logical (3): SUPPORTS, CONTRADICTS, CONTRASTS
    - Reference (2): REFERS_TO, DEFINES
    - Sequential (2): FOLLOWS, CONTINUES
    - Version (4): DUPLICATE_OF, VARIANT_OF, SUPERSEDES, DERIVED_FROM
    - Cross-modal (1): VISUALIZES
    """

    # Structural relationships (Tier 1)
    HAS_SECTION = "HAS_SECTION"  # Structure -> Structure
    HAS_CONTENT = "HAS_CONTENT"  # Structure -> Content
    NEXT_SIBLING = "NEXT_SIBLING"  # Node -> Node (reading order)

    # Semantic similarity (Tier 2)
    SIMILAR_TO = "SIMILAR_TO"  # Content -> Content

    # Elaboration relationships
    ELABORATES = "ELABORATES"  # Content -> Content (expands on)
    SUMMARIZES = "SUMMARIZES"  # Content -> Content (condenses)
    EXEMPLIFIES = "EXEMPLIFIES"  # Content -> Content (provides example)

    # Logical relationships
    SUPPORTS = "SUPPORTS"  # Content -> Content (evidence for)
    CONTRADICTS = "CONTRADICTS"  # Content -> Content (opposes)
    CONTRASTS = "CONTRASTS"  # Content -> Content (compares differences)

    # Reference relationships
    REFERS_TO = "REFERS_TO"  # Any -> Any (explicit reference)
    DEFINES = "DEFINES"  # Paragraph -> Content (definition)

    # Sequential relationships
    FOLLOWS = "FOLLOWS"  # Any -> Any (logical/temporal sequence)
    CONTINUES = "CONTINUES"  # Content -> Content (same content split)

    # Version control relationships
    DUPLICATE_OF = "DUPLICATE_OF"  # Content -> Content (exact/near copy)
    VARIANT_OF = "VARIANT_OF"  # Content -> Content (translation, format)
    SUPERSEDES = "SUPERSEDES"  # Content -> Content (replaces/updates)
    DERIVED_FROM = "DERIVED_FROM"  # Content -> Content (based on)

    # Cross-modal relationships
    VISUALIZES = "VISUALIZES"  # Image/Table -> Paragraph

    # Structure-only relationships (Tier 1)
    STRUCTURALLY_EQUIVALENT = "STRUCTURALLY_EQUIVALENT"  # Structure -> Structure


class RelationConstraint(BaseModel):
    """Constraint for a relationship type.

    Defines which node types can participate as source/target
    and the semantic properties of the relationship.
    """

    relation_type: RelationType | str = Field(..., description="The relationship type")
    source_node_types: list[str] = Field(
        ..., description="Valid source node types (NodeType values)"
    )
    target_node_types: list[str] = Field(
        ..., description="Valid target node types (NodeType values)"
    )
    is_symmetric: bool = Field(default=False, description="A->B implies B->A")
    is_transitive: bool = Field(default=False, description="A->B, B->C implies A->C")
    description: str = Field(default="", description="Human-readable description")
    tiers: list[str] = Field(
        default_factory=lambda: [TIER_2_CONTENT],
        description="List of processing tiers this relation applies to",
    )

    def applies_to_tier(self, tier: str) -> bool:
        """Check if constraint applies to a specific tier."""
        return tier in self.tiers

    def applies_to_any_tier(self, tiers: list[str]) -> bool:
        """Check if constraint applies to any of the given tiers."""
        return any(t in self.tiers for t in tiers)

    def is_valid_pair(self, source_type: str, target_type: str) -> bool:
        """Check if source/target pair is valid for this constraint."""
        return source_type in self.source_node_types and target_type in self.target_node_types


class BaseOntology(BaseModel):
    """Base ontology with common relationships.

    Provides the foundation for domain-specific ontologies.
    Domain ontologies inherit base relations and add domain-specific ones.
    """

    name: str = Field(default="base", description="Ontology identifier")
    display_name: str = Field(default="Base Ontology", description="Human-readable name")
    relations: list[RelationConstraint] = Field(
        default_factory=list, description="Relationship definitions"
    )

    def get_relation(self, relation_type: str) -> RelationConstraint | None:
        """Get a relation constraint by type."""
        for r in self.relations:
            if str(r.relation_type) == relation_type or (
                hasattr(r.relation_type, "value") and r.relation_type.value == relation_type
            ):
                return r
        return None

    def get_valid_relations(self, source_type: str, target_type: str) -> list[RelationType | str]:
        """Get valid relation types between two node types."""
        return [
            r.relation_type for r in self.relations if r.is_valid_pair(source_type, target_type)
        ]

    def get_relations_for_tier(self, tier: str) -> list[RelationConstraint]:
        """Get all relations that apply to a specific tier.

        Args:
            tier: Tier string (tier_1_structure, tier_2_content, tier_3_entity)

        Returns:
            List of RelationConstraint that apply to the tier
        """
        return [r for r in self.relations if r.applies_to_tier(tier)]

    def get_relations_for_tiers(self, tiers: list[str]) -> list[RelationConstraint]:
        """Get all relations that apply to any of the given tiers.

        Args:
            tiers: List of tier strings

        Returns:
            List of RelationConstraint that apply to at least one tier
        """
        return [r for r in self.relations if r.applies_to_any_tier(tiers)]

    def to_llm_prompt_section(self, tier: str | None = None) -> str:
        """Build prompt section for LLM relationship extraction.

        Args:
            tier: Optional tier to filter relations by. If None, includes all relations.

        Returns a formatted string describing available relationships
        for use in LLM prompts.
        """
        relations = self.get_relations_for_tier(tier) if tier else self.relations

        if not relations:
            return "(No relationships defined)"

        lines = ["## Available Relationship Types\n"]
        lines.append("Only use relationships from this list:\n")

        for r in relations:
            rel_type = (
                r.relation_type.value
                if hasattr(r.relation_type, "value")
                else str(r.relation_type)
            )
            line = f"- **{rel_type}**: {r.description}"
            if r.is_symmetric:
                line += " (symmetric)"
            if r.is_transitive:
                line += " (transitive)"
            lines.append(line)

            # Add source/target constraints
            sources = ", ".join(r.source_node_types)
            targets = ", ".join(r.target_node_types)
            lines.append(f"  - Source: [{sources}] -> Target: [{targets}]")

        return "\n".join(lines)

    def validate_relation(
        self,
        relation_type: str,
        source_type: str,
        target_type: str,
    ) -> tuple[bool, str]:
        """Validate if a relation is allowed by this ontology.

        Returns:
            (is_valid, error_message)
        """
        constraint = self.get_relation(relation_type)
        if constraint is None:
            return False, f"Unknown relation type: {relation_type}"

        if not constraint.is_valid_pair(source_type, target_type):
            return False, (
                f"Invalid node types for {relation_type}: {source_type} -> {target_type}"
            )

        return True, ""


# Node type constants (strings to avoid circular import)
# These match NodeType enum values in arango_client.models.enums
PARAGRAPH = "paragraph"
IMAGE = "image"
TABLE = "table"
STRUCTURE = "structure"

CONTENT_TYPES = [PARAGRAPH, IMAGE, TABLE]
STRUCTURE_TYPES = [STRUCTURE]
ALL_TYPES = CONTENT_TYPES + STRUCTURE_TYPES


# Tier constants for relation classification
TIER_1_STRUCTURE = "tier_1_structure"
TIER_2_CONTENT = "tier_2_content"
TIER_3_ENTITY = "tier_3_entity"


# Base relations - domain-agnostic, used by all ontologies
BASE_RELATIONS: list[RelationConstraint] = [
    # =========================================================================
    # Tier 1 ONLY - Structure exclusive
    # =========================================================================
    RelationConstraint(
        relation_type=RelationType.HAS_SECTION,
        source_node_types=STRUCTURE_TYPES,
        target_node_types=STRUCTURE_TYPES,
        description="Parent structure contains child structure",
        tiers=[TIER_1_STRUCTURE],
    ),
    RelationConstraint(
        relation_type=RelationType.HAS_CONTENT,
        source_node_types=STRUCTURE_TYPES,
        target_node_types=CONTENT_TYPES,
        description="Structure contains content",
        tiers=[TIER_1_STRUCTURE],
    ),
    RelationConstraint(
        relation_type=RelationType.NEXT_SIBLING,
        source_node_types=ALL_TYPES,
        target_node_types=ALL_TYPES,
        description="Next element in reading order",
        tiers=[TIER_1_STRUCTURE],
    ),
    RelationConstraint(
        relation_type=RelationType.STRUCTURALLY_EQUIVALENT,
        source_node_types=STRUCTURE_TYPES,
        target_node_types=STRUCTURE_TYPES,
        is_symmetric=True,
        description="Structurally equivalent nodes with high similarity (would merge if allowed)",
        tiers=[TIER_1_STRUCTURE],
    ),
    # =========================================================================
    # SHARED - Tier 1 + Tier 2
    # =========================================================================
    RelationConstraint(
        relation_type=RelationType.SIMILAR_TO,
        source_node_types=ALL_TYPES,
        target_node_types=ALL_TYPES,
        is_symmetric=True,
        description="Semantically similar content",
        tiers=[TIER_1_STRUCTURE, TIER_2_CONTENT],
    ),
    RelationConstraint(
        relation_type=RelationType.FOLLOWS,
        source_node_types=ALL_TYPES,
        target_node_types=ALL_TYPES,
        is_transitive=True,
        description="Follows in logical or temporal sequence",
        tiers=[TIER_1_STRUCTURE, TIER_2_CONTENT],
    ),
    RelationConstraint(
        relation_type=RelationType.REFERS_TO,
        source_node_types=ALL_TYPES,
        target_node_types=ALL_TYPES,
        description="Explicit reference or citation",
        tiers=[TIER_1_STRUCTURE, TIER_2_CONTENT],
    ),
    RelationConstraint(
        relation_type=RelationType.DUPLICATE_OF,
        source_node_types=ALL_TYPES,
        target_node_types=ALL_TYPES,
        is_symmetric=True,
        is_transitive=True,
        description="Exact or near-duplicate content",
        tiers=[TIER_1_STRUCTURE, TIER_2_CONTENT],
    ),
    RelationConstraint(
        relation_type=RelationType.VARIANT_OF,
        source_node_types=ALL_TYPES,
        target_node_types=ALL_TYPES,
        is_symmetric=True,
        description="Different version (translation, format change)",
        tiers=[TIER_1_STRUCTURE, TIER_2_CONTENT],
    ),
    # =========================================================================
    # Tier 2 ONLY - Content exclusive
    # =========================================================================
    RelationConstraint(
        relation_type=RelationType.ELABORATES,
        source_node_types=CONTENT_TYPES,
        target_node_types=CONTENT_TYPES,
        is_transitive=True,
        description="Provides additional detail or explanation",
        tiers=[TIER_2_CONTENT],
    ),
    RelationConstraint(
        relation_type=RelationType.SUMMARIZES,
        source_node_types=[PARAGRAPH],
        target_node_types=CONTENT_TYPES,
        is_transitive=True,
        description="Condenses or summarizes content",
        tiers=[TIER_2_CONTENT],
    ),
    RelationConstraint(
        relation_type=RelationType.EXEMPLIFIES,
        source_node_types=CONTENT_TYPES,
        target_node_types=[PARAGRAPH],
        description="Provides concrete example or illustration",
        tiers=[TIER_2_CONTENT],
    ),
    RelationConstraint(
        relation_type=RelationType.SUPPORTS,
        source_node_types=CONTENT_TYPES,
        target_node_types=CONTENT_TYPES,
        is_transitive=True,
        description="Provides evidence or reasoning in favor",
        tiers=[TIER_2_CONTENT],
    ),
    RelationConstraint(
        relation_type=RelationType.CONTRADICTS,
        source_node_types=CONTENT_TYPES,
        target_node_types=CONTENT_TYPES,
        is_symmetric=True,
        description="Directly opposes or refutes",
        tiers=[TIER_2_CONTENT],
    ),
    RelationConstraint(
        relation_type=RelationType.CONTRASTS,
        source_node_types=CONTENT_TYPES,
        target_node_types=CONTENT_TYPES,
        is_symmetric=True,
        description="Highlights differences in comparison",
        tiers=[TIER_2_CONTENT],
    ),
    RelationConstraint(
        relation_type=RelationType.DEFINES,
        source_node_types=[PARAGRAPH],
        target_node_types=CONTENT_TYPES,
        description="Provides definition for term or concept",
        tiers=[TIER_2_CONTENT],
    ),
    RelationConstraint(
        relation_type=RelationType.CONTINUES,
        source_node_types=CONTENT_TYPES,
        target_node_types=CONTENT_TYPES,
        is_transitive=True,
        description="Continues same content that was split",
        tiers=[TIER_2_CONTENT],
    ),
    RelationConstraint(
        relation_type=RelationType.SUPERSEDES,
        source_node_types=CONTENT_TYPES,
        target_node_types=CONTENT_TYPES,
        is_transitive=True,
        description="Replaces or updates older content",
        tiers=[TIER_2_CONTENT],
    ),
    RelationConstraint(
        relation_type=RelationType.DERIVED_FROM,
        source_node_types=CONTENT_TYPES,
        target_node_types=CONTENT_TYPES,
        is_transitive=True,
        description="Created based on other content",
        tiers=[TIER_2_CONTENT],
    ),
    RelationConstraint(
        relation_type=RelationType.VISUALIZES,
        source_node_types=[IMAGE, TABLE],
        target_node_types=[PARAGRAPH],
        description="Visual representation of textual content",
        tiers=[TIER_2_CONTENT],
    ),
]


# Default base ontology instance
BASE_ONTOLOGY = BaseOntology(
    name="base",
    display_name="Base Ontology",
    relations=BASE_RELATIONS,
)
