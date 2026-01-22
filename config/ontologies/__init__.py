"""Ontology registry for domain-specific relationship schemas.

Provides:
- get_ontology(name): Get ontology by name
- get_current_ontology(): Get ontology from ONTOLOGY_NAME env var
- ONTOLOGY_REGISTRY: Dict of available ontologies

Pattern follows config/domains/__init__.py for consistency.
"""

from __future__ import annotations

import os
from collections.abc import Callable
from functools import lru_cache

# Re-export commonly used types (at top to avoid E402)
from .base import (
    BASE_ONTOLOGY,
    BASE_RELATIONS,
    TIER_1_STRUCTURE,
    TIER_2_CONTENT,
    TIER_3_ENTITY,
    BaseOntology,
    RelationConstraint,
    RelationType,
)
from .education import (
    EDUCATION_ONTOLOGY,
    EDUCATION_RELATION_CONSTRAINTS,
    EducationRelationType,
)
from .entity_tier import (
    ENTITY_ONTOLOGY,
    ENTITY_RELATION_CONSTRAINTS,
    EntityRelationType,
)
from .extension import (
    RELATION_PROPOSAL_PROMPT,
    RELATION_VALIDATION_PROMPT,
    ExtensionResult,
    OntologyExtender,
    ProposalStatus,
    ProposedRelation,
)


def _load_base() -> BaseOntology:
    """Lazy load base ontology."""
    return BASE_ONTOLOGY


def _load_education() -> BaseOntology:
    """Lazy load education ontology."""
    return EDUCATION_ONTOLOGY


def _load_entity() -> BaseOntology:
    """Lazy load entity tier ontology."""
    return ENTITY_ONTOLOGY


# Registry of available ontologies (lazy loading)
ONTOLOGY_REGISTRY: dict[str, Callable[[], BaseOntology]] = {
    "base": _load_base,
    "education": _load_education,
    "entity": _load_entity,
}


@lru_cache(maxsize=8)
def get_ontology(name: str) -> BaseOntology:
    """Get ontology by name.

    Args:
        name: Ontology identifier (e.g., "base", "education", "entity")

    Returns:
        BaseOntology instance

    Raises:
        ValueError: If ontology name is not found
    """
    if name not in ONTOLOGY_REGISTRY:
        available = ", ".join(ONTOLOGY_REGISTRY.keys())
        raise ValueError(f"Unknown ontology: '{name}'. Available: {available}")

    return ONTOLOGY_REGISTRY[name]()


def get_current_ontology() -> BaseOntology:
    """Get ontology from ONTOLOGY_NAME environment variable.

    Defaults to "education" if not set.

    Returns:
        BaseOntology instance
    """
    name = os.getenv("ONTOLOGY_NAME", "education")
    return get_ontology(name)


def list_ontologies() -> list[str]:
    """List available ontology names."""
    return list(ONTOLOGY_REGISTRY.keys())


__all__ = [
    # Functions
    "get_ontology",
    "get_current_ontology",
    "list_ontologies",
    # Registry
    "ONTOLOGY_REGISTRY",
    # Tier constants
    "TIER_1_STRUCTURE",
    "TIER_2_CONTENT",
    "TIER_3_ENTITY",
    # Base ontology
    "BaseOntology",
    "RelationType",
    "RelationConstraint",
    "BASE_ONTOLOGY",
    "BASE_RELATIONS",
    # Education ontology
    "EducationRelationType",
    "EDUCATION_ONTOLOGY",
    "EDUCATION_RELATION_CONSTRAINTS",
    # Entity tier ontology
    "EntityRelationType",
    "ENTITY_ONTOLOGY",
    "ENTITY_RELATION_CONSTRAINTS",
    # Extension system
    "OntologyExtender",
    "ProposedRelation",
    "ProposalStatus",
    "ExtensionResult",
    "RELATION_PROPOSAL_PROMPT",
    "RELATION_VALIDATION_PROMPT",
]
