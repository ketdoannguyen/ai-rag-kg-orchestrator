"""Domain schemas registry.

This module provides:
- Domain schema definitions
- Registry for easy domain switching
- Helper to get current domain from environment

Usage:
    # Get domain from environment (DOMAIN_NAME)
    from config.domains import get_current_domain
    schema = get_current_domain()

    # Or get specific domain
    from config.domains import get_domain
    schema = get_domain("education")

    # Add new domain:
    # 1. Create config/domains/your_domain.py with YOUR_DOMAIN_SCHEMA
    # 2. Register in DOMAIN_REGISTRY below
"""

from __future__ import annotations

import os
from collections.abc import Callable

from .base import DomainSchema, MetadataField

# =============================================================================
# Domain Registry
# =============================================================================
# Lazy loading to avoid circular imports
# To add new domain: add entry here + create the schema file


def _load_education() -> DomainSchema:
    from .education import EDUCATION_SCHEMA

    return EDUCATION_SCHEMA


DOMAIN_REGISTRY: dict[str, Callable[[], DomainSchema]] = {
    "education": _load_education,
    # Add more domains here:
    # "ecommerce": _load_ecommerce,
    # "healthcare": _load_healthcare,
}


# =============================================================================
# Public API
# =============================================================================


def get_domain(name: str) -> DomainSchema:
    """Get domain schema by name.

    Args:
        name: Domain name (e.g., "education")

    Returns:
        DomainSchema for the requested domain

    Raises:
        KeyError: If domain not found
    """
    if name not in DOMAIN_REGISTRY:
        available = list(DOMAIN_REGISTRY.keys())
        raise KeyError(f"Domain '{name}' not found. Available: {available}")
    return DOMAIN_REGISTRY[name]()


def get_current_domain() -> DomainSchema:
    """Get current domain from environment.

    Uses DOMAIN_NAME environment variable, defaults to 'education'.

    Returns:
        DomainSchema for current domain
    """
    name = os.environ.get("DOMAIN_NAME", "education")
    return get_domain(name)


def list_available_domains() -> list[str]:
    """List all registered domain names."""
    return list(DOMAIN_REGISTRY.keys())


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    # Base classes
    "DomainSchema",
    "MetadataField",
    # Registry functions
    "get_domain",
    "get_current_domain",
    "list_available_domains",
    # Convenience - direct schema access
    "EDUCATION_SCHEMA",
]


# Lazy import for convenience (avoids loading all schemas upfront)
def __getattr__(name: str):
    if name == "EDUCATION_SCHEMA":
        from .education import EDUCATION_SCHEMA

        return EDUCATION_SCHEMA
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
