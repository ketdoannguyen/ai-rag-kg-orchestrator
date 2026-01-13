"""
AQL query templates for ArangoDB operations.

This module exports all AQL queries used by handlers.
All queries are centralized here - NO inline AQL in handlers.

Usage:
    >>> from arango_client.aql import GET_DIRECT_CHILDREN, VECTOR_SEARCH_CONTENT
    >>> cursor = db.aql.execute(GET_DIRECT_CHILDREN, bind_vars={...})
"""

from .knowledge_queries import (
    # CRUD Queries
    CRUD_GET_ALL,
    CRUD_GET_BY_FILTER,
    CRUD_COUNT_WITH_FILTER,
    # Graph Traversal
    GET_DIRECT_CHILDREN,
    GET_DESCENDANTS,
    GET_ANCESTORS,
    GET_PARENT,
    DELETE_SUBTREE,
    # Vector Search
    VECTOR_SEARCH_CONTENT,
    VECTOR_SEARCH_CONTENT_WITH_SOURCE,
    VECTOR_SEARCH_STRUCTURE,
    VECTOR_SEARCH_STRUCTURE_WITH_SOURCE,
    HYBRID_SEARCH,
    FULLTEXT_SEARCH,
    # Mining
    MINING_GET_BY_FREQUENCY,
    MINING_INCREMENT_FREQUENCY,
    # Context Expansion
    GET_PATH_TO_ROOT,
    FIND_RELATED_CONTEXT,
)

__all__ = [
    # CRUD Queries
    "CRUD_GET_ALL",
    "CRUD_GET_BY_FILTER",
    "CRUD_COUNT_WITH_FILTER",
    # Graph Traversal
    "GET_DIRECT_CHILDREN",
    "GET_DESCENDANTS",
    "GET_ANCESTORS",
    "GET_PARENT",
    "DELETE_SUBTREE",
    # Vector Search
    "VECTOR_SEARCH_CONTENT",
    "VECTOR_SEARCH_CONTENT_WITH_SOURCE",
    "VECTOR_SEARCH_STRUCTURE",
    "VECTOR_SEARCH_STRUCTURE_WITH_SOURCE",
    "HYBRID_SEARCH",
    "FULLTEXT_SEARCH",
    # Mining
    "MINING_GET_BY_FREQUENCY",
    "MINING_INCREMENT_FREQUENCY",
    # Context Expansion
    "GET_PATH_TO_ROOT",
    "FIND_RELATED_CONTEXT",
]
