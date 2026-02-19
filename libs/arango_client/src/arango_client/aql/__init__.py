"""
AQL query templates for ArangoDB operations.

This module exports all AQL queries used by handlers.
All queries are centralized here - NO inline AQL in handlers.

Usage:
    >>> from arango_client.aql import GET_DIRECT_CHILDREN, VECTOR_SEARCH_CONTENT
    >>> cursor = db.aql.execute(GET_DIRECT_CHILDREN, bind_vars={...})
"""

from .knowledge_queries import (
    # BM25 Search
    BM25_SEARCH_CONTENT,
    BM25_SEARCH_CONTENT_WITH_SOURCE,
    CRUD_COUNT_WITH_FILTER,
    # CRUD Queries
    CRUD_GET_ALL,
    CRUD_GET_BY_FILTER,
    DELETE_SUBTREE,
    FIND_RELATED_CONTEXT,
    FULLTEXT_SEARCH,
    FULLTEXT_SEARCH_BOOSTED,
    FULLTEXT_SEARCH_OPTIMIZED,
    GET_ANCESTORS,
    GET_DESCENDANTS,
    # Graph Traversal
    GET_DIRECT_CHILDREN,
    GET_NEXT_SIBLING,
    # Level-Based Queries
    GET_NODES_AT_LEVEL,
    GET_PARENT,
    # Context Expansion
    GET_PATH_TO_ROOT,
    GET_PREV_SIBLING,
    GET_SIBLING_CHAIN,
    # Sibling Navigation
    GET_SIBLINGS,
    GET_STRUCTURE_AT_LEVEL,
    HYBRID_SEARCH,
    # Mining
    MINING_GET_BY_FREQUENCY,
    MINING_INCREMENT_FREQUENCY,
    # Vector Search
    VECTOR_SEARCH_CONTENT,
    VECTOR_SEARCH_CONTENT_WITH_SOURCE,
    VECTOR_SEARCH_STRUCTURE,
    VECTOR_SEARCH_STRUCTURE_WITH_SOURCE,
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
    # Sibling Navigation
    "GET_SIBLINGS",
    "GET_NEXT_SIBLING",
    "GET_PREV_SIBLING",
    "GET_SIBLING_CHAIN",
    # Level-Based Queries
    "GET_NODES_AT_LEVEL",
    "GET_STRUCTURE_AT_LEVEL",
    # Vector Search
    "VECTOR_SEARCH_CONTENT",
    "VECTOR_SEARCH_CONTENT_WITH_SOURCE",
    "VECTOR_SEARCH_STRUCTURE",
    "VECTOR_SEARCH_STRUCTURE_WITH_SOURCE",
    "HYBRID_SEARCH",
    "FULLTEXT_SEARCH",
    "FULLTEXT_SEARCH_OPTIMIZED",
    "FULLTEXT_SEARCH_BOOSTED",
    # BM25 Search
    "BM25_SEARCH_CONTENT",
    "BM25_SEARCH_CONTENT_WITH_SOURCE",
    # Mining
    "MINING_GET_BY_FREQUENCY",
    "MINING_INCREMENT_FREQUENCY",
    # Context Expansion
    "GET_PATH_TO_ROOT",
    "FIND_RELATED_CONTEXT",
]
