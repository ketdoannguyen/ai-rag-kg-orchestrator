"""
Search handler for ArangoDB vector and hybrid search.

Provides vector similarity search and hybrid search combining
attribute filters with vector similarity.

All AQL queries are imported from the centralized aql/ module.
"""

from __future__ import annotations

import uuid
from typing import Any

from arango.database import StandardDatabase

from ..aql import (
    BM25_SEARCH_CONTENT,
    BM25_SEARCH_CONTENT_WITH_SOURCE,
    HYBRID_SEARCH,
    VECTOR_SEARCH_CONTENT,
    VECTOR_SEARCH_CONTENT_WITH_SOURCE,
    VECTOR_SEARCH_STRUCTURE,
    VECTOR_SEARCH_STRUCTURE_WITH_SOURCE,
)
from ..collections.schema import KNOWLEDGE_VECTOR_VIEW_NAME
from ..models.enums import NodeType
from ..models.knowledge import KnowledgeNode
from ..utils import doc_to_knowledge_node


class SearchResult:
    """Container for search results with score."""

    def __init__(self, node: KnowledgeNode, score: float):
        self.node = node
        self.score = score

    def __repr__(self) -> str:
        return f"SearchResult(node_id={self.node.id}, score={self.score:.4f})"


class SearchHandler:
    """
    Handler for vector similarity and hybrid search operations.

    This handler provides:
    - Content embedding vector search (for detail nodes)
    - Structure embedding vector search (for structure nodes)
    - Hybrid search (attribute filters + vector similarity)

    Example:
        >>> handler = SearchHandler(db)
        >>> results = handler.vector_search_content(query_vector, limit=10)
        >>> for result in results:
        ...     print(f"{result.node.id}: {result.score}")
    """

    def __init__(self, db: StandardDatabase):
        """Initialize SearchHandler."""
        self.db = db
        self.view_name = KNOWLEDGE_VECTOR_VIEW_NAME

    def vector_search_content(
        self,
        query_vector: list[float],
        limit: int = 10,
        min_score: float = 0.0,
        source_id: uuid.UUID | str | None = None,
        node_types: list[NodeType | str] | None = None,
    ) -> list[SearchResult]:
        """
        Search for similar content using content embeddings.

        Args:
            query_vector: Query embedding vector
            limit: Maximum number of results
            min_score: Minimum similarity score (0.0-1.0)
            source_id: Optional filter by data source
            node_types: Optional filter by node types

        Returns:
            List of SearchResult with nodes and scores
        """
        if node_types is None:
            node_types = [NodeType.PARAGRAPH, NodeType.IMAGE, NodeType.TABLE]

        type_values = [t.value if isinstance(t, NodeType) else t for t in node_types]

        bind_vars = {
            "@view": self.view_name,
            "query_vector": query_vector,
            "min_score": min_score,
            "limit": limit,
            "node_types": type_values,
        }

        # Use appropriate query based on source filter
        if source_id:
            query = VECTOR_SEARCH_CONTENT_WITH_SOURCE
            bind_vars["source_id"] = str(source_id)
        else:
            query = VECTOR_SEARCH_CONTENT

        cursor = self.db.aql.execute(query, bind_vars=bind_vars)

        results = []
        for item in cursor:
            node = doc_to_knowledge_node(item["doc"])
            results.append(SearchResult(node, item["score"]))

        return results

    def bm25_search_content(
        self,
        query_text: str,
        limit: int = 10,
        source_id: uuid.UUID | str | None = None,
        node_types: list[NodeType | str] | None = None,
    ) -> list[SearchResult]:
        """Search for content using BM25 full-text scoring.

        Uses TOKENS() ALL IN for token-level matching with BM25 scoring
        from ArangoSearch. Works well for multi-word queries.

        Args:
            query_text: Search query text
            limit: Maximum number of results
            source_id: Optional filter by data source
            node_types: Optional filter by node types

        Returns:
            List of SearchResult with nodes and BM25 scores
        """
        if node_types is None:
            node_types = [NodeType.PARAGRAPH, NodeType.IMAGE, NodeType.TABLE]

        type_values = [t.value if isinstance(t, NodeType) else t for t in node_types]

        bind_vars: dict[str, Any] = {
            "@view": self.view_name,
            "search_term": query_text,
            "limit": limit,
            "node_types": type_values,
        }

        if source_id:
            query = BM25_SEARCH_CONTENT_WITH_SOURCE
            bind_vars["source_id"] = str(source_id)
        else:
            query = BM25_SEARCH_CONTENT

        cursor = self.db.aql.execute(query, bind_vars=bind_vars)

        results = []
        for item in cursor:
            node = doc_to_knowledge_node(item["doc"])
            results.append(SearchResult(node, item["score"]))

        return results

    def vector_search_structure(
        self,
        query_vector: list[float],
        limit: int = 10,
        min_score: float = 0.0,
        source_id: uuid.UUID | str | None = None,
    ) -> list[SearchResult]:
        """
        Search for similar structure nodes using structure embeddings.

        Args:
            query_vector: Query embedding vector
            limit: Maximum number of results
            min_score: Minimum similarity score (0.0-1.0)
            source_id: Optional filter by data source

        Returns:
            List of SearchResult with nodes and scores
        """
        bind_vars = {
            "@view": self.view_name,
            "query_vector": query_vector,
            "min_score": min_score,
            "limit": limit,
        }

        # Use appropriate query based on source filter
        if source_id:
            query = VECTOR_SEARCH_STRUCTURE_WITH_SOURCE
            bind_vars["source_id"] = str(source_id)
        else:
            query = VECTOR_SEARCH_STRUCTURE

        cursor = self.db.aql.execute(query, bind_vars=bind_vars)

        results = []
        for item in cursor:
            node = doc_to_knowledge_node(item["doc"])
            results.append(SearchResult(node, item["score"]))

        return results

    def hybrid_search(
        self,
        query_vector: list[float],
        filters: dict[str, Any],
        limit: int = 10,
        min_score: float = 0.0,
    ) -> list[SearchResult]:
        """
        Hybrid search combining attribute filters with vector similarity.

        Args:
            query_vector: Query embedding vector
            filters: Attribute filters (e.g., {"node_type": "paragraph"})
            limit: Maximum number of results
            min_score: Minimum similarity score

        Returns:
            List of SearchResult with nodes and scores
        """
        # Build filter conditions
        filter_conditions = []
        bind_vars = {
            "@view": self.view_name,
            "query_vector": query_vector,
            "min_score": min_score,
            "limit": limit,
        }

        for i, (field, value) in enumerate(filters.items()):
            var_name = f"filter_{i}"
            if isinstance(value, list):
                filter_conditions.append(f"doc.{field} IN @{var_name}")
            else:
                filter_conditions.append(f"doc.{field} == @{var_name}")
            bind_vars[var_name] = value if not isinstance(value, uuid.UUID) else str(value)

        filter_clause = " AND ".join(filter_conditions) if filter_conditions else "true"

        # Use centralized AQL template with filter_clause injection
        query = HYBRID_SEARCH.format(filter_clause=filter_clause)

        cursor = self.db.aql.execute(query, bind_vars=bind_vars)

        results = []
        for item in cursor:
            node = doc_to_knowledge_node(item["doc"])
            results.append(SearchResult(node, item["score"]))

        return results

    def fulltext_search(
        self,
        search_term: str,
        limit: int = 10,
        source_id: uuid.UUID | str | None = None,
    ) -> list[KnowledgeNode]:
        """
        Full-text search on text and title attributes using ArangoSearch view.

        Args:
            search_term: Search term
            limit: Maximum number of results
            source_id: Optional filter by data source

        Returns:
            List of matching KnowledgeNode instances
        """
        source_filter = ""
        bind_vars: dict[str, Any] = {
            "@view": self.view_name,
            "search_term": f"%{search_term.lower()}%",
            "limit": limit,
        }

        if source_id:
            source_filter = "FILTER doc.source_id == @source_id"
            bind_vars["source_id"] = str(source_id)

        # Use view-based search across all collections
        query = f"""
        FOR doc IN @@view
          SEARCH ANALYZER(true, "identity")
          FILTER (
            LIKE(doc.attributes.text, @search_term, true) OR
            LIKE(doc.attributes.title, @search_term, true) OR
            LIKE(doc.text, @search_term, true) OR
            LIKE(doc.summary, @search_term, true)
          )
          {source_filter}
          LIMIT @limit
          RETURN doc
        """

        cursor = self.db.aql.execute(query, bind_vars=bind_vars)
        return [doc_to_knowledge_node(doc) for doc in cursor]
