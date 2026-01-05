"""
Centralized AQL query templates for ArangoDB operations.

This module contains ALL AQL queries used across handlers.
No inline AQL should exist in handler files - all queries must be defined here.

Query Naming Convention:
- Use UPPERCASE_SNAKE_CASE for query names
- Prefix with category: CRUD_, GRAPH_, VECTOR_, MINING_

Query Parameters:
- Use @param for simple bind variables
- Use @@collection for collection name bind variables
- Dynamic parts (sort, filter) are injected by handlers using f-strings
"""

from __future__ import annotations


# =============================================================================
# CRUD Queries (Used by BaseHandler and subclasses)
# =============================================================================

# Get all documents with pagination and sorting
# Note: collection_name, sort_field, sort_dir are injected via f-string
CRUD_GET_ALL = """
FOR doc IN {collection}
  SORT doc.{sort_field} {sort_dir}
  LIMIT @offset, @limit
  RETURN doc
"""

# Get documents by filter with pagination and sorting
# Note: collection_name, filter_clause, sort_field, sort_dir are injected via f-string
CRUD_GET_BY_FILTER = """
FOR doc IN {collection}
  FILTER {filter_clause}
  SORT doc.{sort_field} {sort_dir}
  LIMIT @offset, @limit
  RETURN doc
"""

# Count documents with filter
# Note: collection_name, filter_clause are injected via f-string
CRUD_COUNT_WITH_FILTER = """
RETURN LENGTH(
  FOR doc IN {collection}
    FILTER {filter_clause}
    RETURN 1
)
"""


# =============================================================================
# Graph Traversal Queries (Used by KnowledgeHandler)
# =============================================================================

# Get direct children of a node via graph traversal
GET_DIRECT_CHILDREN = """
FOR v, e IN 1..1 OUTBOUND @start_vertex GRAPH @graph_name
  FILTER e.type IN @edge_types
  RETURN v
"""

# Get all descendants of a node (deep traversal)
GET_DESCENDANTS = """
FOR v, e IN 1..@max_depth OUTBOUND @start_vertex GRAPH @graph_name
  FILTER e.type IN @edge_types
  RETURN DISTINCT v
"""

# Get ancestors of a node (upward traversal)
GET_ANCESTORS = """
FOR v, e IN 1..@max_depth INBOUND @start_vertex GRAPH @graph_name
  FILTER e.type IN @edge_types
  RETURN DISTINCT v
"""

# Get parent of a node (single hop up)
GET_PARENT = """
FOR v, e IN 1..1 INBOUND @start_vertex GRAPH @graph_name
  FILTER e.type IN ["HAS_SECTION", "HAS_CONTENT"]
  LIMIT 1
  RETURN v
"""

# Delete subtree (node and all descendants with edges)
DELETE_SUBTREE = """
LET descendants = (
  FOR v IN 0..100 OUTBOUND @start_vertex GRAPH @graph_name
    RETURN v._key
)

LET edges = (
  FOR key IN descendants
    FOR e IN @@edge_collection
      FILTER e._from == CONCAT(@collection_prefix, key) 
          OR e._to == CONCAT(@collection_prefix, key)
      RETURN e._key
)

LET deleted_edges = (
  FOR key IN edges
    REMOVE {_key: key} IN @@edge_collection
    RETURN 1
)

LET deleted_nodes = (
  FOR key IN descendants
    REMOVE {_key: key} IN @@node_collection
    RETURN 1
)

RETURN {deleted_nodes: LENGTH(deleted_nodes), deleted_edges: LENGTH(deleted_edges)}
"""


# =============================================================================
# Vector Search Queries (Used by SearchHandler)
# =============================================================================

# Vector similarity search on content embeddings (for detail nodes)
VECTOR_SEARCH_CONTENT = """
FOR doc IN @@view
  SEARCH ANALYZER(doc.embedding != null, "identity")
  FILTER doc.node_type IN @node_types
  LET score = COSINE_SIMILARITY(doc.embedding, @query_vector)
  FILTER score >= @min_score
  SORT score DESC
  LIMIT @limit
  RETURN {doc: doc, score: score}
"""

# Vector search with source filter
VECTOR_SEARCH_CONTENT_WITH_SOURCE = """
FOR doc IN @@view
  SEARCH ANALYZER(doc.embedding != null, "identity")
  FILTER doc.node_type IN @node_types
  FILTER doc.source_id == @source_id
  LET score = COSINE_SIMILARITY(doc.embedding, @query_vector)
  FILTER score >= @min_score
  SORT score DESC
  LIMIT @limit
  RETURN {doc: doc, score: score}
"""

# Vector similarity search on structure embeddings (for structure nodes)
VECTOR_SEARCH_STRUCTURE = """
FOR doc IN @@view
  SEARCH ANALYZER(doc.embedding != null, "identity")
  FILTER doc.node_type == "structure"
  LET score = COSINE_SIMILARITY(doc.embedding, @query_vector)
  FILTER score >= @min_score
  SORT score DESC
  LIMIT @limit
  RETURN {doc: doc, score: score}
"""

# Vector search structure with source filter
VECTOR_SEARCH_STRUCTURE_WITH_SOURCE = """
FOR doc IN @@view
  SEARCH ANALYZER(doc.embedding != null, "identity")
  FILTER doc.node_type == "structure"
  FILTER doc.source_id == @source_id
  LET score = COSINE_SIMILARITY(doc.embedding, @query_vector)
  FILTER score >= @min_score
  SORT score DESC
  LIMIT @limit
  RETURN {doc: doc, score: score}
"""

# Hybrid search: attribute filters + vector similarity
# Note: filter_clause is injected via f-string
HYBRID_SEARCH = """
FOR doc IN @@view
  SEARCH ANALYZER(doc.embedding != null, "identity")
  FILTER {filter_clause}
  LET score = COSINE_SIMILARITY(doc.embedding, @query_vector)
  FILTER score >= @min_score
  SORT score DESC
  LIMIT @limit
  RETURN {{doc: doc, score: score}}
"""

# Full-text search on text and title attributes
# Note: collection_name, source_filter are injected via f-string
FULLTEXT_SEARCH = """
FOR doc IN {collection}
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


# =============================================================================
# Mining Queries (Used by MiningHandler)
# =============================================================================

# Get FAQs by minimum frequency
MINING_GET_BY_FREQUENCY = """
FOR doc IN @@collection
  FILTER doc.frequency >= @min_frequency
  SORT doc.frequency DESC
  LIMIT @limit
  RETURN doc
"""

# Increment frequency atomically
MINING_INCREMENT_FREQUENCY = """
FOR doc IN @@collection
  FILTER doc._key == @key
  UPDATE doc WITH {
    frequency: doc.frequency + 1
  } IN @@collection
  RETURN NEW
"""


# =============================================================================
# Context Expansion Queries (For RAG)
# =============================================================================

# Get path from node to root (for breadcrumb context)
GET_PATH_TO_ROOT = """
FOR v, e, p IN 1..100 INBOUND @start_vertex GRAPH @graph_name
  FILTER e.type IN ["HAS_SECTION", "HAS_CONTENT"]
  OPTIONS {order: "bfs"}
  RETURN v
"""

# Find related nodes through graph (for context expansion in RAG)
FIND_RELATED_CONTEXT = """
LET node = DOCUMENT(@node_id)
LET ancestors = (
  FOR v IN 1..3 INBOUND @node_id GRAPH @graph_name
    RETURN v
)
LET siblings = (
  FOR v IN 1..1 INBOUND @node_id GRAPH @graph_name
    FOR s IN 1..1 OUTBOUND v GRAPH @graph_name
      FILTER s._id != @node_id
      RETURN s
)
RETURN {
  node: node,
  ancestors: ancestors,
  siblings: siblings
}
"""
