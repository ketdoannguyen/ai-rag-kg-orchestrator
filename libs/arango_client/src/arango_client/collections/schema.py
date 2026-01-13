"""
Collection and index schema definitions for ArangoDB.

This module defines all collection names, edge definitions, indexes,
and ArangoSearch view configurations.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


# =============================================================================
# Collection Names
# =============================================================================
class Collections:
    """Document collection names."""
    
    KNOWLEDGE_NODES = "knowledge_nodes"
    CHAT_SESSIONS = "chat_sessions"
    CHAT_MESSAGES = "chat_messages"
    FEEDBACKS = "feedbacks"
    USERS = "users"
    DATA_SOURCES = "data_sources"
    SUGGESTED_FAQS = "suggested_faqs"


class EdgeCollections:
    """Edge collection names for graph relationships."""
    
    KNOWLEDGE_EDGES = "knowledge_edges"


# =============================================================================
# Relationship Types (stored in edge documents)
# =============================================================================
class RelationshipTypes:
    """Types of relationships in the knowledge graph."""
    
    HAS_SECTION = "HAS_SECTION"    # Structure/Root -> Structure child
    HAS_CONTENT = "HAS_CONTENT"    # Structure/Root -> Detail node
    NEXT_SIBLING = "NEXT_SIBLING"  # Sibling -> Next sibling


# =============================================================================
# Graph Definition
# =============================================================================
KNOWLEDGE_GRAPH_NAME = "knowledge_graph"

KNOWLEDGE_GRAPH_DEFINITION = {
    "name": KNOWLEDGE_GRAPH_NAME,
    "edge_definitions": [
        {
            "edge_collection": EdgeCollections.KNOWLEDGE_EDGES,
            "from_vertex_collections": [Collections.KNOWLEDGE_NODES],
            "to_vertex_collections": [Collections.KNOWLEDGE_NODES],
        }
    ],
}


# =============================================================================
# Index Definitions
# =============================================================================
@dataclass
class IndexDefinition:
    """Index definition for a collection."""
    
    collection: str
    type: str  # "persistent", "hash", "fulltext", "geo", "ttl"
    fields: list[str]
    unique: bool = False
    sparse: bool = False
    name: str | None = None


# Standard indexes for each collection
INDEXES: list[IndexDefinition] = [
    # Knowledge nodes indexes
    IndexDefinition(
        collection=Collections.KNOWLEDGE_NODES,
        type="persistent",
        fields=["source_id"],
        name="idx_knowledge_source_id",
    ),
    IndexDefinition(
        collection=Collections.KNOWLEDGE_NODES,
        type="persistent",
        fields=["parent_id"],
        name="idx_knowledge_parent_id",
    ),
    IndexDefinition(
        collection=Collections.KNOWLEDGE_NODES,
        type="persistent",
        fields=["node_type"],
        name="idx_knowledge_node_type",
    ),
    IndexDefinition(
        collection=Collections.KNOWLEDGE_NODES,
        type="persistent",
        fields=["created_at"],
        name="idx_knowledge_created_at",
    ),
    
    # Chat sessions indexes
    IndexDefinition(
        collection=Collections.CHAT_SESSIONS,
        type="persistent",
        fields=["user_identifier"],
        name="idx_chat_sessions_user",
    ),
    
    # Chat messages indexes
    IndexDefinition(
        collection=Collections.CHAT_MESSAGES,
        type="persistent",
        fields=["session_id"],
        name="idx_chat_messages_session",
    ),
    IndexDefinition(
        collection=Collections.CHAT_MESSAGES,
        type="persistent",
        fields=["created_at"],
        name="idx_chat_messages_created_at",
    ),
    
    # Feedbacks indexes
    IndexDefinition(
        collection=Collections.FEEDBACKS,
        type="persistent",
        fields=["message_id"],
        name="idx_feedbacks_message",
    ),
    IndexDefinition(
        collection=Collections.FEEDBACKS,
        type="persistent",
        fields=["status"],
        name="idx_feedbacks_status",
    ),
    
    # Users indexes
    IndexDefinition(
        collection=Collections.USERS,
        type="persistent",
        fields=["username"],
        unique=True,
        name="idx_users_username",
    ),
    
    # Data sources indexes
    IndexDefinition(
        collection=Collections.DATA_SOURCES,
        type="persistent",
        fields=["hash_content"],
        sparse=True,
        name="idx_data_sources_hash",
    ),
    IndexDefinition(
        collection=Collections.DATA_SOURCES,
        type="persistent",
        fields=["type"],
        name="idx_data_sources_type",
    ),
    
    # Suggested FAQs indexes
    IndexDefinition(
        collection=Collections.SUGGESTED_FAQS,
        type="persistent",
        fields=["status"],
        name="idx_faqs_status",
    ),
    IndexDefinition(
        collection=Collections.SUGGESTED_FAQS,
        type="persistent",
        fields=["frequency"],
        name="idx_faqs_frequency",
    ),
    
    # Edge collection indexes
    IndexDefinition(
        collection=EdgeCollections.KNOWLEDGE_EDGES,
        type="persistent",
        fields=["type"],
        name="idx_edges_type",
    ),
]


# =============================================================================
# ArangoSearch View Configuration for Vector Search
# =============================================================================
KNOWLEDGE_VECTOR_VIEW_NAME = "knowledge_vector_view"

KNOWLEDGE_VECTOR_VIEW_DEFINITION: dict[str, Any] = {
    "links": {
        Collections.KNOWLEDGE_NODES: {
            "includeAllFields": False,
            "fields": {
                # Unified embedding field (used by both structure and detail nodes)
                "embedding": {
                    "analyzers": ["identity"],
                },
                # Filterable fields
                "node_type": {},
                "source_id": {},
                "level": {},  # Renamed from level_down
                "created_at": {},
                # Text fields for full-text search
                "summary": {
                    "analyzers": ["text_en", "identity"],
                },
                "text": {
                    "analyzers": ["text_en", "identity"],
                },
                # Nested attributes for filtering
                "attributes": {
                    "nested": {
                        "title": {
                            "analyzers": ["text_en", "identity"],
                        },
                        "text": {
                            "analyzers": ["text_en", "identity"],
                        },
                    },
                },
            },
        },
    },
    # View-level settings for better performance
    "consolidationIntervalMsec": 10000,
    "consolidationPolicy": {
        "type": "tier",
        "segmentsMin": 1,
        "segmentsMax": 10,
        "segmentsBytesFloor": 2097152,  # 2MB
        "segmentsBytesMax": 5368709120,  # 5GB
    },
}


# =============================================================================
# Vector Index Configuration (ArangoDB 3.12+)
# For approximate nearest neighbor search (Milvus-level performance)
# =============================================================================
KNOWLEDGE_VECTOR_INDEX_DEFINITION: dict[str, Any] = {
    "type": "vector",
    "name": "idx_embedding_vector",
    "fields": ["embedding"],
    "params": {
        "metric": "cosine",      # Cosine similarity
        "dimensions": 768,       # Default dimension (update for OpenAI: 1536)
        "nLists": 32,            # Number of clusters for IVF
    },
}


# =============================================================================
# All document collections (for initialization)
# =============================================================================
ALL_DOCUMENT_COLLECTIONS = [
    Collections.KNOWLEDGE_NODES,
    Collections.CHAT_SESSIONS,
    Collections.CHAT_MESSAGES,
    Collections.FEEDBACKS,
    Collections.USERS,
    Collections.DATA_SOURCES,
    Collections.SUGGESTED_FAQS,
]

ALL_EDGE_COLLECTIONS = [
    EdgeCollections.KNOWLEDGE_EDGES,
]
