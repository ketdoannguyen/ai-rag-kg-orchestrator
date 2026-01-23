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

    # Unified content collection (Tier 2 - PRIMARY)
    CONTENT_NODES = "content_nodes"

    # Structure collections (Tier 1)
    STRUCTURE_NODES = "structure_nodes"
    ROOT_NODES = "root_nodes"

    # Entity collection (Tier 3 - scaffold, logic to be implemented)
    ENTITY_NODES = "entity_nodes"

    # Ontology collection (relation types with embeddings)
    ONTOLOGY_RELATIONS = "ontology_relations"

    # Other collections
    CHAT_SESSIONS = "chat_sessions"
    CHAT_MESSAGES = "chat_messages"
    FEEDBACKS = "feedbacks"
    USERS = "users"
    DATA_SOURCES = "data_sources"
    SUGGESTED_FAQS = "suggested_faqs"


# All knowledge node collections (for graph and search)
KNOWLEDGE_COLLECTIONS = [
    Collections.CONTENT_NODES,
    Collections.STRUCTURE_NODES,
    Collections.ROOT_NODES,
    Collections.ENTITY_NODES,
]

# Content collections only (Tier 2)
CONTENT_COLLECTIONS = [
    Collections.CONTENT_NODES,
]

# Structure collections only (Tier 1)
STRUCTURE_COLLECTIONS = [
    Collections.STRUCTURE_NODES,
    Collections.ROOT_NODES,
]

# Entity collections (Tier 3)
ENTITY_COLLECTIONS = [
    Collections.ENTITY_NODES,
]


class EdgeCollections:
    """Edge collection names for graph relationships."""

    KNOWLEDGE_EDGES = "knowledge_edges"


# =============================================================================
# Relationship Types (stored in edge documents)
# =============================================================================
class RelationshipTypes:
    """Types of relationships in the knowledge graph."""

    HAS_SECTION = "HAS_SECTION"  # Structure/Root -> Structure child
    HAS_CONTENT = "HAS_CONTENT"  # Structure/Root -> Detail node
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
            "from_vertex_collections": KNOWLEDGE_COLLECTIONS,
            "to_vertex_collections": KNOWLEDGE_COLLECTIONS,
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
def _generate_knowledge_indexes() -> list[IndexDefinition]:
    """Generate indexes for all knowledge collections."""
    indexes = []
    for collection in KNOWLEDGE_COLLECTIONS:
        indexes.extend(
            [
                IndexDefinition(
                    collection=collection,
                    type="persistent",
                    fields=["source_id"],
                    name=f"idx_{collection}_source_id",
                ),
                IndexDefinition(
                    collection=collection,
                    type="persistent",
                    fields=["parent_id"],
                    name=f"idx_{collection}_parent_id",
                ),
                IndexDefinition(
                    collection=collection,
                    type="persistent",
                    fields=["node_type"],
                    name=f"idx_{collection}_node_type",
                ),
                IndexDefinition(
                    collection=collection,
                    type="persistent",
                    fields=["created_at"],
                    name=f"idx_{collection}_created_at",
                ),
            ]
        )
    return indexes


INDEXES: list[IndexDefinition] = _generate_knowledge_indexes() + [
    # Edge collection composite indexes for graph traversal optimization
    IndexDefinition(
        collection=EdgeCollections.KNOWLEDGE_EDGES,
        type="persistent",
        fields=["_from", "type"],
        name="idx_edges_from_type",
    ),
    IndexDefinition(
        collection=EdgeCollections.KNOWLEDGE_EDGES,
        type="persistent",
        fields=["_to", "type"],
        name="idx_edges_to_type",
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
    # Edge collection indexes (single field - for filtering only)
    IndexDefinition(
        collection=EdgeCollections.KNOWLEDGE_EDGES,
        type="persistent",
        fields=["type"],
        name="idx_edges_type",
    ),
]


# =============================================================================
# Level Index for Knowledge Collections (structure navigation)
# =============================================================================
def _generate_level_indexes() -> list[IndexDefinition]:
    """Generate level indexes for knowledge collections."""
    indexes = []
    for collection in KNOWLEDGE_COLLECTIONS:
        indexes.append(
            IndexDefinition(
                collection=collection,
                type="persistent",
                fields=["level"],
                name=f"idx_{collection}_level",
            )
        )
    return indexes


# Add level indexes to main indexes
INDEXES.extend(_generate_level_indexes())


# =============================================================================
# ArangoSearch View Configuration for Vector Search
# =============================================================================
KNOWLEDGE_VECTOR_VIEW_NAME = "knowledge_vector_view"


def _generate_view_links() -> dict[str, Any]:
    """Generate view links for all knowledge collections.

    Note: embedding field is NOT indexed in the view because:
    1. ArangoSearch views use COSINE_SIMILARITY() which reads raw embedding values
    2. The "identity" analyzer does not support array fields
    3. Vector search uses document-level embedding access, not analyzer-based indexing
    """
    field_config = {
        "includeAllFields": False,
        "fields": {
            # Removed embedding from view - use COSINE_SIMILARITY() on raw field instead
            "node_type": {},
            "source_id": {},
            "level": {},
            "created_at": {},
            "summary": {"analyzers": ["text_en", "identity"]},
            "text": {"analyzers": ["text_en", "identity"]},
            "attributes": {
                "nested": {
                    "title": {"analyzers": ["text_en", "identity"]},
                    "text": {"analyzers": ["text_en", "identity"]},
                },
            },
        },
        "storeValues": "id",  # Store document IDs for faster retrieval
    }
    return {collection: field_config.copy() for collection in KNOWLEDGE_COLLECTIONS}


KNOWLEDGE_VECTOR_VIEW_DEFINITION: dict[str, Any] = {
    "links": _generate_view_links(),
    "consolidationIntervalMsec": 10000,
    "consolidationPolicy": {
        "type": "tier",
        "segmentsMin": 1,
        "segmentsMax": 10,
        "segmentsBytesFloor": 2097152,
        "segmentsBytesMax": 5368709120,
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
        "metric": "cosine",  # Cosine similarity
        "dimensions": 768,  # Default dimension (update for OpenAI: 1536)
        "nLists": 32,  # Number of clusters for IVF
    },
}


# =============================================================================
# All document collections (for initialization)
# =============================================================================
ALL_DOCUMENT_COLLECTIONS = KNOWLEDGE_COLLECTIONS + [
    Collections.ONTOLOGY_RELATIONS,
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
