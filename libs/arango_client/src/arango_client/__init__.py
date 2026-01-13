"""
ArangoDB Client Library.

A multi-model database client that replaces PostgreSQL + Neo4j with ArangoDB.
Provides document store, graph store, and vector search capabilities.
"""

from .client import ArangoClient
from .utils import (
    doc_to_knowledge_node,
    parse_node_from_dict,
    set_parent_ids,
    set_positions,
    serialize_tree_to_dict,
    validate_tree_and_attach_source,
    compute_tree_statistics,
)

# Re-export handlers for convenience
from .handlers import (
    KnowledgeNodeHandler,
    KnowledgeHandler,
    ChatSessionHandler,
    ChatMessageHandler,
    FeedbackHandler,
    ChatHandler,
    UserHandler,
    DataSourceHandler,
    SystemAdminHandler,
    SuggestedFAQHandler,
    MiningHandler,
    SearchHandler,
)

# Re-export models for convenience
from .models import (
    # Enums
    FAQStatus,
    FeedbackStatus,
    NodeType,
    SourceType,
    UserRole,
    # Knowledge nodes
    KnowledgeNode,
    DetailNode,
    ImageNode,
    ParagraphNode,
    TableNode,
    RootNode,
    StructureNode,
    # Chat
    ChatMessage,
    ChatSession,
    Feedback,
    # System admin
    DataSource,
    User,
    # Mining
    SuggestedFAQ,
    # Attributes/Metadata
    ImageAttributes,
    ImageMetadata,
    ParagraphAttributes,
    ParagraphMetadata,
    StructureAttributes,
    StructureMetadata,
    TableAttributes,
    TableMetadata,
)

__all__ = [
    # Client
    "ArangoClient",
    # Utilities
    "doc_to_knowledge_node",
    "parse_node_from_dict",
    "set_parent_ids",
    "set_positions",
    "serialize_tree_to_dict",
    "validate_tree_and_attach_source",
    "compute_tree_statistics",
    # Handlers
    "KnowledgeNodeHandler",
    "KnowledgeHandler",
    "ChatSessionHandler",
    "ChatMessageHandler",
    "FeedbackHandler",
    "ChatHandler",
    "UserHandler",
    "DataSourceHandler",
    "SystemAdminHandler",
    "SuggestedFAQHandler",
    "MiningHandler",
    "SearchHandler",
    # Models - Enums
    "FAQStatus",
    "FeedbackStatus",
    "NodeType",
    "SourceType",
    "UserRole",
    # Models - Knowledge
    "KnowledgeNode",
    "DetailNode",
    "ImageNode",
    "ParagraphNode",
    "TableNode",
    "RootNode",
    "StructureNode",
    # Models - Chat
    "ChatMessage",
    "ChatSession",
    "Feedback",
    # Models - System Admin
    "DataSource",
    "User",
    # Models - Mining
    "SuggestedFAQ",
    # Models - Attributes/Metadata
    "ImageAttributes",
    "ImageMetadata",
    "ParagraphAttributes",
    "ParagraphMetadata",
    "StructureAttributes",
    "StructureMetadata",
    "TableAttributes",
    "TableMetadata",
]
__version__ = "0.1.0"

