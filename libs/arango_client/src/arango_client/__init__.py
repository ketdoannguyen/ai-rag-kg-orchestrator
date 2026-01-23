"""
ArangoDB Client Library.

A multi-model database client that replaces PostgreSQL + Neo4j with ArangoDB.
Provides document store, graph store, and vector search capabilities.
"""

from .client import ArangoClient

# Re-export handlers for convenience
from .handlers import (
    ChatHandler,
    ChatMessageHandler,
    ChatSessionHandler,
    DataSourceHandler,
    FeedbackHandler,
    KnowledgeHandler,
    KnowledgeNodeHandler,
    MiningHandler,
    SearchHandler,
    SuggestedFAQHandler,
    SystemAdminHandler,
    UserHandler,
)

# Re-export models for convenience
from .models import (
    # Chat
    ChatMessage,
    ChatSession,
    # System admin
    DataSource,
    DetailNode,
    # Enums
    FAQStatus,
    Feedback,
    FeedbackStatus,
    # Attributes/Metadata
    ImageAttributes,
    ImageMetadata,
    ImageNode,
    # Knowledge nodes
    KnowledgeNode,
    NodeType,
    ParagraphAttributes,
    ParagraphMetadata,
    ParagraphNode,
    RootNode,
    SourceType,
    StructureAttributes,
    StructureMetadata,
    StructureNode,
    # Mining
    SuggestedFAQ,
    TableAttributes,
    TableMetadata,
    TableNode,
    User,
    UserRole,
)
from .utils import (
    compute_tree_statistics,
    doc_to_knowledge_node,
    parse_node_from_dict,
    serialize_tree_to_dict,
    set_parent_ids,
    set_positions,
    validate_tree_and_attach_source,
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
