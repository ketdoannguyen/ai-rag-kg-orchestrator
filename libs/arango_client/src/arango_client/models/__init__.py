"""Pydantic models for ArangoDB."""

from .attributes_metadata import (
    ImageAttributes,
    ImageMetadata,
    ParagraphAttributes,
    ParagraphMetadata,
    StructureAttributes,
    StructureMetadata,
    TableAttributes,
    TableMetadata,
)
from .chat import ChatMessage, ChatSession, Feedback
from .content_nodes import ContentNode
from .detail_nodes import DetailNode, ImageNode, ParagraphNode, TableNode
from .entity_nodes import EntityNode, EntityType
from .enums import (
    FAQStatus,
    FeedbackStatus,
    NodeType,
    SourceType,
    UserRole,
)
from .knowledge import KnowledgeNode
from .mining import SuggestedFAQ
from .ontology import OntologyRelation, RelationStatus, RelationTier
from .structure_nodes import RootNode, StructureNode
from .system_admin import DataSource, User

__all__ = [
    # Enums
    "FAQStatus",
    "FeedbackStatus",
    "NodeType",
    "SourceType",
    "UserRole",
    "EntityType",  # Tier 3
    "RelationStatus",  # Ontology
    "RelationTier",  # Ontology tier classification
    # Attributes/Metadata
    "ImageAttributes",
    "ImageMetadata",
    "ParagraphAttributes",
    "ParagraphMetadata",
    "StructureAttributes",
    "StructureMetadata",
    "TableAttributes",
    "TableMetadata",
    # Knowledge nodes
    "KnowledgeNode",
    "ContentNode",  # Tier 2 unified content node
    "EntityNode",  # Tier 3 entity node
    "DetailNode",
    "ImageNode",
    "ParagraphNode",
    "TableNode",
    "RootNode",
    "StructureNode",
    # Ontology
    "OntologyRelation",
    # Chat
    "ChatMessage",
    "ChatSession",
    "Feedback",
    # System admin
    "DataSource",
    "User",
    # Mining
    "SuggestedFAQ",
]
