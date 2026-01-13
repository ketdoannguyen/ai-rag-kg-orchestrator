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
from .detail_nodes import DetailNode, ImageNode, ParagraphNode, TableNode
from .enums import (
    FAQStatus,
    FeedbackStatus,
    NodeType,
    SourceType,
    UserRole,
)
from .knowledge import KnowledgeNode
from .mining import SuggestedFAQ
from .structure_nodes import RootNode, StructureNode
from .system_admin import DataSource, User

__all__ = [
    # Enums
    "FAQStatus",
    "FeedbackStatus",
    "NodeType",
    "SourceType",
    "UserRole",
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
    "DetailNode",
    "ImageNode",
    "ParagraphNode",
    "TableNode",
    "RootNode",
    "StructureNode",
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
