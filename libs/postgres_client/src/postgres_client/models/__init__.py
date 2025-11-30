"""PostgreSQL client models matching the SQL schema."""

# Pydantic models (for data validation and serialization)
from .base import KnowledgeNode
from .detail_nodes import ImageNode, ParagraphNode, TableNode
from .enums import (
    FAQStatus,
    FeedbackStatus,
    NodeType,
    SemanticLevel,
    SourceType,
    UserRole,
)
from .structure_nodes import (
    CorpusNode,
    StructureNode,
    SubtopicNode,
    TopicNode,
)


__all__ = [
    # Pydantic Base
    "KnowledgeNode",
    # Enums
    "UserRole",
    "SourceType",
    "FeedbackStatus",
    "FAQStatus",
    "SemanticLevel",
    "NodeType",
    # Pydantic Detail nodes
    "ParagraphNode",
    "ImageNode",
    "TableNode",
    # Pydantic Structure nodes
    "StructureNode",
    "CorpusNode",
    "TopicNode",
    "SubtopicNode",
]

