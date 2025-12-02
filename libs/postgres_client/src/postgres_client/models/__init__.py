"""PostgreSQL client models matching the SQL schema."""

# Pydantic models (for data validation and serialization)
from .knowledge import KnowledgeNode
from .detail_nodes import ImageNode, ParagraphNode, TableNode
from .enums import (
    FAQStatus,
    FeedbackStatus,
    NodeType,
    SourceType,
    UserRole,
)
from .structure_nodes import StructureNode
from .system_admin import DataSource, User
from .embeddings import EmbeddingContent, EmbeddingStructure
from .chat import ChatMessage, ChatSession, Feedback
from .mining import SuggestedFAQ

# ORM classes (for database mapping) - imported from orm_models
from ..orm_models import (
    Base,
    DataSourceORM,
    EmbeddingContentORM,
    EmbeddingStructureORM,
    KnowledgeNodeORM,
    UserORM,
    ChatSessionORM,
    ChatMessageORM,
    FeedbackORM,
    SuggestedFAQORM,
)

# Converter utilities (ONLY for KnowledgeNode - other models use to_orm()/from_orm() methods)
from ..utils.knowledge_node_converter import (
    orm_to_pydantic,  # Only for KnowledgeNodeORM
    pydantic_to_orm,  # Only for KnowledgeNode
)


__all__ = [
    # Pydantic Base
    "KnowledgeNode",
    # Enums
    "UserRole",
    "SourceType",
    "FeedbackStatus",
    "FAQStatus",
    "NodeType",
    # Pydantic Detail nodes
    "ParagraphNode",
    "ImageNode",
    "TableNode",
    # Pydantic Structure node
    "StructureNode",
    # Pydantic System Admin
    "User",
    "DataSource",
    # Pydantic Embeddings
    "EmbeddingStructure",
    "EmbeddingContent",
    # Pydantic Chat
    "ChatSession",
    "ChatMessage",
    "Feedback",
    # Pydantic Mining
    "SuggestedFAQ",
    # ORM Base
    "Base",
    # ORM Classes - System Admin
    "UserORM",
    "DataSourceORM",
    # ORM Classes - Knowledge
    "KnowledgeNodeORM",
    "EmbeddingStructureORM",
    "EmbeddingContentORM",
    # ORM Classes - Chat
    "ChatSessionORM",
    "ChatMessageORM",
    "FeedbackORM",
    # ORM Classes - Mining
    "SuggestedFAQORM",
    # Converter utilities
    "pydantic_to_orm",
    "orm_to_pydantic",
]

