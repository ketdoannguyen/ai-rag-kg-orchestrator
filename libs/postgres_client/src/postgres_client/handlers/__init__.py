"""CRUD handlers for PostgreSQL models."""

from .base_handler import BaseHandler
from .chat_handler import (
    ChatHandler,
    ChatSessionHandler,
    ChatMessageHandler,
    FeedbackHandler,
)
from .system_admin_handler import (
    SystemAdminHandler,
    UserHandler,
    DataSourceHandler,
)
from .knowledge_handler import (
    KnowledgeHandler,
    KnowledgeNodeHandler,
)
from .embedding_handler import (
    EmbeddingHandler,
    EmbeddingStructureHandler,
    EmbeddingContentHandler,
)
from .mining_handler import (
    MiningHandler,
    SuggestedFAQHandler,
)

__all__ = [
    # Base
    "BaseHandler",
    # Chat - Grouped
    "ChatHandler",
    # Chat - Individual
    "ChatSessionHandler",
    "ChatMessageHandler",
    "FeedbackHandler",
    # System Admin - Grouped
    "SystemAdminHandler",
    # System Admin - Individual
    "UserHandler",
    "DataSourceHandler",
    # Knowledge - Grouped
    "KnowledgeHandler",
    # Knowledge - Individual
    "KnowledgeNodeHandler",
    # Embedding - Grouped
    "EmbeddingHandler",
    # Embedding - Individual
    "EmbeddingStructureHandler",
    "EmbeddingContentHandler",
    # Mining - Grouped
    "MiningHandler",
    # Mining - Individual
    "SuggestedFAQHandler",
]
