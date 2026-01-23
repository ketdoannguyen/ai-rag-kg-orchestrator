"""Handlers for ArangoDB operations."""

from .base_extraction_handler import BaseExtractionHandler, ExtractionDecision
from .base_handler import BaseHandler
from .chat_handler import (
    ChatHandler,
    ChatMessageHandler,
    ChatSessionHandler,
    FeedbackHandler,
)
from .content_extraction_handler import (
    ContentExtractionHandler,
    ExtractionResult,
    SimilarNode,
)
from .knowledge_handler import KnowledgeHandler, KnowledgeNodeHandler
from .mining_handler import MiningHandler, SuggestedFAQHandler
from .ontology_handler import OntologyHandler
from .search_handler import SearchHandler, SearchResult
from .structure_extraction_handler import (
    STRUCTURALLY_EQUIVALENT_RELATION,
    SimilarStructure,
    StructureExtractionHandler,
)
from .system_admin_handler import (
    DataSourceHandler,
    SystemAdminHandler,
    UserHandler,
)

__all__ = [
    # Base
    "BaseHandler",
    "BaseExtractionHandler",
    # Knowledge
    "KnowledgeNodeHandler",
    "KnowledgeHandler",
    # Extraction - Content (Tier 2)
    "ContentExtractionHandler",
    "ExtractionDecision",
    "ExtractionResult",
    "SimilarNode",
    # Extraction - Structure (Tier 1)
    "StructureExtractionHandler",
    "SimilarStructure",
    "STRUCTURALLY_EQUIVALENT_RELATION",
    # Ontology
    "OntologyHandler",
    # Chat
    "ChatSessionHandler",
    "ChatMessageHandler",
    "FeedbackHandler",
    "ChatHandler",
    # System Admin
    "UserHandler",
    "DataSourceHandler",
    "SystemAdminHandler",
    # Mining
    "SuggestedFAQHandler",
    "MiningHandler",
    # Search
    "SearchHandler",
    "SearchResult",
]
