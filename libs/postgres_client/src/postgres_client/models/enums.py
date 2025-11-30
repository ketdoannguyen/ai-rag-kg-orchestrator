"""Enums matching the SQL schema definitions."""

from enum import Enum


class UserRole(str, Enum):
    """User roles in the system."""

    ADMIN = "admin"
    EDITOR = "editor"
    VIEWER = "viewer"


class SourceType(str, Enum):
    """Source types for data ingestion."""

    PDF = "pdf"
    WEB = "web"
    FACEBOOK = "facebook"
    FAQ_FILE = "faq_file"


class FeedbackStatus(str, Enum):
    """Status of user feedback."""

    PENDING = "pending"
    REVIEWED = "reviewed"
    IGNORED = "ignored"


class FAQStatus(str, Enum):
    """Status of FAQ suggestions."""

    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"
    ARCHIVED = "archived"


class SemanticLevel(str, Enum):
    """
    Semantic levels for knowledge nodes - used by AI Router.
    
    - CORPUS: Document/Collection level (e.g., Handbook 2025)
    - TOPIC: Major topic level (e.g., Chapter, Part, Web Domain)
    - SUBTOPIC: Sub-topic level (e.g., Lesson, Section I, Clause)
    - DETAIL: Detail level (e.g., Paragraph, Item a/b/c, FAQ question)
    """

    CORPUS = "CORPUS"
    TOPIC = "TOPIC"
    SUBTOPIC = "SUBTOPIC"
    DETAIL = "DETAIL"


class NodeType(str, Enum):
    """Node type classification for knowledge nodes."""

    PARAGRAPH = "paragraph"
    IMAGE = "image"
    TABLE = "table"
    SECTION = "section"
    CORPUS = "corpus"
    TOPIC = "topic"
    SUBTOPIC = "subtopic"

