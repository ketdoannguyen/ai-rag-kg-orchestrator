"""Enums matching the data schema definitions."""

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


class NodeType(str, Enum):
    """Node type classification for knowledge nodes."""

    PARAGRAPH = "paragraph"
    IMAGE = "image"
    TABLE = "table"
    STRUCTURE = "structure"
