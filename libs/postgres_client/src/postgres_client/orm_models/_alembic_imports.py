"""
File này được dùng bởi Alembic để import tất cả ORM models
mà không trigger circular import từ __init__.py.

KHÔNG import file này từ __init__.py hoặc bất kỳ đâu khác!
Chỉ dùng trong migrations/env.py
"""
from .base import Base

# Import tất cả ORM models để đảm bảo metadata được đăng ký
from .system_admin import DataSourceORM, UserORM  # noqa: F401
from .knowledge import (  # noqa: F401
    EmbeddingContentORM,
    EmbeddingStructureORM,
    KnowledgeNodeORM,
)
from .chat import ChatMessageORM, ChatSessionORM, FeedbackORM  # noqa: F401
from .mining import SuggestedFAQORM  # noqa: F401

__all__ = ["Base"]

