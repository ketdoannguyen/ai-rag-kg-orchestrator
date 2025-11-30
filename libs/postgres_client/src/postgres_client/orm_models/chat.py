"""
ORM classes for Chat Operations tables.

Tables:
- chat_sessions: Chat conversation sessions
- chat_messages: Individual messages in a chat session
- feedbacks: User feedback on chat messages
"""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any

from sqlalchemy import (
    Boolean,
    CheckConstraint,
    DateTime,
    ForeignKey,
    String,
    Text,
    UUID,
    Enum as SQLEnum,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column, relationship

from ..models.enums import FeedbackStatus
from .base import Base


class ChatSessionORM(Base):
    """ORM class for chat_sessions table."""

    __tablename__ = 'chat_sessions'

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
    )
    user_identifier: Mapped[str | None] = mapped_column(
        String(100),
        nullable=True,
        comment="Session Cookie hoặc User ID",
    )
    started_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=datetime.utcnow,
    )

    # Relationships
    messages = relationship(
        "ChatMessageORM",
        back_populates="session",
        cascade="all, delete-orphan",
    )


class ChatMessageORM(Base):
    """ORM class for chat_messages table."""

    __tablename__ = 'chat_messages'

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
    )
    session_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey('chat_sessions.id', ondelete='CASCADE'),
        nullable=False,
    )
    role: Mapped[str] = mapped_column(
        String(20),
        nullable=False,
    )
    content: Mapped[str] = mapped_column(
        Text,
        nullable=False,
    )
    context_data: Mapped[dict[str, Any]] = mapped_column(
        JSONB,
        server_default='{}',
        comment="Context quan trọng cho Debug & Mining: "
                "{ 'retrieved_ids': ['uuid1', 'uuid2'], 'source': 'cache/rag', 'latency': 0.5 }",
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=datetime.utcnow,
    )

    # Relationships
    session = relationship(
        "ChatSessionORM",
        back_populates="messages",
    )
    feedbacks = relationship(
        "FeedbackORM",
        back_populates="message",
        cascade="all, delete-orphan",
    )

    # Constraints
    __table_args__ = (
        CheckConstraint(
            "role IN ('user', 'assistant')",
            name='check_role_valid',
        ),
    )


class FeedbackORM(Base):
    """ORM class for feedbacks table."""

    __tablename__ = 'feedbacks'

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
    )
    message_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey('chat_messages.id', ondelete='CASCADE'),
        nullable=False,
    )
    is_positive: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
    )
    comment: Mapped[str | None] = mapped_column(
        Text,
        nullable=True,
    )
    status: Mapped[FeedbackStatus] = mapped_column(
        SQLEnum(FeedbackStatus, name='feedback_status'),
        default=FeedbackStatus.PENDING,
    )
    reviewed_by: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey('users.id'),
        nullable=True,
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=datetime.utcnow,
    )

    # Relationships
    message = relationship(
        "ChatMessageORM",
        back_populates="feedbacks",
    )
    reviewer = relationship(
        "UserORM",
        foreign_keys=[reviewed_by],
        back_populates="reviewed_feedbacks",
    )

