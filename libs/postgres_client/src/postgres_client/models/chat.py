"""
Pydantic models for Chat Operations.

These models represent ChatSession, ChatMessage, and Feedback entities.
"""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from .enums import FeedbackStatus


class ChatSession(BaseModel):
    """Pydantic model for chat_sessions table."""

    id: uuid.UUID = Field(
        default_factory=uuid.uuid4,
        description="Primary key"
    )
    user_identifier: str | None = Field(
        None,
        max_length=100,
        description="Session Cookie hoặc User ID"
    )
    started_at: datetime = Field(
        default_factory=datetime.utcnow,
        description="Session start timestamp"
    )

    model_config = ConfigDict(
        json_encoders={
            uuid.UUID: str
        }
    )

    def to_orm(self):
        """Convert Pydantic model to ORM instance."""
        from ..orm_models.chat import ChatSessionORM
        return ChatSessionORM(
            id=self.id,
            user_identifier=self.user_identifier,
            started_at=self.started_at,
        )

    @classmethod
    def from_orm(cls, orm_instance):
        """Create Pydantic model from ORM instance."""
        return cls(
            id=orm_instance.id,
            user_identifier=orm_instance.user_identifier,
            started_at=orm_instance.started_at,
        )


class ChatMessage(BaseModel):
    """Pydantic model for chat_messages table."""

    id: uuid.UUID = Field(
        default_factory=uuid.uuid4,
        description="Primary key"
    )
    session_id: uuid.UUID = Field(
        ...,
        description="Reference to chat_sessions.id"
    )
    role: Literal["user", "assistant"] = Field(
        ...,
        description="Message role"
    )
    content: str = Field(
        ...,
        description="Message content"
    )
    context_data: dict[str, Any] = Field(
        default_factory=dict,
        description="Context for Debug & Mining: "
                   "{ 'retrieved_ids': ['uuid1', 'uuid2'], 'source': 'cache/rag', 'latency': 0.5 }"
    )
    created_at: datetime = Field(
        default_factory=datetime.utcnow,
        description="Creation timestamp"
    )

    model_config = ConfigDict(
        json_encoders={
            uuid.UUID: str
        }
    )

    def to_orm(self):
        """Convert Pydantic model to ORM instance."""
        from ..orm_models.chat import ChatMessageORM
        return ChatMessageORM(
            id=self.id,
            session_id=self.session_id,
            role=self.role,
            content=self.content,
            context_data=self.context_data,
            created_at=self.created_at,
        )

    @classmethod
    def from_orm(cls, orm_instance):
        """Create Pydantic model from ORM instance."""
        return cls(
            id=orm_instance.id,
            session_id=orm_instance.session_id,
            role=orm_instance.role,
            content=orm_instance.content,
            context_data=orm_instance.context_data or {},
            created_at=orm_instance.created_at,
        )


class Feedback(BaseModel):
    """Pydantic model for feedbacks table."""

    id: uuid.UUID = Field(
        default_factory=uuid.uuid4,
        description="Primary key"
    )
    message_id: uuid.UUID = Field(
        ...,
        description="Reference to chat_messages.id"
    )
    is_positive: bool = Field(
        ...,
        description="Whether feedback is positive"
    )
    comment: str | None = Field(
        None,
        description="Optional comment"
    )
    status: FeedbackStatus = Field(
        default=FeedbackStatus.PENDING,
        description="Feedback status"
    )
    reviewed_by: uuid.UUID | None = Field(
        None,
        description="User who reviewed this feedback"
    )
    created_at: datetime = Field(
        default_factory=datetime.utcnow,
        description="Creation timestamp"
    )

    model_config = ConfigDict(
        use_enum_values=True,
        json_encoders={
            uuid.UUID: str
        }
    )

    def to_orm(self):
        """Convert Pydantic model to ORM instance."""
        from ..orm_models.chat import FeedbackORM
        return FeedbackORM(
            id=self.id,
            message_id=self.message_id,
            is_positive=self.is_positive,
            comment=self.comment,
            status=self.status,
            reviewed_by=self.reviewed_by,
            created_at=self.created_at,
        )

    @classmethod
    def from_orm(cls, orm_instance):
        """Create Pydantic model from ORM instance."""
        return cls(
            id=orm_instance.id,
            message_id=orm_instance.message_id,
            is_positive=orm_instance.is_positive,
            comment=orm_instance.comment,
            status=orm_instance.status,
            reviewed_by=orm_instance.reviewed_by,
            created_at=orm_instance.created_at,
        )

