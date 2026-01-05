"""Pydantic models for Chat Operations in ArangoDB."""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from .enums import FeedbackStatus


class ChatSession(BaseModel):
    """Pydantic model for chat_sessions collection."""

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

    def to_document(self) -> dict[str, Any]:
        """Convert to ArangoDB document."""
        return {
            "_key": str(self.id),
            "user_identifier": self.user_identifier,
            "started_at": self.started_at.isoformat(),
        }

    @classmethod
    def from_document(cls, doc: dict[str, Any]) -> "ChatSession":
        """Create from ArangoDB document."""
        data = dict(doc)
        if "_key" in data:
            data["id"] = data.pop("_key")
        data.pop("_id", None)
        data.pop("_rev", None)
        
        if isinstance(data.get("started_at"), str):
            data["started_at"] = datetime.fromisoformat(data["started_at"])
        
        return cls(**data)


class ChatMessage(BaseModel):
    """Pydantic model for chat_messages collection."""

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

    def to_document(self) -> dict[str, Any]:
        """Convert to ArangoDB document."""
        return {
            "_key": str(self.id),
            "session_id": str(self.session_id),
            "role": self.role,
            "content": self.content,
            "context_data": self.context_data,
            "created_at": self.created_at.isoformat(),
        }

    @classmethod
    def from_document(cls, doc: dict[str, Any]) -> "ChatMessage":
        """Create from ArangoDB document."""
        data = dict(doc)
        if "_key" in data:
            data["id"] = data.pop("_key")
        data.pop("_id", None)
        data.pop("_rev", None)
        
        if isinstance(data.get("created_at"), str):
            data["created_at"] = datetime.fromisoformat(data["created_at"])
        
        return cls(**data)


class Feedback(BaseModel):
    """Pydantic model for feedbacks collection."""

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

    def to_document(self) -> dict[str, Any]:
        """Convert to ArangoDB document."""
        return {
            "_key": str(self.id),
            "message_id": str(self.message_id),
            "is_positive": self.is_positive,
            "comment": self.comment,
            "status": self.status if isinstance(self.status, str) else self.status.value,
            "reviewed_by": str(self.reviewed_by) if self.reviewed_by else None,
            "created_at": self.created_at.isoformat(),
        }

    @classmethod
    def from_document(cls, doc: dict[str, Any]) -> "Feedback":
        """Create from ArangoDB document."""
        data = dict(doc)
        if "_key" in data:
            data["id"] = data.pop("_key")
        data.pop("_id", None)
        data.pop("_rev", None)
        
        if isinstance(data.get("created_at"), str):
            data["created_at"] = datetime.fromisoformat(data["created_at"])
        
        return cls(**data)
