"""Pydantic models for System Administration in ArangoDB."""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from .enums import SourceType, UserRole


class User(BaseModel):
    """Pydantic model for users collection."""

    id: uuid.UUID = Field(
        default_factory=uuid.uuid4,
        description="Primary key"
    )
    username: str = Field(
        ...,
        max_length=50,
        description="Unique username"
    )
    hashed_password: str = Field(
        ...,
        max_length=255,
        description="Hashed password"
    )
    full_name: str | None = Field(
        None,
        max_length=100,
        description="Full name of the user"
    )
    role: UserRole = Field(
        default=UserRole.EDITOR,
        description="User role"
    )
    is_active: bool = Field(
        default=True,
        description="Whether the user is active"
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
            "username": self.username,
            "hashed_password": self.hashed_password,
            "full_name": self.full_name,
            "role": self.role if isinstance(self.role, str) else self.role.value,
            "is_active": self.is_active,
            "created_at": self.created_at.isoformat(),
        }

    @classmethod
    def from_document(cls, doc: dict[str, Any]) -> "User":
        """Create from ArangoDB document."""
        data = dict(doc)
        if "_key" in data:
            data["id"] = data.pop("_key")
        data.pop("_id", None)
        data.pop("_rev", None)
        
        if isinstance(data.get("created_at"), str):
            data["created_at"] = datetime.fromisoformat(data["created_at"])
        
        return cls(**data)


class DataSource(BaseModel):
    """Pydantic model for data_sources collection."""

    id: uuid.UUID = Field(
        default_factory=uuid.uuid4,
        description="Primary key"
    )
    name: str = Field(
        ...,
        max_length=255,
        description="Display name (e.g., 'Quy chế 2025.pdf')"
    )
    type: SourceType = Field(
        ...,
        description="Source type"
    )
    config: dict[str, Any] = Field(
        default_factory=dict,
        description="Technical configuration (minio_path, url, fanpage_id)"
    )
    hash_content: str | None = Field(
        None,
        max_length=64,
        description="MD5 Hash to avoid duplicate uploads"
    )
    status: str = Field(
        default='processing',
        max_length=20,
        description="Status: processing, active, error"
    )
    created_by: uuid.UUID | None = Field(
        None,
        description="User who created this source"
    )
    created_at: datetime = Field(
        default_factory=datetime.utcnow,
        description="Creation timestamp"
    )
    updated_at: datetime = Field(
        default_factory=datetime.utcnow,
        description="Last update timestamp"
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
            "name": self.name,
            "type": self.type if isinstance(self.type, str) else self.type.value,
            "config": self.config,
            "hash_content": self.hash_content,
            "status": self.status,
            "created_by": str(self.created_by) if self.created_by else None,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
        }

    @classmethod
    def from_document(cls, doc: dict[str, Any]) -> "DataSource":
        """Create from ArangoDB document."""
        data = dict(doc)
        if "_key" in data:
            data["id"] = data.pop("_key")
        data.pop("_id", None)
        data.pop("_rev", None)
        
        if isinstance(data.get("created_at"), str):
            data["created_at"] = datetime.fromisoformat(data["created_at"])
        if isinstance(data.get("updated_at"), str):
            data["updated_at"] = datetime.fromisoformat(data["updated_at"])
        
        return cls(**data)
