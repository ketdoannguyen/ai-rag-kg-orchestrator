"""
Pydantic models for System Administration.

These models represent User and DataSource entities.
"""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from .enums import SourceType, UserRole


class User(BaseModel):
    """Pydantic model for users table."""

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

    def to_orm(self):
        """Convert Pydantic model to ORM instance."""
        from ..orm_models.system_admin import UserORM
        return UserORM(
            id=self.id,
            username=self.username,
            hashed_password=self.hashed_password,
            full_name=self.full_name,
            role=self.role,
            is_active=self.is_active,
            created_at=self.created_at,
        )

    @classmethod
    def from_orm(cls, orm_instance):
        """Create Pydantic model from ORM instance."""
        return cls(
            id=orm_instance.id,
            username=orm_instance.username,
            hashed_password=orm_instance.hashed_password,
            full_name=orm_instance.full_name,
            role=orm_instance.role,
            is_active=orm_instance.is_active,
            created_at=orm_instance.created_at,
        )


class DataSource(BaseModel):
    """Pydantic model for data_sources table."""

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

    def to_orm(self):
        """Convert Pydantic model to ORM instance."""
        from ..orm_models.system_admin import DataSourceORM
        return DataSourceORM(
            id=self.id,
            name=self.name,
            type=self.type,
            config=self.config,
            hash_content=self.hash_content,
            status=self.status,
            created_by=self.created_by,
            created_at=self.created_at,
            updated_at=self.updated_at,
        )

    @classmethod
    def from_orm(cls, orm_instance):
        """Create Pydantic model from ORM instance."""
        return cls(
            id=orm_instance.id,
            name=orm_instance.name,
            type=orm_instance.type,
            config=orm_instance.config or {},
            hash_content=orm_instance.hash_content,
            status=orm_instance.status,
            created_by=orm_instance.created_by,
            created_at=orm_instance.created_at,
            updated_at=orm_instance.updated_at,
        )

