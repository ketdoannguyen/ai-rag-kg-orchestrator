"""
ORM classes for System Administration tables.

Tables:
- users: User accounts and authentication
- data_sources: Source documents/files
"""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any

from sqlalchemy import (
    Boolean,
    DateTime,
    ForeignKey,
    String,
    Text,
    UUID,
    Enum as SQLEnum,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column, relationship

from ..models.enums import SourceType, UserRole
from .base import Base


class UserORM(Base):
    """ORM class for users table."""

    __tablename__ = 'users'

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
    )
    username: Mapped[str] = mapped_column(
        String(50),
        unique=True,
        nullable=False,
    )
    hashed_password: Mapped[str] = mapped_column(
        String(255),
        nullable=False,
    )
    full_name: Mapped[str | None] = mapped_column(
        String(100),
        nullable=True,
    )
    role: Mapped[UserRole] = mapped_column(
        SQLEnum(UserRole, name='user_role'),
        default=UserRole.EDITOR,
    )
    is_active: Mapped[bool] = mapped_column(
        Boolean,
        default=True,
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=datetime.utcnow,
    )

    # Relationships
    created_sources = relationship(
        "DataSourceORM",
        back_populates="creator",
        foreign_keys="DataSourceORM.created_by",
    )
    reviewed_feedbacks = relationship(
        "FeedbackORM",
        back_populates="reviewer",
        foreign_keys="FeedbackORM.reviewed_by",
    )


class DataSourceORM(Base):
    """ORM class for data_sources table."""

    __tablename__ = 'data_sources'

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
    )
    name: Mapped[str] = mapped_column(
        String(255),
        nullable=False,
    )
    type: Mapped[SourceType] = mapped_column(
        SQLEnum(SourceType, name='source_type'),
        nullable=False,
    )
    config: Mapped[dict[str, Any]] = mapped_column(
        JSONB,
        server_default='{}',
    )
    hash_content: Mapped[str | None] = mapped_column(
        String(64),
        nullable=True,
    )
    status: Mapped[str] = mapped_column(
        String(20),
        default='processing',
    )
    created_by: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey('users.id'),
        nullable=True,
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=datetime.utcnow,
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=datetime.utcnow,
        onupdate=datetime.utcnow,
    )

    # Relationships
    creator = relationship(
        "UserORM",
        foreign_keys=[created_by],
        back_populates="created_sources",
    )
    nodes = relationship(
        "KnowledgeNodeORM",
        back_populates="source",
        cascade="all, delete-orphan",
    )

