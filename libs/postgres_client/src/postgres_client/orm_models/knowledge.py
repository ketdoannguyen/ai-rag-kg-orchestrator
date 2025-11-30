"""
ORM classes for Knowledge Core tables.

Tables:
- knowledge_nodes: Core knowledge graph nodes
- embeddings_structure: Structure embeddings (CORPUS, TOPIC, SUBTOPIC)
- embeddings_content: Content embeddings (DETAIL chunks)
"""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any, List, Optional

from pgvector.sqlalchemy import Vector
from sqlalchemy import (
    DateTime,
    ForeignKey,
    Integer,
    Text,
    UUID,
    Enum as SQLEnum,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column, relationship, backref

from ..models.enums import NodeType, SemanticLevel
from .base import Base


class KnowledgeNodeORM(Base):
    """
    ORM class for knowledge_nodes table.
    
    This is the core table for the knowledge graph.
    """

    __tablename__ = 'knowledge_nodes'

    # Core Columns
    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
    )
    source_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey('data_sources.id', ondelete='CASCADE'),
        nullable=False,
    )
    parent_id: Mapped[Optional[uuid.UUID]] = mapped_column(
        ForeignKey('knowledge_nodes.id', ondelete='CASCADE'),
        nullable=True,
    )

    # Classification
    semantic_level: Mapped[SemanticLevel] = mapped_column(
        SQLEnum(SemanticLevel, name='semantic_level_enum'),
        nullable=False,
    )
    physical_level: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        default=1,
    )

    # Flexible Data (JSONB fields store Pydantic model data)
    attributes: Mapped[dict[str, Any]] = mapped_column(
        JSONB,
        server_default='{}',
    )
    # Note: 'metadata' is reserved by SQLAlchemy, so we use 'node_metadata' as attribute name
    # but keep the database column name as 'metadata'
    node_metadata: Mapped[dict[str, Any]] = mapped_column(
        JSONB,
        name='metadata',  # Database column name remains 'metadata'
        server_default='{}',
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=datetime.utcnow,
    )

    # Relationships
    children = relationship(
        "KnowledgeNodeORM",
        backref=backref("parent", remote_side=[id]),
        foreign_keys=[parent_id],
    )
    source = relationship(
        "DataSourceORM",
        back_populates="nodes",
    )

    # One-to-One Relationships (Embedding tách bảng)
    structure_embedding = relationship(
        "EmbeddingStructureORM",
        uselist=False,
        back_populates="node",
        cascade="all, delete-orphan",
    )
    content_embedding = relationship(
        "EmbeddingContentORM",
        uselist=False,
        back_populates="node",
        cascade="all, delete-orphan",
    )



class EmbeddingStructureORM(Base):
    """
    ORM class for embeddings_structure table.
    
    Lớp 1: Dành cho MUC (Topic/Subtopic)
    Chỉ chứa nodes có semantic_level là CORPUS, TOPIC, SUBTOPIC
    """

    __tablename__ = 'embeddings_structure'

    node_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey('knowledge_nodes.id', ondelete='CASCADE'),
        primary_key=True,
    )
    summary_text: Mapped[Optional[str]] = mapped_column(Text)
    vector: Mapped[Optional[List[float]]] = mapped_column(Vector(1024))

    node = relationship(
        "KnowledgeNodeORM",
        back_populates="structure_embedding",
    )


class EmbeddingContentORM(Base):
    """
    ORM class for embeddings_content table.
    
    Lớp 2: Dành cho KHOI (Detail/Chunk)
    Chỉ chứa nodes có semantic_level là DETAIL
    """

    __tablename__ = 'embeddings_content'

    node_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey('knowledge_nodes.id', ondelete='CASCADE'),
        primary_key=True,
    )
    chunk_text: Mapped[str] = mapped_column(Text, nullable=False)
    vector: Mapped[Optional[List[float]]] = mapped_column(Vector(1024))
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=datetime.utcnow,
        onupdate=datetime.utcnow,
    )

    node = relationship(
        "KnowledgeNodeORM",
        back_populates="content_embedding",
    )

