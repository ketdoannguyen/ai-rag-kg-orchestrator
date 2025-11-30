"""
Pydantic models for Embedding tables.

These models represent EmbeddingStructure and EmbeddingContent entities.
"""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import List, Optional

from pydantic import BaseModel, ConfigDict, Field


class EmbeddingStructure(BaseModel):
    """
    Pydantic model for embeddings_structure table.
    
    Lớp 1: Dành cho MUC (Topic/Subtopic)
    Chỉ chứa nodes có semantic_level là CORPUS, TOPIC, SUBTOPIC
    """

    node_id: uuid.UUID = Field(
        ...,
        description="Primary key - references knowledge_nodes.id"
    )
    summary_text: str | None = Field(
        None,
        description="Summary text of the branch content"
    )
    vector: List[float] | None = Field(
        None,
        description="Vector embedding (1024 dimensions)"
    )

    model_config = ConfigDict(
        json_encoders={
            uuid.UUID: str
        }
    )

    def to_orm(self):
        """Convert Pydantic model to ORM instance."""
        from ..orm_models.knowledge import EmbeddingStructureORM
        return EmbeddingStructureORM(
            node_id=self.node_id,
            summary_text=self.summary_text,
            vector=self.vector,
        )

    @classmethod
    def from_orm(cls, orm_instance):
        """Create Pydantic model from ORM instance."""
        return cls(
            node_id=orm_instance.node_id,
            summary_text=orm_instance.summary_text,
            vector=list(orm_instance.vector) if orm_instance.vector is not None else None,
        )


class EmbeddingContent(BaseModel):
    """
    Pydantic model for embeddings_content table.
    
    Lớp 2: Dành cho KHOI (Detail/Chunk)
    Chỉ chứa nodes có semantic_level là DETAIL
    """

    node_id: uuid.UUID = Field(
        ...,
        description="Primary key - references knowledge_nodes.id"
    )
    chunk_text: str = Field(
        ...,
        description="Original content (Admin edits here)"
    )
    vector: List[float] | None = Field(
        None,
        description="Vector embedding (1024 dimensions)"
    )
    updated_at: datetime = Field(
        default_factory=datetime.utcnow,
        description="Last update timestamp"
    )

    model_config = ConfigDict(
        json_encoders={
            uuid.UUID: str
        }
    )

    def to_orm(self):
        """Convert Pydantic model to ORM instance."""
        from ..orm_models.knowledge import EmbeddingContentORM
        return EmbeddingContentORM(
            node_id=self.node_id,
            chunk_text=self.chunk_text,
            vector=self.vector,
            updated_at=self.updated_at,
        )

    @classmethod
    def from_orm(cls, orm_instance):
        """Create Pydantic model from ORM instance."""
        return cls(
            node_id=orm_instance.node_id,
            chunk_text=orm_instance.chunk_text,
            vector=list(orm_instance.vector) if orm_instance.vector is not None else None,
            updated_at=orm_instance.updated_at,
        )

