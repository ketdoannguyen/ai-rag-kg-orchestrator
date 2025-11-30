"""Base KnowledgeNode model matching the SQL schema."""

from __future__ import annotations

import uuid
from typing import Any

from pydantic import BaseModel, Field, model_validator

from .enums import SemanticLevel


class KnowledgeNode(BaseModel):
    """
    Base knowledge node model matching the knowledge_nodes table.
    
    This is the core model that all other node types inherit from.
    The attributes and metadata fields are JSONB in SQL and will be
    structured differently in derived classes.
    
    Derived classes should override attributes and metadata with
    specific Pydantic models.
    """

    id: uuid.UUID = Field(
        default_factory=uuid.uuid4,
        description="Primary key (GOLDEN KEY) - UUID"
    )
    source_id: uuid.UUID | None = Field(
        None,
        description="Reference to data_sources table"
    )
    parent_id: uuid.UUID | None = Field(
        None,
        description="Parent node ID for tree structure"
    )

    # Classification: Structure & Semantics
    semantic_level: SemanticLevel = Field(
        ...,
        description="Semantic level used by AI Router (TOPIC vs DETAIL)"
    )
    physical_level: int = Field(
        default=1,
        description="Physical level for UI indentation. For DETAIL nodes, must be -1. For structure nodes (CORPUS, TOPIC, SUBTOPIC), must be >= 1."
    )
    
    @model_validator(mode="after")
    def validate_physical_level(self) -> "KnowledgeNode":
        """
        Validate that physical_level matches semantic_level:
        - DETAIL nodes must have physical_level = -1
        - Structure nodes (CORPUS, TOPIC, SUBTOPIC) must have physical_level >= 1
        """
        if self.semantic_level == SemanticLevel.DETAIL:
            if self.physical_level != -1:
                raise ValueError(
                    f"DETAIL nodes must have physical_level = -1, got {self.physical_level}"
                )
        else:
            # Structure nodes (CORPUS, TOPIC, SUBTOPIC)
            if self.physical_level < 1:
                raise ValueError(
                    f"Structure nodes (CORPUS, TOPIC, SUBTOPIC) must have physical_level >= 1, "
                    f"got {self.physical_level}"
                )
        return self

    # Flexible data fields (JSONB in SQL)
    # These will be overridden in derived classes with specific models
    attributes: dict[str, Any] = Field(
        default_factory=dict,
        description="Intrinsic attributes (Title, Code, etc.)"
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Citation metadata (Page number, Original URL, Position, etc.)"
    )
    
    children: list[KnowledgeNode] = Field(
        default_factory=list,
        description="Nested nodes (paragraphs, images, tables, etc.)",
    )

    class Config:
        """Pydantic configuration."""

        use_enum_values = True
        json_encoders = {
            uuid.UUID: str
        }

