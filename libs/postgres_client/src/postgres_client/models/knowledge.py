"""Base KnowledgeNode model matching the SQL schema."""

from __future__ import annotations

import uuid
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .enums import NodeType


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

    # Classification: Node type & tree levels
    node_type: NodeType = Field(
        ...,
        description="Node type used by AI Router and UI (STRUCTURE, PARAGRAPH, IMAGE, TABLE, etc.)",
    )
    level_down: int = Field(
        default=1,
        ge=0,
        description="Tree depth from root (1 = root, 0 = detail nodes temporarily).",
    )
    level_up: int = Field(
        default=1,
        ge=0,
        description=(
            "Tree distance from this node up to the farthest detail node on the path "
            "from root to detail. Example for a linear path of 4 nodes: "
            "root=(1,4), next=(2,3), next=(3,2), detail=(4,1). "
            "Detail nodes temporarily use (0,0)."
        ),
    )
    
    @model_validator(mode="after")
    def validate_levels(self) -> "KnowledgeNode":
        """
        Basic invariants for level_down/level_up on a single node.

        Global tree-level consistency (constant level_down + level_up along a root->detail
        path) is validated in tree utilities like the ingest workflow.
        
        Note: Detail nodes (PARAGRAPH, IMAGE, TABLE) can temporarily have level_down=0 and level_up=0.
        """
        if self.level_down < 0:
            raise ValueError(f"level_down must be >= 0, got {self.level_down}")
        if self.level_up < 0:
            raise ValueError(f"level_up must be >= 0, got {self.level_up}")
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

    model_config = ConfigDict(
        use_enum_values=True,
        json_encoders={
            uuid.UUID: str
        }
    )

    def to_orm(self):
        """
        Convert Pydantic model to ORM instance.
        
        This method uses the converter utility to ensure consistency.
        When Pydantic model structure changes, update utils/knowledge_node_converter.py.
        """
        from ..utils.knowledge_node_converter import pydantic_to_orm
        return pydantic_to_orm(self)

    @classmethod
    def from_orm(cls, orm_instance):
        """
        Create Pydantic model from ORM instance.
        
        This method uses the converter utility to ensure consistency.
        When Pydantic model structure changes, update utils/knowledge_node_converter.py.
        """
        from ..utils.knowledge_node_converter import orm_to_pydantic
        return orm_to_pydantic(orm_instance)

