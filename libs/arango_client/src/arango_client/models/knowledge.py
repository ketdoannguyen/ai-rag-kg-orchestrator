"""Base KnowledgeNode model for ArangoDB."""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .enums import NodeType


class KnowledgeNode(BaseModel):
    """
    Base knowledge node model for ArangoDB.

    This is the core model that all other node types inherit from.
    Contains ONLY shared fields. Type-specific fields (embedding, summary, text, children)
    are defined in subclasses (StructureNode, DetailNode).

    ArangoDB-specific fields:
    - _key: Document key (uses string UUID)
    - _id: Full document ID (collection/key)
    - _rev: Document revision

    Subclasses:
    - StructureNode: Has embedding, summary, children
    - DetailNode (ParagraphNode, ImageNode, TableNode): Has embedding, text
    """

    # Core fields
    id: uuid.UUID = Field(
        default_factory=uuid.uuid4, description="Primary key (GOLDEN KEY) - UUID"
    )
    source_id: uuid.UUID | None = Field(None, description="Reference to data_sources collection")
    parent_id: uuid.UUID | None = Field(None, description="Parent node ID for tree structure")

    # Classification: Node type & tree level
    node_type: NodeType = Field(
        ...,
        description="Node type used by AI Router and UI (STRUCTURE, PARAGRAPH, IMAGE, TABLE, etc.)",
    )
    level: int = Field(
        default=0,
        ge=0,
        description="Tree depth from root (1 = root, 0 = unset/detail nodes).",
    )
    domain: str | None = Field(
        default=None,
        description="Domain identifier for filtering (e.g., 'education', 'ecommerce')",
    )
    is_reviewed: bool = Field(
        default=False,
        description="Whether this node has been reviewed for deduplication. "
        "Nodes with is_reviewed=true are considered stable/canonical.",
    )

    @model_validator(mode="after")
    def validate_level(self) -> KnowledgeNode:
        """Basic invariant for level."""
        if self.level < 0:
            raise ValueError(f"level must be >= 0, got {self.level}")
        return self

    # Flexible data fields (stored as nested objects in ArangoDB)
    attributes: dict[str, Any] = Field(
        default_factory=dict, description="Intrinsic attributes (Title, Code, etc.)"
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Citation metadata (Page number, Original URL, Position, etc.)",
    )

    # Timestamps
    created_at: datetime = Field(default_factory=datetime.utcnow, description="Creation timestamp")
    updated_at: datetime = Field(
        default_factory=datetime.utcnow, description="Last update timestamp"
    )

    model_config = ConfigDict(use_enum_values=True, json_encoders={uuid.UUID: str})

    def to_document(self) -> dict[str, Any]:
        """
        Convert Pydantic model to ArangoDB document.

        Returns:
            Dictionary suitable for ArangoDB insertion.
            Uses _key for document ID (string UUID).
        """
        # Exclude children from serialization (they're transient)
        exclude_fields = {"children"} if hasattr(self, "children") else set()
        data = self.model_dump(exclude=exclude_fields)

        # Convert UUIDs to strings
        data["_key"] = str(data.pop("id"))
        if data.get("source_id"):
            data["source_id"] = str(data["source_id"])
        if data.get("parent_id"):
            data["parent_id"] = str(data["parent_id"])

        # Convert attributes/metadata if they're Pydantic models
        if hasattr(data.get("attributes"), "model_dump"):
            data["attributes"] = data["attributes"].model_dump()
        if hasattr(data.get("metadata"), "model_dump"):
            data["metadata"] = data["metadata"].model_dump()

        # Convert datetime to ISO format
        if data.get("created_at"):
            data["created_at"] = data["created_at"].isoformat()
        if data.get("updated_at"):
            data["updated_at"] = data["updated_at"].isoformat()

        return data

    @classmethod
    def from_document(cls, doc: dict[str, Any]) -> KnowledgeNode:
        """
        Create Pydantic model from ArangoDB document.

        Args:
            doc: ArangoDB document dictionary

        Returns:
            KnowledgeNode instance (or appropriate subclass)
        """
        # Convert _key back to id
        data = dict(doc)
        if "_key" in data:
            data["id"] = data.pop("_key")

        # Remove ArangoDB internal fields
        data.pop("_id", None)
        data.pop("_rev", None)

        # Parse datetime strings
        if isinstance(data.get("created_at"), str):
            data["created_at"] = datetime.fromisoformat(data["created_at"])
        if isinstance(data.get("updated_at"), str):
            data["updated_at"] = datetime.fromisoformat(data["updated_at"])

        # Initialize children as empty list if subclass has it
        if hasattr(cls, "children"):
            data["children"] = []

        return cls(**data)
