"""Pydantic models for Knowledge Mining in ArangoDB."""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any, List

from pydantic import BaseModel, ConfigDict, Field

from .enums import FAQStatus


class SuggestedFAQ(BaseModel):
    """
    Pydantic model for suggested_faqs collection.
    
    Stores suggested FAQs extracted from chat message clusters.
    """

    id: uuid.UUID = Field(
        default_factory=uuid.uuid4,
        description="Primary key"
    )

    # Cluster information
    cluster_vector: List[float] | None = Field(
        None,
        description="Cluster center vector for deduplication"
    )
    representative_question: str = Field(
        ...,
        description="Display question for Admin"
    )
    frequency: int = Field(
        default=1,
        description="Number of people who asked"
    )

    # Suggested content
    answer_draft: str | None = Field(
        None,
        description="Draft answer (Admin will edit)"
    )
    source_node_ids: dict[str, Any] | None = Field(
        None,
        description="List of source chunk IDs (for invalidation)"
    )

    status: FAQStatus = Field(
        default=FAQStatus.PENDING,
        description="FAQ status"
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
            "cluster_vector": self.cluster_vector,
            "representative_question": self.representative_question,
            "frequency": self.frequency,
            "answer_draft": self.answer_draft,
            "source_node_ids": self.source_node_ids,
            "status": self.status if isinstance(self.status, str) else self.status.value,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
        }

    @classmethod
    def from_document(cls, doc: dict[str, Any]) -> "SuggestedFAQ":
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
