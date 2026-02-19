"""Merged nodes log model for tracking merge operations.

This collection stores audit records of node merge operations,
enabling benchmark evaluation of merge quality without affecting
the main content_nodes schema.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator


class MergedNodeLog(BaseModel):
    """Record of a merge operation for benchmark evaluation.

    Stores full text of both source and target nodes at merge time,
    along with merge decision metadata. This enables:
    1. Benchmark evaluation of merge accuracy
    2. Audit trail for debugging
    3. Distinction between auto_merge vs llm_merge
    """

    # Source node info (node đã bị merge/xóa)
    source_id: str = Field(
        ...,
        description="Original full ID of merged node: content_nodes/xxx",
    )
    source_text: str = Field(
        ...,
        description="Full text of merged node (preserved for benchmark)",
    )
    source_type: str = Field(
        ...,
        description="Node type: paragraph, image, table",
    )
    source_parent_id: str | None = Field(
        default=None,
        description="Parent structure node ID for context",
    )
    source_source_id: str | None = Field(
        default=None,
        description="Data source ID (e.g., document origin)",
    )

    # Target node info (node được giữ lại)
    target_id: str = Field(
        ...,
        description="Full ID of target node: content_nodes/yyy",
    )
    target_text: str = Field(
        ...,
        description="Snapshot of target text at merge time",
    )

    # Merge decision info
    similarity: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Cosine similarity score at merge time",
    )

    @field_validator("similarity", mode="before")
    @classmethod
    def clamp_similarity(cls, v: float) -> float:
        """Clamp float32 cosine similarity epsilon to [0, 1].

        ArangoDB vector search uses float32 internally, which can produce
        values like 1.0000001192092896 due to IEEE 754 precision limits.
        Tolerance of 1e-6 (~10x float32 machine epsilon) catches
        accumulation of rounding errors while still detecting real bugs.
        """
        _EPSILON = 1e-6
        if v > 1.0 and v <= 1.0 + _EPSILON:
            return 1.0
        if v < 0.0 and v >= -_EPSILON:
            return 0.0
        return v

    merge_reason: Literal["auto_merge", "llm_merge"] = Field(
        ...,
        description="auto_merge (similarity >= threshold) or llm_merge (LLM decided)",
    )
    llm_reasoning: str | None = Field(
        default=None,
        description="LLM reasoning if merge_reason is llm_merge",
    )

    # Metadata
    merged_at: datetime = Field(
        default_factory=lambda: datetime.now(UTC),
        description="Timestamp of merge operation",
    )

    def to_document(self) -> dict[str, Any]:
        """Convert to ArangoDB document format."""
        doc = self.model_dump(mode="json")
        # Let ArangoDB auto-generate _key
        return doc

    @classmethod
    def from_document(cls, doc: dict[str, Any]) -> MergedNodeLog:
        """Create from ArangoDB document."""
        data = doc.copy()
        # Remove ArangoDB system fields
        data.pop("_key", None)
        data.pop("_id", None)
        data.pop("_rev", None)
        return cls(**data)
