"""
Pydantic models for Knowledge Mining.

These models represent SuggestedFAQ entities.
"""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any, List, Optional

from pydantic import BaseModel, ConfigDict, Field

from .enums import FAQStatus


class SuggestedFAQ(BaseModel):
    """
    Pydantic model for suggested_faqs table.
    
    Stores suggested FAQs extracted from chat message clusters.
    """

    id: uuid.UUID = Field(
        default_factory=uuid.uuid4,
        description="Primary key"
    )

    # Thông tin Cluster (Cụm câu hỏi)
    cluster_vector: List[float] | None = Field(
        None,
        description="Vector trung tâm để check trùng"
    )
    representative_question: str = Field(
        ...,
        description="Câu hỏi hiển thị cho Admin"
    )
    frequency: int = Field(
        default=1,
        description="Số lượng người hỏi"
    )

    # Nội dung đề xuất
    answer_draft: str | None = Field(
        None,
        description="Câu trả lời mẫu (Admin sẽ sửa)"
    )
    source_node_ids: dict[str, Any] | None = Field(
        None,
        description="Danh sách ID chunk nguồn (để Invalidation)"
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

    def to_orm(self):
        """Convert Pydantic model to ORM instance."""
        from ..orm_models.mining import SuggestedFAQORM
        return SuggestedFAQORM(
            id=self.id,
            cluster_vector=self.cluster_vector,
            representative_question=self.representative_question,
            frequency=self.frequency,
            answer_draft=self.answer_draft,
            source_node_ids=self.source_node_ids,
            status=self.status,
            created_at=self.created_at,
            updated_at=self.updated_at,
        )

    @classmethod
    def from_orm(cls, orm_instance):
        """Create Pydantic model from ORM instance."""
        return cls(
            id=orm_instance.id,
            cluster_vector=list(orm_instance.cluster_vector) if orm_instance.cluster_vector is not None else None,
            representative_question=orm_instance.representative_question,
            frequency=orm_instance.frequency,
            answer_draft=orm_instance.answer_draft,
            source_node_ids=orm_instance.source_node_ids,
            status=orm_instance.status,
            created_at=orm_instance.created_at,
            updated_at=orm_instance.updated_at,
        )

