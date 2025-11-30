"""
ORM classes for Knowledge Mining tables.

Tables:
- suggested_faqs: Suggested FAQs from chat message clustering
"""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any, List, Optional

from pgvector.sqlalchemy import Vector
from sqlalchemy import (
    DateTime,
    Integer,
    Text,
    UUID,
    Enum as SQLEnum,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column

from ..models.enums import FAQStatus
from .base import Base


class SuggestedFAQORM(Base):
    """
    ORM class for suggested_faqs table.
    
    Stores suggested FAQs extracted from chat message clusters.
    """

    __tablename__ = 'suggested_faqs'

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
    )

    # Thông tin Cluster (Cụm câu hỏi)
    cluster_vector: Mapped[Optional[List[float]]] = mapped_column(
        Vector(1024),
        nullable=True,
        comment="Vector trung tâm để check trùng",
    )
    representative_question: Mapped[str] = mapped_column(
        Text,
        nullable=False,
        comment="Câu hỏi hiển thị cho Admin",
    )
    frequency: Mapped[int] = mapped_column(
        Integer,
        default=1,
        comment="Số lượng người hỏi",
    )

    # Nội dung đề xuất
    answer_draft: Mapped[str | None] = mapped_column(
        Text,
        nullable=True,
        comment="Câu trả lời mẫu (Admin sẽ sửa)",
    )
    source_node_ids: Mapped[dict[str, Any] | None] = mapped_column(
        JSONB,
        nullable=True,
        comment="Danh sách ID chunk nguồn (để Invalidation)",
    )

    status: Mapped[FAQStatus] = mapped_column(
        SQLEnum(FAQStatus, name='faq_status'),
        default=FAQStatus.PENDING,
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

