"""
CRUD handlers for Mining models.

Handles SuggestedFAQ operations.
"""

from __future__ import annotations

import uuid
from typing import Optional, List, Any
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from .base_handler import BaseHandler
from ..models.mining import SuggestedFAQ
from ..orm_models.mining import SuggestedFAQORM
from ..models.enums import FAQStatus


class SuggestedFAQHandler(BaseHandler[SuggestedFAQ, SuggestedFAQORM]):
    """Handler for SuggestedFAQ CRUD operations."""

    def __init__(self, session: AsyncSession):
        """Initialize SuggestedFAQ handler."""
        super().__init__(session, SuggestedFAQ, SuggestedFAQORM)

    async def get_by_status(
        self,
        status: FAQStatus,
        limit: Optional[int] = None,
        order_by: Optional[Any] = None,
    ) -> List[SuggestedFAQ]:
        """
        Get suggested FAQs by status.
        
        Args:
            status: FAQ status (pending, approved, rejected, archived)
            limit: Maximum number of FAQs to return
            order_by: Column to order by (default: created_at desc)
            
        Returns:
            List of SuggestedFAQ instances
        """
        if order_by is None:
            order_by = SuggestedFAQORM.created_at.desc()
        
        return await self.get_by_filter(
            {"status": status},
            limit=limit,
            order_by=order_by,
        )

    async def get_pending(
        self,
        limit: Optional[int] = None,
    ) -> List[SuggestedFAQ]:
        """
        Get all pending suggested FAQs.
        
        Args:
            limit: Maximum number of FAQs to return
            
        Returns:
            List of pending SuggestedFAQ instances
        """
        return await self.get_by_status(FAQStatus.PENDING, limit=limit)

    async def get_approved(
        self,
        limit: Optional[int] = None,
    ) -> List[SuggestedFAQ]:
        """
        Get all approved suggested FAQs.
        
        Args:
            limit: Maximum number of FAQs to return
            
        Returns:
            List of approved SuggestedFAQ instances
        """
        return await self.get_by_status(FAQStatus.APPROVED, limit=limit)

    async def get_by_frequency(
        self,
        min_frequency: int = 1,
        limit: Optional[int] = None,
        order_by: Optional[Any] = None,
    ) -> List[SuggestedFAQ]:
        """
        Get suggested FAQs by minimum frequency.
        
        Args:
            min_frequency: Minimum frequency (number of people who asked)
            limit: Maximum number of FAQs to return
            order_by: Column to order by (default: frequency desc)
            
        Returns:
            List of SuggestedFAQ instances
        """
        if order_by is None:
            order_by = SuggestedFAQORM.frequency.desc()
        
        stmt = select(self.orm_class).where(
            SuggestedFAQORM.frequency >= min_frequency
        )
        
        stmt = stmt.order_by(order_by)
        
        if limit is not None:
            stmt = stmt.limit(limit)
        
        result = await self.session.execute(stmt)
        orm_instances = result.scalars().all()
        
        return [
            self.pydantic_class.from_orm(orm_instance)
            for orm_instance in orm_instances
        ]

    async def approve(self, faq_id: uuid.UUID) -> Optional[SuggestedFAQ]:
        """
        Approve a suggested FAQ.
        
        Args:
            faq_id: Suggested FAQ ID
            
        Returns:
            Updated SuggestedFAQ instance or None if not found
        """
        return await self.update(faq_id, {"status": FAQStatus.APPROVED})

    async def reject(self, faq_id: uuid.UUID) -> Optional[SuggestedFAQ]:
        """
        Reject a suggested FAQ.
        
        Args:
            faq_id: Suggested FAQ ID
            
        Returns:
            Updated SuggestedFAQ instance or None if not found
        """
        return await self.update(faq_id, {"status": FAQStatus.REJECTED})

    async def archive(self, faq_id: uuid.UUID) -> Optional[SuggestedFAQ]:
        """
        Archive a suggested FAQ.
        
        Args:
            faq_id: Suggested FAQ ID
            
        Returns:
            Updated SuggestedFAQ instance or None if not found
        """
        return await self.update(faq_id, {"status": FAQStatus.ARCHIVED})

    async def increment_frequency(self, faq_id: uuid.UUID) -> Optional[SuggestedFAQ]:
        """
        Increment frequency count for a suggested FAQ.
        
        Args:
            faq_id: Suggested FAQ ID
            
        Returns:
            Updated SuggestedFAQ instance or None if not found
        """
        orm_instance = await self.session.get(self.orm_class, faq_id)
        if orm_instance is None:
            return None
        
        orm_instance.frequency += 1
        await self.session.commit()
        await self.session.refresh(orm_instance)
        
        return self.pydantic_class.from_orm(orm_instance)

    async def update_answer_draft(
        self,
        faq_id: uuid.UUID,
        answer_draft: str,
    ) -> Optional[SuggestedFAQ]:
        """
        Update answer draft for a suggested FAQ.
        
        Args:
            faq_id: Suggested FAQ ID
            answer_draft: Answer draft text (Admin will edit)
            
        Returns:
            Updated SuggestedFAQ instance or None if not found
        """
        return await self.update(faq_id, {"answer_draft": answer_draft})


# Convenience class that groups all mining handlers
class MiningHandler:
    """
    Convenience class grouping all mining-related handlers.
    
    Usage:
        >>> handler = MiningHandler(session)
        >>> faq = await handler.faqs.create(SuggestedFAQ(...))
    """

    def __init__(self, session: AsyncSession):
        """Initialize all mining handlers."""
        self.session = session
        self.faqs = SuggestedFAQHandler(session)

