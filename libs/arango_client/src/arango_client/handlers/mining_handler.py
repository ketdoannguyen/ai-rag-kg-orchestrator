"""
Mining handler for ArangoDB.

Handles SuggestedFAQ operations.
All AQL queries are imported from the centralized aql/ module.
"""

from __future__ import annotations

import uuid

from arango.database import StandardDatabase

from ..aql import MINING_GET_BY_FREQUENCY, MINING_INCREMENT_FREQUENCY
from ..collections.schema import Collections
from ..models.enums import FAQStatus
from ..models.mining import SuggestedFAQ
from .base_handler import BaseHandler


class SuggestedFAQHandler(BaseHandler[SuggestedFAQ]):
    """Handler for SuggestedFAQ CRUD operations."""

    def __init__(self, db: StandardDatabase):
        """Initialize SuggestedFAQ handler."""
        super().__init__(db, SuggestedFAQ, Collections.SUGGESTED_FAQS)

    def get_by_status(
        self,
        status: FAQStatus | str,
        limit: int | None = None,
        offset: int | None = None,
    ) -> list[SuggestedFAQ]:
        """Get suggested FAQs by status."""
        status_value = status.value if isinstance(status, FAQStatus) else status
        return self.get_by_filter(
            {"status": status_value},
            limit=limit,
            offset=offset,
            sort_desc=True,
        )

    def get_pending(self, limit: int | None = None) -> list[SuggestedFAQ]:
        """Get all pending suggested FAQs."""
        return self.get_by_status(FAQStatus.PENDING, limit=limit)

    def get_approved(self, limit: int | None = None) -> list[SuggestedFAQ]:
        """Get all approved suggested FAQs."""
        return self.get_by_status(FAQStatus.APPROVED, limit=limit)

    def get_by_frequency(
        self,
        min_frequency: int = 1,
        limit: int | None = None,
    ) -> list[SuggestedFAQ]:
        """
        Get suggested FAQs by minimum frequency.

        Args:
            min_frequency: Minimum frequency (number of people who asked)
            limit: Maximum number of FAQs to return

        Returns:
            List of SuggestedFAQ instances
        """
        cursor = self.db.aql.execute(
            MINING_GET_BY_FREQUENCY,
            bind_vars={
                "@collection": self.collection_name,
                "min_frequency": min_frequency,
                "limit": limit or 100,
            },
        )
        return [SuggestedFAQ.from_document(doc) for doc in cursor]

    def approve(self, faq_id: uuid.UUID | str) -> SuggestedFAQ | None:
        """Approve a suggested FAQ."""
        return self.update(faq_id, {"status": FAQStatus.APPROVED.value})

    def reject(self, faq_id: uuid.UUID | str) -> SuggestedFAQ | None:
        """Reject a suggested FAQ."""
        return self.update(faq_id, {"status": FAQStatus.REJECTED.value})

    def archive(self, faq_id: uuid.UUID | str) -> SuggestedFAQ | None:
        """Archive a suggested FAQ."""
        return self.update(faq_id, {"status": FAQStatus.ARCHIVED.value})

    def increment_frequency(self, faq_id: uuid.UUID | str) -> SuggestedFAQ | None:
        """
        Increment frequency count for a suggested FAQ.

        Args:
            faq_id: Suggested FAQ ID

        Returns:
            Updated SuggestedFAQ instance or None if not found
        """
        cursor = self.db.aql.execute(
            MINING_INCREMENT_FREQUENCY,
            bind_vars={
                "key": str(faq_id),
                "@collection": self.collection_name,
            },
        )

        result = next(cursor, None)
        if result is None:
            return None

        return SuggestedFAQ.from_document(result)

    def update_answer_draft(
        self,
        faq_id: uuid.UUID | str,
        answer_draft: str,
    ) -> SuggestedFAQ | None:
        """Update answer draft for a suggested FAQ."""
        return self.update(faq_id, {"answer_draft": answer_draft})


class MiningHandler:
    """Convenience class grouping all mining-related handlers."""

    def __init__(self, db: StandardDatabase):
        """Initialize all mining handlers."""
        self.db = db
        self.faqs = SuggestedFAQHandler(db)
