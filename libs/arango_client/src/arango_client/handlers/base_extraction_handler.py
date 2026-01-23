"""Base extraction handler for common extraction logic.

Provides shared functionality for ContentExtractionHandler and StructureExtractionHandler:
- Relation allowance caching
- Review marking
- Basic setup
"""

from __future__ import annotations

import asyncio
from enum import Enum
from functools import partial
from typing import TYPE_CHECKING

from arango.database import StandardDatabase
from loguru import logger

if TYPE_CHECKING:
    from config.extraction import ExtractionSettings

    from ..models.ontology import RelationTier
    from .ontology_handler import OntologyHandler


class ExtractionDecision(str, Enum):
    """Decision from extraction process."""

    MERGE = "merge"
    CREATE_RELATION = "create_relation"
    KEEP_SEPARATE = "keep_separate"


class BaseExtractionHandler:
    """Base class for extraction handlers."""

    def __init__(
        self,
        db: StandardDatabase,
        settings: ExtractionSettings | None = None,
        ontology_handler: OntologyHandler | None = None,
        log_name: str = "extraction_handler",
    ) -> None:
        """Initialize base extraction handler.

        Args:
            db: ArangoDB database connection
            settings: Extraction settings (lazy loaded if not provided)
            ontology_handler: Handler for ontology operations (lazy loaded if not provided)
            log_name: Logger name binding
        """
        self.db = db
        self._settings = settings
        self._ontology_handler = ontology_handler
        self._allowed_relations_cache: frozenset[str] | None = None
        self._log = logger.bind(handler=log_name)

    @property
    def settings(self) -> ExtractionSettings:
        """Get extraction settings (lazy load)."""
        if self._settings is None:
            from config.extraction import get_extraction_settings

            self._settings = get_extraction_settings()
        return self._settings

    @property
    def ontology_handler(self) -> OntologyHandler:
        """Get ontology handler (lazy load)."""
        if self._ontology_handler is None:
            from .ontology_handler import OntologyHandler

            self._ontology_handler = OntologyHandler(self.db)
        return self._ontology_handler

    def decide_action(self, similarity_score: float) -> ExtractionDecision:
        """Determine action based on similarity score."""
        if similarity_score >= self.settings.EXTRACTION_AUTO_MERGE_THRESHOLD:
            return ExtractionDecision.MERGE
        elif similarity_score >= self.settings.EXTRACTION_LLM_THRESHOLD:
            return ExtractionDecision.CREATE_RELATION
        else:
            return ExtractionDecision.KEEP_SEPARATE

    def _get_allowed_relations_for_tier(
        self,
        tier: RelationTier,
        fallback_relations: frozenset[str],
    ) -> frozenset[str]:
        """Get allowed relations for a specific tier from database (cached).

        Uses dynamic query to OntologyHandler.
        Falls back to provided fallback_relations if database is unavailable.
        """
        if self._allowed_relations_cache is not None:
            return self._allowed_relations_cache

        try:
            relations = self.ontology_handler.get_by_tier(tier)
            if not relations:
                self._log.warning(
                    f"No {tier.value} relations in database, using fallback",
                    tier=tier.value,
                )
                return fallback_relations

            self._allowed_relations_cache = frozenset(r.id for r in relations)
            self._log.debug(
                "Loaded allowed relations from database",
                tier=tier.value,
                count=len(self._allowed_relations_cache),
            )
            return self._allowed_relations_cache
        except Exception as e:
            self._log.warning(
                "Failed to query ontology, using fallback",
                error=str(e),
                tier=tier.value,
            )
            return fallback_relations

    def invalidate_cache(self) -> None:
        """Invalidate the allowed relations cache.

        Call this after ontology updates to refresh the cache.
        """
        self._allowed_relations_cache = None

    async def mark_as_reviewed(self, node_ids: list[str], collection_name: str) -> int:
        """Mark nodes as reviewed for extraction.

        Args:
            node_ids: List of node keys to mark
            collection_name: Name of collection containing nodes

        Returns:
            Number of nodes updated
        """
        if not node_ids:
            return 0

        query = """
        FOR key IN @keys
            UPDATE { _key: key, is_reviewed: true } IN @@collection
            OPTIONS { ignoreErrors: true }
            RETURN 1
        """

        bind_vars = {
            "keys": node_ids,
            "@collection": collection_name,
        }

        cursor = await asyncio.to_thread(
            partial(self.db.aql.execute, query, bind_vars=bind_vars),
        )
        return sum(1 for _ in cursor)
