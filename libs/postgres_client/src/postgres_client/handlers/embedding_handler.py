"""
CRUD handlers for Embedding models.

Handles EmbeddingStructure and EmbeddingContent operations.
"""

from __future__ import annotations

import uuid
from typing import Optional, List, Any
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from .base_handler import BaseHandler
from ..models.embeddings import EmbeddingStructure, EmbeddingContent
from ..orm_models.knowledge import EmbeddingStructureORM, EmbeddingContentORM


class EmbeddingStructureHandler(BaseHandler[EmbeddingStructure, EmbeddingStructureORM]):
    """Handler for EmbeddingStructure CRUD operations."""

    def __init__(self, session: AsyncSession):
        """Initialize EmbeddingStructure handler."""
        super().__init__(session, EmbeddingStructure, EmbeddingStructureORM)

    async def get_by_node_id(self, node_id: uuid.UUID) -> Optional[EmbeddingStructure]:
        """
        Get embedding structure by node ID.
        
        Args:
            node_id: Knowledge node ID
            
        Returns:
            EmbeddingStructure instance or None if not found
        """
        return await self.get_by_id(node_id)

    async def get_with_vectors(
        self,
        limit: Optional[int] = None,
        order_by: Optional[Any] = None,
    ) -> List[EmbeddingStructure]:
        """
        Get embedding structures that have vectors.
        
        Args:
            limit: Maximum number of records to return
            order_by: Column to order by
            
        Returns:
            List of EmbeddingStructure instances with vectors
        """
        stmt = select(self.orm_class).where(
            EmbeddingStructureORM.vector.isnot(None)
        )
        
        if order_by is not None:
            stmt = stmt.order_by(order_by)
        
        if limit is not None:
            stmt = stmt.limit(limit)
        
        result = await self.session.execute(stmt)
        orm_instances = result.scalars().all()
        
        return [
            self.pydantic_class.from_orm(orm_instance)
            for orm_instance in orm_instances
        ]

    async def update_vector(
        self,
        node_id: uuid.UUID,
        vector: List[float],
    ) -> Optional[EmbeddingStructure]:
        """
        Update embedding vector for a node.
        
        Args:
            node_id: Knowledge node ID
            vector: Vector embedding (1024 dimensions)
            
        Returns:
            Updated EmbeddingStructure instance or None if not found
        """
        return await self.update(node_id, {"vector": vector})

    async def update_summary(
        self,
        node_id: uuid.UUID,
        summary_text: str,
    ) -> Optional[EmbeddingStructure]:
        """
        Update summary text for a node.
        
        Args:
            node_id: Knowledge node ID
            summary_text: Summary text of the branch content
            
        Returns:
            Updated EmbeddingStructure instance or None if not found
        """
        return await self.update(node_id, {"summary_text": summary_text})


class EmbeddingContentHandler(BaseHandler[EmbeddingContent, EmbeddingContentORM]):
    """Handler for EmbeddingContent CRUD operations."""

    def __init__(self, session: AsyncSession):
        """Initialize EmbeddingContent handler."""
        super().__init__(session, EmbeddingContent, EmbeddingContentORM)

    async def get_by_node_id(self, node_id: uuid.UUID) -> Optional[EmbeddingContent]:
        """
        Get embedding content by node ID.
        
        Args:
            node_id: Knowledge node ID
            
        Returns:
            EmbeddingContent instance or None if not found
        """
        return await self.get_by_id(node_id)

    async def get_with_vectors(
        self,
        limit: Optional[int] = None,
        order_by: Optional[Any] = None,
    ) -> List[EmbeddingContent]:
        """
        Get embedding contents that have vectors.
        
        Args:
            limit: Maximum number of records to return
            order_by: Column to order by
            
        Returns:
            List of EmbeddingContent instances with vectors
        """
        stmt = select(self.orm_class).where(
            EmbeddingContentORM.vector.isnot(None)
        )
        
        if order_by is not None:
            stmt = stmt.order_by(order_by)
        
        if limit is not None:
            stmt = stmt.limit(limit)
        
        result = await self.session.execute(stmt)
        orm_instances = result.scalars().all()
        
        return [
            self.pydantic_class.from_orm(orm_instance)
            for orm_instance in orm_instances
        ]

    async def update_vector(
        self,
        node_id: uuid.UUID,
        vector: List[float],
    ) -> Optional[EmbeddingContent]:
        """
        Update embedding vector for a node.
        
        Args:
            node_id: Knowledge node ID
            vector: Vector embedding (1024 dimensions)
            
        Returns:
            Updated EmbeddingContent instance or None if not found
        """
        return await self.update(node_id, {"vector": vector})

    async def update_chunk_text(
        self,
        node_id: uuid.UUID,
        chunk_text: str,
    ) -> Optional[EmbeddingContent]:
        """
        Update chunk text for a node.
        
        Args:
            node_id: Knowledge node ID
            chunk_text: Original content (Admin edits here)
            
        Returns:
            Updated EmbeddingContent instance or None if not found
        """
        return await self.update(node_id, {"chunk_text": chunk_text})


# Convenience class that groups all embedding handlers
class EmbeddingHandler:
    """
    Convenience class grouping all embedding-related handlers.
    
    Usage:
        >>> handler = EmbeddingHandler(session)
        >>> structure = await handler.structures.create(EmbeddingStructure(...))
        >>> content = await handler.contents.create(EmbeddingContent(...))
    """

    def __init__(self, session: AsyncSession):
        """Initialize all embedding handlers."""
        self.session = session
        self.structures = EmbeddingStructureHandler(session)
        self.contents = EmbeddingContentHandler(session)

