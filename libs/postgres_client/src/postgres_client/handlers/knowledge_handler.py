"""
CRUD handlers for Knowledge models.

Handles KnowledgeNode and its subtypes (ParagraphNode, ImageNode, TableNode,
StructureNode) operations.

Note: KnowledgeNode uses special converter functions (pydantic_to_orm, orm_to_pydantic)
instead of the standard to_orm()/from_orm() methods.
"""

from __future__ import annotations

import uuid
from typing import Optional, List, Any, Type, Union
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func

from .base_handler import BaseHandler
from ..models.knowledge import KnowledgeNode
from ..models.detail_nodes import ParagraphNode, ImageNode, TableNode
from ..models.structure_nodes import StructureNode
from ..orm_models.knowledge import KnowledgeNodeORM
from ..models.enums import NodeType
from ..utils.knowledge_node_converter import pydantic_to_orm, orm_to_pydantic


class KnowledgeNodeHandler:
    """
    Handler for KnowledgeNode CRUD operations.
    
    This handler uses special converter functions for KnowledgeNode because
    it has multiple subtypes and complex conversion logic.
    """

    def __init__(self, session: AsyncSession):
        """Initialize KnowledgeNode handler."""
        self.session = session
        self.orm_class = KnowledgeNodeORM

    async def create(
        self,
        pydantic_model: Union[
            KnowledgeNode,
            ParagraphNode,
            ImageNode,
            TableNode,
            StructureNode,
        ],
    ) -> Union[
        KnowledgeNode,
        ParagraphNode,
        ImageNode,
        TableNode,
        StructureNode,
    ]:
        """
        Create a new knowledge node from Pydantic model.
        
        Args:
            pydantic_model: Validated KnowledgeNode Pydantic model instance
                           (can be any subtype: ParagraphNode, StructureNode, etc.)
            
        Returns:
            Created Pydantic model instance (with database-generated fields)
            
        Example:
            >>> handler = KnowledgeNodeHandler(session)
            >>> paragraph = ParagraphNode(
            ...     source_id=uuid.uuid4(),
            ...     parent_id=None,
            ...     attributes={"text": "This is a paragraph about AI."},
            ...     metadata={"page": 1, "bbox": [10, 20, 100, 200]},
            ... )
            >>> created = await handler.create(paragraph)
        """
        # Convert Pydantic to ORM using special converter
        orm_instance = pydantic_to_orm(pydantic_model)
        
        # Add to session and commit
        self.session.add(orm_instance)
        await self.session.commit()
        await self.session.refresh(orm_instance)
        
        # Convert back to Pydantic using special converter
        return orm_to_pydantic(orm_instance)

    async def get_by_id(
        self,
        id: uuid.UUID,
        pydantic_class: Optional[Type[KnowledgeNode]] = None,
    ) -> Optional[Union[
        KnowledgeNode,
        ParagraphNode,
        ImageNode,
        TableNode,
        StructureNode,
    ]]:
        """
        Get a knowledge node by ID.
        
        Args:
            id: Primary key (UUID)
            pydantic_class: Optional specific Pydantic class to use.
                           If None, will be inferred from ORM data.
            
        Returns:
            Pydantic model instance or None if not found
            
        Example:
            >>> handler = KnowledgeNodeHandler(session)
            >>> node = await handler.get_by_id(uuid.UUID("..."))
        """
        orm_instance = await self.session.get(self.orm_class, id)
        if orm_instance is None:
            return None
        
        # Convert to Pydantic using special converter
        return orm_to_pydantic(orm_instance, pydantic_class)

    async def get_all(
        self,
        limit: Optional[int] = None,
        offset: Optional[int] = None,
        order_by: Optional[Any] = None,
        pydantic_class: Optional[Type[KnowledgeNode]] = None,
    ) -> List[Union[
        KnowledgeNode,
        ParagraphNode,
        ImageNode,
        TableNode,
        StructureNode,
    ]]:
        """
        Get all knowledge nodes with optional pagination and ordering.
        
        Args:
            limit: Maximum number of records to return
            offset: Number of records to skip
            order_by: Column to order by (e.g., KnowledgeNodeORM.created_at.desc())
            pydantic_class: Optional specific Pydantic class to use.
            
        Returns:
            List of Pydantic model instances
            
        Example:
            >>> handler = KnowledgeNodeHandler(session)
            >>> nodes = await handler.get_all(limit=10, order_by=KnowledgeNodeORM.created_at.desc())
        """
        stmt = select(self.orm_class)
        
        if order_by is not None:
            stmt = stmt.order_by(order_by)
        
        if offset is not None:
            stmt = stmt.offset(offset)
        
        if limit is not None:
            stmt = stmt.limit(limit)
        
        result = await self.session.execute(stmt)
        orm_instances = result.scalars().all()
        
        # Convert to Pydantic using special converter
        return [
            orm_to_pydantic(orm_instance, pydantic_class)
            for orm_instance in orm_instances
        ]

    async def get_by_filter(
        self,
        filters: dict[str, Any],
        limit: Optional[int] = None,
        offset: Optional[int] = None,
        order_by: Optional[Any] = None,
        pydantic_class: Optional[Type[KnowledgeNode]] = None,
    ) -> List[Union[
        KnowledgeNode,
        ParagraphNode,
        ImageNode,
        TableNode,
        StructureNode,
    ]]:
        """
        Get knowledge nodes by filter criteria.
        
        Args:
            filters: Dictionary of field: value pairs to filter by
            limit: Maximum number of records to return
            offset: Number of records to skip
            order_by: Column to order by
            pydantic_class: Optional specific Pydantic class to use.
            
        Returns:
            List of Pydantic model instances matching filters
            
        Example:
            >>> handler = KnowledgeNodeHandler(session)
            >>> nodes = await handler.get_by_filter(
            ...     {"source_id": source_id, "node_type": NodeType.PARAGRAPH},
            ...     order_by=KnowledgeNodeORM.created_at.desc()
            ... )
        """
        stmt = select(self.orm_class)
        
        # Apply filters
        for field, value in filters.items():
            if hasattr(self.orm_class, field):
                stmt = stmt.where(getattr(self.orm_class, field) == value)
        
        if order_by is not None:
            stmt = stmt.order_by(order_by)
        
        if offset is not None:
            stmt = stmt.offset(offset)
        
        if limit is not None:
            stmt = stmt.limit(limit)
        
        result = await self.session.execute(stmt)
        orm_instances = result.scalars().all()
        
        # Convert to Pydantic using special converter
        return [
            orm_to_pydantic(orm_instance, pydantic_class)
            for orm_instance in orm_instances
        ]

    async def get_by_source(
        self,
        source_id: uuid.UUID,
        limit: Optional[int] = None,
        order_by: Optional[Any] = None,
    ) -> List[Union[
        KnowledgeNode,
        ParagraphNode,
        ImageNode,
        TableNode,
        StructureNode,
    ]]:
        """
        Get knowledge nodes by source ID.
        
        Args:
            source_id: Data source ID
            limit: Maximum number of nodes to return
            order_by: Column to order by (default: created_at desc)
            
        Returns:
            List of KnowledgeNode instances
        """
        if order_by is None:
            order_by = KnowledgeNodeORM.created_at.desc()
        
        return await self.get_by_filter(
            {"source_id": source_id},
            limit=limit,
            order_by=order_by,
        )

    async def get_by_node_type(
        self,
        node_type: NodeType,
        limit: Optional[int] = None,
        order_by: Optional[Any] = None,
    ) -> List[Union[
        KnowledgeNode,
        ParagraphNode,
        ImageNode,
        TableNode,
        StructureNode,
    ]]:
        """
        Get knowledge nodes by node type.
        
        Args:
            node_type: Node type (STRUCTURE, PARAGRAPH, IMAGE, TABLE, etc.)
            limit: Maximum number of nodes to return
            order_by: Column to order by (default: created_at desc)
            
        Returns:
            List of KnowledgeNode instances
        """
        if order_by is None:
            order_by = KnowledgeNodeORM.created_at.desc()
        
        return await self.get_by_filter(
            {"node_type": node_type},
            limit=limit,
            order_by=order_by,
        )

    async def get_by_parent(
        self,
        parent_id: uuid.UUID,
        limit: Optional[int] = None,
        order_by: Optional[Any] = None,
    ) -> List[Union[
        KnowledgeNode,
        ParagraphNode,
        ImageNode,
        TableNode,
        StructureNode,
    ]]:
        """
        Get child nodes by parent ID.
        
        Args:
            parent_id: Parent node ID
            limit: Maximum number of nodes to return
            order_by: Column to order by (default: created_at asc)
            
        Returns:
            List of child KnowledgeNode instances
        """
        if order_by is None:
            order_by = KnowledgeNodeORM.created_at.asc()
        
        return await self.get_by_filter(
            {"parent_id": parent_id},
            limit=limit,
            order_by=order_by,
        )

    async def get_children(
        self,
        node_id: uuid.UUID,
        limit: Optional[int] = None,
    ) -> List[Union[
        KnowledgeNode,
        ParagraphNode,
        ImageNode,
        TableNode,
        StructureNode,
    ]]:
        """
        Get direct children of a node (using relationship).
        
        Args:
            node_id: Parent node ID
            limit: Maximum number of children to return
            
        Returns:
            List of child KnowledgeNode instances
        """
        orm_instance = await self.session.get(self.orm_class, node_id)
        if orm_instance is None:
            return []
        
        # Need to access relationship - may need to eager load
        from sqlalchemy.orm import selectinload
        stmt = select(self.orm_class).where(self.orm_class.id == node_id).options(selectinload(self.orm_class.children))
        result = await self.session.execute(stmt)
        orm_instance = result.scalar_one_or_none()
        
        if orm_instance is None:
            return []
        
        children = orm_instance.children
        if limit is not None:
            children = children[:limit]
        
        return [
            orm_to_pydantic(child)
            for child in children
        ]

    async def update(
        self,
        id: uuid.UUID,
        update_data: dict[str, Any],
    ) -> Optional[Union[
        KnowledgeNode,
        ParagraphNode,
        ImageNode,
        TableNode,
        StructureNode,
    ]]:
        """
        Update a knowledge node by ID.
        
        Args:
            id: Primary key (UUID)
            update_data: Dictionary of fields to update
            
        Returns:
            Updated Pydantic model instance or None if not found
            
        Example:
            >>> handler = KnowledgeNodeHandler(session)
            >>> updated = await handler.update(node_id, {"attributes": {"text": "Updated text"}})
        """
        orm_instance = await self.session.get(self.orm_class, id)
        if orm_instance is None:
            return None
        
        # Update fields
        for field, value in update_data.items():
            if hasattr(orm_instance, field):
                setattr(orm_instance, field, value)
        
        await self.session.commit()
        await self.session.refresh(orm_instance)
        
        # Convert to Pydantic using special converter
        return orm_to_pydantic(orm_instance)

    async def update_by_model(
        self,
        pydantic_model: Union[
            KnowledgeNode,
            ParagraphNode,
            ImageNode,
            TableNode,
            StructureNode,
        ],
    ) -> Optional[Union[
        KnowledgeNode,
        ParagraphNode,
        ImageNode,
        TableNode,
        StructureNode,
    ]]:
        """
        Update a knowledge node using a Pydantic model.
        
        Args:
            pydantic_model: Pydantic model instance with updated data
            
        Returns:
            Updated Pydantic model instance or None if not found
            
        Example:
            >>> handler = KnowledgeNodeHandler(session)
            >>> paragraph = ParagraphNode(id=existing_id, ...)
            >>> updated = await handler.update_by_model(paragraph)
        """
        if not hasattr(pydantic_model, "id"):
            raise ValueError("Pydantic model must have an 'id' field for update")
        
        id = pydantic_model.id
        orm_instance = await self.session.get(self.orm_class, id)
        if orm_instance is None:
            return None
        
        # Convert Pydantic to ORM to get updated values
        updated_orm = pydantic_to_orm(pydantic_model)
        
        # Update all fields from the new ORM instance, excluding id and created_at
        for column in self.orm_class.__table__.columns:
            if column.name not in ("id", "created_at"):  # Don't update primary key or created_at
                setattr(orm_instance, column.name, getattr(updated_orm, column.name))
        
        await self.session.commit()
        await self.session.refresh(orm_instance)
        
        # Convert to Pydantic using special converter
        return orm_to_pydantic(orm_instance)

    async def delete(self, id: uuid.UUID) -> bool:
        """
        Delete a knowledge node by ID.
        
        Args:
            id: Primary key (UUID)
            
        Returns:
            True if deleted, False if not found
            
        Example:
            >>> handler = KnowledgeNodeHandler(session)
            >>> deleted = await handler.delete(node_id)
        """
        orm_instance = await self.session.get(self.orm_class, id)
        if orm_instance is None:
            return False
        
        self.session.delete(orm_instance)
        await self.session.commit()
        return True

    async def count(self, filters: Optional[dict[str, Any]] = None) -> int:
        """
        Count knowledge nodes, optionally filtered.
        
        Args:
            filters: Optional dictionary of field: value pairs to filter by
            
        Returns:
            Number of nodes matching criteria
            
        Example:
            >>> handler = KnowledgeNodeHandler(session)
            >>> count = await handler.count({"source_id": source_id})
        """
        stmt = select(func.count()).select_from(self.orm_class)
        
        if filters:
            for field, value in filters.items():
                if hasattr(self.orm_class, field):
                    stmt = stmt.where(getattr(self.orm_class, field) == value)
        
        result = await self.session.execute(stmt)
        return result.scalar_one()


# Convenience class that groups all knowledge handlers
class KnowledgeHandler:
    """
    Convenience class grouping all knowledge-related handlers.
    
    Usage:
        >>> handler = KnowledgeHandler(session)
        >>> paragraph = await handler.nodes.create(ParagraphNode(...))
        >>> section = await handler.nodes.create(StructureNode(...))
    """

    def __init__(self, session: AsyncSession):
        """Initialize all knowledge handlers."""
        self.session = session
        self.nodes = KnowledgeNodeHandler(session)

