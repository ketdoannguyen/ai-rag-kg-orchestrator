"""
Base handler class with common CRUD operations.

This provides a foundation for all model-specific handlers.
"""

from __future__ import annotations

from typing import Generic, TypeVar, Type, Optional, List, Any, Dict
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

# Type variables for generic handler
PydanticModel = TypeVar("PydanticModel")
ORMModel = TypeVar("ORMModel")


class BaseHandler(Generic[PydanticModel, ORMModel]):
    """
    Base handler class providing common CRUD operations.
    
    This class uses generics to work with any Pydantic model and its corresponding ORM model.
    Subclasses should specify the Pydantic and ORM types.
    
    Pattern:
        1. Input: Pydantic model
        2. Convert to ORM using to_orm() method
        3. Use SQLAlchemy AsyncSession for database operations
        4. Convert back to Pydantic using from_orm() method
    """

    def __init__(
        self,
        session: AsyncSession,
        pydantic_class: Type[PydanticModel],
        orm_class: Type[ORMModel],
    ):
        """
        Initialize handler with session and model classes.
        
        Args:
            session: SQLAlchemy async session
            pydantic_class: Pydantic model class
            orm_class: ORM model class
        """
        self.session = session
        self.pydantic_class = pydantic_class
        self.orm_class = orm_class

    async def create(self, pydantic_model: PydanticModel) -> PydanticModel:
        """
        Create a new record from Pydantic model.
        
        Args:
            pydantic_model: Validated Pydantic model instance
            
        Returns:
            Created Pydantic model instance (with database-generated fields)
            
        Example:
            >>> handler = ChatSessionHandler(session)
            >>> session_model = ChatSession(user_identifier="user123")
            >>> created = await handler.create(session_model)
        """
        # Convert Pydantic to ORM
        if hasattr(pydantic_model, "to_orm"):
            orm_instance = pydantic_model.to_orm()
        else:
            raise ValueError(
                f"Pydantic model {self.pydantic_class.__name__} must have to_orm() method"
            )
        
        # Add to session and commit
        self.session.add(orm_instance)
        await self.session.commit()
        await self.session.refresh(orm_instance)
        
        # Convert back to Pydantic
        if hasattr(self.pydantic_class, "from_orm"):
            return self.pydantic_class.from_orm(orm_instance)
        else:
            raise ValueError(
                f"Pydantic model {self.pydantic_class.__name__} must have from_orm() classmethod"
            )

    async def get_by_id(self, id: Any) -> Optional[PydanticModel]:
        """
        Get a record by ID.
        
        Args:
            id: Primary key value (usually UUID)
            
        Returns:
            Pydantic model instance or None if not found
            
        Example:
            >>> handler = ChatSessionHandler(session)
            >>> session = await handler.get_by_id(uuid.UUID("..."))
        """
        orm_instance = await self.session.get(self.orm_class, id)
        if orm_instance is None:
            return None
        
        # Convert to Pydantic
        if hasattr(self.pydantic_class, "from_orm"):
            return self.pydantic_class.from_orm(orm_instance)
        else:
            raise ValueError(
                f"Pydantic model {self.pydantic_class.__name__} must have from_orm() classmethod"
            )

    async def get_all(
        self,
        limit: Optional[int] = None,
        offset: Optional[int] = None,
        order_by: Optional[Any] = None,
    ) -> List[PydanticModel]:
        """
        Get all records with optional pagination and ordering.
        
        Args:
            limit: Maximum number of records to return
            offset: Number of records to skip
            order_by: Column to order by (e.g., ModelORM.created_at.desc())
            
        Returns:
            List of Pydantic model instances
            
        Example:
            >>> handler = ChatSessionHandler(session)
            >>> sessions = await handler.get_all(limit=10, order_by=ChatSessionORM.started_at.desc())
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
        
        # Convert to Pydantic
        if hasattr(self.pydantic_class, "from_orm"):
            return [
                self.pydantic_class.from_orm(orm_instance)
                for orm_instance in orm_instances
            ]
        else:
            raise ValueError(
                f"Pydantic model {self.pydantic_class.__name__} must have from_orm() classmethod"
            )

    async def get_by_filter(
        self,
        filters: Dict[str, Any],
        limit: Optional[int] = None,
        offset: Optional[int] = None,
        order_by: Optional[Any] = None,
    ) -> List[PydanticModel]:
        """
        Get records by filter criteria.
        
        Args:
            filters: Dictionary of field: value pairs to filter by
            limit: Maximum number of records to return
            offset: Number of records to skip
            order_by: Column to order by
            
        Returns:
            List of Pydantic model instances matching filters
            
        Example:
            >>> handler = ChatMessageHandler(session)
            >>> messages = await handler.get_by_filter(
            ...     {"session_id": session_id, "role": "user"},
            ...     order_by=ChatMessageORM.created_at.desc()
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
        
        # Convert to Pydantic
        if hasattr(self.pydantic_class, "from_orm"):
            return [
                self.pydantic_class.from_orm(orm_instance)
                for orm_instance in orm_instances
            ]
        else:
            raise ValueError(
                f"Pydantic model {self.pydantic_class.__name__} must have from_orm() classmethod"
            )

    async def update(
        self,
        id: Any,
        update_data: Dict[str, Any],
    ) -> Optional[PydanticModel]:
        """
        Update a record by ID.
        
        Args:
            id: Primary key value
            update_data: Dictionary of fields to update
            
        Returns:
            Updated Pydantic model instance or None if not found
            
        Example:
            >>> handler = ChatSessionHandler(session)
            >>> updated = await handler.update(session_id, {"user_identifier": "new_user"})
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
        
        # Convert to Pydantic
        if hasattr(self.pydantic_class, "from_orm"):
            return self.pydantic_class.from_orm(orm_instance)
        else:
            raise ValueError(
                f"Pydantic model {self.pydantic_class.__name__} must have from_orm() classmethod"
            )

    async def update_by_model(
        self,
        pydantic_model: PydanticModel,
    ) -> Optional[PydanticModel]:
        """
        Update a record using a Pydantic model (updates all fields).
        
        Args:
            pydantic_model: Pydantic model instance with updated data
            
        Returns:
            Updated Pydantic model instance or None if not found
            
        Example:
            >>> handler = ChatSessionHandler(session)
            >>> session_model = ChatSession(id=existing_id, user_identifier="new_user", ...)
            >>> updated = await handler.update_by_model(session_model)
        """
        # Get ID from model (assuming it has an 'id' field)
        if not hasattr(pydantic_model, "id"):
            raise ValueError("Pydantic model must have an 'id' field for update")
        
        id = pydantic_model.id
        orm_instance = await self.session.get(self.orm_class, id)
        if orm_instance is None:
            return None
        
        # Convert Pydantic to ORM to get updated values
        if hasattr(pydantic_model, "to_orm"):
            updated_orm = pydantic_model.to_orm()
        else:
            raise ValueError(
                f"Pydantic model {self.pydantic_class.__name__} must have to_orm() method"
            )
        
        # Update all fields from the new ORM instance
        for column in self.orm_class.__table__.columns:
            if column.name != "id":  # Don't update primary key
                setattr(orm_instance, column.name, getattr(updated_orm, column.name))
        
        await self.session.commit()
        await self.session.refresh(orm_instance)
        
        # Convert to Pydantic
        if hasattr(self.pydantic_class, "from_orm"):
            return self.pydantic_class.from_orm(orm_instance)
        else:
            raise ValueError(
                f"Pydantic model {self.pydantic_class.__name__} must have from_orm() classmethod"
            )

    async def delete(self, id: Any) -> bool:
        """
        Delete a record by ID.
        
        Args:
            id: Primary key value
            
        Returns:
            True if deleted, False if not found
            
        Example:
            >>> handler = ChatSessionHandler(session)
            >>> deleted = await handler.delete(session_id)
        """
        orm_instance = await self.session.get(self.orm_class, id)
        if orm_instance is None:
            return False
        
        self.session.delete(orm_instance)
        await self.session.commit()
        return True

    async def count(self, filters: Optional[Dict[str, Any]] = None) -> int:
        """
        Count records, optionally filtered.
        
        Args:
            filters: Optional dictionary of field: value pairs to filter by
            
        Returns:
            Number of records matching criteria
            
        Example:
            >>> handler = ChatMessageHandler(session)
            >>> count = await handler.count({"session_id": session_id})
        """
        from sqlalchemy import func
        
        stmt = select(func.count()).select_from(self.orm_class)
        
        if filters:
            for field, value in filters.items():
                if hasattr(self.orm_class, field):
                    stmt = stmt.where(getattr(self.orm_class, field) == value)
        
        result = await self.session.execute(stmt)
        return result.scalar_one()

