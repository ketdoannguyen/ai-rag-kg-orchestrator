"""
CRUD handlers for System Administration models.

Handles User and DataSource operations.
"""

from __future__ import annotations

import uuid
from typing import Optional, List, Any
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from .base_handler import BaseHandler
from ..models.system_admin import User, DataSource
from ..orm_models.system_admin import UserORM, DataSourceORM
from ..models.enums import UserRole, SourceType


class UserHandler(BaseHandler[User, UserORM]):
    """Handler for User CRUD operations."""

    def __init__(self, session: AsyncSession):
        """Initialize User handler."""
        super().__init__(session, User, UserORM)

    async def get_by_username(self, username: str) -> Optional[User]:
        """
        Get user by username.
        
        Args:
            username: Username to search for
            
        Returns:
            User instance or None if not found
        """
        stmt = select(self.orm_class).where(
            UserORM.username == username
        )
        result = await self.session.execute(stmt)
        orm_instance = result.scalar_one_or_none()
        
        if orm_instance is None:
            return None
        
        return self.pydantic_class.from_orm(orm_instance)

    async def get_by_role(
        self,
        role: UserRole,
        limit: Optional[int] = None,
        order_by: Optional[Any] = None,
    ) -> List[User]:
        """
        Get users by role.
        
        Args:
            role: User role (admin, editor, viewer)
            limit: Maximum number of users to return
            order_by: Column to order by (default: created_at desc)
            
        Returns:
            List of User instances
        """
        if order_by is None:
            order_by = UserORM.created_at.desc()
        
        return await self.get_by_filter(
            {"role": role},
            limit=limit,
            order_by=order_by,
        )

    async def get_active_users(
        self,
        limit: Optional[int] = None,
        order_by: Optional[Any] = None,
    ) -> List[User]:
        """
        Get all active users.
        
        Args:
            limit: Maximum number of users to return
            order_by: Column to order by (default: created_at desc)
            
        Returns:
            List of active User instances
        """
        if order_by is None:
            order_by = UserORM.created_at.desc()
        
        return await self.get_by_filter(
            {"is_active": True},
            limit=limit,
            order_by=order_by,
        )

    async def deactivate(self, user_id: uuid.UUID) -> Optional[User]:
        """
        Deactivate a user.
        
        Args:
            user_id: User ID to deactivate
            
        Returns:
            Updated User instance or None if not found
        """
        return await self.update(user_id, {"is_active": False})

    async def activate(self, user_id: uuid.UUID) -> Optional[User]:
        """
        Activate a user.
        
        Args:
            user_id: User ID to activate
            
        Returns:
            Updated User instance or None if not found
        """
        return await self.update(user_id, {"is_active": True})


class DataSourceHandler(BaseHandler[DataSource, DataSourceORM]):
    """Handler for DataSource CRUD operations."""

    def __init__(self, session: AsyncSession):
        """Initialize DataSource handler."""
        super().__init__(session, DataSource, DataSourceORM)

    async def get_by_type(
        self,
        source_type: SourceType,
        limit: Optional[int] = None,
        order_by: Optional[Any] = None,
    ) -> List[DataSource]:
        """
        Get data sources by type.
        
        Args:
            source_type: Source type (pdf, web, facebook, faq_file)
            limit: Maximum number of sources to return
            order_by: Column to order by (default: created_at desc)
            
        Returns:
            List of DataSource instances
        """
        if order_by is None:
            order_by = DataSourceORM.created_at.desc()
        
        return await self.get_by_filter(
            {"type": source_type},
            limit=limit,
            order_by=order_by,
        )

    async def get_by_status(
        self,
        status: str,
        limit: Optional[int] = None,
        order_by: Optional[Any] = None,
    ) -> List[DataSource]:
        """
        Get data sources by status.
        
        Args:
            status: Source status (processing, active, error)
            limit: Maximum number of sources to return
            order_by: Column to order by (default: created_at desc)
            
        Returns:
            List of DataSource instances
        """
        if order_by is None:
            order_by = DataSourceORM.created_at.desc()
        
        return await self.get_by_filter(
            {"status": status},
            limit=limit,
            order_by=order_by,
        )

    async def get_by_creator(
        self,
        created_by: uuid.UUID,
        limit: Optional[int] = None,
        order_by: Optional[Any] = None,
    ) -> List[DataSource]:
        """
        Get data sources by creator.
        
        Args:
            created_by: User ID who created the source
            limit: Maximum number of sources to return
            order_by: Column to order by (default: created_at desc)
            
        Returns:
            List of DataSource instances
        """
        if order_by is None:
            order_by = DataSourceORM.created_at.desc()
        
        return await self.get_by_filter(
            {"created_by": created_by},
            limit=limit,
            order_by=order_by,
        )

    async def get_by_hash(self, hash_content: str) -> Optional[DataSource]:
        """
        Get data source by content hash (to avoid duplicates).
        
        Args:
            hash_content: MD5 hash of content
            
        Returns:
            DataSource instance or None if not found
        """
        stmt = select(self.orm_class).where(
            DataSourceORM.hash_content == hash_content
        )
        result = await self.session.execute(stmt)
        orm_instance = result.scalar_one_or_none()
        
        if orm_instance is None:
            return None
        
        return self.pydantic_class.from_orm(orm_instance)

    async def update_status(
        self,
        source_id: uuid.UUID,
        status: str,
    ) -> Optional[DataSource]:
        """
        Update data source status.
        
        Args:
            source_id: Data source ID
            status: New status (processing, active, error)
            
        Returns:
            Updated DataSource instance or None if not found
        """
        return await self.update(source_id, {"status": status})


# Convenience class that groups all system admin handlers
class SystemAdminHandler:
    """
    Convenience class grouping all system admin-related handlers.
    
    Usage:
        >>> handler = SystemAdminHandler(session)
        >>> user = await handler.users.create(User(...))
        >>> source = await handler.sources.create(DataSource(...))
    """

    def __init__(self, session: AsyncSession):
        """Initialize all system admin handlers."""
        self.session = session
        self.users = UserHandler(session)
        self.sources = DataSourceHandler(session)

