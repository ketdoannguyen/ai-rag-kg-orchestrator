"""
System administration handler for ArangoDB.

Handles User and DataSource operations.
"""

from __future__ import annotations

import uuid

from arango.database import StandardDatabase

from ..collections.schema import Collections
from ..models.enums import SourceType, UserRole
from ..models.system_admin import DataSource, User
from .base_handler import BaseHandler


class UserHandler(BaseHandler[User]):
    """Handler for User CRUD operations."""

    def __init__(self, db: StandardDatabase):
        """Initialize User handler."""
        super().__init__(db, User, Collections.USERS)

    def get_by_username(self, username: str) -> User | None:
        """
        Get user by username.

        Args:
            username: Username to search for

        Returns:
            User instance or None if not found
        """
        results = self.get_by_filter({"username": username}, limit=1)
        return results[0] if results else None

    def get_by_role(
        self,
        role: UserRole | str,
        limit: int | None = None,
        offset: int | None = None,
    ) -> list[User]:
        """
        Get users by role.

        Args:
            role: User role (admin, editor, viewer)
            limit: Maximum number of users to return
            offset: Number of records to skip

        Returns:
            List of User instances
        """
        role_value = role.value if isinstance(role, UserRole) else role
        return self.get_by_filter(
            {"role": role_value},
            limit=limit,
            offset=offset,
            sort_desc=True,
        )

    def get_active_users(
        self,
        limit: int | None = None,
        offset: int | None = None,
    ) -> list[User]:
        """
        Get all active users.

        Args:
            limit: Maximum number of users to return
            offset: Number of records to skip

        Returns:
            List of active User instances
        """
        return self.get_by_filter(
            {"is_active": True},
            limit=limit,
            offset=offset,
            sort_desc=True,
        )

    def deactivate(self, user_id: uuid.UUID | str) -> User | None:
        """
        Deactivate a user.

        Args:
            user_id: User ID to deactivate

        Returns:
            Updated User instance or None if not found
        """
        return self.update(user_id, {"is_active": False})

    def activate(self, user_id: uuid.UUID | str) -> User | None:
        """
        Activate a user.

        Args:
            user_id: User ID to activate

        Returns:
            Updated User instance or None if not found
        """
        return self.update(user_id, {"is_active": True})


class DataSourceHandler(BaseHandler[DataSource]):
    """Handler for DataSource CRUD operations."""

    def __init__(self, db: StandardDatabase):
        """Initialize DataSource handler."""
        super().__init__(db, DataSource, Collections.DATA_SOURCES)

    def get_by_type(
        self,
        source_type: SourceType | str,
        limit: int | None = None,
        offset: int | None = None,
    ) -> list[DataSource]:
        """
        Get data sources by type.

        Args:
            source_type: Source type (pdf, web, facebook, faq_file)
            limit: Maximum number of sources to return
            offset: Number of records to skip

        Returns:
            List of DataSource instances
        """
        type_value = source_type.value if isinstance(source_type, SourceType) else source_type
        return self.get_by_filter(
            {"type": type_value},
            limit=limit,
            offset=offset,
            sort_desc=True,
        )

    def get_by_status(
        self,
        status: str,
        limit: int | None = None,
        offset: int | None = None,
    ) -> list[DataSource]:
        """
        Get data sources by status.

        Args:
            status: Source status (processing, active, error)
            limit: Maximum number of sources to return
            offset: Number of records to skip

        Returns:
            List of DataSource instances
        """
        return self.get_by_filter(
            {"status": status},
            limit=limit,
            offset=offset,
            sort_desc=True,
        )

    def get_by_creator(
        self,
        created_by: uuid.UUID | str,
        limit: int | None = None,
        offset: int | None = None,
    ) -> list[DataSource]:
        """
        Get data sources by creator.

        Args:
            created_by: User ID who created the source
            limit: Maximum number of sources to return
            offset: Number of records to skip

        Returns:
            List of DataSource instances
        """
        return self.get_by_filter(
            {"created_by": str(created_by)},
            limit=limit,
            offset=offset,
            sort_desc=True,
        )

    def get_by_hash(self, hash_content: str) -> DataSource | None:
        """
        Get data source by content hash (to avoid duplicates).

        Args:
            hash_content: MD5 hash of content

        Returns:
            DataSource instance or None if not found
        """
        results = self.get_by_filter({"hash_content": hash_content}, limit=1)
        return results[0] if results else None

    def update_status(
        self,
        source_id: uuid.UUID | str,
        status: str,
    ) -> DataSource | None:
        """
        Update data source status.

        Args:
            source_id: Data source ID
            status: New status (processing, active, error)

        Returns:
            Updated DataSource instance or None if not found
        """
        return self.update(source_id, {"status": status})


# Convenience class that groups all system admin handlers
class SystemAdminHandler:
    """
    Convenience class grouping all system admin-related handlers.

    Usage:
        >>> handler = SystemAdminHandler(db)
        >>> user = handler.users.create(User(...))
        >>> source = handler.sources.create(DataSource(...))
    """

    def __init__(self, db: StandardDatabase):
        """Initialize all system admin handlers."""
        self.db = db
        self.users = UserHandler(db)
        self.sources = DataSourceHandler(db)
