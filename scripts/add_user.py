"""Temporary script to add a user to the database.

This script creates a user with the specified UUID to fix foreign key constraint violations.
"""

from __future__ import annotations

import argparse
import asyncio
import sys
import uuid
from pathlib import Path

# Ensure repo root on sys.path when running as script
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from loguru import logger
from postgres_client.handlers.system_admin_handler import SystemAdminHandler
from postgres_client.models.enums import UserRole
from postgres_client.models.system_admin import User
from workflows.dependencies import init_postgres_client


async def add_user(
    *,
    user_id: uuid.UUID,
    username: str | None = None,
    full_name: str | None = None,
    role: UserRole = UserRole.EDITOR,
) -> User:
    """Add a user to the database."""
    pg_client = init_postgres_client()
    
    async with pg_client.sessionmaker() as session:
        handler = SystemAdminHandler(session)
        
        # Check if user already exists
        existing_user = await handler.users.get_by_id(user_id)
        if existing_user:
            logger.info(f"User with ID {user_id} already exists: {existing_user.username}")
            return existing_user
        
        # Create new user
        user_model = User(
            id=user_id,
            username=username or f"user_{user_id.hex[:8]}",
            hashed_password="temp_password_hash",  # Temporary password hash
            full_name=full_name or f"User {user_id.hex[:8]}",
            role=role,
            is_active=True,
        )
        
        created_user = await handler.users.create(user_model)
        logger.success(f"✅ Created User: {created_user.username} (ID: {created_user.id})")
        return created_user


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Add a user to the database (temporary script to fix foreign key constraints)"
    )
    parser.add_argument(
        "--user-id",
        required=True,
        help="UUID of the user to create (e.g., 550e8400-e29b-41d4-a716-446655440000)"
    )
    parser.add_argument(
        "--username",
        default=None,
        help="Username for the user (default: auto-generated)"
    )
    parser.add_argument(
        "--full-name",
        default=None,
        help="Full name for the user (default: auto-generated)"
    )
    parser.add_argument(
        "--role",
        default="EDITOR",
        choices=["ADMIN", "EDITOR", "VIEWER"],
        help="User role (default: EDITOR)"
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    user_id = uuid.UUID(args.user_id)
    role = UserRole[args.role]
    
    try:
        user = asyncio.run(
            add_user(
                user_id=user_id,
                username=args.username,
                full_name=args.full_name,
                role=role,
            )
        )
        logger.info(f"User created successfully: {user.model_dump()}")
    except Exception as exc:
        logger.exception(f"Failed to create user: {exc}")
        sys.exit(1)


if __name__ == "__main__":
    main()

