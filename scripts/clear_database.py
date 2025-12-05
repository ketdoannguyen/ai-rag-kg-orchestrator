#!/usr/bin/env python3
"""Script to clear all data from PostgreSQL database.

This script deletes all data from all tables in the correct order
to respect foreign key constraints. Useful after schema migrations
that change vector dimensions (e.g., from 1024 to 1536).

Usage:
    python scripts/clear_database.py
    python scripts/clear_database.py --confirm
"""

import sys
from pathlib import Path

from loguru import logger
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from workflows.dependencies import init_postgres_client
from postgres_client import PostgresClient


async def clear_all_tables(pg_client: PostgresClient) -> None:
    """
    Clear all data from all tables in the correct order.
    
    Order is important to respect foreign key constraints:
    1. Child tables first (feedbacks, chat_messages, embeddings)
    2. Parent tables (knowledge_nodes, chat_sessions, data_sources)
    3. Independent tables (suggested_faqs, users)
    """
    async with pg_client.sessionmaker() as session:
        try:
            logger.info("Starting database cleanup...")
            
            # Disable foreign key checks temporarily (PostgreSQL doesn't support this directly,
            # so we'll delete in the correct order instead)
            
            # Step 1: Delete child records first
            logger.info("Deleting feedbacks...")
            count_result = await session.execute(text("SELECT COUNT(*) FROM feedbacks"))
            count = count_result.scalar()
            await session.execute(text("DELETE FROM feedbacks"))
            logger.info(f"  Deleted {count} feedbacks")
            
            logger.info("Deleting chat_messages...")
            count_result = await session.execute(text("SELECT COUNT(*) FROM chat_messages"))
            count = count_result.scalar()
            await session.execute(text("DELETE FROM chat_messages"))
            logger.info(f"  Deleted {count} chat_messages")
            
            logger.info("Deleting embeddings_content...")
            count_result = await session.execute(text("SELECT COUNT(*) FROM embeddings_content"))
            count = count_result.scalar()
            await session.execute(text("DELETE FROM embeddings_content"))
            logger.info(f"  Deleted {count} embeddings_content")
            
            logger.info("Deleting embeddings_structure...")
            count_result = await session.execute(text("SELECT COUNT(*) FROM embeddings_structure"))
            count = count_result.scalar()
            await session.execute(text("DELETE FROM embeddings_structure"))
            logger.info(f"  Deleted {count} embeddings_structure")
            
            # Step 2: Delete parent records
            logger.info("Deleting knowledge_nodes...")
            count_result = await session.execute(text("SELECT COUNT(*) FROM knowledge_nodes"))
            count = count_result.scalar()
            await session.execute(text("DELETE FROM knowledge_nodes"))
            logger.info(f"  Deleted {count} knowledge_nodes")
            
            logger.info("Deleting chat_sessions...")
            count_result = await session.execute(text("SELECT COUNT(*) FROM chat_sessions"))
            count = count_result.scalar()
            await session.execute(text("DELETE FROM chat_sessions"))
            logger.info(f"  Deleted {count} chat_sessions")
            
            # Step 3: Delete independent tables
            logger.info("Deleting suggested_faqs...")
            count_result = await session.execute(text("SELECT COUNT(*) FROM suggested_faqs"))
            count = count_result.scalar()
            await session.execute(text("DELETE FROM suggested_faqs"))
            logger.info(f"  Deleted {count} suggested_faqs")
            
            logger.info("Deleting data_sources...")
            count_result = await session.execute(text("SELECT COUNT(*) FROM data_sources"))
            count = count_result.scalar()
            await session.execute(text("DELETE FROM data_sources"))
            logger.info(f"  Deleted {count} data_sources")
            
            logger.info("Deleting users...")
            count_result = await session.execute(text("SELECT COUNT(*) FROM users"))
            count = count_result.scalar()
            await session.execute(text("DELETE FROM users"))
            logger.info(f"  Deleted {count} users")
            
            # Commit all deletions
            await session.commit()
            logger.success("✅ All data cleared successfully!")
            
        except Exception as e:
            await session.rollback()
            logger.error(f"❌ Error during database cleanup: {e}")
            raise


async def verify_empty(pg_client: PostgresClient) -> None:
    """Verify that all tables are empty."""
    async with pg_client.sessionmaker() as session:
        tables = [
            "feedbacks",
            "chat_messages",
            "chat_sessions",
            "embeddings_content",
            "embeddings_structure",
            "knowledge_nodes",
            "suggested_faqs",
            "data_sources",
            "users",
        ]
        
        logger.info("\nVerifying tables are empty...")
        all_empty = True
        for table in tables:
            result = await session.execute(text(f"SELECT COUNT(*) FROM {table}"))
            count = result.scalar()
            if count > 0:
                logger.warning(f"⚠️  {table}: {count} rows remaining")
                all_empty = False
            else:
                logger.info(f"✅ {table}: empty")
        
        if all_empty:
            logger.success("\n✅ All tables are empty!")
        else:
            logger.warning("\n⚠️  Some tables still contain data")


async def main() -> None:
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Clear all data from PostgreSQL database",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Preview what will be deleted (requires confirmation)
  python scripts/clear_database.py
  
  # Skip confirmation prompt
  python scripts/clear_database.py --confirm
  
  # Verify tables are empty after clearing
  python scripts/clear_database.py --verify-only
        """
    )
    parser.add_argument(
        "--confirm",
        action="store_true",
        help="Skip confirmation prompt (use with caution!)",
    )
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="Only verify tables are empty, don't delete anything",
    )
    
    args = parser.parse_args()
    
    if args.verify_only:
        logger.info("Verification mode: checking if tables are empty...")
        pg_client = init_postgres_client()
        try:
            await verify_empty(pg_client)
        finally:
            await pg_client.close()
        return
    
    # Confirmation prompt
    if not args.confirm:
        logger.warning("⚠️  WARNING: This will delete ALL data from the database!")
        logger.warning("⚠️  This action cannot be undone!")
        response = input("\nType 'yes' to confirm: ")
        if response.lower() != "yes":
            logger.info("Operation cancelled.")
            sys.exit(0)
    
    # Initialize postgres client
    pg_client = init_postgres_client()
    
    try:
        # Clear all tables
        await clear_all_tables(pg_client)
        
        # Verify everything is empty
        await verify_empty(pg_client)
        
    except Exception as e:
        logger.error(f"Failed to clear database: {e}")
        sys.exit(1)
    finally:
        await pg_client.close()


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())

