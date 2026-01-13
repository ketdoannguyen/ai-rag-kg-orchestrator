#!/usr/bin/env python3
"""Script to clear all data from ArangoDB database.

This script deletes all data from all collections in ArangoDB.
Useful after testing or when you need to reset the database.

Usage:
    python scripts/clear_arango_database.py
    python scripts/clear_arango_database.py --confirm
    python scripts/clear_arango_database.py --verify-only
"""

import sys
from pathlib import Path

from loguru import logger

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from workflows.dependencies import init_arango_client
from arango_client import ArangoClient


# Collections to clear (order matters for edges first, then nodes)
EDGE_COLLECTIONS = [
    "knowledge_edges",
]

DOCUMENT_COLLECTIONS = [
    "feedbacks",
    "chat_messages",
    "chat_sessions",
    "suggested_faqs",
    "knowledge_nodes",
    "data_sources",
    "users",
]


def clear_collection(db, collection_name: str) -> int:
    """Clear all documents from a collection. Returns count deleted."""
    try:
        if not db.has_collection(collection_name):
            logger.warning(f"  Collection '{collection_name}' does not exist, skipping")
            return 0
        
        collection = db.collection(collection_name)
        count = collection.count()
        
        if count > 0:
            collection.truncate()
            logger.info(f"  Deleted {count} documents from {collection_name}")
        else:
            logger.info(f"  {collection_name}: already empty")
        
        return count
    except Exception as e:
        logger.error(f"  Error clearing {collection_name}: {e}")
        return 0


def clear_all_collections(client: ArangoClient) -> dict:
    """Clear all data from all collections."""
    db = client.db
    stats = {"edges_deleted": 0, "documents_deleted": 0, "collections_cleared": 0}
    
    logger.info("Starting ArangoDB cleanup...")
    
    # Step 1: Delete edges first (foreign key relationships)
    logger.info("\n[Step 1/2] Clearing edge collections...")
    for collection_name in EDGE_COLLECTIONS:
        count = clear_collection(db, collection_name)
        stats["edges_deleted"] += count
        stats["collections_cleared"] += 1
    
    # Step 2: Delete documents
    logger.info("\n[Step 2/2] Clearing document collections...")
    for collection_name in DOCUMENT_COLLECTIONS:
        count = clear_collection(db, collection_name)
        stats["documents_deleted"] += count
        stats["collections_cleared"] += 1
    
    return stats


def verify_empty(client: ArangoClient) -> bool:
    """Verify that all collections are empty."""
    db = client.db
    all_collections = EDGE_COLLECTIONS + DOCUMENT_COLLECTIONS
    
    logger.info("\nVerifying collections are empty...")
    all_empty = True
    
    for collection_name in all_collections:
        if not db.has_collection(collection_name):
            logger.info(f"  ✅ {collection_name}: does not exist")
            continue
            
        collection = db.collection(collection_name)
        count = collection.count()
        
        if count > 0:
            logger.warning(f"  ⚠️  {collection_name}: {count} documents remaining")
            all_empty = False
        else:
            logger.info(f"  ✅ {collection_name}: empty")
    
    if all_empty:
        logger.success("\n✅ All collections are empty!")
    else:
        logger.warning("\n⚠️  Some collections still contain data")
    
    return all_empty


def show_stats(client: ArangoClient) -> None:
    """Show current database statistics."""
    db = client.db
    all_collections = EDGE_COLLECTIONS + DOCUMENT_COLLECTIONS
    
    logger.info("\n📊 Current database statistics:")
    total = 0
    
    for collection_name in all_collections:
        if not db.has_collection(collection_name):
            continue
        
        collection = db.collection(collection_name)
        count = collection.count()
        total += count
        if count > 0:
            logger.info(f"  {collection_name}: {count} documents")
    
    logger.info(f"\n  Total: {total} documents")


def main() -> None:
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Clear all data from ArangoDB database",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Preview what will be deleted (requires confirmation)
  python scripts/clear_arango_database.py
  
  # Skip confirmation prompt
  python scripts/clear_arango_database.py --confirm
  
  # Only show current statistics
  python scripts/clear_arango_database.py --stats
  
  # Verify collections are empty
  python scripts/clear_arango_database.py --verify-only
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
        help="Only verify collections are empty, don't delete anything",
    )
    parser.add_argument(
        "--stats",
        action="store_true",
        help="Show current database statistics",
    )
    
    args = parser.parse_args()
    
    # Initialize arango client
    client = init_arango_client()
    
    try:
        if args.stats:
            show_stats(client)
            return
        
        if args.verify_only:
            logger.info("Verification mode: checking if collections are empty...")
            verify_empty(client)
            return
        
        # Show current stats
        show_stats(client)
        
        # Confirmation prompt
        if not args.confirm:
            logger.warning("\n⚠️  WARNING: This will delete ALL data from ArangoDB!")
            logger.warning("⚠️  This action cannot be undone!")
            response = input("\nType 'yes' to confirm: ")
            if response.lower() != "yes":
                logger.info("Operation cancelled.")
                sys.exit(0)
        
        # Clear all collections
        stats = clear_all_collections(client)
        
        # Summary
        logger.success(f"\n✅ Cleanup complete!")
        logger.info(f"   Edges deleted: {stats['edges_deleted']}")
        logger.info(f"   Documents deleted: {stats['documents_deleted']}")
        
        # Verify everything is empty
        verify_empty(client)
        
    except Exception as e:
        logger.error(f"Failed to clear database: {e}")
        sys.exit(1)
    finally:
        client.close()


if __name__ == "__main__":
    main()
