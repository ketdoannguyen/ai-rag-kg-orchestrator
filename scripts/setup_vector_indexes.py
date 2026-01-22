"""Setup ArangoDB indexes for efficient similarity search.

This script creates:
1. Vector indexes (if ArangoDB Enterprise 3.12+) for ANN search
2. Persistent indexes for query optimization (is_reviewed, source_id, level)

Usage:
    python scripts/setup_vector_indexes.py

Note on Vector Indexes:
- ArangoDB 3.12 Community: COSINE_SIMILARITY() works but no native ANN index
- ArangoDB 3.12 Enterprise: Native vector indexes with HNSW/IVF algorithms
- Current implementation uses COSINE_SIMILARITY() which is optimized in 3.12+
"""

from __future__ import annotations

import sys
from pathlib import Path

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from arango.database import StandardDatabase
from arango.exceptions import IndexCreateError
from loguru import logger

from config.extraction import get_extraction_settings
from libs.arango_client.src.arango_client.collections.schema import Collections
from workflows.dependencies import init_arango_client


def create_vector_index(
    db: StandardDatabase,
    collection_name: str,
    embedding_field: str = "embedding",
    dimensions: int = 1536,
    name: str | None = None,
) -> bool:
    """Create a vector index on a collection.

    Attempts to create native vector index (Enterprise 3.12+).
    Falls back gracefully if not supported.

    Args:
        db: ArangoDB database instance
        collection_name: Name of the collection
        embedding_field: Name of the field containing embeddings
        dimensions: Number of embedding dimensions
        name: Optional name for the index

    Returns:
        True if index created successfully or already exists
    """
    log = logger.bind(collection=collection_name, field=embedding_field)

    try:
        collection = db.collection(collection_name)
        index_name = name or f"vec_idx_{embedding_field}"

        # Check if vector index already exists
        existing_indexes = collection.indexes()
        for idx in existing_indexes:
            if idx.get("type") == "vector" and idx.get("name") == index_name:
                log.info("Vector index already exists", index_id=idx.get("id"))
                return True

        # Try to create native vector index (Enterprise 3.12+)
        try:
            collection.add_index(
                {
                    "type": "vector",
                    "name": index_name,
                    "fields": [embedding_field],
                    "params": {
                        "metric": "cosine",
                        "dimensions": dimensions,
                        "nLists": 100,  # Number of clusters for IVF
                    },
                    "inBackground": True,
                }
            )
            log.info(
                "Created native vector index (Enterprise)",
                index_name=index_name,
                dimensions=dimensions,
            )
            return True

        except IndexCreateError as e:
            error_msg = str(e).lower()

            if "unknown index type" in error_msg or "not supported" in error_msg:
                # Vector index not supported (Community Edition)
                log.warning(
                    "Native vector index not supported (requires Enterprise Edition). "
                    "COSINE_SIMILARITY() will still work but without ANN optimization.",
                )
                return False

            if "already exists" in error_msg or "duplicate" in error_msg:
                log.info("Vector index already exists")
                return True

            # Other errors - log and continue
            log.warning(
                "Could not create vector index",
                error=str(e),
            )
            return False

    except Exception as e:
        log.exception("Failed to create vector index", error=str(e))
        return False


def create_dedup_support_indexes(db: StandardDatabase) -> dict[str, bool]:
    """Create supporting indexes for deduplication queries.

    Creates persistent indexes that speed up common deduplication queries:
    - is_reviewed filter
    - source_id filter
    - level filter (for structure nodes)
    """
    log = logger.bind(operation="create_support_indexes")
    results = {}

    # Content nodes indexes
    try:
        content_collection = db.collection(Collections.CONTENT_NODES)

        # Check if index exists
        existing = {idx.get("name") for idx in content_collection.indexes()}

        if "idx_content_dedup" not in existing:
            content_collection.add_index(
                {
                    "type": "persistent",
                    "name": "idx_content_dedup",
                    "fields": ["is_reviewed", "source_id"],
                    "inBackground": True,
                }
            )
            log.info("Created content_nodes dedup index")
        else:
            log.info("Content dedup index already exists")

        results["content_dedup_index"] = True

    except IndexCreateError as e:
        if "already exists" in str(e).lower() or "duplicate" in str(e).lower():
            results["content_dedup_index"] = True
            log.info("Content dedup index already exists")
        else:
            results["content_dedup_index"] = False
            log.warning("Failed to create content index", error=str(e))

    # Structure nodes indexes
    try:
        structure_collection = db.collection(Collections.STRUCTURE_NODES)

        # Check if index exists
        existing = {idx.get("name") for idx in structure_collection.indexes()}

        if "idx_structure_dedup" not in existing:
            structure_collection.add_index(
                {
                    "type": "persistent",
                    "name": "idx_structure_dedup",
                    "fields": ["is_reviewed", "level", "source_id"],
                    "inBackground": True,
                }
            )
            log.info("Created structure_nodes dedup index")
        else:
            log.info("Structure dedup index already exists")

        results["structure_dedup_index"] = True

    except IndexCreateError as e:
        if "already exists" in str(e).lower() or "duplicate" in str(e).lower():
            results["structure_dedup_index"] = True
            log.info("Structure dedup index already exists")
        else:
            results["structure_dedup_index"] = False
            log.warning("Failed to create structure index", error=str(e))

    return results


def get_arangodb_info(db: StandardDatabase) -> dict:
    """Get ArangoDB server information."""
    try:
        version = db.version()
        engine = db.engine()
        return {
            "version": version.get("version", "unknown"),
            "license": version.get("license", "unknown"),
            "engine": engine.get("name", "unknown"),
        }
    except Exception as e:
        logger.warning("Could not get ArangoDB info", error=str(e))
        return {"version": "unknown", "license": "unknown", "engine": "unknown"}


def main() -> int:
    """Main entry point for setup script."""
    logger.info("Setting up ArangoDB indexes for extraction")

    # Get settings
    settings = get_extraction_settings()
    dimensions = settings.EXTRACTION_EMBEDDING_DIMENSIONS

    # Initialize client
    client = init_arango_client()
    db = client.db

    # Get ArangoDB info
    info = get_arangodb_info(db)
    logger.info(
        "Connected to ArangoDB",
        database=db.name,
        version=info["version"],
        license=info["license"],
        engine=info["engine"],
    )

    # Check if Enterprise Edition
    is_enterprise = info["license"].lower() == "enterprise"
    if not is_enterprise:
        logger.warning(
            "ArangoDB Community Edition detected. "
            "Native vector indexes require Enterprise Edition. "
            "COSINE_SIMILARITY() will still work for similarity search."
        )

    results = {}

    # Try to create vector indexes (will fail gracefully on Community)
    logger.info("Attempting to create vector indexes...")

    results["content_vector"] = create_vector_index(
        db=db,
        collection_name=Collections.CONTENT_NODES,
        embedding_field="embedding",
        dimensions=dimensions,
        name="vec_idx_content_embedding",
    )

    results["structure_vector"] = create_vector_index(
        db=db,
        collection_name=Collections.STRUCTURE_NODES,
        embedding_field="embedding",
        dimensions=dimensions,
        name="vec_idx_structure_embedding",
    )

    # Create supporting indexes (always works)
    logger.info("Creating supporting indexes...")
    support_results = create_dedup_support_indexes(db)
    results.update(support_results)

    # Print summary
    logger.info("=" * 60)
    logger.info("Index Setup Summary")
    logger.info("=" * 60)
    logger.info(f"ArangoDB Version: {info['version']} ({info['license']})")
    logger.info("-" * 60)

    success_count = sum(1 for v in results.values() if v)
    total_count = len(results)

    for name, success in results.items():
        status = "✓" if success else "✗"
        logger.info(f"  {status} {name}")

    logger.info("-" * 60)
    logger.info(f"Total: {success_count}/{total_count} indexes created/verified")

    if not is_enterprise:
        logger.info("")
        logger.info("Note: Vector indexes not available (Community Edition).")
        logger.info("Similarity search uses COSINE_SIMILARITY() which is still efficient.")
        logger.info("For ANN optimization at massive scale, consider Enterprise Edition.")

    logger.info("=" * 60)

    # Close client
    client.close()

    # Return success if at least support indexes were created
    return 0 if results.get("content_dedup_index") and results.get("structure_dedup_index") else 1


if __name__ == "__main__":
    sys.exit(main())
