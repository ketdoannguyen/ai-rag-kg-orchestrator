"""Script to delete all data from Neo4j database."""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

# Ensure repo root on sys.path
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from loguru import logger
from workflows.dependencies import init_neo4j_client


async def clear_neo4j() -> None:
    """Delete all nodes and relationships from Neo4j."""
    client = init_neo4j_client()
    driver = client.async_driver
    
    async with driver.session() as session:
        # Count nodes before deletion
        count_result = await session.run("MATCH (n) RETURN count(n) as count")
        record = await count_result.single()
        node_count = record["count"] if record else 0
        
        logger.info(f"Found {node_count} nodes in Neo4j")
        
        if node_count == 0:
            logger.info("Database is already empty")
            return
        
        # Delete all nodes and relationships
        logger.warning("Deleting all nodes and relationships...")
        result = await session.run("MATCH (n) DETACH DELETE n")
        await result.consume()
        
        # Verify deletion
        verify_result = await session.run("MATCH (n) RETURN count(n) as count")
        verify_record = await verify_result.single()
        remaining = verify_record["count"] if verify_record else 0
        
        if remaining == 0:
            logger.info(f"Successfully deleted {node_count} nodes")
        else:
            logger.error(f"Deletion incomplete: {remaining} nodes remaining")
    
    await driver.close()


def main() -> None:
    """Entry point."""
    logger.info("Starting Neo4j cleanup script")
    asyncio.run(clear_neo4j())
    logger.info("Done")


if __name__ == "__main__":
    main()
