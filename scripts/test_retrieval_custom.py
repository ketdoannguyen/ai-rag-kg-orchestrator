#!/usr/bin/env python
"""
Custom retrieval test script for ArangoDB.
Checks for nodes without embeddings, deletes them, and tests vector search.
"""

import asyncio
import logging
import os
import sys

# Add project root to Python path
sys.path.append(os.getcwd())

from arango_client.handlers.search_handler import SearchHandler

from workflows.dependencies import init_arango_client

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


async def main():
    logger.info("Initializing ArangoClient...")
    try:
        client = init_arango_client()
        logger.info(f"Connected to ArangoDB: {client.db.name}")
    except Exception as e:
        logger.error(f"Failed to connect to ArangoDB: {e}")
        return

    # 1. Check for nodes without embeddings
    logger.info("Step 1: Checking for nodes without embeddings...")

    # Collections to check
    cols_to_check = ["paragraph_nodes", "structure_nodes", "image_nodes", "table_nodes"]
    total_missing = 0

    for col_name in cols_to_check:
        if not client.db.has_collection(col_name):
            continue

        logger.info(f"Checking collection: {col_name}")
        aql_missing_embedding = f"""
        FOR doc IN {col_name}
            FILTER doc.embedding == null OR LENGTH(doc.embedding) == 0
            RETURN doc
        """

        try:
            cursor = client.db.aql.execute(aql_missing_embedding)
            missing_nodes = [doc for doc in cursor]

            if missing_nodes:
                logger.info(f"Found {len(missing_nodes)} nodes without embeddings in {col_name}.")
                total_missing += len(missing_nodes)

                # Delete them
                count = 0
                for node in missing_nodes:
                    try:
                        doc_id = node["_id"]
                        client.db.delete_document(doc_id)
                        count += 1
                    except Exception as e:
                        logger.error(f"Failed to delete {node.get('_id')}: {e}")

                logger.info(f"Deleted {count} nodes from {col_name}.")
        except Exception as e:
            logger.error(f"Error checking {col_name}: {e}")

    if total_missing == 0:
        logger.info("No nodes without embeddings found in any collection. Clean state.")

    # 2. Check total node count
    logger.info("Step 2: Checking total node count...")
    # Count per collection to be sure
    total_nodes = 0
    collections = ["paragraph_nodes", "image_nodes", "table_nodes", "structure_nodes"]
    for col in collections:
        if client.db.has_collection(col):
            count = client.db.collection(col).count()
            logger.info(f"Collection {col}: {count} nodes")
            total_nodes += count

    logger.info(f"Total knowledge nodes in DB: {total_nodes}")

    if total_nodes == 0:
        logger.warning("Database is empty. Please run ingestion tests first to populate data.")
        return

    # 3. Test Retrieval
    logger.info("Step 3: Testing Vector Search (Retrieval mechanism)...")

    # Find a node WITH an embedding to use as query
    sample_node = None

    # Try finding sample in paragraph_nodes first (most likely for content search)
    # Then others
    for col_name in cols_to_check:
        if not client.db.has_collection(col_name):
            continue

        aql_get_embedded = f"""
        FOR doc IN {col_name}
            FILTER doc.embedding != null AND LENGTH(doc.embedding) > 0
            LIMIT 1
            RETURN doc
        """

        try:
            cursor = client.db.aql.execute(aql_get_embedded)
            for doc in cursor:
                sample_node = doc
                break

            if sample_node:
                logger.info(f"Found sample node in {col_name}")
                break
        except Exception as e:
            logger.error(f"Error querying for sample node in {col_name}: {e}")

    if sample_node:
        logger.info(
            f"Found sample node {sample_node['_key']} (Type: {sample_node.get('node_type')}) with embedding."
        )
        embedding = sample_node["embedding"]

        # Test using the SearchHandler
        logger.info("Running vector_search_content with the sample embedding...")

        try:
            # Manually instantiate SearchHandler as client wrapper doesn't expose it
            search_handler = SearchHandler(client.db)

            results = search_handler.vector_search_content(query_vector=embedding, limit=5)

            logger.info(f"Search returned {len(results)} results.")

            print("\n--- Search Results ---")
            for i, res in enumerate(results):
                # Basic validation: The sample node itself should be the top result (score ~1.0)
                is_self = (
                    res.node.id == sample_node["_id"].split("/")[-1]
                    if "/" in sample_node["_id"]
                    else res.node.id == sample_node["_key"]
                )
                marker = "[SELF]" if is_self else ""

                # Extract content/text
                content_text = "No content"
                if hasattr(res.node, "text") and res.node.text:
                    content_text = res.node.text
                elif hasattr(res.node, "attributes"):
                    attrs = res.node.attributes
                    if hasattr(attrs, "text") and attrs.text:
                        content_text = attrs.text
                    elif isinstance(attrs, dict) and attrs.get("text"):
                        content_text = attrs.get("text")

                content_preview = (content_text[:70] + "...") if content_text else "Empty content"

                print(f"{i + 1}. Score: {res.score:.4f} | ID: {res.node.id} {marker}")
                print(f"   Type: {res.node.node_type}")
                print(f"   Content: {content_preview}")
                print("---")

            # Validation check
            if len(results) > 0:
                top_score = results[0].score
                if top_score > 0.99:
                    logger.info(
                        "SUCCESS: Top result matches query vector (Score ~ 1.0). Retrieval is working."
                    )
                else:
                    logger.warning(
                        f"WARNING: Top result score is {top_score}. Expected ~1.0 for self-match."
                    )

        except Exception as e:
            logger.error(f"Vector search failed: {e}")
            import traceback

            traceback.print_exc()

    else:
        logger.warning("No nodes with embeddings found. Cannot test vector search.")
        logger.warning("Please run: python -m tests.workflows.test_ingest_file")


if __name__ == "__main__":
    asyncio.run(main())
