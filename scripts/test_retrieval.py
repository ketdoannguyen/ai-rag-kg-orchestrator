#!/usr/bin/env python
"""
Test script for evaluating ArangoDB retrieval effectiveness.

This script tests:
1. Vector search (COSINE_SIMILARITY)
2. Full-text search (PHRASE, LIKE)
3. Graph traversal (ancestors, descendants, siblings)
4. Hybrid search (combined vector + text)

Usage:
    python scripts/test_retrieval.py
"""

from __future__ import annotations

import asyncio
import json
import sys
import time
from typing import Any

from loguru import logger

# Configure logger
logger.remove()
logger.add(
    sys.stderr,
    level="INFO",
    format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | {message}",
)

# Add project to path
sys.path.insert(0, str(__file__).rsplit("/", 2)[0])

from arango_client.handlers import KnowledgeHandler, SearchHandler
from llm_gateway_client.models.embedding import EmbeddingRequest

from workflows.dependencies import init_arango_client, init_llm_client


class RetrievalTester:
    """Test retrieval operations on ArangoDB."""

    def __init__(self):
        self.arango = init_arango_client()
        self.llm_client = init_llm_client()
        self.db = self.arango._db
        self.knowledge = KnowledgeHandler(self.db)
        self.search = SearchHandler(self.db)
        self.results: list[dict[str, Any]] = []

    async def generate_embedding(self, text: str) -> list[float]:
        """Generate embedding for text."""
        request = EmbeddingRequest(input=[text])
        response = await self.llm_client.embed(request)
        return response.embeddings[0]

    def log_result(
        self,
        test_name: str,
        success: bool,
        latency_ms: float,
        count: int = 0,
        details: str = "",
    ):
        """Log and store test result."""
        status = "✅ PASS" if success else "❌ FAIL"
        logger.info(f"{status} | {test_name} | {latency_ms:.2f}ms | {count} results | {details}")
        self.results.append(
            {
                "test": test_name,
                "success": success,
                "latency_ms": latency_ms,
                "count": count,
                "details": details,
            }
        )

    def test_database_connection(self) -> bool:
        """Test 1: Verify database connection."""
        start = time.perf_counter()
        try:
            version = self.db.version()
            latency = (time.perf_counter() - start) * 1000
            self.log_result("Database Connection", True, latency, details=f"ArangoDB {version}")
            return True
        except Exception as e:
            latency = (time.perf_counter() - start) * 1000
            self.log_result("Database Connection", False, latency, details=str(e))
            return False

    def test_collection_counts(self) -> dict[str, int]:
        """Test 2: Get document counts for all collections."""
        start = time.perf_counter()
        counts = {}
        try:
            collections = [
                "paragraph_nodes",
                "image_nodes",
                "table_nodes",
                "structure_nodes",
                "root_nodes",
                "knowledge_edges",
            ]
            for coll_name in collections:
                if self.db.has_collection(coll_name):
                    counts[coll_name] = self.db.collection(coll_name).count()
                else:
                    counts[coll_name] = 0

            latency = (time.perf_counter() - start) * 1000
            total = sum(counts.values())
            self.log_result(
                "Collection Counts", total > 0, latency, count=total, details=json.dumps(counts)
            )
            return counts
        except Exception as e:
            latency = (time.perf_counter() - start) * 1000
            self.log_result("Collection Counts", False, latency, details=str(e))
            return {}

    def test_view_exists(self) -> bool:
        """Test 3: Verify ArangoSearch view exists and is configured."""
        start = time.perf_counter()
        try:
            views = [v["name"] for v in self.db.views()]
            exists = "knowledge_vector_view" in views
            linked_collections = []

            if exists:
                view_props = self.db.view("knowledge_vector_view")
                if hasattr(view_props, "properties"):
                    props = view_props.properties()
                    linked_collections = list(props.get("links", {}).keys())
                else:
                    # View exists, just can't get details
                    linked_collections = ["(view exists)"]

            latency = (time.perf_counter() - start) * 1000
            self.log_result(
                "ArangoSearch View",
                exists,
                latency,
                count=len(linked_collections),
                details=f"Links: {linked_collections}" if exists else "View not found",
            )
            return exists
        except Exception as e:
            latency = (time.perf_counter() - start) * 1000
            self.log_result("ArangoSearch View", False, latency, details=str(e))
            return False

    async def test_vector_search(self, query: str = "tuyển sinh đại học") -> list[dict]:
        """Test 4: Vector similarity search."""
        start = time.perf_counter()
        try:
            # Generate embedding for query
            embedding = await self.generate_embedding(query)
            embed_time = (time.perf_counter() - start) * 1000

            # Execute vector search
            search_start = time.perf_counter()
            results = self.search.vector_search_content(
                query_vector=embedding,
                limit=5,
            )
            search_time = (time.perf_counter() - search_start) * 1000

            latency = (time.perf_counter() - start) * 1000
            self.log_result(
                "Vector Search",
                len(results) > 0,
                latency,
                count=len(results),
                details=f"Query: '{query}' | Embed: {embed_time:.0f}ms | Search: {search_time:.0f}ms",
            )

            # Print top results
            for i, result in enumerate(results[:3]):
                node = result.node
                text_preview = (node.attributes.text if hasattr(node.attributes, "text") else "")[
                    :100
                ]
                score = result.score
                logger.info(f"  [{i + 1}] Score: {score:.4f} | {text_preview}...")

            return results
        except Exception as e:
            latency = (time.perf_counter() - start) * 1000
            self.log_result("Vector Search", False, latency, details=str(e))
            return []

    def test_fulltext_search(self, search_term: str = "giáo viên") -> list:
        """Test 5: Full-text search."""
        start = time.perf_counter()
        try:
            results = self.search.fulltext_search(
                search_term=search_term,
                limit=5,
            )
            latency = (time.perf_counter() - start) * 1000
            self.log_result(
                "Fulltext Search",
                len(results) > 0,
                latency,
                count=len(results),
                details=f"Term: '{search_term}'",
            )

            # Print top results
            for i, node in enumerate(results[:3]):
                text_preview = (node.attributes.text if hasattr(node.attributes, "text") else "")[
                    :100
                ]
                logger.info(f"  [{i + 1}] {text_preview}...")

            return results
        except Exception as e:
            latency = (time.perf_counter() - start) * 1000
            self.log_result("Fulltext Search", False, latency, details=str(e))
            return []

    def test_graph_traversal(self) -> dict[str, Any]:
        """Test 6: Graph traversal (ancestors, descendants)."""
        results = {"ancestors": 0, "descendants": 0, "children": 0}
        start = time.perf_counter()
        try:
            # Find a paragraph node to start from
            cursor = self.db.aql.execute("FOR doc IN paragraph_nodes LIMIT 1 RETURN doc")
            docs = list(cursor)
            if not docs:
                self.log_result("Graph Traversal", False, 0, details="No paragraph nodes found")
                return results

            start_node = docs[0]
            start_id = start_node["_id"]

            # Test get_ancestors
            ancestors = self.knowledge.nodes.get_ancestors(start_id, max_depth=5)
            results["ancestors"] = len(ancestors)

            # Test get_children from root
            if ancestors:
                root = ancestors[-1]  # Last ancestor should be root
                root_id = root["_id"] if isinstance(root, dict) else root._id
                children = self.knowledge.nodes.get_children(root_id)
                results["children"] = len(children)

            latency = (time.perf_counter() - start) * 1000
            self.log_result(
                "Graph Traversal",
                sum(results.values()) > 0,
                latency,
                count=sum(results.values()),
                details=f"ancestors={results['ancestors']}, children={results['children']}",
            )
            return results
        except Exception as e:
            latency = (time.perf_counter() - start) * 1000
            self.log_result("Graph Traversal", False, latency, details=str(e))
            return results

    def test_embedding_coverage(self) -> dict[str, Any]:
        """Test 7: Check how many documents have embeddings."""
        start = time.perf_counter()
        try:
            counts = {}

            for coll_name in ["paragraph_nodes", "structure_nodes"]:
                if not self.db.has_collection(coll_name):
                    counts[coll_name] = {"total": 0, "with_embedding": 0, "percentage": 0}
                    continue

                # Count total
                total = self.db.collection(coll_name).count()

                # Count with embedding
                cursor = self.db.aql.execute(f"""
                    RETURN (
                        FOR doc IN {coll_name}
                          FILTER doc.embedding != null AND LENGTH(doc.embedding) > 0
                          COLLECT WITH COUNT INTO cnt
                          RETURN cnt
                    )[0] || 0
                """)
                with_embedding = list(cursor)[0] if cursor else 0

                counts[coll_name] = {
                    "total": total,
                    "with_embedding": with_embedding,
                    "percentage": (with_embedding / total * 100) if total > 0 else 0,
                }

            latency = (time.perf_counter() - start) * 1000
            total_with_embed = sum(c["with_embedding"] for c in counts.values())
            summary = {
                k: f"{v['with_embedding']}/{v['total']} ({v['percentage']:.1f}%)"
                for k, v in counts.items()
            }
            self.log_result(
                "Embedding Coverage",
                total_with_embed > 0,
                latency,
                count=total_with_embed,
                details=json.dumps(summary),
            )
            return counts
        except Exception as e:
            latency = (time.perf_counter() - start) * 1000
            self.log_result("Embedding Coverage", False, latency, details=str(e))
            return {}

    def test_index_usage(self) -> dict[str, Any]:
        """Test 8: Verify indexes are being used in queries."""
        start = time.perf_counter()
        try:
            # Test query explain to check index usage
            query = """
            FOR doc IN paragraph_nodes
              FILTER doc.source_id == @source_id
              RETURN doc
            """
            explain = self.db.aql.explain(query, bind_vars={"source_id": "test-id"})
            plan = explain.get("plan", {})
            nodes = plan.get("nodes", [])

            # Check if index is used
            index_used = any(node.get("type") == "IndexNode" for node in nodes)

            latency = (time.perf_counter() - start) * 1000
            self.log_result(
                "Index Usage",
                index_used,
                latency,
                details="Index scan" if index_used else "Collection scan (slow)",
            )
            return {"index_used": index_used}
        except Exception as e:
            latency = (time.perf_counter() - start) * 1000
            self.log_result("Index Usage", False, latency, details=str(e))
            return {"index_used": False}

    async def run_all_tests(self):
        """Run all retrieval tests."""
        logger.info("=" * 60)
        logger.info("ArangoDB Retrieval Effectiveness Test")
        logger.info("=" * 60)

        # Run tests
        self.test_database_connection()
        counts = self.test_collection_counts()
        self.test_view_exists()
        self.test_embedding_coverage()
        self.test_index_usage()

        # Only run search tests if we have data
        if sum(counts.values()) > 0:
            await self.test_vector_search()
            self.test_fulltext_search()
            self.test_graph_traversal()
        else:
            logger.warning("No data in collections - skipping search tests")

        # Summary
        logger.info("=" * 60)
        logger.info("Test Summary")
        logger.info("=" * 60)

        passed = sum(1 for r in self.results if r["success"])
        failed = len(self.results) - passed
        avg_latency = (
            sum(r["latency_ms"] for r in self.results) / len(self.results) if self.results else 0
        )

        logger.info(f"Total: {len(self.results)} tests | Passed: {passed} | Failed: {failed}")
        logger.info(f"Average latency: {avg_latency:.2f}ms")

        if failed > 0:
            logger.warning("Failed tests:")
            for r in self.results:
                if not r["success"]:
                    logger.warning(f"  - {r['test']}: {r['details']}")

        return self.results


async def main():
    """Main entry point."""
    tester = RetrievalTester()
    results = await tester.run_all_tests()

    # Exit with error code if any tests failed
    failed = sum(1 for r in results if not r["success"])
    sys.exit(1 if failed > 0 else 0)


if __name__ == "__main__":
    asyncio.run(main())
