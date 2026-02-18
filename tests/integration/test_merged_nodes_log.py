"""Integration tests for merged_nodes_log functionality.

These tests require a running ArangoDB instance.
They will be skipped if ArangoDB is not available.
"""

from __future__ import annotations

import os

import pytest

# Check if ArangoDB is available
ARANGO_URL = os.getenv("ARANGO_URL", "http://localhost:8529")
ARANGO_TEST_DB = os.getenv("ARANGO_TEST_DATABASE", "test_merged_nodes_log")

# Try to connect to ArangoDB
try:
    from arango import ArangoClient as ArangoDriver

    _driver = ArangoDriver(hosts=ARANGO_URL)
    _sys_db = _driver.db(
        "_system",
        username=os.getenv("ARANGO_USER", "root"),
        password=os.getenv("ARANGO_PASSWORD", ""),
    )
    _sys_db.properties()  # Test connection
    ARANGO_AVAILABLE = True
except Exception:
    ARANGO_AVAILABLE = False

pytestmark = pytest.mark.skipif(
    not ARANGO_AVAILABLE,
    reason="ArangoDB not available",
)


@pytest.fixture(scope="module")
def test_db():
    """Create a test database for integration tests."""
    from arango import ArangoClient as ArangoDriver

    driver = ArangoDriver(hosts=ARANGO_URL)
    sys_db = driver.db(
        "_system",
        username=os.getenv("ARANGO_USER", "root"),
        password=os.getenv("ARANGO_PASSWORD", ""),
    )

    # Create test database
    if not sys_db.has_database(ARANGO_TEST_DB):
        sys_db.create_database(ARANGO_TEST_DB)

    db = driver.db(
        ARANGO_TEST_DB,
        username=os.getenv("ARANGO_USER", "root"),
        password=os.getenv("ARANGO_PASSWORD", ""),
    )

    yield db

    # Cleanup: drop test database
    if sys_db.has_database(ARANGO_TEST_DB):
        sys_db.delete_database(ARANGO_TEST_DB)


@pytest.fixture
def arango_client(test_db):
    """Create ArangoClient connected to test database."""
    from arango_client import ArangoClient

    client = ArangoClient(
        url=ARANGO_URL,
        database=ARANGO_TEST_DB,
        username=os.getenv("ARANGO_USER", "root"),
        password=os.getenv("ARANGO_PASSWORD", ""),
        auto_init=True,
    )
    yield client


class TestCollectionCreation:
    """Tests for merged_nodes_log collection creation."""

    def test_merged_nodes_log_collection_created(self, arango_client):
        """Test that merged_nodes_log collection is created on init."""
        from arango_client.collections.schema import Collections

        assert arango_client.db.has_collection(Collections.MERGED_NODES_LOG)

    def test_collection_is_document_type(self, arango_client):
        """Test that merged_nodes_log is a document collection (not edge)."""
        from arango_client.collections.schema import Collections

        collection = arango_client.db.collection(Collections.MERGED_NODES_LOG)
        properties = collection.properties()
        assert properties["type"] == 2  # 2 = document, 3 = edge


class TestMergedNodeLogInsert:
    """Tests for inserting MergedNodeLog records."""

    def test_insert_auto_merge_log(self, arango_client):
        """Test inserting an auto_merge log record."""
        from arango_client.collections.schema import Collections
        from arango_client.models import MergedNodeLog

        log = MergedNodeLog(
            source_id="content_nodes/test_source_1",
            source_text="Source text for testing",
            source_type="paragraph",
            source_parent_id="structure_nodes/parent1",
            source_source_id="data_sources/doc1",
            target_id="content_nodes/test_target_1",
            target_text="Target text for testing",
            similarity=0.85,
            merge_reason="auto_merge",
        )

        collection = arango_client.db.collection(Collections.MERGED_NODES_LOG)
        result = collection.insert(log.to_document())

        assert "_key" in result
        assert "_id" in result

        # Verify document exists
        doc = collection.get(result["_key"])
        assert doc is not None
        assert doc["source_id"] == "content_nodes/test_source_1"
        assert doc["similarity"] == 0.85
        assert doc["merge_reason"] == "auto_merge"

    def test_insert_llm_merge_log(self, arango_client):
        """Test inserting an llm_merge log record with reasoning."""
        from arango_client.collections.schema import Collections
        from arango_client.models import MergedNodeLog

        reasoning = "LLM determined these texts are semantically equivalent duplicates"

        log = MergedNodeLog(
            source_id="content_nodes/test_source_2",
            source_text="Source text for LLM merge test",
            source_type="paragraph",
            target_id="content_nodes/test_target_2",
            target_text="Target text for LLM merge test",
            similarity=0.65,
            merge_reason="llm_merge",
            llm_reasoning=reasoning,
        )

        collection = arango_client.db.collection(Collections.MERGED_NODES_LOG)
        result = collection.insert(log.to_document())

        # Verify document
        doc = collection.get(result["_key"])
        assert doc["merge_reason"] == "llm_merge"
        assert doc["llm_reasoning"] == reasoning
        assert doc["similarity"] == 0.65

    def test_query_merge_logs_by_reason(self, arango_client):
        """Test querying merge logs filtered by merge_reason."""
        from arango_client.collections.schema import Collections
        from arango_client.models import MergedNodeLog

        # Insert both types
        collection = arango_client.db.collection(Collections.MERGED_NODES_LOG)

        auto_log = MergedNodeLog(
            source_id="content_nodes/query_test_auto",
            source_text="Auto merge test",
            source_type="paragraph",
            target_id="content_nodes/query_test_target",
            target_text="Target",
            similarity=0.90,
            merge_reason="auto_merge",
        )
        collection.insert(auto_log.to_document())

        llm_log = MergedNodeLog(
            source_id="content_nodes/query_test_llm",
            source_text="LLM merge test",
            source_type="paragraph",
            target_id="content_nodes/query_test_target2",
            target_text="Target 2",
            similarity=0.55,
            merge_reason="llm_merge",
            llm_reasoning="Test reasoning",
        )
        collection.insert(llm_log.to_document())

        # Query by merge_reason
        query = """
        FOR m IN @@collection
            FILTER m.merge_reason == @reason
            RETURN m
        """

        auto_results = list(
            arango_client.db.aql.execute(
                query,
                bind_vars={
                    "@collection": Collections.MERGED_NODES_LOG,
                    "reason": "auto_merge",
                },
            )
        )

        llm_results = list(
            arango_client.db.aql.execute(
                query,
                bind_vars={
                    "@collection": Collections.MERGED_NODES_LOG,
                    "reason": "llm_merge",
                },
            )
        )

        assert len(auto_results) >= 1
        assert len(llm_results) >= 1
        assert all(r["merge_reason"] == "auto_merge" for r in auto_results)
        assert all(r["merge_reason"] == "llm_merge" for r in llm_results)

    def test_from_document_reconstruction(self, arango_client):
        """Test that MergedNodeLog can be reconstructed from database document."""
        from arango_client.collections.schema import Collections
        from arango_client.models import MergedNodeLog

        original = MergedNodeLog(
            source_id="content_nodes/reconstruct_test",
            source_text="Original source text",
            source_type="table",
            source_parent_id="structure_nodes/section",
            source_source_id="data_sources/file",
            target_id="content_nodes/reconstruct_target",
            target_text="Original target text",
            similarity=0.78,
            merge_reason="auto_merge",
        )

        collection = arango_client.db.collection(Collections.MERGED_NODES_LOG)
        result = collection.insert(original.to_document())

        # Fetch and reconstruct
        doc = collection.get(result["_key"])
        reconstructed = MergedNodeLog.from_document(doc)

        assert reconstructed.source_id == original.source_id
        assert reconstructed.source_text == original.source_text
        assert reconstructed.source_type == original.source_type
        assert reconstructed.similarity == original.similarity
        assert reconstructed.merge_reason == original.merge_reason
