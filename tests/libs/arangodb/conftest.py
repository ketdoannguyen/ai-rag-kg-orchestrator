"""
Pytest configuration and fixtures for ArangoDB integration tests.

Provides test database setup/teardown and common fixtures.
Supports conditional teardown via KEEP_TEST_DB environment variable.
"""

import os
import uuid

import pytest
from arango_client import ArangoClient

# Test database name (isolated from production)
TEST_DATABASE = "test"
TEST_ARANGO_URL = "http://localhost:8529"
TEST_ARANGO_USER = "root"
TEST_ARANGO_PASSWORD = "test"


@pytest.fixture(scope="session")
def arango_client():
    """
    Create ArangoDB client for the test session.

    Creates a fresh test database before tests.
    Drops it after all tests complete UNLESS KEEP_TEST_DB=true.

    Environment Variables:
        KEEP_TEST_DB: Set to "true" to preserve test database for inspection.
    """
    # Create client connected to test database
    client = ArangoClient(
        url=TEST_ARANGO_URL,
        database=TEST_DATABASE,
        username=TEST_ARANGO_USER,
        password=TEST_ARANGO_PASSWORD,
        auto_init=True,
    )

    yield client

    # Conditional Teardown: Check KEEP_TEST_DB environment variable
    keep_db = os.getenv("KEEP_TEST_DB", "").lower() == "true"

    if keep_db:
        print(f"\n✅ Test DB preserved at '{TEST_DATABASE}' for inspection")
        print(f"   URL: {TEST_ARANGO_URL}")
        print(f"   Database: {TEST_DATABASE}")
        print(f"   Username: {TEST_ARANGO_USER}")
        print(f"   Password: {TEST_ARANGO_PASSWORD}")
    else:
        # Default behavior: drop test database for clean environment
        client.drop_database(confirm=True)

    client.close()


@pytest.fixture
def db(arango_client):
    """Get the database instance."""
    return arango_client.db


@pytest.fixture
def clean_collections(db):
    """Clean all collections before each test."""
    from arango_client.collections import ALL_DOCUMENT_COLLECTIONS, ALL_EDGE_COLLECTIONS

    for coll_name in ALL_DOCUMENT_COLLECTIONS + ALL_EDGE_COLLECTIONS:
        if db.has_collection(coll_name):
            db.collection(coll_name).truncate()

    yield

    # Also clean after test (only if not keeping DB)
    keep_db = os.getenv("KEEP_TEST_DB", "").lower() == "true"
    if not keep_db:
        for coll_name in ALL_DOCUMENT_COLLECTIONS + ALL_EDGE_COLLECTIONS:
            if db.has_collection(coll_name):
                db.collection(coll_name).truncate()


@pytest.fixture
def sample_uuid():
    """Generate a sample UUID for tests."""
    return uuid.uuid4()
