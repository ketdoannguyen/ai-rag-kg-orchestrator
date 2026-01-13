"""
Integration tests for ArangoClient connection and initialization.

Tests that the client properly connects and auto-creates schema.
"""

import pytest
from arango_client import ArangoClient
from arango_client.collections import (
    Collections,
    EdgeCollections,
    KNOWLEDGE_GRAPH_NAME,
    KNOWLEDGE_VECTOR_VIEW_NAME,
)


class TestClientConnection:
    """Tests for client connection and initialization."""

    def test_client_connects_successfully(self, arango_client):
        """Test that client connects to ArangoDB."""
        assert arango_client is not None
        assert arango_client.db is not None

    def test_client_creates_document_collections(self, db):
        """Test that all document collections are created."""
        expected = [
            Collections.KNOWLEDGE_NODES,
            Collections.CHAT_SESSIONS,
            Collections.CHAT_MESSAGES,
            Collections.FEEDBACKS,
            Collections.USERS,
            Collections.DATA_SOURCES,
            Collections.SUGGESTED_FAQS,
        ]
        for coll_name in expected:
            assert db.has_collection(coll_name), f"Missing collection: {coll_name}"

    def test_client_creates_edge_collections(self, db):
        """Test that edge collections are created."""
        assert db.has_collection(EdgeCollections.KNOWLEDGE_EDGES)

    def test_client_creates_graph(self, db):
        """Test that knowledge graph is created."""
        assert db.has_graph(KNOWLEDGE_GRAPH_NAME)

    def test_client_creates_vector_view(self, db):
        """Test that ArangoSearch view for vectors is created."""
        views = [v["name"] for v in db.views()]
        assert KNOWLEDGE_VECTOR_VIEW_NAME in views


class TestClientErrors:
    """Tests for error handling."""

    def test_wrong_password_raises_error(self):
        """Test that wrong password raises authentication error."""
        with pytest.raises(Exception):  # arango.exceptions.ServerConnectionError
            ArangoClient(
                url="http://localhost:8529",
                database="test_db",
                username="root",
                password="wrong_password_12345",
                auto_init=True,
            )

    def test_wrong_host_raises_error(self):
        """Test that wrong host raises connection error."""
        with pytest.raises(Exception):
            ArangoClient(
                url="http://nonexistent-host:8529",
                database="test_db",
                username="root",
                password="",
                auto_init=True,
            )
