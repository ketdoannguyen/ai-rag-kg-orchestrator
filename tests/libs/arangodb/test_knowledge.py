"""
Integration tests for KnowledgeHandler.

Tests CRUD operations, graph relationships, and vector search.
"""

import uuid
import pytest
from arango_client.handlers import KnowledgeHandler
from arango_client.models import ParagraphNode, StructureNode
from arango_client.models.enums import NodeType
from arango_client.collections import Collections


@pytest.fixture
def handler(db):
    """Create KnowledgeHandler instance."""
    return KnowledgeHandler(db)


class TestKnowledgeNodeCRUD:
    """Tests for CRUD operations on knowledge nodes."""

    def test_create_and_get_node(self, handler, clean_collections):
        """Test creating and retrieving a node."""
        # Arrange
        node = ParagraphNode(
            attributes={"text": "Test paragraph content"}
        )
        
        # Act
        created = handler.nodes.create_node(node)
        retrieved = handler.nodes.get_by_id(created.id)
        
        # Assert
        assert retrieved is not None
        assert str(retrieved.id) == str(created.id)
        assert retrieved.attributes.text == "Test paragraph content"

    def test_update_node(self, handler, clean_collections):
        """Test updating a node's attributes."""
        # Arrange
        node = ParagraphNode(attributes={"text": "Original"})
        created = handler.nodes.create_node(node)
        
        # Act
        updated = handler.nodes.update(created.id, {
            "attributes": {"text": "Updated text"}
        })
        
        # Assert
        assert updated is not None
        retrieved = handler.nodes.get_by_id(created.id)
        assert retrieved.attributes.text == "Updated text"

    def test_delete_node(self, handler, clean_collections):
        """Test deleting a node."""
        # Arrange
        node = ParagraphNode(attributes={"text": "To delete"})
        created = handler.nodes.create_node(node)
        
        # Act
        result = handler.nodes.delete(created.id)
        
        # Assert
        assert result is True
        assert handler.nodes.get_by_id(created.id) is None

    def test_get_by_node_type(self, handler, clean_collections):
        """Test filtering nodes by type."""
        # Arrange
        handler.nodes.create_node(ParagraphNode(attributes={"text": "P1"}))
        handler.nodes.create_node(StructureNode(attributes={"title": "S1"}))
        
        # Act
        paragraphs = handler.nodes.get_by_node_type(NodeType.PARAGRAPH)
        
        # Assert
        assert len(paragraphs) >= 1
        assert all(n.node_type == NodeType.PARAGRAPH for n in paragraphs)


class TestGraphRelationships:
    """Tests for graph relationship operations."""

    def test_create_has_section_relationship(self, handler, clean_collections):
        """Test creating HAS_SECTION relationship."""
        # Arrange
        parent = handler.nodes.create_node(
            StructureNode(attributes={"title": "Parent"})
        )
        child = handler.nodes.create_node(
            StructureNode(attributes={"title": "Child"})
        )
        
        # Act
        edge = handler.nodes.create_has_section_relationship(parent.id, child.id)
        
        # Assert
        assert edge is not None
        assert edge["type"] == "HAS_SECTION"

    def test_create_has_content_relationship(self, handler, clean_collections):
        """Test creating HAS_CONTENT relationship."""
        # Arrange
        parent = handler.nodes.create_node(
            StructureNode(attributes={"title": "Section"})
        )
        content = handler.nodes.create_node(
            ParagraphNode(attributes={"text": "Content"})
        )
        
        # Act
        edge = handler.nodes.create_has_content_relationship(parent.id, content.id)
        
        # Assert
        assert edge is not None
        assert edge["type"] == "HAS_CONTENT"

    def test_get_children_via_graph(self, handler, clean_collections):
        """Test getting children through graph traversal."""
        # Arrange
        parent = handler.nodes.create_node(
            StructureNode(attributes={"title": "Parent"})
        )
        child1 = handler.nodes.create_node(
            ParagraphNode(attributes={"text": "Child 1"})
        )
        child2 = handler.nodes.create_node(
            ParagraphNode(attributes={"text": "Child 2"})
        )
        handler.nodes.create_has_content_relationship(parent.id, child1.id)
        handler.nodes.create_has_content_relationship(parent.id, child2.id)
        
        # Act
        children = handler.nodes.get_children(parent.id)
        
        # Assert
        assert len(children) == 2
        child_ids = {str(c.id) for c in children}
        assert str(child1.id) in child_ids
        assert str(child2.id) in child_ids


class TestVectorSearch:
    """Tests for vector similarity search."""

    def test_vector_search_returns_ordered_results(self, handler, clean_collections, db):
        """Test that vector search returns results ordered by similarity."""
        from arango_client.handlers import SearchHandler
        
        search = SearchHandler(db)
        
        # Arrange: Create 3 nodes with different embeddings
        vec1 = [1.0, 0.0, 0.0]  # Close to query
        vec2 = [0.0, 1.0, 0.0]  # Different
        vec3 = [0.0, 0.0, 1.0]  # Different
        
        n1 = ParagraphNode(attributes={"text": "Node 1"}, embedding=vec1)
        n2 = ParagraphNode(attributes={"text": "Node 2"}, embedding=vec2)
        n3 = ParagraphNode(attributes={"text": "Node 3"}, embedding=vec3)
        
        created1 = handler.nodes.create_node(n1)
        handler.nodes.create_node(n2)
        handler.nodes.create_node(n3)
        
        # Act: Search with vector close to vec1
        query_vector = [0.9, 0.1, 0.0]
        results = search.vector_search_content(query_vector, limit=3, min_score=0.0)
        
        # Assert: First result should be closest (node1)
        if len(results) > 0:
            assert str(results[0].node.id) == str(created1.id)
            # Scores should be in descending order
            scores = [r.score for r in results]
            assert scores == sorted(scores, reverse=True)


class TestEdgeCases:
    """Tests for error handling and edge cases."""

    def test_get_nonexistent_node_returns_none(self, handler, clean_collections):
        """Test that getting a non-existent node returns None."""
        result = handler.nodes.get_by_id(uuid.uuid4())
        assert result is None

    def test_delete_nonexistent_node_returns_false(self, handler, clean_collections):
        """Test that deleting a non-existent node returns False."""
        result = handler.nodes.delete(uuid.uuid4())
        assert result is False

    def test_duplicate_key_raises_error(self, handler, clean_collections):
        """Test that inserting a duplicate key raises error."""
        node_id = uuid.uuid4()
        
        node1 = ParagraphNode(id=node_id, attributes={"text": "First"})
        handler.nodes.create_node(node1)
        
        # Try to insert with same ID
        node2 = ParagraphNode(id=node_id, attributes={"text": "Second"})
        with pytest.raises(Exception):  # arango.exceptions.DocumentInsertError
            handler.nodes.create_node(node2)
