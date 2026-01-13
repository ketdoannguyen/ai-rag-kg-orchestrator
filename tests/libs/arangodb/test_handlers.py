"""
Comprehensive integration tests for all ArangoDB handlers.

Tests use REAL database connections (no mocks) following the pattern
established in test_seed_visual_data.py.

Run with: pytest tests/libs/arangodb/test_handlers.py -v
"""

import uuid

import pytest
from arango_client.collections import RelationshipTypes
from arango_client.handlers import (
    ChatHandler,
    KnowledgeHandler,
    MiningHandler,
    SearchHandler,
    SystemAdminHandler,
)
from arango_client.models import (
    ChatMessage,
    ChatSession,
    DataSource,
    FAQStatus,
    Feedback,
    FeedbackStatus,
    NodeType,
    ParagraphNode,
    SourceType,
    StructureNode,
    SuggestedFAQ,
    User,
    UserRole,
)

# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def knowledge_handler(db):
    """Create KnowledgeHandler instance."""
    return KnowledgeHandler(db)


@pytest.fixture
def search_handler(db):
    """Create SearchHandler instance."""
    return SearchHandler(db)


@pytest.fixture
def chat_handler(db):
    """Create ChatHandler instance."""
    return ChatHandler(db)


@pytest.fixture
def mining_handler(db):
    """Create MiningHandler instance."""
    return MiningHandler(db)


@pytest.fixture
def admin_handler(db):
    """Create SystemAdminHandler instance."""
    return SystemAdminHandler(db)


# =============================================================================
# Knowledge Handler Tests
# =============================================================================


class TestKnowledgeNodeOperations:
    """Tests for knowledge node CRUD and graph operations."""

    def test_create_paragraph_node(self, knowledge_handler, clean_collections):
        """Test creating a paragraph node."""
        node = ParagraphNode(attributes={"text": "Test paragraph."})
        created = knowledge_handler.nodes.create_node(node)

        assert created is not None
        assert created.node_type == NodeType.PARAGRAPH
        assert created.attributes.text == "Test paragraph."

    def test_create_structure_node(self, knowledge_handler, clean_collections):
        """Test creating a structure node."""
        node = StructureNode(level=2, attributes={"title": "Chapter 1"})
        created = knowledge_handler.nodes.create_node(node)

        assert created is not None
        assert created.node_type == NodeType.STRUCTURE
        assert created.level == 2

    def test_get_by_id_returns_correct_type(self, knowledge_handler, clean_collections):
        """Test that get_by_id returns correct node type."""
        para = knowledge_handler.nodes.create_node(ParagraphNode(attributes={"text": "Paragraph"}))
        struct = knowledge_handler.nodes.create_node(
            StructureNode(level=2, attributes={"title": "Section"})
        )

        retrieved_para = knowledge_handler.nodes.get_by_id(para.id)
        retrieved_struct = knowledge_handler.nodes.get_by_id(struct.id)

        assert isinstance(retrieved_para, ParagraphNode)
        assert isinstance(retrieved_struct, StructureNode)

    def test_update_node(self, knowledge_handler, clean_collections):
        """Test updating a node."""
        node = knowledge_handler.nodes.create_node(ParagraphNode(attributes={"text": "Original"}))
        knowledge_handler.nodes.update(node.id, {"attributes": {"text": "Updated"}})
        retrieved = knowledge_handler.nodes.get_by_id(node.id)

        assert retrieved.attributes.text == "Updated"

    def test_delete_node(self, knowledge_handler, clean_collections):
        """Test deleting a node."""
        node = knowledge_handler.nodes.create_node(ParagraphNode(attributes={"text": "To delete"}))
        result = knowledge_handler.nodes.delete(node.id)

        assert result is True
        assert knowledge_handler.nodes.get_by_id(node.id) is None

    def test_get_by_node_type(self, knowledge_handler, clean_collections):
        """Test filtering by node type."""
        knowledge_handler.nodes.create_node(ParagraphNode(attributes={"text": "P1"}))
        knowledge_handler.nodes.create_node(ParagraphNode(attributes={"text": "P2"}))
        knowledge_handler.nodes.create_node(StructureNode(level=2, attributes={"title": "S1"}))

        paragraphs = knowledge_handler.nodes.get_by_node_type(NodeType.PARAGRAPH)

        assert len(paragraphs) >= 2
        assert all(p.node_type == NodeType.PARAGRAPH for p in paragraphs)

    def test_exists(self, knowledge_handler, clean_collections):
        """Test checking if node exists."""
        node = knowledge_handler.nodes.create_node(ParagraphNode(attributes={"text": "Test"}))

        assert knowledge_handler.nodes.exists(node.id) is True
        assert knowledge_handler.nodes.exists(uuid.uuid4()) is False


class TestGraphRelationships:
    """Tests for graph relationship operations."""

    def test_create_has_section_relationship(self, knowledge_handler, clean_collections):
        """Test creating HAS_SECTION relationship."""
        parent = knowledge_handler.nodes.create_node(
            StructureNode(level=1, attributes={"title": "Root"})
        )
        child = knowledge_handler.nodes.create_node(
            StructureNode(level=2, attributes={"title": "Chapter"})
        )

        edge = knowledge_handler.nodes.create_has_section_relationship(parent.id, child.id)

        assert edge["type"] == RelationshipTypes.HAS_SECTION

    def test_create_has_content_relationship(self, knowledge_handler, clean_collections):
        """Test creating HAS_CONTENT relationship."""
        section = knowledge_handler.nodes.create_node(
            StructureNode(level=2, attributes={"title": "Intro"})
        )
        paragraph = knowledge_handler.nodes.create_node(
            ParagraphNode(attributes={"text": "Welcome."})
        )

        edge = knowledge_handler.nodes.create_has_content_relationship(section.id, paragraph.id)

        assert edge["type"] == RelationshipTypes.HAS_CONTENT

    def test_get_children(self, knowledge_handler, clean_collections):
        """Test getting children of a node."""
        parent = knowledge_handler.nodes.create_node(
            StructureNode(level=1, attributes={"title": "Parent"})
        )
        child1 = knowledge_handler.nodes.create_node(ParagraphNode(attributes={"text": "Child 1"}))
        child2 = knowledge_handler.nodes.create_node(ParagraphNode(attributes={"text": "Child 2"}))

        knowledge_handler.nodes.create_has_content_relationship(parent.id, child1.id)
        knowledge_handler.nodes.create_has_content_relationship(parent.id, child2.id)

        children = knowledge_handler.nodes.get_children(parent.id)

        assert len(children) == 2

    def test_get_descendants(self, knowledge_handler, clean_collections):
        """Test getting all descendants (deep traversal)."""
        root = knowledge_handler.nodes.create_node(
            StructureNode(level=1, attributes={"title": "Root"})
        )
        chapter = knowledge_handler.nodes.create_node(
            StructureNode(level=2, attributes={"title": "Chapter"})
        )
        paragraph = knowledge_handler.nodes.create_node(
            ParagraphNode(attributes={"text": "Content"})
        )

        knowledge_handler.nodes.create_has_section_relationship(root.id, chapter.id)
        knowledge_handler.nodes.create_has_content_relationship(chapter.id, paragraph.id)

        descendants = knowledge_handler.nodes.get_descendants(root.id)

        assert len(descendants) >= 2

    def test_get_ancestors(self, knowledge_handler, clean_collections):
        """Test getting ancestors (upward traversal)."""
        root = knowledge_handler.nodes.create_node(
            StructureNode(level=1, attributes={"title": "Root"})
        )
        chapter = knowledge_handler.nodes.create_node(
            StructureNode(level=2, attributes={"title": "Chapter"})
        )
        paragraph = knowledge_handler.nodes.create_node(
            ParagraphNode(attributes={"text": "Content"})
        )

        knowledge_handler.nodes.create_has_section_relationship(root.id, chapter.id)
        knowledge_handler.nodes.create_has_content_relationship(chapter.id, paragraph.id)

        ancestors = knowledge_handler.nodes.get_ancestors(paragraph.id)

        assert len(ancestors) >= 2

    def test_get_parent(self, knowledge_handler, clean_collections):
        """Test getting immediate parent."""
        parent = knowledge_handler.nodes.create_node(
            StructureNode(level=1, attributes={"title": "Parent"})
        )
        child = knowledge_handler.nodes.create_node(ParagraphNode(attributes={"text": "Child"}))
        knowledge_handler.nodes.create_has_content_relationship(parent.id, child.id)

        found_parent = knowledge_handler.nodes.get_parent(child.id)

        assert found_parent is not None
        assert str(found_parent.id) == str(parent.id)


class TestEmbeddings:
    """Tests for embedding operations."""

    def test_update_embedding(self, knowledge_handler, clean_collections):
        """Test updating embeddings on a node."""
        node = knowledge_handler.nodes.create_node(ParagraphNode(attributes={"text": "Test"}))

        vector = [0.1, 0.2, 0.3]
        knowledge_handler.nodes.update_embedding(node.id, vector)
        retrieved = knowledge_handler.nodes.get_by_id(node.id)

        assert retrieved.embedding == vector

    def test_create_with_embedding(self, knowledge_handler, clean_collections):
        """Test creating a node with embedding."""
        node = ParagraphNode(attributes={"text": "Embedded"})
        vector = [0.5, 0.6, 0.7]

        created = knowledge_handler.nodes.create_with_embedding(node, vector)

        assert created.embedding == vector


# =============================================================================
# Search Handler Tests
# =============================================================================


class TestSearchOperations:
    """Tests for vector and hybrid search."""

    def test_vector_search_content(self, knowledge_handler, search_handler, clean_collections):
        """Test vector search on content nodes."""
        knowledge_handler.nodes.create_node(
            ParagraphNode(attributes={"text": "ML"}, embedding=[1.0, 0.0, 0.0])
        )
        knowledge_handler.nodes.create_node(
            ParagraphNode(attributes={"text": "DL"}, embedding=[0.9, 0.1, 0.0])
        )

        results = search_handler.vector_search_content([0.95, 0.05, 0.0], limit=3)

        if len(results) > 0:
            assert results[0].score >= results[-1].score

    def test_hybrid_search_with_type_filter(
        self, knowledge_handler, search_handler, clean_collections
    ):
        """Test hybrid search with node type filter."""
        knowledge_handler.nodes.create_node(
            ParagraphNode(attributes={"text": "Para"}, embedding=[1.0, 0.0, 0.0])
        )
        knowledge_handler.nodes.create_node(
            StructureNode(level=2, attributes={"title": "Struct"}, embedding=[1.0, 0.0, 0.0])
        )

        results = search_handler.hybrid_search(
            query_vector=[1.0, 0.0, 0.0], filters={"node_type": "paragraph"}, limit=10
        )

        for result in results:
            assert result.node.node_type == NodeType.PARAGRAPH

    def test_fulltext_search(self, knowledge_handler, search_handler, clean_collections):
        """Test full-text search.

        Note: ArangoSearch view has consolidation interval (~10s default).
        We need to wait briefly for new documents to be indexed.
        """
        import time

        knowledge_handler.nodes.create_node(
            ParagraphNode(attributes={"text": "Machine learning algorithms"})
        )

        # Wait for ArangoSearch view consolidation
        time.sleep(2)

        results = search_handler.fulltext_search("machine")

        assert any("machine" in str(r.attributes.text).lower() for r in results)


# =============================================================================
# Chat Handler Tests
# =============================================================================


class TestChatOperations:
    """Tests for chat session, message, and feedback operations."""

    def test_create_session(self, chat_handler, clean_collections):
        """Test creating a chat session."""
        session = ChatSession(user_identifier="user123")
        created = chat_handler.sessions.create(session)

        assert created is not None
        assert created.user_identifier == "user123"

    def test_get_sessions_by_user(self, chat_handler, clean_collections):
        """Test getting sessions by user identifier."""
        chat_handler.sessions.create(ChatSession(user_identifier="alice"))
        chat_handler.sessions.create(ChatSession(user_identifier="alice"))
        chat_handler.sessions.create(ChatSession(user_identifier="bob"))

        alice_sessions = chat_handler.sessions.get_by_user_identifier("alice")

        assert len(alice_sessions) == 2

    def test_create_message(self, chat_handler, clean_collections):
        """Test creating a message in a session."""
        session = chat_handler.sessions.create(ChatSession(user_identifier="user123"))
        message = ChatMessage(session_id=session.id, role="user", content="Hello!")

        created = chat_handler.messages.create(message)

        assert created is not None
        assert created.role == "user"

    def test_get_messages_by_session(self, chat_handler, clean_collections):
        """Test getting all messages in a session."""
        session = chat_handler.sessions.create(ChatSession(user_identifier="user"))
        chat_handler.messages.create(ChatMessage(session_id=session.id, role="user", content="Hi"))
        chat_handler.messages.create(
            ChatMessage(session_id=session.id, role="assistant", content="Hello!")
        )

        messages = chat_handler.messages.get_by_session(session.id)

        assert len(messages) == 2

    def test_get_messages_by_role(self, chat_handler, clean_collections):
        """Test filtering messages by role."""
        session = chat_handler.sessions.create(ChatSession(user_identifier="user"))
        chat_handler.messages.create(ChatMessage(session_id=session.id, role="user", content="Q1"))
        chat_handler.messages.create(ChatMessage(session_id=session.id, role="user", content="Q2"))
        chat_handler.messages.create(
            ChatMessage(session_id=session.id, role="assistant", content="A1")
        )

        user_messages = chat_handler.messages.get_by_role(session.id, "user")

        assert len(user_messages) == 2

    def test_create_feedback(self, chat_handler, clean_collections):
        """Test creating feedback for a message."""
        session = chat_handler.sessions.create(ChatSession(user_identifier="user"))
        message = chat_handler.messages.create(
            ChatMessage(session_id=session.id, role="assistant", content="Reply")
        )

        feedback = Feedback(message_id=message.id, is_positive=True, comment="Great!")
        created = chat_handler.feedbacks.create(feedback)

        assert created is not None
        assert created.is_positive is True

    def test_get_pending_feedbacks(self, chat_handler, clean_collections):
        """Test getting pending feedbacks."""
        session = chat_handler.sessions.create(ChatSession(user_identifier="user"))
        msg = chat_handler.messages.create(
            ChatMessage(session_id=session.id, role="assistant", content="Test")
        )
        chat_handler.feedbacks.create(
            Feedback(message_id=msg.id, is_positive=False, status=FeedbackStatus.PENDING)
        )

        pending = chat_handler.feedbacks.get_pending_feedbacks()

        assert len(pending) >= 1


# =============================================================================
# Mining Handler Tests
# =============================================================================


class TestMiningOperations:
    """Tests for suggested FAQ operations."""

    def test_create_faq(self, mining_handler, clean_collections):
        """Test creating a suggested FAQ."""
        faq = SuggestedFAQ(representative_question="How to reset password?", frequency=1)
        created = mining_handler.faqs.create(faq)

        assert created is not None
        assert created.representative_question == "How to reset password?"

    def test_get_by_status(self, mining_handler, clean_collections):
        """Test filtering FAQs by status."""
        mining_handler.faqs.create(
            SuggestedFAQ(representative_question="Q1", status=FAQStatus.PENDING)
        )
        mining_handler.faqs.create(
            SuggestedFAQ(representative_question="Q2", status=FAQStatus.APPROVED)
        )

        pending = mining_handler.faqs.get_by_status(FAQStatus.PENDING)

        assert all(f.status == FAQStatus.PENDING for f in pending)

    def test_approve_faq(self, mining_handler, clean_collections):
        """Test approving a FAQ."""
        faq = mining_handler.faqs.create(SuggestedFAQ(representative_question="Good question?"))
        approved = mining_handler.faqs.approve(faq.id)

        assert approved.status == FAQStatus.APPROVED

    def test_reject_faq(self, mining_handler, clean_collections):
        """Test rejecting a FAQ."""
        faq = mining_handler.faqs.create(SuggestedFAQ(representative_question="Bad question?"))
        rejected = mining_handler.faqs.reject(faq.id)

        assert rejected.status == FAQStatus.REJECTED

    def test_increment_frequency(self, mining_handler, clean_collections):
        """Test incrementing FAQ frequency."""
        faq = mining_handler.faqs.create(
            SuggestedFAQ(representative_question="Popular?", frequency=5)
        )
        updated = mining_handler.faqs.increment_frequency(faq.id)

        assert updated.frequency == 6

    def test_get_by_frequency(self, mining_handler, clean_collections):
        """Test getting FAQs by minimum frequency."""
        mining_handler.faqs.create(SuggestedFAQ(representative_question="Rare", frequency=1))
        mining_handler.faqs.create(SuggestedFAQ(representative_question="Common", frequency=10))
        mining_handler.faqs.create(
            SuggestedFAQ(representative_question="Very Common", frequency=50)
        )

        frequent = mining_handler.faqs.get_by_frequency(min_frequency=10)

        assert all(f.frequency >= 10 for f in frequent)


# =============================================================================
# System Admin Handler Tests
# =============================================================================


class TestAdminOperations:
    """Tests for user and data source operations."""

    def test_create_user(self, admin_handler, clean_collections):
        """Test creating a user."""
        user = User(username="john", hashed_password="hash123", role=UserRole.EDITOR)
        created = admin_handler.users.create(user)

        assert created is not None
        assert created.username == "john"

    def test_get_by_username(self, admin_handler, clean_collections):
        """Test getting user by username."""
        admin_handler.users.create(User(username="alice", hashed_password="hash123"))

        found = admin_handler.users.get_by_username("alice")

        assert found is not None
        assert found.username == "alice"

    def test_activate_deactivate(self, admin_handler, clean_collections):
        """Test activating and deactivating users."""
        user = admin_handler.users.create(
            User(username="bob", hashed_password="hash123", is_active=True)
        )

        deactivated = admin_handler.users.deactivate(user.id)
        assert deactivated.is_active is False

        activated = admin_handler.users.activate(user.id)
        assert activated.is_active is True

    def test_create_data_source(self, admin_handler, clean_collections):
        """Test creating a data source."""
        ds = DataSource(name="Test PDF", type=SourceType.PDF, config={"path": "/test.pdf"})
        created = admin_handler.sources.create(ds)

        assert created is not None
        assert created.type == SourceType.PDF

    def test_get_data_sources_by_status(self, admin_handler, clean_collections):
        """Test filtering data sources by status."""
        admin_handler.sources.create(
            DataSource(name="Pending", type=SourceType.WEB, config={}, status="processing")
        )
        admin_handler.sources.create(
            DataSource(name="Done", type=SourceType.WEB, config={}, status="active")
        )

        pending = admin_handler.sources.get_by_status("processing")

        assert all(ds.status == "processing" for ds in pending)


# =============================================================================
# Bulk and Edge Case Tests
# =============================================================================


class TestBulkAndEdgeCases:
    """Tests for bulk operations and edge cases."""

    def test_bulk_create_nodes(self, knowledge_handler, clean_collections):
        """Test bulk creating multiple nodes."""
        nodes = [ParagraphNode(attributes={"text": f"P{i}"}) for i in range(5)]
        created = knowledge_handler.nodes.bulk_create(nodes)

        assert len(created) == 5

    def test_bulk_delete_nodes(self, knowledge_handler, clean_collections):
        """Test bulk deleting multiple nodes."""
        nodes = [
            knowledge_handler.nodes.create_node(ParagraphNode(attributes={"text": f"P{i}"}))
            for i in range(3)
        ]
        ids = [n.id for n in nodes]
        deleted_count = knowledge_handler.nodes.bulk_delete(ids)

        assert deleted_count == 3

    def test_get_nonexistent_returns_none(self, knowledge_handler, clean_collections):
        """Test getting non-existent ID returns None."""
        result = knowledge_handler.nodes.get_by_id(uuid.uuid4())
        assert result is None

    def test_delete_nonexistent_returns_false(self, knowledge_handler, clean_collections):
        """Test deleting non-existent ID returns False."""
        result = knowledge_handler.nodes.delete(uuid.uuid4())
        assert result is False

    def test_update_nonexistent_returns_none(self, knowledge_handler, clean_collections):
        """Test updating non-existent ID returns None."""
        result = knowledge_handler.nodes.update(uuid.uuid4(), {"text": "Updated"})
        assert result is None
