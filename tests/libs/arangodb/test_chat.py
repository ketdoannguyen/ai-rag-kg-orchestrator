"""
Integration tests for ChatHandler.

Tests chat sessions, messages, and feedback operations.
"""

import pytest
from arango_client.handlers import ChatHandler
from arango_client.models import ChatMessage, ChatSession, Feedback


@pytest.fixture
def handler(db):
    """Create ChatHandler instance."""
    return ChatHandler(db)


class TestChatSessions:
    """Tests for chat session operations."""

    def test_create_and_get_session(self, handler, clean_collections):
        """Test creating and retrieving a session."""
        # Arrange
        session = ChatSession(user_identifier="user123")

        # Act
        created = handler.sessions.create(session)
        retrieved = handler.sessions.get_by_id(created.id)

        # Assert
        assert retrieved is not None
        assert retrieved.user_identifier == "user123"

    def test_get_sessions_by_user(self, handler, clean_collections):
        """Test filtering sessions by user identifier."""
        # Arrange
        handler.sessions.create(ChatSession(user_identifier="alice"))
        handler.sessions.create(ChatSession(user_identifier="bob"))
        handler.sessions.create(ChatSession(user_identifier="alice"))

        # Act
        alice_sessions = handler.sessions.get_by_user_identifier("alice")

        # Assert
        assert len(alice_sessions) == 2
        assert all(s.user_identifier == "alice" for s in alice_sessions)


class TestChatMessages:
    """Tests for chat message operations."""

    def test_create_message(self, handler, clean_collections):
        """Test creating a chat message."""
        # Arrange
        session = handler.sessions.create(ChatSession(user_identifier="user1"))
        message = ChatMessage(session_id=session.id, role="user", content="Hello, world!")

        # Act
        created = handler.messages.create(message)

        # Assert
        assert created is not None
        assert created.role == "user"
        assert created.content == "Hello, world!"

    def test_get_messages_in_chronological_order(self, handler, clean_collections):
        """Test that messages are returned in chronological order."""
        import time

        # Arrange
        session = handler.sessions.create(ChatSession(user_identifier="user1"))

        msg1 = handler.messages.create(
            ChatMessage(session_id=session.id, role="user", content="Hi")
        )
        time.sleep(0.01)  # Ensure different timestamps
        msg2 = handler.messages.create(
            ChatMessage(session_id=session.id, role="assistant", content="Hello!")
        )
        time.sleep(0.01)
        msg3 = handler.messages.create(
            ChatMessage(session_id=session.id, role="user", content="How are you?")
        )

        # Act
        messages = handler.messages.get_by_session(session.id)

        # Assert
        assert len(messages) == 3
        assert messages[0].content == "Hi"
        assert messages[1].content == "Hello!"
        assert messages[2].content == "How are you?"

    def test_get_messages_by_role(self, handler, clean_collections):
        """Test filtering messages by role."""
        # Arrange
        session = handler.sessions.create(ChatSession(user_identifier="user1"))
        handler.messages.create(ChatMessage(session_id=session.id, role="user", content="Q1"))
        handler.messages.create(ChatMessage(session_id=session.id, role="assistant", content="A1"))
        handler.messages.create(ChatMessage(session_id=session.id, role="user", content="Q2"))

        # Act
        user_messages = handler.messages.get_by_role(session.id, "user")

        # Assert
        assert len(user_messages) == 2
        assert all(m.role == "user" for m in user_messages)


class TestFeedback:
    """Tests for feedback operations."""

    def test_create_feedback(self, handler, clean_collections):
        """Test creating feedback for a message."""
        # Arrange
        session = handler.sessions.create(ChatSession(user_identifier="user1"))
        message = handler.messages.create(
            ChatMessage(session_id=session.id, role="assistant", content="Response")
        )
        feedback = Feedback(message_id=message.id, is_positive=True, comment="Great answer!")

        # Act
        created = handler.feedbacks.create(feedback)

        # Assert
        assert created is not None
        assert created.is_positive is True
        assert created.comment == "Great answer!"

    def test_get_pending_feedbacks(self, handler, clean_collections):
        """Test retrieving pending feedbacks."""
        # Arrange
        session = handler.sessions.create(ChatSession(user_identifier="user1"))
        message = handler.messages.create(
            ChatMessage(session_id=session.id, role="assistant", content="Response")
        )
        handler.feedbacks.create(Feedback(message_id=message.id, is_positive=False, comment="Bad"))

        # Act
        pending = handler.feedbacks.get_pending_feedbacks()

        # Assert
        assert len(pending) >= 1
        assert all(f.status == "pending" for f in pending)
