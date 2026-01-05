"""
Chat handler for ArangoDB.

Handles ChatSession, ChatMessage, and Feedback operations.
"""

from __future__ import annotations

import uuid
from typing import Any, Optional

from arango.database import StandardDatabase

from ..collections.schema import Collections
from ..models.chat import ChatSession, ChatMessage, Feedback
from ..models.enums import FeedbackStatus
from .base_handler import BaseHandler


class ChatSessionHandler(BaseHandler[ChatSession]):
    """Handler for ChatSession CRUD operations."""

    def __init__(self, db: StandardDatabase):
        """Initialize ChatSession handler."""
        super().__init__(db, ChatSession, Collections.CHAT_SESSIONS)

    def get_by_user_identifier(
        self,
        user_identifier: str,
        limit: Optional[int] = None,
        offset: Optional[int] = None,
    ) -> list[ChatSession]:
        """
        Get chat sessions by user identifier.
        
        Args:
            user_identifier: User identifier (session cookie or user ID)
            limit: Maximum number of sessions to return
            offset: Number of records to skip
            
        Returns:
            List of ChatSession instances
        """
        return self.get_by_filter(
            {"user_identifier": user_identifier},
            limit=limit,
            offset=offset,
            sort_field="started_at",
            sort_desc=True,
        )

    def get_messages_by_session(
        self,
        session_id: uuid.UUID | str,
        limit: Optional[int] = None,
    ) -> list[ChatMessage]:
        """
        Get all messages for a chat session.
        
        Args:
            session_id: Chat session ID
            limit: Maximum number of messages to return
            
        Returns:
            List of ChatMessage instances
        """
        message_handler = ChatMessageHandler(self.db)
        return message_handler.get_by_session(session_id, limit=limit)


class ChatMessageHandler(BaseHandler[ChatMessage]):
    """Handler for ChatMessage CRUD operations."""

    def __init__(self, db: StandardDatabase):
        """Initialize ChatMessage handler."""
        super().__init__(db, ChatMessage, Collections.CHAT_MESSAGES)

    def get_by_session(
        self,
        session_id: uuid.UUID | str,
        limit: Optional[int] = None,
        offset: Optional[int] = None,
    ) -> list[ChatMessage]:
        """
        Get messages by session ID.
        
        Args:
            session_id: Chat session ID
            limit: Maximum number of messages to return
            offset: Number of records to skip
            
        Returns:
            List of ChatMessage instances (sorted by creation time ascending)
        """
        return self.get_by_filter(
            {"session_id": str(session_id)},
            limit=limit,
            offset=offset,
            sort_field="created_at",
            sort_desc=False,  # Ascending for chronological order
        )

    def get_by_role(
        self,
        session_id: uuid.UUID | str,
        role: str,
        limit: Optional[int] = None,
    ) -> list[ChatMessage]:
        """
        Get messages by session ID and role.
        
        Args:
            session_id: Chat session ID
            role: Message role ("user" or "assistant")
            limit: Maximum number of messages to return
            
        Returns:
            List of ChatMessage instances
        """
        return self.get_by_filter(
            {"session_id": str(session_id), "role": role},
            limit=limit,
            sort_field="created_at",
            sort_desc=False,
        )

    def get_feedbacks_by_message(
        self,
        message_id: uuid.UUID | str,
    ) -> list[Feedback]:
        """
        Get all feedbacks for a message.
        
        Args:
            message_id: Chat message ID
            
        Returns:
            List of Feedback instances
        """
        feedback_handler = FeedbackHandler(self.db)
        return feedback_handler.get_by_filter({"message_id": str(message_id)})


class FeedbackHandler(BaseHandler[Feedback]):
    """Handler for Feedback CRUD operations."""

    def __init__(self, db: StandardDatabase):
        """Initialize Feedback handler."""
        super().__init__(db, Feedback, Collections.FEEDBACKS)

    def get_by_message(
        self,
        message_id: uuid.UUID | str,
    ) -> list[Feedback]:
        """
        Get feedbacks by message ID.
        
        Args:
            message_id: Chat message ID
            
        Returns:
            List of Feedback instances
        """
        return self.get_by_filter({"message_id": str(message_id)})

    def get_by_status(
        self,
        status: FeedbackStatus | str,
        limit: Optional[int] = None,
        offset: Optional[int] = None,
    ) -> list[Feedback]:
        """
        Get feedbacks by status.
        
        Args:
            status: Feedback status (pending, reviewed, ignored)
            limit: Maximum number of feedbacks to return
            offset: Number of records to skip
            
        Returns:
            List of Feedback instances
        """
        status_value = status.value if isinstance(status, FeedbackStatus) else status
        return self.get_by_filter(
            {"status": status_value},
            limit=limit,
            offset=offset,
            sort_desc=True,
        )

    def get_pending_feedbacks(
        self,
        limit: Optional[int] = None,
    ) -> list[Feedback]:
        """
        Get all pending feedbacks.
        
        Args:
            limit: Maximum number of feedbacks to return
            
        Returns:
            List of pending Feedback instances
        """
        return self.get_by_status(FeedbackStatus.PENDING, limit=limit)


# Convenience class that groups all chat handlers
class ChatHandler:
    """
    Convenience class grouping all chat-related handlers.
    
    Usage:
        >>> handler = ChatHandler(db)
        >>> session = handler.sessions.create(ChatSession(...))
        >>> message = handler.messages.create(ChatMessage(...))
        >>> feedback = handler.feedbacks.create(Feedback(...))
    """

    def __init__(self, db: StandardDatabase):
        """Initialize all chat handlers."""
        self.db = db
        self.sessions = ChatSessionHandler(db)
        self.messages = ChatMessageHandler(db)
        self.feedbacks = FeedbackHandler(db)
