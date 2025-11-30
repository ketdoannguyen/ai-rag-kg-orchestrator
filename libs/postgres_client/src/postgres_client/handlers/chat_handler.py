"""
CRUD handlers for Chat models.

Handles ChatSession, ChatMessage, and Feedback operations.
"""

from __future__ import annotations

import uuid
from typing import Optional, List, Any
from sqlalchemy.ext.asyncio import AsyncSession

from .base_handler import BaseHandler
from ..models.chat import ChatSession, ChatMessage, Feedback
from ..orm_models.chat import ChatSessionORM, ChatMessageORM, FeedbackORM


class ChatSessionHandler(BaseHandler[ChatSession, ChatSessionORM]):
    """Handler for ChatSession CRUD operations."""

    def __init__(self, session: AsyncSession):
        """Initialize ChatSession handler."""
        super().__init__(session, ChatSession, ChatSessionORM)

    async def get_by_user_identifier(
        self,
        user_identifier: str,
        limit: Optional[int] = None,
        order_by: Optional[Any] = None,
    ) -> List[ChatSession]:
        """
        Get chat sessions by user identifier.
        
        Args:
            user_identifier: User identifier (session cookie or user ID)
            limit: Maximum number of sessions to return
            order_by: Column to order by (default: started_at desc)
            
        Returns:
            List of ChatSession instances
        """
        if order_by is None:
            order_by = ChatSessionORM.started_at.desc()
        
        return await self.get_by_filter(
            {"user_identifier": user_identifier},
            limit=limit,
            order_by=order_by,
        )

    async def get_messages_by_session(
        self,
        session_id: uuid.UUID,
        limit: Optional[int] = None,
        order_by: Optional[Any] = None,
    ) -> List[ChatMessage]:
        """
        Get all messages for a chat session.
        
        Args:
            session_id: Chat session ID
            limit: Maximum number of messages to return
            order_by: Column to order by (default: created_at asc)
            
        Returns:
            List of ChatMessage instances
        """
        message_handler = ChatMessageHandler(self.session)
        if order_by is None:
            order_by = ChatMessageORM.created_at.asc()
        
        return await message_handler.get_by_filter(
            {"session_id": session_id},
            limit=limit,
            order_by=order_by,
        )


class ChatMessageHandler(BaseHandler[ChatMessage, ChatMessageORM]):
    """Handler for ChatMessage CRUD operations."""

    def __init__(self, session: AsyncSession):
        """Initialize ChatMessage handler."""
        super().__init__(session, ChatMessage, ChatMessageORM)

    async def get_by_session(
        self,
        session_id: uuid.UUID,
        limit: Optional[int] = None,
        order_by: Optional[Any] = None,
    ) -> List[ChatMessage]:
        """
        Get messages by session ID.
        
        Args:
            session_id: Chat session ID
            limit: Maximum number of messages to return
            order_by: Column to order by (default: created_at asc)
            
        Returns:
            List of ChatMessage instances
        """
        if order_by is None:
            order_by = ChatMessageORM.created_at.asc()
        
        return await self.get_by_filter(
            {"session_id": session_id},
            limit=limit,
            order_by=order_by,
        )

    async def get_by_role(
        self,
        session_id: uuid.UUID,
        role: str,
        limit: Optional[int] = None,
    ) -> List[ChatMessage]:
        """
        Get messages by session ID and role.
        
        Args:
            session_id: Chat session ID
            role: Message role ("user" or "assistant")
            limit: Maximum number of messages to return
            
        Returns:
            List of ChatMessage instances
        """
        return await self.get_by_filter(
            {"session_id": session_id, "role": role},
            limit=limit,
            order_by=ChatMessageORM.created_at.asc(),
        )

    async def get_feedbacks_by_message(
        self,
        message_id: uuid.UUID,
    ) -> List[Feedback]:
        """
        Get all feedbacks for a message.
        
        Args:
            message_id: Chat message ID
            
        Returns:
            List of Feedback instances
        """
        feedback_handler = FeedbackHandler(self.session)
        return await feedback_handler.get_by_filter({"message_id": message_id})


class FeedbackHandler(BaseHandler[Feedback, FeedbackORM]):
    """Handler for Feedback CRUD operations."""

    def __init__(self, session: AsyncSession):
        """Initialize Feedback handler."""
        super().__init__(session, Feedback, FeedbackORM)

    async def get_by_message(
        self,
        message_id: uuid.UUID,
    ) -> List[Feedback]:
        """
        Get feedbacks by message ID.
        
        Args:
            message_id: Chat message ID
            
        Returns:
            List of Feedback instances
        """
        return await self.get_by_filter({"message_id": message_id})

    async def get_by_status(
        self,
        status: str,
        limit: Optional[int] = None,
        order_by: Optional[Any] = None,
    ) -> List[Feedback]:
        """
        Get feedbacks by status.
        
        Args:
            status: Feedback status (pending, reviewed, ignored)
            limit: Maximum number of feedbacks to return
            order_by: Column to order by (default: created_at desc)
            
        Returns:
            List of Feedback instances
        """
        if order_by is None:
            order_by = FeedbackORM.created_at.desc()
        
        return await self.get_by_filter(
            {"status": status},
            limit=limit,
            order_by=order_by,
        )

    async def get_pending_feedbacks(
        self,
        limit: Optional[int] = None,
    ) -> List[Feedback]:
        """
        Get all pending feedbacks.
        
        Args:
            limit: Maximum number of feedbacks to return
            
        Returns:
            List of pending Feedback instances
        """
        from ..models.enums import FeedbackStatus
        return await self.get_by_status(FeedbackStatus.PENDING, limit=limit)


# Convenience class that groups all chat handlers
class ChatHandler:
    """
    Convenience class grouping all chat-related handlers.
    
    Usage:
        >>> handler = ChatHandler(session)
        >>> session = await handler.sessions.create(ChatSession(...))
        >>> message = await handler.messages.create(ChatMessage(...))
        >>> feedback = await handler.feedbacks.create(Feedback(...))
    """

    def __init__(self, session: AsyncSession):
        """Initialize all chat handlers."""
        self.session = session
        self.sessions = ChatSessionHandler(session)
        self.messages = ChatMessageHandler(session)
        self.feedbacks = FeedbackHandler(session)

