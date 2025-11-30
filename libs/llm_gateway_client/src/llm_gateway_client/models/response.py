from typing import Any

from pydantic import BaseModel

from .request import GatewayChatTurn
from .tool import ToolCall, ToolCallChunk


class InvokeResponse(BaseModel):
    """
    A unified response model from an LLM invocation.
    The response can be a message, a tool call, or both.
    """

    message: GatewayChatTurn | None = None
    tool_calls: list[ToolCall] | None = None
    finish_reason: str
    usage: dict[str, int]


class InvokeStreamChunk(BaseModel):
    """
    A single chunk of a streaming response from an LLM invocation.
    This is designed to be compatible with LangGraph's streaming model.
    """

    content_delta: str | None = None
    tool_call_chunks: list[ToolCallChunk] | None = None
    finish_reason: str | None = None
    usage: dict[str, Any] | None = None
