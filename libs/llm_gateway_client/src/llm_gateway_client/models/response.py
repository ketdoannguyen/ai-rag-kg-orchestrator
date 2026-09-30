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
    """The counters normalized to prompt/completion/total, and zero-filled.

    A count the provider left out and a count it reported as zero both read ``0``
    here, so this view alone cannot tell a bill from silence; ``stated_usage``
    carries the provider's own envelope for that.
    """
    model: str | None = None
    """The model the provider reported it served, or ``None`` when it named none.

    ``GatewayInvokeRequest.model`` is the name that was asked for. A gateway may
    resolve an unmatched name to a model of its own choosing, so only the name in
    the response is evidence of who answered.
    """
    stated_usage: dict[str, Any] | None = None
    """The usage envelope exactly as the provider stated it, or ``None`` when it
    stated none.

    Nothing is normalized away: a count the provider left out stays out instead of
    reading as a zero it never claimed, and cache detail the provider stated —
    under ``prompt_tokens_details`` or flat — stays where the provider put it.
    """
    attempts: int = 1
    """How many times the gateway itself called the provider for this request.

    The gateway retries on its own behalf, so one request can be several calls to
    the provider. An answer that came back after a retry holds the usage of its
    last attempt alone, and whether the attempts before it were billed is not
    something the response can say — so a caller that needs the whole bill has to
    know they happened. One is what a call answered on its first attempt reports,
    and what a response built by hand states unless it says otherwise.
    """


class InvokeStreamChunk(BaseModel):
    """
    A single chunk of a streaming response from an LLM invocation.
    This is designed to be compatible with LangGraph's streaming model.
    """

    content_delta: str | None = None
    tool_call_chunks: list[ToolCallChunk] | None = None
    finish_reason: str | None = None
    usage: dict[str, Any] | None = None
