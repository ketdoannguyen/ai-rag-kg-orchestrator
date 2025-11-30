from typing import Any, Literal

from pydantic import BaseModel, Field

from .tool import ToolCall


class GatewayChatTurn(BaseModel):
    """
    Represents a single turn in a chat conversation, independent of the underlying provider's format.
    This is part of the stable, public-facing API.
    """

    role: Literal["system", "user", "assistant", "tool"]
    content: str
    tool_call_id: str | None = None
    tool_calls: list[ToolCall] | None = None


class GatewayInvokeRequest(BaseModel):
    """
    Defines the stable, public-facing request model for the gateway's `invoke` and `astream` methods.
    It decouples the client from provider-specific message and model option formats.
    """

    messages: list[GatewayChatTurn]
    model: str | None = None
    tools: list[dict[str, Any]] | None = None
    provider_options: dict[str, Any] | None = Field(default_factory=dict)
