from typing import Any

from pydantic import BaseModel


class ToolCall(BaseModel):
    """
    Represents a complete tool call requested by the model.
    This is part of the stable, public-facing API.
    """

    id: str
    name: str
    arguments: dict[str, Any]


class ToolCallChunk(BaseModel):
    """Represents a chunk of a tool call, typically received during streaming."""

    id: str | None = None
    name: str | None = None
    arguments_delta: str  # JSON string delta
    index: int | None = None
