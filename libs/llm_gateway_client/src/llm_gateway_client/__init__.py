"""
LLM Gateway Client
A standardized, provider-agnostic client for interacting with LLM and embedding models.
"""

__version__ = "0.1.0"

from .client import AIGatewayClient
from .exceptions import (
    ConfigurationError,
    GatewayAPIError,
    ModelNotFoundError,
)
from .models.config import GatewayModelConfig
from .models.embedding import EmbeddingRequest, EmbeddingResponse
from .models.request import GatewayChatTurn, GatewayInvokeRequest
from .models.response import (
    InvokeResponse,
    InvokeStreamChunk,
)
from .models.tool import ToolCall, ToolCallChunk
from .utils import MessageConverter

__all__ = [
    "AIGatewayClient",
    "ConfigurationError",
    "EmbeddingRequest",
    "EmbeddingResponse",
    "GatewayModelConfig",
    "GatewayAPIError",
    "GatewayChatTurn",
    "GatewayInvokeRequest",
    "InvokeResponse",
    "InvokeStreamChunk",
    "MessageConverter",
    "ModelNotFoundError",
    "ToolCall",
    "ToolCallChunk",
]
