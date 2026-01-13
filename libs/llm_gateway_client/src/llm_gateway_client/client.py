from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any

from litellm import acompletion, aembedding
from loguru import logger
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from .exceptions import ConfigurationError, GatewayAPIError, ModelNotFoundError
from .models.config import GatewayModelConfig
from .models.embedding import EmbeddingRequest, EmbeddingResponse
from .models.request import GatewayInvokeRequest
from .models.response import InvokeResponse, InvokeStreamChunk
from .registry import ModelRegistry
from .utils import MessageConverter


class AIGatewayClient:
    """Provider-agnostic gateway built on top of LiteLLM."""

    def __init__(
        self,
        chat_models: dict[str, GatewayModelConfig | dict[str, Any] | str] | None = None,
        embedding_models: dict[str, GatewayModelConfig | dict[str, Any] | str] | None = None,
        default_tools: list[dict[str, Any]] | None = None,
    ):
        self.chat_registry = ModelRegistry("chat", chat_models)
        self.embedding_registry = ModelRegistry("embedding", embedding_models)
        self.default_tools = default_tools or []

        logger.info(
            "AIGatewayClient configured with {chat} chat models and {embed} embedding models.",
            chat=len(self.chat_registry.entries),
            embed=len(self.embedding_registry.entries),
        )

    def bind_tools(self, tools: list[dict[str, Any]]) -> "AIGatewayClient":
        """Return a new client instance with default tools applied to every request."""
        combined_tools = [*self.default_tools, *tools]
        return AIGatewayClient(
            chat_models=self.chat_registry.entries,
            embedding_models=self.embedding_registry.entries,
            default_tools=combined_tools,
        )

    def _prepare_params(
        self,
        provider_options: dict[str, Any] | None,
        model_config: GatewayModelConfig,
        tools: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        params = model_config.to_litellm_params()
        if provider_options:
            params.update(provider_options)
        if tools:
            params["tools"] = tools
        return params

    def _resolve_tools(self, request: GatewayInvokeRequest) -> list[dict[str, Any]] | None:
        if request.tools:
            return request.tools
        if self.default_tools:
            return self.default_tools
        return None

    def _to_dict(self, payload: Any) -> dict[str, Any]:
        if isinstance(payload, dict):
            return payload
        if hasattr(payload, "model_dump"):
            return payload.model_dump()
        if hasattr(payload, "dict"):
            return payload.dict()
        raise GatewayAPIError(f"Unsupported response type: {type(payload)}")

    def _normalize_usage(self, usage: dict[str, Any] | None) -> dict[str, int]:
        usage = usage or {}
        return {
            "prompt_tokens": int(usage.get("prompt_tokens") or 0),
            "completion_tokens": int(usage.get("completion_tokens") or 0),
            "total_tokens": int(usage.get("total_tokens") or 0),
        }

    def _first_choice(self, response: dict[str, Any]) -> dict[str, Any]:
        choices = response.get("choices") or []
        if not choices:
            raise GatewayAPIError("LLM provider returned no choices.")
        return choices[0]

    @retry(
        wait=wait_exponential(multiplier=1, min=2, max=60),
        stop=stop_after_attempt(5),
        retry=retry_if_exception_type(GatewayAPIError),
        before_sleep=lambda state: logger.warning(
            "Retrying chat completion due to error: {error}. Attempt #{attempt}, sleeping {wait:.2f}s.",
            error=state.outcome.exception() if state.outcome else "unknown",
            attempt=state.attempt_number,
            wait=state.next_action.sleep if state.next_action else 0,
        ),
    )
    async def invoke(self, request: GatewayInvokeRequest) -> InvokeResponse:
        try:
            chat_config = self.chat_registry.get(request.model)
            messages = MessageConverter.to_litellm_messages(request.messages)
            params = self._prepare_params(request.provider_options, chat_config, self._resolve_tools(request))
            response = await acompletion(
                model=request.model or chat_config.model,
                messages=messages,
                **params,
            )
            response_dict = self._to_dict(response)
            choice = self._first_choice(response_dict)
            message_payload = choice.get("message") or {}
            message, tool_calls = MessageConverter.parse_assistant_message(message_payload)

            return InvokeResponse(
                message=message,
                tool_calls=tool_calls,
                finish_reason=choice.get("finish_reason") or "unknown",
                usage=self._normalize_usage(response_dict.get("usage")),
            )
        except (ConfigurationError, ModelNotFoundError):
            raise
        except Exception as exc:
            logger.exception("Invoke failed: {error}", error=exc)
            raise GatewayAPIError(f"Chat invocation failed: {exc}") from exc

    @retry(
        wait=wait_exponential(multiplier=1, min=2, max=60),
        stop=stop_after_attempt(5),
        retry=retry_if_exception_type(GatewayAPIError),
        before_sleep=lambda state: logger.warning(
            "Retrying streaming call due to error: {error}. Attempt #{attempt}, sleeping {wait:.2f}s.",
            error=state.outcome.exception() if state.outcome else "unknown",
            attempt=state.attempt_number,
            wait=state.next_action.sleep if state.next_action else 0,
        ),
    )
    async def astream(self, request: GatewayInvokeRequest) -> AsyncIterator[InvokeStreamChunk]:
        try:
            chat_config = self.chat_registry.get(request.model)
            messages = MessageConverter.to_litellm_messages(request.messages)
            params = self._prepare_params(request.provider_options, chat_config, self._resolve_tools(request))
            params["stream"] = True

            async for chunk in acompletion(
                model=request.model or chat_config.model,
                messages=messages,
                **params,
            ):
                chunk_dict = self._to_dict(chunk)
                yield MessageConverter.parse_stream_chunk(chunk_dict)
        except (ConfigurationError, ModelNotFoundError):
            raise
        except Exception as exc:
            logger.exception("Streaming invoke failed: {error}", error=exc)
            raise GatewayAPIError(f"Streaming invocation failed: {exc}") from exc

    @retry(
        wait=wait_exponential(multiplier=1, min=2, max=60),
        stop=stop_after_attempt(5),
        retry=retry_if_exception_type(GatewayAPIError),
        before_sleep=lambda state: logger.warning(
            "Retrying embedding call due to error: {error}. Attempt #{attempt}, sleeping {wait:.2f}s.",
            error=state.outcome.exception() if state.outcome else "unknown",
            attempt=state.attempt_number,
            wait=state.next_action.sleep if state.next_action else 0,
        ),
    )
    async def embed(self, request: EmbeddingRequest) -> EmbeddingResponse:
        """Generate embeddings for input texts.

        Uses LiteLLM's aembedding with api_base for OpenRouter/OpenAI-compatible APIs.
        """
        embedding_config = self.embedding_registry.get(request.model)
        try:
            params = self._prepare_params(request.provider_options, embedding_config)
            response = await aembedding(
                model=request.model or embedding_config.model,
                input=request.input,
                encoding_format="float",
                **params,
            )
            response_dict = self._to_dict(response)
            data = response_dict.get("data") or []
            if not data:
                raise GatewayAPIError("Embedding provider returned no vectors.")
            embeddings: list[list[float]] = []
            for record in data:
                vector = record.get("embedding")
                if not isinstance(vector, list):
                    raise GatewayAPIError("Embedding vector missing or invalid.")
                embeddings.append(vector)
            return EmbeddingResponse(
                embeddings=embeddings,
                usage=self._normalize_usage(response_dict.get("usage")),
            )
        except (ConfigurationError, ModelNotFoundError):
            raise
        except Exception as exc:
            logger.exception("Embedding call failed: {error}", error=exc)
            raise GatewayAPIError(f"Embedding generation failed: {exc}") from exc
