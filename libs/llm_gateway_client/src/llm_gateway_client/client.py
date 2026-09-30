"""AI Gateway Client implementation using LiteLLM."""

from __future__ import annotations

from collections.abc import AsyncIterator, Callable, Mapping
from typing import Any

from litellm import acompletion, aembedding
from litellm.exceptions import BadRequestError as LiteLLMBadRequestError
from loguru import logger
from tenacity import (
    AsyncRetrying,
    RetryCallState,
    retry,
    retry_if_not_exception_type,
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

# Non-retryable exceptions - configuration errors that won't be fixed by retry
NON_RETRYABLE_EXCEPTIONS = (ConfigurationError, ModelNotFoundError, LiteLLMBadRequestError)


def _log_retry(what: str) -> Callable[[RetryCallState], None]:
    """A retry log that names the error's type and repeats none of its text.

    A provider error can quote the request it refused, so its text is exactly
    what must not reach a log; the type is what tells a reader whether the retry
    was worth making.
    """

    def log_retry(state: RetryCallState) -> None:
        error = state.outcome.exception() if state.outcome else None
        logger.warning(
            "Retrying {what} after {error_type}. Attempt #{attempt}, sleeping {wait:.2f}s.",
            what=what,
            error_type=type(error).__name__,
            attempt=state.attempt_number,
            wait=state.next_action.sleep if state.next_action else 0,
        )

    return log_retry


#: The retry policy a chat call runs under: three attempts, exponential backoff,
#: configuration errors refused outright. A call copies it, so two calls in
#: flight do not share attempt state; a test patches its ``wait`` to drop the
#: sleep between attempts.
_CHAT_RETRY = AsyncRetrying(
    wait=wait_exponential(multiplier=1, min=2, max=60),
    stop=stop_after_attempt(3),
    retry=retry_if_not_exception_type(NON_RETRYABLE_EXCEPTIONS),
    reraise=True,
    before_sleep=_log_retry("chat completion"),
)


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

    def bind_tools(self, tools: list[dict[str, Any]]) -> AIGatewayClient:
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

    def _stated_usage(self, usage: Any) -> dict[str, Any] | None:
        """The usage envelope exactly as the provider stated it, or None for silence.

        The normalized view fills every counter it does not find with a zero, so a
        count the provider left out reads the same as one it reported as zero.
        This keeps the provider's own envelope — cache detail included — so a
        caller can tell the two apart.
        """
        if not isinstance(usage, Mapping) or not usage:
            return None
        return dict(usage)

    def _served_model(self, response: dict[str, Any]) -> str | None:
        """The model the provider named as the one that answered, if it named one."""
        model = response.get("model")
        return model if isinstance(model, str) and model else None

    def _first_choice(self, response: dict[str, Any]) -> dict[str, Any]:
        choices = response.get("choices") or []
        if not choices:
            raise GatewayAPIError("LLM provider returned no choices.")
        return choices[0]

    async def invoke(self, request: GatewayInvokeRequest) -> InvokeResponse:
        """Call the chat model, retrying, and report what the answer that came back cost.

        The retry policy is copied for this call alone, so two calls in flight do
        not count their attempts against each other, and the count travels on the
        response: an answer that came back after a retry states the usage of its
        last attempt alone, and whether the attempts before it were billed cannot
        be read from it.
        """
        chat_config = self.chat_registry.get(request.model)
        messages = MessageConverter.to_litellm_messages(request.messages)
        params = self._prepare_params(
            request.provider_options, chat_config, self._resolve_tools(request)
        )
        resolved_model = chat_config.model

        attempts = 0

        async def attempt() -> Any:
            nonlocal attempts
            attempts += 1
            return await acompletion(
                model=resolved_model,
                messages=messages,
                **params,
            )

        response: Any = await _CHAT_RETRY.copy()(attempt)
        response_dict = self._to_dict(response)

        # Debug logging for troubleshooting empty responses
        logger.debug(
            "LLM response",
            model=resolved_model,
            response_keys=list(response_dict.keys()) if response_dict else [],
            choices_count=len(response_dict.get("choices", [])),
        )

        choice = self._first_choice(response_dict)

        # Debug logging for choice structure. The content itself is never logged:
        # it is the book page the model was asked to rewrite.
        message_raw = choice.get("message") or {}
        logger.debug(
            "Choice message structure",
            finish_reason=choice.get("finish_reason"),
            message_keys=list(message_raw.keys()) if message_raw else [],
            content_type=type(message_raw.get("content")).__name__,
        )

        message, tool_calls = MessageConverter.parse_assistant_message(message_raw)

        return InvokeResponse(
            message=message,
            tool_calls=tool_calls,
            finish_reason=choice.get("finish_reason") or "unknown",
            usage=self._normalize_usage(response_dict.get("usage")),
            model=self._served_model(response_dict),
            stated_usage=self._stated_usage(response_dict.get("usage")),
            attempts=attempts,
        )

    @retry(
        wait=wait_exponential(multiplier=1, min=2, max=60),
        stop=stop_after_attempt(3),
        retry=retry_if_not_exception_type(NON_RETRYABLE_EXCEPTIONS),
        reraise=True,
        before_sleep=_log_retry("streaming call"),
    )
    async def astream(self, request: GatewayInvokeRequest) -> AsyncIterator[InvokeStreamChunk]:
        chat_config = self.chat_registry.get(request.model)
        messages = MessageConverter.to_litellm_messages(request.messages)
        params = self._prepare_params(
            request.provider_options, chat_config, self._resolve_tools(request)
        )
        params["stream"] = True

        async for chunk in acompletion(
            model=chat_config.model,
            messages=messages,
            **params,
        ):
            chunk_dict = self._to_dict(chunk)
            yield MessageConverter.parse_stream_chunk(chunk_dict)

    @retry(
        wait=wait_exponential(multiplier=1, min=2, max=60),
        stop=stop_after_attempt(3),
        retry=retry_if_not_exception_type(NON_RETRYABLE_EXCEPTIONS),
        reraise=True,
        before_sleep=_log_retry("embedding call"),
    )
    async def embed(self, request: EmbeddingRequest) -> EmbeddingResponse:
        """Generate embeddings for input texts.

        Uses LiteLLM's aembedding with api_base for OpenRouter/OpenAI-compatible APIs.
        Handles model names with 'openrouter/' prefix by stripping it and using api_base.
        """
        embedding_config = self.embedding_registry.get(request.model)
        params = self._prepare_params(request.provider_options, embedding_config)

        # Determine the model name to use
        model_name = embedding_config.model

        # Handle openrouter/ prefix - strip it and let api_base do the routing
        if model_name.startswith("openrouter/"):
            model_name = model_name.replace("openrouter/", "", 1)

        response = await aembedding(
            model=model_name,
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
