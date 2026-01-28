"""LLM Gateway configuration and factory."""

from __future__ import annotations

from llm_gateway_client import AIGatewayClient, GatewayModelConfig, ProviderConfig

from .extraction import get_extraction_settings
from .langfuse import configure_langfuse
from .settings import Settings


def _parse_providers(providers_csv: str | None) -> list[str]:
    """Parse comma-separated provider list into a list of provider names."""
    if not providers_csv:
        return []
    return [p.strip() for p in providers_csv.split(",") if p.strip()]


def _build_provider_config(providers: list[str], allow_fallbacks: bool) -> ProviderConfig | None:
    """Build OpenRouter provider routing config from parsed providers."""
    if not providers:
        return None
    return ProviderConfig(order=providers, allow_fallbacks=allow_fallbacks)


def build_llm_gateway_client(settings: Settings) -> AIGatewayClient:
    """Construct a gateway client from global settings.

    Uses separate configurations for chat and embedding:
    - Chat: LLM_CHAT_* or fallback to LLM_*
    - Embedding: EMBEDDING_* (new) or LLM_EMBEDDING_* (legacy) or LLM_*
    """
    # Configure Langfuse observability before first LLM call
    configure_langfuse(settings)

    # Resolve Chat API config
    chat_api_key = settings.LLM_CHAT_API_KEY or settings.LLM_API_KEY
    chat_api_base = settings.LLM_CHAT_API_BASE or settings.LLM_API_BASE

    if not chat_api_key:
        raise ValueError("LLM_API_KEY or LLM_CHAT_API_KEY must be configured for chat.")

    # Resolve Embedding API config (new EMBEDDING_* takes precedence)
    embedding_api_key = (
        settings.EMBEDDING_API_KEY or settings.LLM_EMBEDDING_API_KEY or settings.LLM_API_KEY
    )
    embedding_api_base = (
        settings.EMBEDDING_API_BASE or settings.LLM_EMBEDDING_API_BASE or settings.LLM_API_BASE
    )
    embedding_model = settings.EMBEDDING_MODEL or settings.LLM_EMBEDDING_MODEL
    embedding_options = settings.EMBEDDING_OPTIONS or settings.LLM_EMBEDDING_OPTIONS

    if not embedding_api_key:
        raise ValueError(
            "EMBEDDING_API_KEY or LLM_EMBEDDING_API_KEY or LLM_API_KEY must be configured."
        )

    # Build provider routing configs
    extraction_settings = get_extraction_settings()
    chat_provider_config = _build_provider_config(
        _parse_providers(settings.LLM_CHAT_PROVIDERS),
        settings.LLM_CHAT_ALLOW_FALLBACKS,
    )
    strong_chat_provider_config = _build_provider_config(
        _parse_providers(extraction_settings.EXTRACTION_STRONG_MODEL_PROVIDER),
        settings.LLM_CHAT_ALLOW_FALLBACKS,
    )
    embedding_provider_config = _build_provider_config(
        _parse_providers(settings.LLM_EMBEDDING_PROVIDERS),
        settings.LLM_EMBEDDING_ALLOW_FALLBACKS,
    )

    chat_config = GatewayModelConfig(
        model=settings.LLM_CHAT_MODEL,
        api_key=chat_api_key,
        api_base=chat_api_base,
        extra=settings.LLM_CHAT_OPTIONS,
        provider=chat_provider_config,
    )
    embed_config = GatewayModelConfig(
        model=embedding_model,
        api_key=embedding_api_key,
        api_base=embedding_api_base,
        extra=embedding_options,
        provider=embedding_provider_config,
    )

    # Strong LLM config for relation validation
    strong_config = GatewayModelConfig(
        model=extraction_settings.EXTRACTION_STRONG_MODEL,
        api_key=chat_api_key,
        api_base=chat_api_base,
        extra=settings.LLM_CHAT_OPTIONS,
        provider=strong_chat_provider_config,
    )

    return AIGatewayClient(
        chat_models={
            "default": chat_config,
            "strong": strong_config,
        },
        embedding_models={"default": embed_config},
    )
