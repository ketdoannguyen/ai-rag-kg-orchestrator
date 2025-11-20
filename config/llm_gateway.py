from llm_gateway_client import AIGatewayClient, GatewayModelConfig

from .settings import Settings

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"


def _resolve_credentials(explicit_key: str | None, fallback: str | None) -> tuple[str | None, str | None]:
    """Return (api_key, api_base) preferring explicit settings, then OpenRouter."""
    if explicit_key:
        return explicit_key, None
    if fallback:
        return fallback, OPENROUTER_BASE_URL
    return None, None


def build_llm_gateway_client(settings: Settings) -> AIGatewayClient:
    """Construct a gateway client from global settings."""

    chat_key, chat_default_base = _resolve_credentials(
        settings.LLM_GATEWAY_CHAT_API_KEY,
        settings.OPENROUTER_API_KEY,
    )
    embed_key, embed_default_base = _resolve_credentials(
        settings.LLM_GATEWAY_EMBEDDING_API_KEY,
        settings.OPENROUTER_API_KEY,
    )

    if not chat_key or not embed_key:
        raise ValueError("OPENROUTER_API_KEY (or LLM-specific override) must be configured.")

    chat_config = GatewayModelConfig(
        model=settings.LLM_GATEWAY_CHAT_MODEL,
        api_key=chat_key,
        api_base=settings.LLM_GATEWAY_CHAT_API_BASE or chat_default_base,
        extra=settings.LLM_GATEWAY_CHAT_OPTIONS,
    )
    embed_config = GatewayModelConfig(
        model=settings.LLM_GATEWAY_EMBEDDING_MODEL,
        api_key=embed_key,
        api_base=settings.LLM_GATEWAY_EMBEDDING_API_BASE or embed_default_base,
        extra=settings.LLM_GATEWAY_EMBEDDING_OPTIONS,
    )

    return AIGatewayClient(
        chat_models={"default": chat_config},
        embedding_models={"default": embed_config},
    )

