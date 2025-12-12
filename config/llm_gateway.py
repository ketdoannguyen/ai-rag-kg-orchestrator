from llm_gateway_client import AIGatewayClient, GatewayModelConfig

from .langfuse import configure_langfuse
from .settings import Settings


def build_llm_gateway_client(settings: Settings) -> AIGatewayClient:
    """Construct a gateway client from global settings.

    Uses unified LLM_API_KEY and LLM_API_BASE, with optional overrides
    for separate chat/embedding configurations.
    """
    # Configure Langfuse observability before first LLM call
    configure_langfuse(settings)

    # Resolve API keys (specific overrides > unified key)
    chat_api_key = settings.LLM_CHAT_API_KEY or settings.LLM_API_KEY
    embedding_api_key = settings.LLM_EMBEDDING_API_KEY or settings.LLM_API_KEY

    if not chat_api_key or not embedding_api_key:
        raise ValueError(
            "LLM_API_KEY (or LLM_CHAT_API_KEY/LLM_EMBEDDING_API_KEY) must be configured."
        )

    # Resolve API base URLs (specific overrides > unified base)
    chat_api_base = settings.LLM_CHAT_API_BASE or settings.LLM_API_BASE
    embedding_api_base = settings.LLM_EMBEDDING_API_BASE or settings.LLM_API_BASE

    chat_config = GatewayModelConfig(
        model=settings.LLM_CHAT_MODEL,
        api_key=chat_api_key,
        api_base=chat_api_base,
        extra=settings.LLM_CHAT_OPTIONS,
    )
    embed_config = GatewayModelConfig(
        model=settings.LLM_EMBEDDING_MODEL,
        api_key=embedding_api_key,
        api_base=embedding_api_base,
        extra=settings.LLM_EMBEDDING_OPTIONS,
    )

    return AIGatewayClient(
        chat_models={"default": chat_config},
        embedding_models={"default": embed_config},
    )
