"""Langfuse observability configuration for LiteLLM.

This module configures LiteLLM to send traces to Langfuse Cloud using the
OpenTelemetry (OTEL) integration. All LLM and embedding calls through the
AIGatewayClient will be automatically traced.
"""

from __future__ import annotations

import os

import litellm
from loguru import logger

from .settings import Settings


def configure_langfuse(settings: Settings) -> bool:
    """
    Configure Langfuse OTEL integration for LiteLLM.

    This function sets up the necessary environment variables and enables
    the langfuse_otel callback in LiteLLM. Once configured, all LLM and
    embedding calls will be automatically traced.

    Args:
        settings: Application settings containing Langfuse credentials.

    Returns:
        True if Langfuse is enabled and configured successfully,
        False otherwise.
    """
    if not settings.LANGFUSE_ENABLED:
        logger.info("Langfuse tracing is disabled (LANGFUSE_ENABLED=false)")
        return False

    if not settings.LANGFUSE_PUBLIC_KEY or not settings.LANGFUSE_SECRET_KEY:
        logger.warning(
            "Langfuse credentials not provided. "
            "Set LANGFUSE_PUBLIC_KEY and LANGFUSE_SECRET_KEY to enable tracing."
        )
        return False

    # LiteLLM's langfuse_otel callback reads these from environment
    os.environ["LANGFUSE_PUBLIC_KEY"] = settings.LANGFUSE_PUBLIC_KEY
    os.environ["LANGFUSE_SECRET_KEY"] = settings.LANGFUSE_SECRET_KEY
    os.environ["LANGFUSE_OTEL_HOST"] = settings.LANGFUSE_OTEL_HOST

    # Enable Langfuse OTEL callback if not already present
    current_callbacks = litellm.callbacks or []
    if "langfuse_otel" not in current_callbacks:
        litellm.callbacks = [*current_callbacks, "langfuse_otel"]

    logger.info(
        "Langfuse tracing enabled",
        extra={"host": settings.LANGFUSE_OTEL_HOST},
    )
    return True
