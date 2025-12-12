#!/usr/bin/env python3
"""Test script to verify Langfuse integration is working.

This script performs:
1. A simple LLM completion call
2. A simple embedding call

Both calls will be traced to Langfuse Cloud if LANGFUSE_PUBLIC_KEY and
LANGFUSE_SECRET_KEY are configured in your .env file.

Usage:
    uv run python scripts/test_langfuse_integration.py
"""

from __future__ import annotations

import asyncio
import sys

from loguru import logger


async def main() -> None:
    """Run test LLM and embedding calls to verify Langfuse tracing."""
    # Import after env is loaded
    from config.settings import get_settings
    from workflows.dependencies import init_llm_client
    from llm_gateway_client.models import (
        EmbeddingRequest,
        GatewayChatTurn,
        GatewayInvokeRequest,
    )

    settings = get_settings()

    # Check Langfuse configuration
    if not settings.LANGFUSE_PUBLIC_KEY or not settings.LANGFUSE_SECRET_KEY:
        logger.warning(
            "Langfuse credentials not found in .env. "
            "Traces will NOT be sent to Langfuse Cloud.\n"
            "Add LANGFUSE_PUBLIC_KEY and LANGFUSE_SECRET_KEY to your .env file."
        )
        proceed = input("Continue anyway? (y/n): ")
        if proceed.lower() != "y":
            sys.exit(0)
    else:
        logger.info(
            "Langfuse configured",
            extra={
                "host": settings.LANGFUSE_OTEL_HOST,
                "enabled": settings.LANGFUSE_ENABLED,
            },
        )

    # Initialize LLM client (this configures Langfuse)
    llm_client = init_llm_client()
    logger.info("LLM Gateway client initialized")

    # Test 1: LLM Completion
    print("\n" + "=" * 60)
    print("TEST 1: LLM Completion")
    print("=" * 60)

    try:
        request = GatewayInvokeRequest(
            messages=[
                GatewayChatTurn(
                    role="user",
                    content="Say 'Hello from Langfuse test!' in Vietnamese.",
                )
            ],
            provider_options={"temperature": 0.7, "max_tokens": 50},
        )
        response = await llm_client.invoke(request)
        print(f"✅ LLM Response: {response.message.content}")
        print(f"   Tokens: {response.usage}")
    except Exception as e:
        print(f"❌ LLM call failed: {e}")
        logger.exception("LLM call failed")

    # Test 2: Embedding
    print("\n" + "=" * 60)
    print("TEST 2: Embedding")
    print("=" * 60)

    try:
        request = EmbeddingRequest(
            input=["This is a test sentence for embedding."],
        )
        response = await llm_client.embed(request)
        vector = response.embeddings[0]
        print(f"✅ Embedding generated: dimension={len(vector)}")
        print(f"   First 5 values: {vector[:5]}")
        print(f"   Tokens: {response.usage}")
    except Exception as e:
        print(f"❌ Embedding call failed: {e}")
        logger.exception("Embedding call failed")

    # Summary
    print("\n" + "=" * 60)
    print("VERIFICATION")
    print("=" * 60)
    if settings.LANGFUSE_PUBLIC_KEY and settings.LANGFUSE_SECRET_KEY:
        print(
            f"\n🔍 Check your traces at: {settings.LANGFUSE_OTEL_HOST}\n"
            "   Go to your project → Traces to see the recorded calls."
        )
    else:
        print("\n⚠️  Langfuse was not configured. No traces were sent.")

    print("\n✨ Test completed!")


if __name__ == "__main__":
    asyncio.run(main())
