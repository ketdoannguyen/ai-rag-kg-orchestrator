#!/usr/bin/env python
"""OpenRouter-only smoke test for llm_gateway_client.

Usage:
    OPENROUTER_API_KEY=sk-or-... uv run python scripts/run_llm_gateway_demo.py

Optional environment variables:
    OPENROUTER_API_BASE     (default: https://openrouter.ai/api/v1)
    OPENROUTER_HTTP_REFERER (default: http://localhost)
    OPENROUTER_APP_NAME     (default: ai-rag-kg-orchestrator)
"""

from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from loguru import logger

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config.settings import get_settings  # noqa: E402
from llm_gateway_client import (
    AIGatewayClient,
    EmbeddingRequest,
    GatewayChatTurn,
    GatewayInvokeRequest,
    GatewayModelConfig,
)

OPENROUTER_BASE = "https://openrouter.ai/api/v1"


def _headers() -> dict[str, str]:
    return {
        "HTTP-Referer": os.environ.get("OPENROUTER_HTTP_REFERER", "http://localhost"),
        "X-Title": os.environ.get("OPENROUTER_APP_NAME", "ai-rag-kg-orchestrator"),
    }


def build_client() -> AIGatewayClient:
    load_dotenv()
    settings = get_settings()
    api_key = settings.OPENROUTER_API_KEY

    if not api_key:
        logger.error("OPENROUTER_API_KEY must be set in the environment or .env file.")
        sys.exit(1)

    api_base = os.environ.get("OPENROUTER_API_BASE", OPENROUTER_BASE)
    headers = _headers()

    chat_cfg = GatewayModelConfig(
        model=settings.LLM_GATEWAY_CHAT_MODEL,
        api_key=api_key,
        api_base=api_base,
        extra={"headers": headers},
    )
    embed_cfg = GatewayModelConfig(
        model=settings.LLM_GATEWAY_EMBEDDING_MODEL,
        api_key=api_key,
        api_base=api_base,
        extra={"headers": headers},
    )

    logger.info("OpenRouter chat model: {model}", model=chat_cfg.model)
    logger.info("OpenRouter embedding model: {model}", model=embed_cfg.model)
    logger.info("API base: {base}", base=api_base)

    return AIGatewayClient(
        chat_models={"default": chat_cfg},
        embedding_models={"default": embed_cfg},
    )


async def demo_chat(client: AIGatewayClient) -> dict[str, Any]:
    request = GatewayInvokeRequest(
        messages=[
            GatewayChatTurn(role="system", content="You are a concise assistant."),
            GatewayChatTurn(
                role="user",
                content="Dùng tiếng Việt và mô tả ngắn gọn về một khám phá khoa học thú vị.",
            ),
        ],
        provider_options={"temperature": 0.4},
    )
    response = await client.invoke(request)
    logger.success(f"Chat:\n {response.message.content}")
    logger.info(f"Usage: {response.usage}")
    return response.model_dump()


async def demo_embeddings(client: AIGatewayClient) -> dict[str, Any]:
    request = EmbeddingRequest(
        input=["Xin chào thế giới", "Hệ thống llm_gateway_client đang hoạt động!"],
    )
    response = await client.embed(request)
    logger.success(f"Embeddings generated: vectors - dim={len(response.embeddings[0])}")
    return response.model_dump()


async def main() -> None:
    client = build_client()
    payload = {
        "chat": await demo_chat(client),
        "embeddings": await demo_embeddings(client),
    }
    logger.info("Payload snapshot:\n%s", payload)


if __name__ == "__main__":
    asyncio.run(main())

