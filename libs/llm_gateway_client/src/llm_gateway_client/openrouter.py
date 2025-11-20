from __future__ import annotations

import json
from typing import Any

import httpx
from loguru import logger

from .models.config import GatewayModelConfig

OPENROUTER_API_BASE = "https://openrouter.ai/api/v1"


class OpenRouterError(RuntimeError):
    """Raised when the direct OpenRouter API call fails."""


def _sanitize_model_name(model: str | None) -> str:
    if not model:
        raise OpenRouterError("OpenRouter fallback requires a model name.")
    if model.startswith("openrouter/"):
        return model.split("/", 1)[1]
    return model


async def fetch_embeddings_via_openrouter(
    config: GatewayModelConfig,
    texts: list[str],
) -> tuple[list[list[float]], dict[str, Any] | None]:
    """Call OpenRouter's embeddings endpoint directly."""

    if not config.api_key:
        raise OpenRouterError("OpenRouter fallback requires api_key to be set.")

    api_base = (config.api_base or OPENROUTER_API_BASE).rstrip("/")
    url = f"{api_base}/embeddings"

    headers = {
        "Authorization": f"Bearer {config.api_key}",
        "Content-Type": "application/json",
    }

    extra_headers = (config.extra or {}).get("headers")
    if extra_headers:
        headers.update(extra_headers)

    payload = {
        "model": _sanitize_model_name(config.model),
        "input": texts,
    }

    logger.info("Calling OpenRouter embeddings fallback at {url}", url=url)

    async with httpx.AsyncClient(timeout=60) as client:
        response = await client.post(url, headers=headers, content=json.dumps(payload))

    if response.status_code >= 400:
        raise OpenRouterError(
            f"OpenRouter embeddings error {response.status_code}: {response.text}",
        )

    data = response.json()
    embedding_records = data.get("data") or []
    embeddings: list[list[float]] = []
    for record in embedding_records:
        vector = record.get("embedding")
        if not isinstance(vector, list):
            raise OpenRouterError("Invalid embedding vector returned by OpenRouter.")
        embeddings.append(vector)

    if not embeddings:
        raise OpenRouterError("OpenRouter returned no embeddings.")

    usage = data.get("usage")
    return embeddings, usage

