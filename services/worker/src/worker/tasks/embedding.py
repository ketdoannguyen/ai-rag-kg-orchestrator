import asyncio

from celery import shared_task

from config.llm_gateway import build_llm_gateway_client
from config.settings import get_settings
from llm_gateway_client import AIGatewayClient, EmbeddingRequest


_GATEWAY_CLIENT: AIGatewayClient = build_llm_gateway_client(get_settings())


@shared_task
def embed_batch_task(texts: list[str]) -> list[list[float]]:
    async def _embed_many() -> list[list[float]]:
        response = await _GATEWAY_CLIENT.embed(EmbeddingRequest(input=texts))
        return response.embeddings

    return asyncio.run(_embed_many())


