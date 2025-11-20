import asyncio

from llm_gateway_client import (
    AIGatewayClient,
    EmbeddingRequest,
    GatewayChatTurn,
    GatewayInvokeRequest,
)
from postgres_client import PostgresClient, rag


def execute_query(
    query: str,
    pg_client: PostgresClient,
    llm_client: AIGatewayClient,
) -> dict:
    top_lop1 = rag.search_lop_1(pg_client.engine, query)
    embed_request = EmbeddingRequest(input=[query])

    context_block = f"Context:\n{top_lop1}\n\nQuestion:\n{query}"
    invoke_request = GatewayInvokeRequest(
        messages=[
            GatewayChatTurn(
                role="system",
                content="You are a concise assistant that answers using the provided context only.",
            ),
            GatewayChatTurn(role="user", content=context_block),
        ],
        provider_options={"temperature": 0.2},
    )

    async def _run_pipeline() -> tuple[list[float], str, dict]:
        embedding_response = await llm_client.embed(embed_request)
        invoke_response = await llm_client.invoke(invoke_request)
        vector = embedding_response.embeddings[0] if embedding_response.embeddings else []
        answer = invoke_response.message.content if invoke_response.message else ""
        metadata = {
            "tool_calls": [tc.model_dump() for tc in (invoke_response.tool_calls or [])],
            "finish_reason": invoke_response.finish_reason,
            "usage": invoke_response.usage,
        }
        return vector, answer, metadata

    embedding_vector, answer_content, answer_meta = asyncio.run(_run_pipeline())
    return {
        "results": top_lop1,
        "embedding": embedding_vector,
        "answer": {
            "content": answer_content,
            **answer_meta,
        },
    }


