"""Entry point for query_retrieval workflow.

Usage:
    from workflows.query_retrieval import start_query_retrieval

    result = await start_query_retrieval({
        "query": "Nguyên nhân thất bại của phong trào Cần Vương?",
        "top_k": 10,
    })
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from loguru import logger

from .models import QueryRetrievalInput, QueryRetrievalOutput
from .workflow import QueryRetrievalWorkflow

if TYPE_CHECKING:
    from arango.database import StandardDatabase
    from llm_gateway_client.client import AIGatewayClient


async def start_query_retrieval(
    input_data: QueryRetrievalInput | dict[str, Any],
    *,
    db: StandardDatabase | None = None,
    llm_client: AIGatewayClient | None = None,
    graph_name: str = "knowledge_graph",
) -> QueryRetrievalOutput:
    """Entry point for query_retrieval workflow.

    Args:
        input_data: Query retrieval input
        db: ArangoDB database connection (optional, loaded from dependencies)
        llm_client: LLM client (optional, loaded from dependencies)
        graph_name: Knowledge graph name

    Returns:
        QueryRetrievalOutput with classification and results
    """
    _log = logger.bind(workflow="query_retrieval")

    # Convert dict to dataclass if needed
    if isinstance(input_data, dict):
        input_data = QueryRetrievalInput(**input_data)

    # Load dependencies if not provided
    if db is None:
        from workflows.dependencies import get_arango_db

        db = get_arango_db()

    if llm_client is None:
        from workflows.dependencies import get_llm_client

        llm_client = get_llm_client()

    _log.info(
        "Starting query_retrieval workflow",
        query=input_data.query[:50] + "..." if len(input_data.query) > 50 else input_data.query,
        top_k=input_data.top_k,
    )

    workflow = QueryRetrievalWorkflow(
        db=db,
        llm_client=llm_client,
        graph_name=graph_name,
    )

    result = await workflow.run(input_data)

    _log.info(
        "Query retrieval completed",
        query_type=result.classification.query_type.value,
        strategy=result.classification.strategy.value,
        total_results=result.total_results,
        total_time_ms=f"{result.total_time_ms:.0f}",
    )

    return result
