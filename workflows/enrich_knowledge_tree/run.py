"""Entry point for enrich_knowledge_tree workflow."""

from __future__ import annotations

from typing import Any, Dict

from llm_gateway_client import AIGatewayClient
from postgres_client.models.knowledge import KnowledgeNode

from workflows.dependencies import init_llm_client

from .enricher import KnowledgeTreeEnricher
from .models import EnrichKnowledgeTreeInput, EnrichKnowledgeTreeOutput


async def start_enrich_knowledge_tree(
    payload: EnrichKnowledgeTreeInput | Dict[str, Any] | KnowledgeNode,
    *,
    llm_client: AIGatewayClient | None = None,
) -> Dict[str, Any]:
    """
    Entry point for the enrich_knowledge_tree workflow.

    Args:
        payload: Workflow input - can be:
            - EnrichKnowledgeTreeInput (Pydantic model)
            - plain dict (will be unpacked into EnrichKnowledgeTreeInput)
            - KnowledgeNode (interpreted as knowledge_tree)
        llm_client: Optional LLM gateway client (defaults to init_llm_client())

    Returns:
        Dictionary representation of EnrichKnowledgeTreeOutput
    """
    if isinstance(payload, EnrichKnowledgeTreeInput):
        input_payload = payload
    elif isinstance(payload, KnowledgeNode):
        input_payload = EnrichKnowledgeTreeInput(knowledge_tree=payload)
    else:
        input_payload = EnrichKnowledgeTreeInput(**payload)

    gateway_client = llm_client or init_llm_client()

    enricher = KnowledgeTreeEnricher(llm_client=gateway_client)
    result = await enricher.enrich(input_payload)
    
    if hasattr(result, "model_dump"):
        return result.model_dump(mode="json")
    return result.dict()
