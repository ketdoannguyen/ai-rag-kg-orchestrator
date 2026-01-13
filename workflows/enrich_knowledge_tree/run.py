"""Entry point for enrich_knowledge_tree workflow."""

from __future__ import annotations

from typing import Any

from arango_client import KnowledgeNode, StructureNode, serialize_tree_to_dict
from llm_gateway_client import AIGatewayClient

from workflows.dependencies import init_llm_client

from .enricher import KnowledgeTreeEnricher
from .models import EnrichKnowledgeTreeInput


async def start_enrich_knowledge_tree(
    payload: EnrichKnowledgeTreeInput | dict[str, Any] | KnowledgeNode,
    *,
    llm_client: AIGatewayClient | None = None,
) -> dict[str, Any]:
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
    elif isinstance(payload, (KnowledgeNode, StructureNode)):
        input_payload = EnrichKnowledgeTreeInput(knowledge_tree=payload)
    else:
        input_payload = EnrichKnowledgeTreeInput(**payload)

    gateway_client = llm_client or init_llm_client()

    enricher = KnowledgeTreeEnricher(llm_client=gateway_client)
    result = await enricher.enrich(input_payload)

    # Custom serialization to properly handle nested children in enriched_tree
    # Pydantic's model_dump() doesn't correctly serialize children subclasses
    if hasattr(result, "model_dump"):
        output_dict = result.model_dump(mode="json")
        # Re-serialize enriched_tree using serialize_tree_to_dict for proper children
        if result.enriched_tree is not None:
            output_dict["enriched_tree"] = serialize_tree_to_dict(result.enriched_tree)
        return output_dict
    return result.dict()
