"""Simple async pipeline for web ingestion → enrich → save → embed → persist embeddings."""

from __future__ import annotations

import argparse
import asyncio
import sys
import uuid
from pathlib import Path
from typing import Any, List, Sequence
from uuid import UUID

# Ensure repo root on sys.path when running as script (before local imports)
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from loguru import logger
from postgres_client import parse_node_from_dict
from workflows.dependencies import init_llm_client, init_postgres_client
from workflows.enrich_knowledge_tree import (
    EnrichKnowledgeTreeInput,
    KnowledgeTreeEnricher,
)
from workflows.embed_nodes import EmbedNodesInput, start_embed_nodes
from workflows.embed_nodes.helper import extract_text_for_embedding
from workflows.ingest_web import start_ingest_web
from workflows.save_embeddings_to_postgres import SaveEmbeddingsInput, start_save_embeddings_to_postgres
from workflows.save_knowledge_to_neo4j import SaveKnowledgeToNeo4jInput, start_save_knowledge_to_neo4j
from workflows.save_knowledge_to_postgres import SaveKnowledgeInput, start_save_knowledge_to_postgres


async def _process_single_result(
    result: dict[str, Any],
    *,
    embed_model: str | None,
    embed_batch_size: int,
    pg_client,
) -> dict[str, Any]:
    """Handle the downstream steps for one successful web ingestion result."""
    status = result.get("status")
    root_node_dict = result.get("root_node")
    data_source = result.get("data_source")
    
    if status != "success" or not root_node_dict or not data_source:
        return {"status": status, "error": result.get("error_message")}

    data_source_id = UUID(data_source["id"]) if isinstance(data_source, dict) else data_source.id
    
    # Parse root_node from dict (same as test)
    root_node = parse_node_from_dict(root_node_dict)

    # Enrich knowledge tree
    llm_client = init_llm_client()
    enricher = KnowledgeTreeEnricher(llm_client=llm_client)
    enrich_out = await enricher.enrich(EnrichKnowledgeTreeInput(knowledge_tree=root_node))

    # Save knowledge to postgres and neo4j in parallel
    from workflows.dependencies import init_neo4j_client
    
    neo4j_client = init_neo4j_client()
    save_knowledge_task = start_save_knowledge_to_postgres(
        SaveKnowledgeInput(
            enriched_tree=enrich_out.enriched_tree,
            data_source_id=data_source_id,
        ),
        pg_client=pg_client,
    )
    save_neo4j_task = start_save_knowledge_to_neo4j(
        SaveKnowledgeToNeo4jInput(
            enriched_tree=enrich_out.enriched_tree,
        ),
        neo4j_client=neo4j_client,
    )
    
    # Run both in parallel, handle errors separately
    save_knowledge, save_neo4j = await asyncio.gather(
        save_knowledge_task,
        save_neo4j_task,
        return_exceptions=True,
    )
    
    # Handle postgres result
    if isinstance(save_knowledge, Exception):
        logger.error("Failed to save knowledge to postgres", error=str(save_knowledge))
        save_knowledge = {"total_nodes_saved": 0, "partial_success": True}
    elif isinstance(save_knowledge, dict):
        pass  # Already a dict
    else:
        save_knowledge = save_knowledge.model_dump(mode="json") if hasattr(save_knowledge, "model_dump") else save_knowledge.dict()
    
    # Handle neo4j result
    if isinstance(save_neo4j, Exception):
        logger.error("Failed to save knowledge to neo4j", error=str(save_neo4j))
        save_neo4j = {"total_nodes_saved": 0, "partial_success": True}
    elif isinstance(save_neo4j, dict):
        pass  # Already a dict
    else:
        save_neo4j = save_neo4j.model_dump(mode="json") if hasattr(save_neo4j, "model_dump") else save_neo4j.dict()
    
    logger.info(
        "Knowledge saved",
        postgres_nodes=save_knowledge.get("total_nodes_saved", 0),
        neo4j_nodes=save_neo4j.get("total_nodes_saved", 0),
    )

    # Query nodes from database (same as test_embed_nodes.py)
    from postgres_client.handlers.knowledge_handler import KnowledgeNodeHandler
    
    async with pg_client.sessionmaker() as session:
        handler = KnowledgeNodeHandler(session)
        saved_nodes = await handler.get_by_source(data_source_id)
    
    # Filter embeddable nodes (same as test)
    embeddable_nodes = [
        node for node in saved_nodes if extract_text_for_embedding(node) is not None
    ]

    if not embeddable_nodes:
        logger.warning(f"No embeddable nodes found for data_source {data_source_id}")
        return {
            "status": "success",
            "data_source_id": str(data_source_id),
            "knowledge_saved": {
                "postgres": {
                    "total_nodes_saved": save_knowledge.get("total_nodes_saved", 0),
                    "partial_success": save_knowledge.get("partial_success", False),
                },
                "neo4j": {
                    "total_nodes_saved": save_neo4j.get("total_nodes_saved", 0),
                    "partial_success": save_neo4j.get("partial_success", False),
                },
            },
            "embeddings_saved": {"structures_saved": 0, "contents_saved": 0},
            "enrich_stats": enrich_out.statistics.model_dump(),
        }

    # Embed nodes
    embed_out = await start_embed_nodes(
        EmbedNodesInput(
            nodes=embeddable_nodes,
            batch_size=embed_batch_size,
            model=embed_model,
        ),
        llm_client=llm_client,
    )

    # Save embeddings
    save_embed_out = await start_save_embeddings_to_postgres(
        SaveEmbeddingsInput(
            structure_embeddings=embed_out["structure_embeddings"],
            content_embeddings=embed_out["content_embeddings"],
        ),
        pg_client=pg_client,
    )

    return {
        "status": "success",
        "data_source_id": str(data_source_id),
        "knowledge_saved": {
            "postgres": {
                "total_nodes_saved": save_knowledge.get("total_nodes_saved", 0),
                "partial_success": save_knowledge.get("partial_success", False),
            },
            "neo4j": {
                "total_nodes_saved": save_neo4j.get("total_nodes_saved", 0),
                "partial_success": save_neo4j.get("partial_success", False),
            },
        },
        "embeddings_saved": {
            "structures_saved": save_embed_out["structures_saved"],
            "contents_saved": save_embed_out["contents_saved"],
        },
        "enrich_stats": enrich_out.statistics.model_dump(),
    }


async def run_pipeline(
    *,
    urls: Sequence[str],
    created_by: UUID,
    embed_model: str | None = None,
    embed_batch_size: int = 10,
    max_concurrent_urls: int | None = None,
) -> List[dict[str, Any]]:
    """Run the full web pipeline for given URLs."""
    pg_client = init_postgres_client()
    
    ingest_input: dict[str, Any] = {
        "urls": list(urls),
        "created_by": created_by,
    }
    if max_concurrent_urls:
        ingest_input["max_concurrent_urls"] = max_concurrent_urls

    ingest_result = await start_ingest_web(ingest_input, pg_client=pg_client)
    results = ingest_result.get("results", []) if isinstance(ingest_result, dict) else ingest_result.results

    async def process_with_error_handling(res: dict[str, Any] | Any) -> dict[str, Any]:
        """Process a single result with error handling."""
        try:
            res_dict = res if isinstance(res, dict) else res.model_dump(mode="json")
            return await _process_single_result(
                res_dict,
                embed_model=embed_model,
                embed_batch_size=embed_batch_size,
                pg_client=pg_client,
            )
        except Exception as exc:  # noqa: BLE001
            logger.exception("Pipeline failed for one URL")
            return {"status": "error", "error": str(exc)}

    # Process all results concurrently
    outputs = await asyncio.gather(*(process_with_error_handling(res) for res in results))
    return list(outputs)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run web pipeline: ingest → enrich → save → embed.")
    parser.add_argument("--urls", nargs="+", required=True, help="List of URLs to ingest")
    parser.add_argument("--created-by", required=True, help="UUID of user who initiates ingestion")
    parser.add_argument("--embed-model", default=None, help="Optional embedding model override")
    parser.add_argument("--embed-batch-size", type=int, default=10, help="Batch size for embedding")
    parser.add_argument("--max-concurrent-urls", type=int, default=None, help="Override max concurrent URLs")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    created_by = uuid.UUID(args.created_by)
    outputs = asyncio.run(
        run_pipeline(
            urls=args.urls,
            created_by=created_by,
            embed_model=args.embed_model,
            embed_batch_size=args.embed_batch_size,
            max_concurrent_urls=args.max_concurrent_urls,
        )
    )
    for item in outputs:
        logger.info(item)


if __name__ == "__main__":
    main()

