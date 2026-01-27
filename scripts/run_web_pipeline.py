"""Simple async pipeline for web ingestion → enrich → save → embed.
python -m scripts.run_web_pipeline  --urls  https://tuyensinh.vku.udn.vn/cntt-ks https://tuyensinh.vku.udn.vn/pt-tuyenthang --created-by 550e8400-e29b-41d4-a716-446655440000"""

from __future__ import annotations

import argparse
import asyncio
import sys
import uuid
from collections.abc import Sequence
from pathlib import Path
from typing import Any
from uuid import UUID

# Ensure repo root on sys.path when running as script (before local imports)
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from arango_client import (
    ArangoClient,
    KnowledgeNodeHandler,
    parse_node_from_dict,
)
from loguru import logger

from workflows.dependencies import init_arango_client, init_llm_client
from workflows.embed_nodes import EmbedNodesInput, start_embed_nodes
from workflows.embed_nodes.helper import extract_text_for_embedding
from workflows.enrich_knowledge_tree import (
    EnrichKnowledgeTreeInput,
    start_enrich_knowledge_tree,
)
from workflows.ingest_web import start_ingest_web
from workflows.save_embeddings_to_arango import (
    SaveEmbeddingsInput,
    start_save_embeddings_to_arango,
)
from workflows.save_knowledge_to_arango import SaveKnowledgeInput, start_save_knowledge_to_arango

llm_client = init_llm_client()
arango_client = init_arango_client()


async def _process_single_result(
    result: dict[str, Any],
    *,
    embed_model: str | None,
    embed_batch_size: int,
    arango_client: ArangoClient,
) -> dict[str, Any]:
    """Handle the downstream steps for one successful web ingestion result."""
    status = result.get("status")
    root_node_dict = result.get("root_node")
    data_source = result.get("data_source")

    if status != "success" or not root_node_dict or not data_source:
        return {"status": status, "error": result.get("error_message")}

    data_source_id = UUID(data_source["id"]) if isinstance(data_source, dict) else data_source.id

    # Parse root_node from dict
    root_node = parse_node_from_dict(root_node_dict)

    # Enrich knowledge tree
    enrich_out = await start_enrich_knowledge_tree(
        EnrichKnowledgeTreeInput(
            knowledge_tree=root_node,
        ),
        llm_client=llm_client,
    )

    # Parse enriched_tree từ dict để giữ children (KnowledgeNode base không có children)
    enriched_tree = parse_node_from_dict(enrich_out["enriched_tree"])

    # Save knowledge to ArangoDB (handles both document and graph)
    save_result = await start_save_knowledge_to_arango(
        SaveKnowledgeInput(
            enriched_tree=enriched_tree,
            data_source_id=data_source_id,
        ),
        arango_client=arango_client,
    )

    # Handle save result
    if isinstance(save_result, Exception):
        logger.error("Failed to save knowledge to ArangoDB", error=str(save_result))
        save_result = {"total_nodes_saved": 0, "partial_success": True}
    elif not isinstance(save_result, dict):
        save_result = (
            save_result.model_dump(mode="json")
            if hasattr(save_result, "model_dump")
            else save_result.dict()
        )

    logger.info(
        "Knowledge saved to ArangoDB",
        total_nodes=save_result.get("total_nodes_saved", 0),
    )

    # Query nodes from database using ArangoDB handler
    handler = KnowledgeNodeHandler(arango_client.db)
    saved_nodes = handler.get_by_source(data_source_id)

    # Filter embeddable nodes
    embeddable_nodes = [
        node for node in saved_nodes if extract_text_for_embedding(node) is not None
    ]

    if not embeddable_nodes:
        logger.warning(f"No embeddable nodes found for data_source {data_source_id}")
        return {
            "status": "success",
            "data_source_id": str(data_source_id),
            "knowledge_saved": {
                "total_nodes_saved": save_result.get("total_nodes_saved", 0),
                "partial_success": save_result.get("partial_success", False),
            },
            "embeddings_saved": {"structures_saved": 0, "contents_saved": 0},
            "enrich_stats": enrich_out["statistics"],
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
    save_embed_out = await start_save_embeddings_to_arango(
        SaveEmbeddingsInput(
            structure_embeddings=embed_out["structure_embeddings"],
            content_embeddings=embed_out["content_embeddings"],
        ),
        arango_client=arango_client,
    )

    return {
        "status": "success",
        "data_source_id": str(data_source_id),
        "knowledge_saved": {
            "total_nodes_saved": save_result.get("total_nodes_saved", 0),
            "partial_success": save_result.get("partial_success", False),
        },
        "embeddings_saved": {
            "structures_saved": save_embed_out.get("structures_saved", 0),
            "contents_saved": save_embed_out.get("contents_saved", 0),
        },
        "enrich_stats": enrich_out["statistics"],
    }


async def run_pipeline(
    *,
    urls: Sequence[str],
    created_by: UUID,
    embed_model: str | None = None,
    embed_batch_size: int = 10,
    max_concurrent_urls: int | None = None,
) -> list[dict[str, Any]]:
    """Run the full web pipeline for given URLs."""
    ingest_input: dict[str, Any] = {
        "urls": list(urls),
        "created_by": created_by,
    }
    if max_concurrent_urls:
        ingest_input["max_concurrent_urls"] = max_concurrent_urls

    ingest_result = await start_ingest_web(ingest_input, arango_client=arango_client)
    results = (
        ingest_result.get("results", [])
        if isinstance(ingest_result, dict)
        else ingest_result.results
    )

    async def process_with_error_handling(res: dict[str, Any] | Any) -> dict[str, Any]:
        """Process a single result with error handling."""
        try:
            res_dict = res if isinstance(res, dict) else res.model_dump(mode="json")
            return await _process_single_result(
                res_dict,
                embed_model=embed_model,
                embed_batch_size=embed_batch_size,
                arango_client=arango_client,
            )
        except Exception as exc:
            logger.exception("Pipeline failed for one URL")
            return {"status": "error", "error": str(exc)}

    # Process all results concurrently
    outputs = await asyncio.gather(*(process_with_error_handling(res) for res in results))
    return list(outputs)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run web pipeline: ingest → enrich → save → embed."
    )
    parser.add_argument("--urls", nargs="+", required=True, help="List of URLs to ingest")
    parser.add_argument("--created-by", required=True, help="UUID of user who initiates ingestion")
    parser.add_argument("--embed-model", default=None, help="Optional embedding model override")
    parser.add_argument(
        "--embed-batch-size", type=int, default=10, help="Batch size for embedding"
    )
    parser.add_argument(
        "--max-concurrent-urls", type=int, default=None, help="Override max concurrent URLs"
    )
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
