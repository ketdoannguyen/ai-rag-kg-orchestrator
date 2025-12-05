"""Simple async pipeline for PDF upload → ingest → enrich → save → embed → persist embeddings."""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path
from typing import Any, Dict
from uuid import UUID

# Ensure repo root on sys.path when running as script
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

# Mitigate asyncio child watcher warnings on subprocess cleanup
try:
    if hasattr(asyncio.get_event_loop_policy(), "set_child_watcher"):
        asyncio.get_event_loop_policy().set_child_watcher(asyncio.FastChildWatcher())
except Exception:
    pass

from loguru import logger
from postgres_client import parse_node_from_dict
from workflows.dependencies import init_llm_client, init_postgres_client
from workflows.enrich_knowledge_tree import EnrichKnowledgeTreeInput, KnowledgeTreeEnricher
from workflows.embed_nodes import EmbedNodesInput, start_embed_nodes
from workflows.embed_nodes.helper import extract_text_for_embedding
from workflows.ingest_file import start_ingest
from workflows.save_embeddings_to_postgres import SaveEmbeddingsInput, start_save_embeddings_to_postgres
from workflows.save_knowledge_to_neo4j import SaveKnowledgeToNeo4jInput, start_save_knowledge_to_neo4j
from workflows.save_knowledge_to_postgres import SaveKnowledgeInput, start_save_knowledge_to_postgres
from workflows.upload_file import (
    FileUploadInput,
    UploadSourceType,
    start_upload_file,
)


async def _maybe_upload_file(
    *,
    file_path: Path,
    created_by: UUID,
    source_type: UploadSourceType,
    pg_client,
) -> UUID:
    """Upload file to create DataSource; return data_source_id."""
    file_bytes = await asyncio.to_thread(file_path.read_bytes)
    upload_input = FileUploadInput(
        file=file_bytes,
        filename=file_path.name,
        source_type=source_type,
        created_by=created_by,
    )
    upload_result = await start_upload_file(upload_input, pg_client=pg_client)
    data_source = upload_result.get("data_source") if isinstance(upload_result, dict) else upload_result.data_source
    data_source_id = UUID(data_source["id"]) if isinstance(data_source, dict) else data_source.id
    duplicate = upload_result.get("duplicate", False) if isinstance(upload_result, dict) else upload_result.duplicate
    if duplicate:
        logger.warning("File duplicate detected; using existing data_source", data_source_id=str(data_source_id))
    return data_source_id


async def _process_data_source(
    *,
    data_source_id: UUID,
    embed_model: str | None,
    embed_batch_size: int,
    pg_client,
) -> Dict[str, Any]:
    """Full pipeline for a single data_source_id."""
    ingest_result = await asyncio.to_thread(start_ingest, {"data_source_id": str(data_source_id)})
    status = ingest_result.get("status") if isinstance(ingest_result, dict) else ingest_result.status
    if status != "complete":
        error = ingest_result.get("error") if isinstance(ingest_result, dict) else getattr(ingest_result, "error", None)
        return {"status": "error", "data_source_id": str(data_source_id), "error": error}

    root_node_dict = ingest_result.get("root_node") if isinstance(ingest_result, dict) else ingest_result.root_node
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
    data_source_id: UUID | None,
    file_path: Path | None,
    created_by: UUID | None,
    source_type: UploadSourceType = UploadSourceType.FILE,
    embed_model: str | None = None,
    embed_batch_size: int = 10,
) -> Dict[str, Any]:
    """Run the PDF pipeline. If file_path is provided, upload first to get data_source_id."""
    pg_client = init_postgres_client()
    
    ds_id = data_source_id
    if ds_id is None and file_path:
        if not created_by:
            raise ValueError("--created-by is required when uploading a file")
        ds_id = await _maybe_upload_file(
            file_path=file_path,
            created_by=created_by,
            source_type=source_type,
            pg_client=pg_client,
        )
    if ds_id is None:
        raise ValueError("Either --data-source-id or --file-path must be provided.")

    return await _process_data_source(
        data_source_id=ds_id,
        embed_model=embed_model,
        embed_batch_size=embed_batch_size,
        pg_client=pg_client,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run PDF pipeline: upload → ingest → enrich → save → embed.")
    parser.add_argument("--file-path", help="Local file path to upload (optional if data_source_id provided)")
    parser.add_argument("--data-source-id", help="Existing data_source_id to ingest (UUID)")
    parser.add_argument("--created-by", help="UUID of user (required when uploading)")
    parser.add_argument("--source-type", default="FILE", choices=[t.name for t in UploadSourceType], help="Upload source type")
    parser.add_argument("--embed-model", default=None, help="Optional embedding model override")
    parser.add_argument("--embed-batch-size", type=int, default=10, help="Batch size for embedding")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    file_path = Path(args.file_path) if args.file_path else None
    ds_id = UUID(args.data_source_id) if args.data_source_id else None
    created_by = UUID(args.created_by) if args.created_by else None
    source_type = UploadSourceType[args.source_type]

    output = asyncio.run(
        run_pipeline(
            data_source_id=ds_id,
            file_path=file_path,
            created_by=created_by,
            source_type=source_type,
            embed_model=args.embed_model,
            embed_batch_size=args.embed_batch_size,
        )
    )
    logger.info(output)


if __name__ == "__main__":
    main()

