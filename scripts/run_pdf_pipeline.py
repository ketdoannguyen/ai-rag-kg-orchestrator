"""Simple async pipeline for PDF upload → ingest → enrich → save → embed → persist embeddings.

Uses ArangoDB for all storage operations (replaced PostgreSQL + Neo4j).
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path
from typing import Any
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

from arango_client import KnowledgeNodeHandler, parse_node_from_dict
from loguru import logger

from workflows.dependencies import init_arango_client, init_llm_client
from workflows.embed_nodes import EmbedNodesInput, start_embed_nodes
from workflows.embed_nodes.helper import extract_text_for_embedding
from workflows.enrich_knowledge_tree import EnrichKnowledgeTreeInput, start_enrich_knowledge_tree
from workflows.enrich_knowledge_tree.models import RefineConfig
from workflows.ingest_file import start_ingest
from workflows.save_embeddings_to_arango import (
    SaveEmbeddingsInput,
    start_save_embeddings_to_arango,
)
from workflows.save_knowledge_to_arango import SaveKnowledgeInput, start_save_knowledge_to_arango
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
    arango_client,
) -> UUID:
    """Upload file to create DataSource; return data_source_id."""
    file_bytes = await asyncio.to_thread(file_path.read_bytes)
    upload_input = FileUploadInput(
        file=file_bytes,
        filename=file_path.name,
        source_type=source_type,
        created_by=created_by,
    )
    upload_result = await start_upload_file(upload_input, arango_client=arango_client)
    data_source = (
        upload_result.get("data_source")
        if isinstance(upload_result, dict)
        else upload_result.data_source
    )
    data_source_id = UUID(data_source["id"]) if isinstance(data_source, dict) else data_source.id
    duplicate = (
        upload_result.get("duplicate", False)
        if isinstance(upload_result, dict)
        else upload_result.duplicate
    )
    if duplicate:
        logger.warning(
            "File duplicate detected; using existing data_source",
            data_source_id=str(data_source_id),
        )
    return data_source_id


async def _process_data_source(
    *,
    data_source_id: UUID,
    embed_model: str | None,
    embed_batch_size: int,
    arango_client,
) -> dict[str, Any]:
    """Full pipeline for a single data_source_id."""
    ingest_result = await asyncio.to_thread(start_ingest, {"data_source_id": str(data_source_id)})
    status = (
        ingest_result.get("status") if isinstance(ingest_result, dict) else ingest_result.status
    )
    if status != "complete":
        error = (
            ingest_result.get("error")
            if isinstance(ingest_result, dict)
            else getattr(ingest_result, "error", None)
        )
        return {"status": "error", "data_source_id": str(data_source_id), "error": error}

    root_node_dict = (
        ingest_result.get("root_node")
        if isinstance(ingest_result, dict)
        else ingest_result.root_node
    )
    root_node = parse_node_from_dict(root_node_dict)

    # Enrich knowledge tree
    llm_client = init_llm_client()

    enrich_out = await start_enrich_knowledge_tree(
        EnrichKnowledgeTreeInput(
            knowledge_tree=root_node,
            # ingest_file already grouped blocks into retrieval-sized chunks,
            # so the detail merge would only re-chunk them and let the LLM
            # rewrite text that must stay verbatim.
            refine_config=RefineConfig(refine_details=False),
        ),
        llm_client=llm_client,
    )

    # Save knowledge to ArangoDB (document + graph)
    save_knowledge = await start_save_knowledge_to_arango(
        SaveKnowledgeInput(
            enriched_tree=enrich_out["enriched_tree"],
            data_source_id=data_source_id,
        ),
        arango_client=arango_client,
    )

    # Handle result
    if isinstance(save_knowledge, Exception):
        logger.error("Failed to save knowledge to ArangoDB", error=str(save_knowledge))
        save_knowledge = {"total_nodes_saved": 0, "partial_success": True}
    elif not isinstance(save_knowledge, dict):
        save_knowledge = (
            save_knowledge.model_dump(mode="json")
            if hasattr(save_knowledge, "model_dump")
            else save_knowledge.dict()
        )

    logger.info(
        "Knowledge saved to ArangoDB",
        nodes_saved=save_knowledge.get("total_nodes_saved", 0),
        relationships=save_knowledge.get("relationships_created", {}),
    )

    # Query nodes from database
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
                "total_nodes_saved": save_knowledge.get("total_nodes_saved", 0),
                "relationships_created": save_knowledge.get("relationships_created", {}),
                "partial_success": save_knowledge.get("partial_success", False),
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

    # Save embeddings to ArangoDB
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
            "total_nodes_saved": save_knowledge.get("total_nodes_saved", 0),
            "relationships_created": save_knowledge.get("relationships_created", {}),
            "partial_success": save_knowledge.get("partial_success", False),
        },
        "embeddings_saved": {
            "structures_saved": save_embed_out["structures_saved"],
            "contents_saved": save_embed_out["contents_saved"],
        },
        "enrich_stats": enrich_out["statistics"],
    }


async def run_pipeline(
    *,
    data_source_id: UUID | None,
    file_path: Path | None,
    created_by: UUID | None,
    source_type: UploadSourceType = UploadSourceType.FILE,
    embed_model: str | None = None,
    embed_batch_size: int = 10,
) -> dict[str, Any]:
    """Run the PDF pipeline. If file_path is provided, upload first to get data_source_id."""
    arango_client = init_arango_client()

    try:
        ds_id = data_source_id
        if ds_id is None and file_path:
            if not created_by:
                raise ValueError("--created-by is required when uploading a file")
            ds_id = await _maybe_upload_file(
                file_path=file_path,
                created_by=created_by,
                source_type=source_type,
                arango_client=arango_client,
            )
        if ds_id is None:
            raise ValueError("Either --data-source-id or --file-path must be provided.")

        return await _process_data_source(
            data_source_id=ds_id,
            embed_model=embed_model,
            embed_batch_size=embed_batch_size,
            arango_client=arango_client,
        )
    finally:
        arango_client.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run PDF pipeline: upload → ingest → enrich → save → embed."
    )
    parser.add_argument(
        "--file-path", help="Local file path to upload (optional if data_source_id provided)"
    )
    parser.add_argument("--data-source-id", help="Existing data_source_id to ingest (UUID)")
    parser.add_argument("--created-by", help="UUID of user (required when uploading)")
    parser.add_argument(
        "--source-type",
        default="FILE",
        choices=[t.name for t in UploadSourceType],
        help="Upload source type",
    )
    parser.add_argument("--embed-model", default=None, help="Optional embedding model override")
    parser.add_argument(
        "--embed-batch-size", type=int, default=10, help="Batch size for embedding"
    )
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
