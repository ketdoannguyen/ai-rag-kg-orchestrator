"""Implementation of save_embeddings_to_arango workflow.

Save embeddings to ArangoDB using KnowledgeNodeHandler.update_embedding().
"""

from __future__ import annotations

import uuid
from typing import Any

from arango_client import ArangoClient, KnowledgeNodeHandler
from loguru import logger
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from workflows.dependencies import init_arango_client
from workflows.embed_nodes.models import EmbeddingData

from .models import SaveEmbeddingsInput, SaveEmbeddingsOutput


async def start_save_embeddings_to_arango(
    payload: SaveEmbeddingsInput | dict[str, Any],
    *,
    arango_client: ArangoClient | None = None,
) -> dict[str, Any]:
    """
    Entry point for the save_embeddings_to_arango workflow.

    Args:
        payload: Workflow input (Pydantic model or plain dict)
        arango_client: Optional ArangoDB client (defaults to settings)

    Returns:
        Dictionary representation of SaveEmbeddingsOutput
    """
    input_payload = (
        payload if isinstance(payload, SaveEmbeddingsInput) else SaveEmbeddingsInput(**payload)
    )
    client = arango_client or init_arango_client()

    workflow = SaveEmbeddingsToArangoWorkflow(arango_client=client)
    result = await workflow.run(input_payload)

    if hasattr(result, "model_dump"):
        return result.model_dump(mode="json")
    return result.dict()


class SaveEmbeddingsToArangoWorkflow:
    """Implements the save_embeddings_to_arango workflow with retry logic."""

    def __init__(
        self,
        *,
        arango_client: ArangoClient,
    ) -> None:
        self._arango_client = arango_client
        self._db = arango_client.db
        self._log = logger.bind(workflow="save_embeddings_to_arango")

    async def run(self, payload: SaveEmbeddingsInput) -> SaveEmbeddingsOutput:
        """Execute the workflow end-to-end."""
        total_embeddings = len(payload.structure_embeddings) + len(payload.content_embeddings)
        self._log.info(
            "Starting save embeddings workflow",
            structure_count=len(payload.structure_embeddings),
            content_count=len(payload.content_embeddings),
            total=total_embeddings,
        )

        if total_embeddings == 0:
            self._log.warning("No embeddings to save")
            return SaveEmbeddingsOutput(
                structures_saved=0,
                contents_saved=0,
                total_saved=0,
                failed_nodes=[],
            )

        handler = KnowledgeNodeHandler(self._db)

        structures_saved = 0
        contents_saved = 0
        failed_nodes: list[uuid.UUID] = []

        # Save structure embeddings
        for embedding in payload.structure_embeddings:
            try:
                await self._save_embedding(handler, embedding)
                structures_saved += 1
            except Exception as exc:
                self._log.error(
                    "Failed to save structure embedding",
                    node_id=str(embedding.node_id),
                    error=str(exc),
                )
                failed_nodes.append(embedding.node_id)

        # Save content embeddings
        for embedding in payload.content_embeddings:
            try:
                await self._save_embedding(handler, embedding)
                contents_saved += 1
            except Exception as exc:
                self._log.error(
                    "Failed to save content embedding",
                    node_id=str(embedding.node_id),
                    error=str(exc),
                )
                failed_nodes.append(embedding.node_id)

        total_saved = structures_saved + contents_saved

        if failed_nodes:
            self._log.warning(
                "Partial success: some embeddings failed to save",
                total=total_embeddings,
                saved=total_saved,
                failed=len(failed_nodes),
            )
        else:
            self._log.info(
                "All embeddings saved successfully",
                total=total_saved,
            )

        return SaveEmbeddingsOutput(
            structures_saved=structures_saved,
            contents_saved=contents_saved,
            total_saved=total_saved,
            failed_nodes=failed_nodes,
        )

    @retry(
        retry=retry_if_exception_type(Exception),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=8),
        reraise=True,
    )
    async def _save_embedding(
        self,
        handler: KnowledgeNodeHandler,
        embedding: EmbeddingData,
    ) -> None:
        """Save an embedding with retry logic."""
        if embedding.vector:
            vector_len = len(embedding.vector)
            self._log.debug(
                f"Saving embedding with {vector_len} dimensions",
                node_id=str(embedding.node_id),
            )

        # Use update_embedding method which handles both create and update
        result = handler.update_embedding(
            node_id=embedding.node_id,
            vector=embedding.vector,
            text=embedding.text,
        )

        if result is None:
            raise ValueError(f"Node {embedding.node_id} not found in database")
