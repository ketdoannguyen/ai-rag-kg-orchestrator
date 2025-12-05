"""Implementation of save_embeddings_to_postgres workflow."""

from __future__ import annotations

import uuid
from typing import Any, Dict, List

from loguru import logger
from postgres_client import PostgresClient
from postgres_client.handlers.embedding_handler import EmbeddingHandler
from postgres_client.models.embeddings import EmbeddingStructure, EmbeddingContent
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from workflows.dependencies import init_postgres_client

from .models import SaveEmbeddingsInput, SaveEmbeddingsOutput


async def start_save_embeddings_to_postgres(
    payload: SaveEmbeddingsInput | Dict[str, Any],
    *,
    pg_client: PostgresClient | None = None,
) -> Dict[str, Any]:
    """
    Entry point for the save_embeddings_to_postgres workflow.

    Args:
        payload: Workflow input (Pydantic model or plain dict)
        pg_client: Optional Postgres client (defaults to settings.DATABASE_URL)

    Returns:
        Dictionary representation of SaveEmbeddingsOutput
    """
    input_payload = payload if isinstance(payload, SaveEmbeddingsInput) else SaveEmbeddingsInput(**payload)
    postgres_client = pg_client or init_postgres_client()

    workflow = SaveEmbeddingsWorkflow(pg_client=postgres_client)
    result = await workflow.run(input_payload)
    if hasattr(result, "model_dump"):
        return result.model_dump(mode="json")
    return result.dict()


class SaveEmbeddingsWorkflow:
    """Implements the save_embeddings_to_postgres workflow with retry logic."""

    def __init__(
        self,
        *,
        pg_client: PostgresClient,
    ) -> None:
        self._sessionmaker = pg_client.sessionmaker
        self._log = logger.bind(workflow="save_embeddings_to_postgres")

    async def run(self, payload: SaveEmbeddingsInput) -> SaveEmbeddingsOutput:
        """Execute the workflow end-to-end."""
        total_embeddings = len(payload.structure_embeddings) + len(payload.content_embeddings)
        self._log.info(
            "Starting save embeddings workflow",
            structure_count=len(payload.structure_embeddings),
            content_count=len(payload.content_embeddings),
            total=total_embeddings,
        )

        # Validate input
        if total_embeddings == 0:
            self._log.warning("No embeddings to save")
            return SaveEmbeddingsOutput(
                structures_saved=0,
                contents_saved=0,
                total_saved=0,
                failed_nodes=[],
            )

        async with self._sessionmaker() as session:
            handler = EmbeddingHandler(session)

            structures_saved = 0
            contents_saved = 0
            failed_nodes: List[uuid.UUID] = []

            # Save structure embeddings
            for embedding in payload.structure_embeddings:
                try:
                    await self._save_structure_embedding(handler, embedding)
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
                    await self._save_content_embedding(handler, embedding)
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
        reraise=False,  # Don't raise, track failures instead
    )
    async def _save_structure_embedding(
        self,
        handler: EmbeddingHandler,
        embedding: EmbeddingStructure,
    ) -> None:
        """Save a structure embedding with retry logic."""
        # Use vector as-is from embedding model (no truncation or padding)
        if embedding.vector:
            vector_len = len(embedding.vector)
            self._log.debug(
                f"Saving structure embedding with {vector_len} dimensions",
                node_id=str(embedding.node_id),
            )
        
        try:
            # Check if embedding already exists
            existing = await handler.structures.get_by_node_id(embedding.node_id)
            if existing:
                # Update existing embedding
                await handler.structures.update_vector(embedding.node_id, embedding.vector)
                if embedding.summary_text:
                    await handler.structures.update_summary(embedding.node_id, embedding.summary_text)
            else:
                # Create new embedding
                await handler.structures.create(embedding)
        except Exception as exc:
            # Rollback session on error
            try:
                await handler.session.rollback()
            except Exception:
                pass  # Ignore rollback errors
            
            error_msg = str(exc)
            self._log.warning(
                "Structure embedding save failed",
                node_id=str(embedding.node_id),
                error=error_msg,
            )
            raise  # Re-raise to trigger retry

    @retry(
        retry=retry_if_exception_type(Exception),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=8),
        reraise=False,  # Don't raise, track failures instead
    )
    async def _save_content_embedding(
        self,
        handler: EmbeddingHandler,
        embedding: EmbeddingContent,
    ) -> None:
        """Save a content embedding with retry logic."""
        # Use vector as-is from embedding model (no truncation or padding)
        if embedding.vector:
            vector_len = len(embedding.vector)
            self._log.debug(
                f"Saving content embedding with {vector_len} dimensions",
                node_id=str(embedding.node_id),
            )
        
        try:
            # Check if embedding already exists
            existing = await handler.contents.get_by_node_id(embedding.node_id)
            if existing:
                # Update existing embedding
                await handler.contents.update_vector(embedding.node_id, embedding.vector)
                await handler.contents.update_chunk_text(embedding.node_id, embedding.chunk_text)
            else:
                # Create new embedding
                await handler.contents.create(embedding)
        except Exception as exc:
            # Rollback session on error
            try:
                await handler.session.rollback()
            except Exception:
                pass  # Ignore rollback errors
            
            error_msg = str(exc)
            self._log.warning(
                "Content embedding save failed",
                node_id=str(embedding.node_id),
                error=error_msg,
            )
            raise  # Re-raise to trigger retry

