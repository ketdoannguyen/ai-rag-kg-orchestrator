"""Extract relations workflow - Embedding-based relation extraction pipeline.

This module provides the entry point for the extract_relations workflow.
For the main workflow implementation, see workflow.py.

Pipeline Overview:
1. Embedding Filter: Filter top-k relations using centroid method
2. LLM Extract: Choose from top-k or propose new relation
3. New Relation Validation:
   - Cosine >= 0.8: Merge with existing relation
   - Cosine < 0.8: LLM strong verification
4. Save edge with extracted relation

Key Settings (from config/extraction.py):
- TOP_K_RELATIONS: Number of relations to filter (default: 15)
- MERGE_THRESHOLD: Cosine threshold for merging (default: 0.8)
- MAX_RETRIES: Max retries for new relation validation (default: 5)
- FALLBACK_RELATION: Fallback when all retries fail (default: SIMILAR_TO)
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from loguru import logger

from config.extraction import ExtractionSettings, get_extraction_settings
from config.ontologies import BaseOntology, get_current_ontology

from .models import ExtractRelationsInput
from .workflow import ExtractRelationsWorkflow

if TYPE_CHECKING:
    from arango.database import StandardDatabase


async def start_extract_relations(
    input_data: ExtractRelationsInput | dict[str, Any],
    *,
    db: StandardDatabase | None = None,
    settings: ExtractionSettings | None = None,
    ontology: BaseOntology | None = None,
) -> dict[str, Any]:
    """Entry point for extract_relations workflow.

    Supports hybrid mode:
    - INLINE mode (default for <= threshold nodes): Runs during ingest
    - BATCH mode (for > threshold nodes): Background job

    Args:
        input_data: Workflow input (Pydantic dataclass or dict)
        db: ArangoDB database instance
        settings: Extraction settings (loaded from config if None)
        ontology: Ontology for relation validation (loaded from config if None)

    Returns:
        Dictionary with extraction results
    """
    _log = logger.bind(workflow="extract_relations")

    # Convert dict to dataclass if needed
    if isinstance(input_data, dict):
        input_data = ExtractRelationsInput(**input_data)

    # Load dependencies
    settings = settings or get_extraction_settings()
    ontology = ontology or get_current_ontology()

    if db is None:
        from workflows.dependencies import get_arango_db

        db = get_arango_db()

    _log.info(
        "Starting extract_relations workflow",
        mode=input_data.mode.value,
        batch_size=input_data.batch_size,
        max_llm_calls=input_data.max_llm_calls,
        dry_run=input_data.dry_run,
    )

    workflow = ExtractRelationsWorkflow(
        db=db,
        settings=settings,
        ontology=ontology,
    )

    result = await workflow.run(input_data)

    return {
        "nodes_processed": result.nodes_processed,
        "nodes_merged": result.nodes_merged,
        "relations_created": result.relations_created,
        "nodes_kept_separate": result.nodes_kept_separate,
        "llm_calls_used": result.llm_calls_used,
        "new_relations_proposed": result.new_relations_proposed,
        "new_relations_merged": result.new_relations_merged,
        "new_relations_approved": result.new_relations_approved,
        "errors": result.errors,
    }
