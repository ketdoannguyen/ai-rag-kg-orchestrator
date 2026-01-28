"""Integration tests for PDF file ingestion pipeline.

Full pipeline: upload_file → ingest_file → enrich_knowledge_tree → save_knowledge_to_arango → embed_nodes → save_embeddings

This test uses pre-generated JSON from Marker dashboard instead of running the heavy Marker model.
All output files are organized in timestamped folders for easy management.
"""

from __future__ import annotations

import asyncio
import json
import unittest
import uuid
from datetime import UTC, datetime
from pathlib import Path
from uuid import UUID

from arango_client import (
    ArangoClient,
    KnowledgeNodeHandler,
    SystemAdminHandler,
    User,
    parse_node_from_dict,
)
from loguru import logger

from config.settings import settings
from workflows.dependencies import get_minio_client, init_arango_client, init_llm_client
from workflows.embed_nodes import start_embed_nodes
from workflows.embed_nodes.helper import extract_text_for_embedding
from workflows.embed_nodes.models import EmbedNodesInput
from workflows.enrich_knowledge_tree import start_enrich_knowledge_tree
from workflows.enrich_knowledge_tree.models import EnrichKnowledgeTreeInput
from workflows.ingest_file import FileIngestionInput, FileIngestionStatus
from workflows.ingest_file.run import FileIngestionWorkflow
from workflows.save_embeddings_to_arango import start_save_embeddings_to_arango
from workflows.save_embeddings_to_arango.models import SaveEmbeddingsInput
from workflows.save_knowledge_to_arango import start_save_knowledge_to_arango
from workflows.save_knowledge_to_arango.models import SaveKnowledgeInput
from workflows.upload_file import FileUploadInput, UploadSourceType, start_upload_file


class IngestFilePipelineTest(unittest.TestCase):
    """Full pipeline integration test for PDF file ingestion.

    Pipeline: upload → ingest → enrich → save → embed → save_embeddings

    Uses pre-generated JSON from Marker dashboard to avoid heavy model execution.
    Output files are organized in timestamped folders for easy management.
    """

    # Test configuration
    SAMPLE_FILE = Path("data/input/SGK-Lich-su-12-Canh-dieu.pdf")
    RAW_JSON = Path("data/ingest_file/raw_json/SGK-Lich-su-12-Canh-dieu.json")
    CREATED_BY = UUID("550e8400-e29b-41d4-a716-446655440000")
    BASE_OUTPUT_DIR = Path("tests/workflows/output")

    @classmethod
    def setUpClass(cls) -> None:
        """Set up test fixtures."""
        if not cls.SAMPLE_FILE.exists():
            raise FileNotFoundError(
                f"Sample file '{cls.SAMPLE_FILE}' is required for the integration test."
            )
        if not cls.RAW_JSON.exists():
            raise FileNotFoundError(
                f"Raw JSON file '{cls.RAW_JSON}' is required for the integration test."
            )

        cls._storage = get_minio_client()
        cls._arango_client: ArangoClient = init_arango_client()
        cls._llm_client = init_llm_client()
        cls._ensure_user_exists()

    @classmethod
    def tearDownClass(cls) -> None:
        """Clean up resources."""
        if hasattr(cls, "_arango_client"):
            cls._arango_client.close()

    @classmethod
    def _ensure_user_exists(cls) -> None:
        """Create test user if not exists."""
        handler = SystemAdminHandler(cls._arango_client.db)
        existing = handler.users.get_by_id(cls.CREATED_BY)
        if existing is None:
            user = User(
                id=cls.CREATED_BY,
                username="ingest_pipeline_tester",
                hashed_password="temporary",
            )
            handler.users.create(user)

    class _TestMarkerClient:
        """Lightweight MarkerClient that reads pre-generated JSON instead of running Marker model."""

        def __init__(self, raw_json_path: Path) -> None:
            self._raw_json_path = raw_json_path

        def convert_pdf(self, file_path: Path):
            data = json.loads(self._raw_json_path.read_text(encoding="utf-8"))
            return {
                "request_id": f"test-ingest-{uuid.uuid4().hex}",
                "payload": {
                    "json": data,
                    "images": [],
                    "metadata": {},
                },
            }

    def _create_output_folder(self) -> Path:
        """Create timestamped output folder for this test run."""
        timestamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
        output_dir = self.BASE_OUTPUT_DIR / timestamp
        output_dir.mkdir(parents=True, exist_ok=True)
        return output_dir

    def _save_json(self, data: dict, filepath: Path) -> None:
        """Save data to JSON file."""
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False, default=str)
        logger.info(f"Saved: {filepath}")

    def test_full_pipeline(self) -> None:
        """Test full pipeline: upload → ingest → enrich → save → embed → save_embeddings.

        All output files are saved to a timestamped folder for easy inspection.
        """
        output_dir = self._create_output_folder()

        print(f"\n{'=' * 80}")
        print("Full PDF Ingestion Pipeline Test")
        print(f"Output folder: {output_dir}")
        print(f"{'=' * 80}")

        # ============ Stage 1: Upload File ============
        print("\n[Stage 1/6] Uploading file to MinIO...")
        upload_payload = FileUploadInput(
            file=self.SAMPLE_FILE.read_bytes(),
            filename=self.SAMPLE_FILE.name,
            source_type=UploadSourceType.FILE,
            created_by=self.CREATED_BY,
            additional_config={"category": "full-pipeline-test"},
        )

        upload_result = asyncio.run(
            start_upload_file(
                upload_payload,
                storage=self._storage,
                arango_client=self._arango_client,
                fail_on_duplicate=False,
            )
        )

        data_source_id = UUID(upload_result["data_source"]["id"])
        self._save_json(upload_result, output_dir / "1_upload_result.json")
        print(f"  ✓ DataSource ID: {data_source_id}")

        # Reset DataSource status for re-ingestion
        handler = SystemAdminHandler(self._arango_client.db)
        handler.sources.update(data_source_id, {"status": "pending"})

        # ============ Stage 2: Ingest File ============
        print("\n[Stage 2/6] Ingesting file (using pre-generated JSON)...")
        workflow = FileIngestionWorkflow(
            storage=self._storage,
            workspace_root=Path(settings.INGEST_FILE_WORKDIR),
            marker_client=self._TestMarkerClient(self.RAW_JSON),
            arango_client=self._arango_client,
        )

        ingest_result = asyncio.run(
            workflow.run(FileIngestionInput(data_source_id=str(data_source_id)))
        )

        self.assertEqual(ingest_result.status, FileIngestionStatus.COMPLETE)

        # Save ingest outputs
        self._save_json(ingest_result.model_dump(mode="json"), output_dir / "2_ingest_result.json")
        self._save_json(ingest_result.root_node, output_dir / "2_ingest_tree.json")

        ingest_stats = ingest_result.statistics
        print(f"  ✓ Parsed: {ingest_stats.get('total_nodes', 0)} nodes")
        print(f"  ✓ Paragraphs: {ingest_stats.get('total_paragraphs', 0)}")

        # Parse root_node for next stage
        root_node = parse_node_from_dict(ingest_result.root_node)

        # ============ Stage 3: Enrich Knowledge Tree ============
        print("\n[Stage 3/6] Enriching knowledge tree...")

        enrich_result = asyncio.run(
            start_enrich_knowledge_tree(
                EnrichKnowledgeTreeInput(
                    knowledge_tree=root_node,
                ),
                llm_client=self._llm_client,
            )
        )

        # Save enrich outputs
        self._save_json(enrich_result, output_dir / "3_enrich_result.json")
        self._save_json(enrich_result["enriched_tree"], output_dir / "3_enriched_tree.json")

        enrich_stats = enrich_result["statistics"]
        print(
            f"  ✓ Paragraphs: {enrich_stats['paragraphs_before']} → {enrich_stats['paragraphs_after']}"
        )
        print(f"  ✓ Structures enriched: {enrich_stats['structures_enriched']}")
        print(
            f"  ✓ LLM calls: detail={enrich_stats['detail_llm_calls']}, structure={enrich_stats['structure_llm_calls']}"
        )

        # Parse enriched tree for next stage
        enriched_tree = parse_node_from_dict(enrich_result["enriched_tree"])

        # ============ Stage 4: Save to ArangoDB ============
        print("\n[Stage 4/6] Saving to ArangoDB...")

        save_result = asyncio.run(
            start_save_knowledge_to_arango(
                SaveKnowledgeInput(
                    enriched_tree=enriched_tree,
                    data_source_id=data_source_id,
                ),
                arango_client=self._arango_client,
            )
        )

        self._save_json(save_result, output_dir / "4_save_result.json")

        relationships = save_result.get("relationships_created", {})
        print(f"  ✓ Nodes saved: {save_result['total_nodes_saved']}")
        print(
            f"  ✓ Relationships: HAS_SECTION={relationships.get('HAS_SECTION', 0)}, "
            f"HAS_CONTENT={relationships.get('HAS_CONTENT', 0)}, "
            f"NEXT_SIBLING={relationships.get('NEXT_SIBLING', 0)}"
        )

        # ============ Stage 5: Embed Nodes ============
        print("\n[Stage 5/6] Embedding nodes...")
        node_handler = KnowledgeNodeHandler(self._arango_client.db)
        saved_nodes = node_handler.get_by_source(data_source_id)

        embeddable = [n for n in saved_nodes if extract_text_for_embedding(n)]
        print(f"  → Embeddable nodes: {len(embeddable)}/{len(saved_nodes)}")

        embed_result = None
        embed_count = 0
        if embeddable:
            embed_result = asyncio.run(
                start_embed_nodes(
                    EmbedNodesInput(nodes=embeddable, batch_size=10),
                    llm_client=self._llm_client,
                )
            )

            self._save_json(embed_result, output_dir / "5_embed_result.json")

            structure_embs = embed_result.get("structure_embeddings", [])
            content_embs = embed_result.get("content_embeddings", [])
            embed_count = len(structure_embs) + len(content_embs)
            print(
                f"  ✓ Embeddings created: {embed_count} (structure={len(structure_embs)}, content={len(content_embs)})"
            )

            # ============ Stage 6: Save Embeddings ============
            print("\n[Stage 6/6] Saving embeddings to ArangoDB...")
            all_embeddings = structure_embs + content_embs

            if all_embeddings:
                emb_save_result = asyncio.run(
                    start_save_embeddings_to_arango(
                        SaveEmbeddingsInput(
                            structure_embeddings=structure_embs,
                            content_embeddings=content_embs,
                        ),
                        arango_client=self._arango_client,
                    )
                )

                self._save_json(emb_save_result, output_dir / "6_save_embeddings_result.json")
                print(f"  ✓ Structure embeddings saved: {emb_save_result['structures_saved']}")
                print(f"  ✓ Content embeddings saved: {emb_save_result['contents_saved']}")
        else:
            print("  ⚠ No embeddable nodes found, skipping embedding stages")

        # ============ Final Summary ============
        summary = {
            "timestamp": datetime.now(UTC).isoformat(),
            "data_source_id": str(data_source_id),
            "source_file": self.SAMPLE_FILE.name,
            "output_folder": str(output_dir),
            "pipeline_stages": {
                "1_upload": {
                    "status": "success",
                    "minio_path": upload_result.get("minio_info", {}).get("path"),
                },
                "2_ingest": {
                    "status": "success",
                    "total_nodes": ingest_stats.get("total_nodes", 0),
                    "total_paragraphs": ingest_stats.get("total_paragraphs", 0),
                    "duration_seconds": ingest_stats.get("processing_duration_seconds", 0),
                },
                "3_enrich": {
                    "status": "success",
                    "paragraphs_before": enrich_stats["paragraphs_before"],
                    "paragraphs_after": enrich_stats["paragraphs_after"],
                    "structures_enriched": enrich_stats["structures_enriched"],
                    "detail_llm_calls": enrich_stats["detail_llm_calls"],
                    "structure_llm_calls": enrich_stats["structure_llm_calls"],
                },
                "4_save": {
                    "status": "success",
                    "nodes_saved": save_result["total_nodes_saved"],
                    "relationships": relationships,
                },
                "5_embed": {
                    "status": "success" if embed_count > 0 else "skipped",
                    "embeddings_created": embed_count,
                },
                "6_save_embeddings": {
                    "status": "success" if embed_count > 0 else "skipped",
                },
            },
        }

        self._save_json(summary, output_dir / "pipeline_summary.json")

        print(f"\n{'=' * 80}")
        print("Pipeline Summary")
        print(f"{'=' * 80}")
        print(f"  • Data Source ID: {data_source_id}")
        print(f"  • Ingest: {ingest_stats.get('total_nodes', 0)} nodes parsed")
        print(
            f"  • Enrich: {enrich_stats['paragraphs_after']} paragraphs, {enrich_stats['structures_enriched']} structures"
        )
        print(
            f"  • Save: {save_result['total_nodes_saved']} nodes + {sum(relationships.values())} relationships"
        )
        print(f"  • Embed: {embed_count} embeddings")
        print(f"  • Output: {output_dir}")
        print(f"{'=' * 80}\n")

        # Assertions
        self.assertGreater(save_result["total_nodes_saved"], 0)


if __name__ == "__main__":
    unittest.main()
