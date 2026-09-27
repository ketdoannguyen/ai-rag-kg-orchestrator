"""Integration tests for extract_relations workflow.

This test validates the relation extraction pipeline by:
1. Ingesting a PDF file and a related web page
2. Running extraction workflow
3. Verifying that similar content is merged or related
4. Running a second time to verify no duplicates remain

Pipeline: ingest_file + ingest_web → extract_relations → verify relations
"""

from __future__ import annotations

import asyncio
import json
import unittest
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

from config.extraction import get_extraction_settings
from config.ontologies import get_current_ontology
from config.settings import settings
from workflows.dependencies import get_minio_client, init_arango_client, init_llm_client
from workflows.embed_nodes import start_embed_nodes
from workflows.embed_nodes.helper import extract_text_for_embedding
from workflows.embed_nodes.models import EmbedNodesInput
from workflows.enrich_knowledge_tree import start_enrich_knowledge_tree
from workflows.enrich_knowledge_tree.models import EnrichKnowledgeTreeInput, RefineConfig
from workflows.extract_relations import (
    ExtractionMode,
    ExtractRelationsInput,
    start_extract_relations,
)
from workflows.ingest_file import FileIngestionInput, FileIngestionStatus
from workflows.ingest_file.run import FileIngestionWorkflow
from workflows.ingest_file.vision_ocr import (
    PageExtraction,
    VisionBlock,
    VisionOCRConfig,
    VisionPage,
)
from workflows.ingest_web import WebIngestionInput, WebIngestionStatus, start_ingest_web
from workflows.save_embeddings_to_arango import start_save_embeddings_to_arango
from workflows.save_embeddings_to_arango.models import SaveEmbeddingsInput
from workflows.save_knowledge_to_arango import start_save_knowledge_to_arango
from workflows.save_knowledge_to_arango.models import SaveKnowledgeInput
from workflows.upload_file import FileUploadInput, UploadSourceType, start_upload_file


class ExtractRelationsPipelineTest(unittest.TestCase):
    """Integration test for extract_relations workflow.

    Tests cross-source relation extraction between file and web content.
    """

    # Test configuration
    SAMPLE_FILE = Path("data/input/vku_tuyen_sinh.pdf")
    # Use a related education URL for cross-source testing
    TEST_URL = "https://vku.udn.vn/tuyen-sinh"
    CREATED_BY = UUID("550e8400-e29b-41d4-a716-446655440000")
    BASE_OUTPUT_DIR = Path("tests/workflows/output")

    @classmethod
    def setUpClass(cls) -> None:
        """Set up test fixtures."""
        if not cls.SAMPLE_FILE.exists():
            raise unittest.SkipTest(
                f"Sample file '{cls.SAMPLE_FILE}' is required for the integration test."
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
                username="extract_relations_tester",
                hashed_password="temporary",
            )
            handler.users.create(user)

    class _FakeVisionClient:
        """Vision client stub: one section per page instead of a real model call."""

        def __init__(self) -> None:
            self.config = VisionOCRConfig(
                api_base="https://fake-vision.invalid/v1/chat/completions",
                model="fake-vision",
            )

        async def extract_page(
            self, png_bytes: bytes, *, page_number: int, state: list[dict]
        ) -> PageExtraction:
            page = VisionPage(
                page=page_number,
                blocks=[
                    VisionBlock(
                        kind="heading",
                        marker=str(page_number),
                        depth=1,
                        text=f"Trang {page_number}",
                    ),
                    VisionBlock(kind="p", text=f"Nội dung trang {page_number}."),
                ],
            )
            return PageExtraction(page=page, attempts=1, prompt_tokens=0, completion_tokens=0)

        async def aclose(self) -> None:
            pass

    def _create_output_folder(self) -> Path:
        """Create timestamped output folder for this test run."""
        timestamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
        output_dir = self.BASE_OUTPUT_DIR / f"extraction_{timestamp}"
        output_dir.mkdir(parents=True, exist_ok=True)
        return output_dir

    def _save_json(self, data: dict, filepath: Path) -> None:
        """Save data to JSON file."""
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False, default=str)
        logger.info(f"Saved: {filepath}")

    async def _ingest_file(self, output_dir: Path) -> tuple[UUID, int]:
        """Ingest PDF file and return data_source_id and node count."""
        print("\n[Stage 1a] Uploading and ingesting file...")

        # Upload
        upload_payload = FileUploadInput(
            file=self.SAMPLE_FILE.read_bytes(),
            filename=self.SAMPLE_FILE.name,
            source_type=UploadSourceType.FILE,
            created_by=self.CREATED_BY,
            additional_config={"category": "extraction-test"},
        )

        upload_result = await start_upload_file(
            upload_payload,
            storage=self._storage,
            arango_client=self._arango_client,
            fail_on_duplicate=False,
        )

        data_source_id = UUID(upload_result["data_source"]["id"])
        self._save_json(upload_result, output_dir / "1a_file_upload_result.json")

        # Reset status
        handler = SystemAdminHandler(self._arango_client.db)
        handler.sources.update(data_source_id, {"status": "pending"})

        # Ingest
        workflow = FileIngestionWorkflow(
            storage=self._storage,
            workspace_root=Path(settings.INGEST_FILE_WORKDIR),
            vision_client=self._FakeVisionClient(),
            arango_client=self._arango_client,
        )

        ingest_result = await workflow.run(FileIngestionInput(data_source_id=str(data_source_id)))

        self.assertEqual(ingest_result.status, FileIngestionStatus.COMPLETE)
        self._save_json(
            ingest_result.model_dump(mode="json"), output_dir / "1a_file_ingest_result.json"
        )

        root_node = parse_node_from_dict(ingest_result.root_node)

        # Enrich
        enrich_result = await start_enrich_knowledge_tree(
            EnrichKnowledgeTreeInput(
                knowledge_tree=root_node,
                refine_config=RefineConfig(refine_details=False),
            ),
            llm_client=self._llm_client,
        )
        self.assertEqual(enrich_result["statistics"]["detail_llm_calls"], 0)
        self._save_json(enrich_result, output_dir / "1a_file_enrich_result.json")

        enriched_tree = parse_node_from_dict(enrich_result["enriched_tree"])

        # Save to ArangoDB
        save_result = await start_save_knowledge_to_arango(
            SaveKnowledgeInput(
                enriched_tree=enriched_tree,
                data_source_id=data_source_id,
            ),
            arango_client=self._arango_client,
        )
        self._save_json(save_result, output_dir / "1a_file_save_result.json")

        print(f"  ✓ File DataSource ID: {data_source_id}")
        print(f"  ✓ Nodes saved: {save_result['total_nodes_saved']}")

        return data_source_id, save_result["total_nodes_saved"]

    async def _ingest_web(self, output_dir: Path) -> tuple[UUID, int]:
        """Ingest web content and return data_source_id and node count."""
        print("\n[Stage 1b] Ingesting web content...")

        ingest_result = await start_ingest_web(
            WebIngestionInput(
                urls=[self.TEST_URL],
                created_by=self.CREATED_BY,
                crawl_config={"respect_robots_txt": False},
                max_url_retries=2,
                per_url_timeout_seconds=60.0,
            ),
            arango_client=self._arango_client,
        )

        self._save_json(ingest_result, output_dir / "1b_web_ingest_result.json")

        url_result = ingest_result["results"][0]
        if url_result["status"] != WebIngestionStatus.SUCCESS:
            print(f"  ⚠ Web ingestion failed: {url_result.get('error', 'Unknown error')}")
            return None, 0

        root_node_dict = url_result["root_node"]
        data_source_id = UUID(url_result["data_source"]["id"])

        root_node = parse_node_from_dict(root_node_dict)

        # Enrich
        enrich_result = await start_enrich_knowledge_tree(
            EnrichKnowledgeTreeInput(
                knowledge_tree=root_node,
            ),
            llm_client=self._llm_client,
        )
        self._save_json(enrich_result, output_dir / "1b_web_enrich_result.json")

        enriched_tree = parse_node_from_dict(enrich_result["enriched_tree"])

        # Save to ArangoDB
        save_result = await start_save_knowledge_to_arango(
            SaveKnowledgeInput(
                enriched_tree=enriched_tree,
                data_source_id=data_source_id,
            ),
            arango_client=self._arango_client,
        )
        self._save_json(save_result, output_dir / "1b_web_save_result.json")

        print(f"  ✓ Web DataSource ID: {data_source_id}")
        print(f"  ✓ Nodes saved: {save_result['total_nodes_saved']}")

        return data_source_id, save_result["total_nodes_saved"]

    async def _embed_nodes(self, data_source_ids: list[UUID], output_dir: Path) -> int:
        """Embed nodes from given sources and save to ArangoDB."""
        print("\n[Stage 2] Embedding nodes...")

        node_handler = KnowledgeNodeHandler(self._arango_client.db)
        all_nodes = []
        for source_id in data_source_ids:
            if source_id:
                all_nodes.extend(node_handler.get_by_source(source_id))

        embeddable = [n for n in all_nodes if extract_text_for_embedding(n)]
        print(f"  → Embeddable nodes: {len(embeddable)}/{len(all_nodes)}")

        if not embeddable:
            print("  ⚠ No embeddable nodes found")
            return 0

        embed_result = await start_embed_nodes(
            EmbedNodesInput(nodes=embeddable, batch_size=10),
            llm_client=self._llm_client,
        )
        self._save_json(embed_result, output_dir / "2_embed_result.json")

        structure_embs = embed_result.get("structure_embeddings", [])
        content_embs = embed_result.get("content_embeddings", [])

        # Save embeddings
        emb_save_result = await start_save_embeddings_to_arango(
            SaveEmbeddingsInput(
                structure_embeddings=structure_embs,
                content_embeddings=content_embs,
            ),
            arango_client=self._arango_client,
        )
        self._save_json(emb_save_result, output_dir / "2_save_embeddings_result.json")

        total_saved = emb_save_result["structures_saved"] + emb_save_result["contents_saved"]
        print(f"  ✓ Embeddings saved: {total_saved}")

        return total_saved

    async def _run_extraction(self, run_number: int, output_dir: Path) -> dict:
        """Run extraction workflow and return results."""
        print(f"\n[Stage 3.{run_number}] Running extraction (run #{run_number})...")

        extraction_settings = get_extraction_settings()
        ontology = get_current_ontology()

        extraction_result = await start_extract_relations(
            ExtractRelationsInput(
                mode=ExtractionMode.INLINE,
                batch_size=100,
                max_llm_calls=50,
                dry_run=False,
            ),
            db=self._arango_client.db,
            settings=extraction_settings,
            ontology=ontology,
        )

        self._save_json(extraction_result, output_dir / f"3_{run_number}_extraction_result.json")

        print(f"  ✓ Nodes processed: {extraction_result['nodes_processed']}")
        print(f"  ✓ Nodes merged: {extraction_result['nodes_merged']}")
        print(f"  ✓ Relations created: {extraction_result['relations_created']}")
        print(f"  ✓ Kept separate: {extraction_result['nodes_kept_separate']}")
        print(f"  ✓ LLM calls: {extraction_result['llm_calls_used']}")

        if extraction_result["errors"]:
            print(f"  ⚠ Errors: {len(extraction_result['errors'])}")

        return extraction_result

    def _verify_relations(self, output_dir: Path) -> dict:
        """Query and verify created relations."""
        print("\n[Stage 4] Verifying relations in ArangoDB...")

        # Query all extraction-created edges
        cursor = self._arango_client.db.aql.execute(
            """
            FOR e IN knowledge_edges
                FILTER e.detection_method == 'llm_verified'
                   OR e.detection_method == 'embedding_auto'
                RETURN {
                    type: e.type,
                    confidence: e.confidence,
                    method: e.detection_method,
                    from: e._from,
                    to: e._to
                }
            """
        )
        extraction_edges = list(cursor)

        # Group by type
        by_type = {}
        for edge in extraction_edges:
            edge_type = edge.get("type", "unknown")
            if edge_type not in by_type:
                by_type[edge_type] = []
            by_type[edge_type].append(edge)

        result = {
            "total_extraction_edges": len(extraction_edges),
            "by_type": {k: len(v) for k, v in by_type.items()},
            "edges": extraction_edges,
        }

        self._save_json(result, output_dir / "4_extraction_relations.json")

        print(f"  ✓ Total extraction edges: {len(extraction_edges)}")
        for edge_type, count in result["by_type"].items():
            print(f"    - {edge_type}: {count}")

        return result

    def test_full_extraction_pipeline(self) -> None:
        """Test full relation extraction pipeline with file and web content.

        Steps:
        1. Ingest PDF file + web page (same topic for testing cross-source relations)
        2. Embed all nodes
        3. Run extraction workflow - first time
        4. Run extraction workflow - second time (should have fewer changes)
        5. Verify relations in ArangoDB
        """
        output_dir = self._create_output_folder()

        print(f"\n{'=' * 80}")
        print("Extract Relations Pipeline Test")
        print(f"Output folder: {output_dir}")
        print(f"{'=' * 80}")

        async def run_pipeline():
            # Stage 1: Ingest both sources
            file_source_id, file_nodes = await self._ingest_file(output_dir)
            web_source_id, web_nodes = await self._ingest_web(output_dir)

            source_ids = [sid for sid in [file_source_id, web_source_id] if sid]
            total_initial_nodes = file_nodes + web_nodes

            print(f"\n  → Total initial nodes: {total_initial_nodes}")

            # Stage 2: Embed nodes
            if source_ids:
                await self._embed_nodes(source_ids, output_dir)

            # Stage 3: Run extraction - first time
            extraction1_result = await self._run_extraction(1, output_dir)

            # Stage 3: Run extraction - second time
            extraction2_result = await self._run_extraction(2, output_dir)

            return extraction1_result, extraction2_result

        extraction1_result, extraction2_result = asyncio.run(run_pipeline())

        # Stage 4: Verify relations
        relations = self._verify_relations(output_dir)

        # ============ Final Summary ============
        summary = {
            "timestamp": datetime.now(UTC).isoformat(),
            "output_folder": str(output_dir),
            "extraction_run_1": extraction1_result,
            "extraction_run_2": extraction2_result,
            "relations": relations,
        }

        self._save_json(summary, output_dir / "pipeline_summary.json")

        print(f"\n{'=' * 80}")
        print("Extract Relations Pipeline Summary")
        print(f"{'=' * 80}")
        print(
            f"  • Run 1: processed={extraction1_result['nodes_processed']}, "
            f"merged={extraction1_result['nodes_merged']}, "
            f"relations={extraction1_result['relations_created']}"
        )
        print(
            f"  • Run 2: processed={extraction2_result['nodes_processed']}, "
            f"merged={extraction2_result['nodes_merged']}, "
            f"relations={extraction2_result['relations_created']}"
        )
        print(f"  • Total extraction edges: {relations['total_extraction_edges']}")
        print(f"  • Output: {output_dir}")
        print(f"{'=' * 80}\n")

        # Assertions
        # Run 2 should have fewer or equal actions than Run 1 (no new duplicates)
        self.assertLessEqual(
            extraction2_result["nodes_merged"],
            extraction1_result["nodes_merged"],
            "Run 2 should not have more merges than Run 1",
        )


if __name__ == "__main__":
    unittest.main()
