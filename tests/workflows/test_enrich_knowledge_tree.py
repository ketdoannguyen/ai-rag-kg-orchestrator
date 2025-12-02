from __future__ import annotations

import asyncio
import json
import unittest
from datetime import datetime
from pathlib import Path
from uuid import UUID

from config.settings import settings
from workflows.dependencies import init_llm_client, init_postgres_client
from workflows.ingest_file import FileIngestionInput, FileIngestionStatus
from workflows.ingest_file.run import FileIngestionWorkflow
from workflows.upload_file import FileUploadInput, UploadSourceType, start_upload_file
from workflows.enrich_knowledge_tree import (
    EnrichKnowledgeTreeInput,
    EnrichKnowledgeTreeOutput,
    KnowledgeTreeEnricher,
)

from postgres_client import PostgresClient, parse_node_from_dict
from postgres_client.handlers.system_admin_handler import SystemAdminHandler
from postgres_client.models.system_admin import User


class EnrichKnowledgeTreeIntegrationTest(unittest.TestCase):
    """Integration test for enrich_knowledge_tree workflow with real services.

    This test:
    - Uses real PostgreSQL (via PostgresClient)
    - Uses real MinIO (via upload_file + ingest_file)
    - Uses real LLM Gateway (no mocking) for enrichment
    """

    SAMPLE_FILE = Path("data/input/SGK-Vat-li-12-KNTT-0-10.pdf")
    RAW_JSON = Path("data/ingest_file/raw_json/SGK-Vat-li-12-KNTT-0-10.json")
    CREATED_BY = UUID("550e8400-e29b-41d4-a716-446655440000")
    OUTPUT_DIR = Path("tests/workflows/output")

    @classmethod
    def setUpClass(cls) -> None:
        if not cls.SAMPLE_FILE.exists():
            raise FileNotFoundError(
                f"Sample file '{cls.SAMPLE_FILE}' is required for the integration test."
            )
        if not cls.RAW_JSON.exists():
            raise FileNotFoundError(
                f"Raw JSON file '{cls.RAW_JSON}' is required for the integration test."
            )

        cls._pg_client: PostgresClient = init_postgres_client()
        cls._sessionmaker = cls._pg_client.sessionmaker
        cls._llm_client = init_llm_client()

        asyncio.run(cls._ensure_user_exists())

    class _TestMarkerClient:
        """Lightweight MarkerClient used only for this integration test.

        Instead of running the heavy Marker model, it loads a pre-generated JSON
        file from disk and returns it in the same shape that the real
        MarkerClient.convert_pdf(...) would produce.
        """

        def __init__(self, raw_json_path: Path) -> None:
            self._raw_json_path = raw_json_path

        def convert_pdf(self, file_path: Path):
            # file_path is accepted for API compatibility but ignored here.
            data = json.loads(self._raw_json_path.read_text(encoding="utf-8"))
            import uuid

            return {
                "request_id": f"test-enrich-{uuid.uuid4().hex}",
                "payload": {
                    "json": data,
                    "images": [],
                    "metadata": {},
                },
            }

    @classmethod
    def tearDownClass(cls) -> None:
        if hasattr(cls, "_pg_client"):
            asyncio.run(cls._pg_client.close())

    @classmethod
    async def _ensure_user_exists(cls) -> None:
        async with cls._sessionmaker() as session:
            handler = SystemAdminHandler(session)
            existing = await handler.users.get_by_id(cls.CREATED_BY)
            if existing is None:
                user = User(
                    id=cls.CREATED_BY,
                    username="enrich_workflow_tester",
                    hashed_password="temporary",
                )
                await handler.users.create(user)

    def test_enrich_on_real_ingested_file(self) -> None:
        """Run full ingest_file + enrich_knowledge_tree on a real PDF and real LLM."""

        # Step 1: Upload file to create DataSource (same as ingest_file test)
        from workflows.dependencies import get_minio_client

        storage = get_minio_client()

        upload_payload = FileUploadInput(
            file=self.SAMPLE_FILE.read_bytes(),
            filename=self.SAMPLE_FILE.name,
            source_type=UploadSourceType.FILE,
            created_by=self.CREATED_BY,
            additional_config={"category": "enrich-integration-test"},
        )

        upload_result = asyncio.run(
            start_upload_file(
                upload_payload,
                storage=storage,
                pg_client=self._pg_client,
                fail_on_duplicate=False,
            )
        )

        self.assertIn("data_source", upload_result)
        data_source_dict = upload_result["data_source"]
        data_source_id = UUID(data_source_dict["id"])

        # Step 1.5: Reset DataSource status to allow re-ingestion (for test re-runs)
        async def _reset_datasource_status():
            async with self._sessionmaker() as session:
                handler = SystemAdminHandler(session)
                await handler.sources.update(
                    data_source_id,
                    {"status": "pending"},
                )

        asyncio.run(_reset_datasource_status())

        # Step 2: Ingest the file using the same lightweight Marker client strategy
        # as the ingest_file integration test (reads pre-generated RAW_JSON).
        workspace_root = Path(settings.INGEST_FILE_WORKDIR)
        marker_client = self._TestMarkerClient(self.RAW_JSON)

        ingest_workflow = FileIngestionWorkflow(
            storage=storage,
            workspace_root=workspace_root,
            marker_client=marker_client,
            pg_client=self._pg_client,
        )

        ingest_input = FileIngestionInput(data_source_id=str(data_source_id))
        ingest_result = asyncio.run(ingest_workflow.run(ingest_input))

        self.assertEqual(ingest_result.status, FileIngestionStatus.COMPLETE)
        self.assertIsNotNone(ingest_result.root_node)

        # Step 3: Parse root_node dict from ingest workflow output into a typed tree
        self.assertIsInstance(ingest_result.root_node, dict)
        root_node = parse_node_from_dict(ingest_result.root_node)

        enricher = KnowledgeTreeEnricher(llm_client=self._llm_client)

        enrich_input = EnrichKnowledgeTreeInput(
            knowledge_tree=root_node,
            # Use a conservative concurrency limit to be gentle on the API
            refine_config={"max_concurrent_sections": 20},
        )

        enrich_result = asyncio.run(enricher.enrich(enrich_input))

        # Basic assertions on enrich result
        self.assertIsNotNone(enrich_result.enriched_tree)
        self.assertIsNotNone(enrich_result.statistics)
        self.assertIsNotNone(enrich_result.changes_summary)

        stats = enrich_result.statistics
        self.assertGreater(stats.paragraphs_before, 0)
        self.assertGreater(stats.structures_enriched, 0)
        self.assertGreater(stats.structure_llm_calls, 0)

        # Step 4: Save enrich output for manual inspection
        self._save_enrich_output(
            data_source_id=data_source_id,
            enrich_result=enrich_result,
        )

    def _save_enrich_output(self, data_source_id: UUID, enrich_result) -> None:
        """Persist enriched tree + stats to JSON for manual review."""
        self.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        base_filename = f"enrich_tree_result_{timestamp}_{data_source_id.hex[:8]}"

        # Enriched tree is a KnowledgeNode Pydantic model
        tree_json = enrich_result.enriched_tree.model_dump_json(
            indent=2,
            exclude_none=False,
        )
        stats_json = json.dumps(
            enrich_result.statistics.model_dump(),
            indent=2,
            ensure_ascii=False,
        )
        changes_json = json.dumps(
            enrich_result.changes_summary.model_dump(),
            indent=2,
            ensure_ascii=False,
        )

        tree_path = self.OUTPUT_DIR / f"{base_filename}_tree.json"
        stats_path = self.OUTPUT_DIR / f"{base_filename}_stats.json"
        changes_path = self.OUTPUT_DIR / f"{base_filename}_changes.json"

        tree_path.write_text(tree_json, encoding="utf-8")
        stats_path.write_text(stats_json, encoding="utf-8")
        changes_path.write_text(changes_json, encoding="utf-8")

        print(f"\n✅ Saved enriched tree to: {tree_path}")
        print(f"   Statistics: {stats_path}")
        print(f"   Changes summary: {changes_path}")


if __name__ == "__main__":
    # Example:
    #   source .venv/bin/activate
    #   pytest tests/workflows/test_enrich_knowledge_tree_integration.py -q
    unittest.main()


