from __future__ import annotations

import asyncio
import json
import unittest
import uuid
from datetime import datetime
from pathlib import Path
from uuid import UUID

from config.settings import settings
from workflows.dependencies import get_minio_client, init_postgres_client
from workflows.upload_file import FileUploadInput, UploadSourceType, start_upload_file
from workflows.ingest_file import FileIngestionInput, FileIngestionStatus
from workflows.ingest_file.run import FileIngestionWorkflow
from postgres_client import PostgresClient
from postgres_client.handlers.system_admin_handler import SystemAdminHandler
from postgres_client.models.system_admin import User


class IngestFileWorkflowIntegrationTest(unittest.TestCase):
    """Integration test for ingest_file workflow using real storage + database.

    This test intentionally avoids running the heavy Marker model on the laptop.
    Instead, it injects a lightweight MarkerClient implementation that reads a
    pre-generated raw JSON file from disk:

        data/ingest_file/raw_json/SGK-Vat-li-12-KNTT-0-10.json

    The rest of the workflow (MinIO download, hash verification, parser, validation,
    DataSource status update) is executed end-to-end against real services.
    """

    SAMPLE_FILE = Path("data/input/SGK-Vat-li-12-KNTT-0-10.pdf")
    RAW_JSON = Path("data/ingest_file/raw_json/SGK-Vat-li-12-KNTT-0-10.json")
    CREATED_BY = UUID("550e8400-e29b-41d4-a716-446655440000")
    OUTPUT_DIR = Path("tests/workflows/output")

    @classmethod
    def setUpClass(cls) -> None:
        if not cls.SAMPLE_FILE.exists():
            raise FileNotFoundError(
                f"Sample file '{cls.SAMPLE_FILE}' is required for the ingest_file integration test."
            )
        if not cls.RAW_JSON.exists():
            raise FileNotFoundError(
                f"Raw JSON file '{cls.RAW_JSON}' is required for the ingest_file integration test."
            )

        cls._storage = get_minio_client()
        cls._pg_client: PostgresClient = init_postgres_client()
        cls._sessionmaker = cls._pg_client.sessionmaker
        asyncio.run(cls._ensure_user_exists())

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
                    username="ingest_workflow_tester",
                    hashed_password="temporary",
                )
                await handler.users.create(user)

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
            return {
                "request_id": f"test-ingest-{uuid.uuid4().hex}",
                "payload": {
                    "json": data,
                    "images": [],
                    "metadata": {},
                },
            }

    async def _fetch_datasource(self, source_id: UUID):
        async with self._sessionmaker() as session:
            handler = SystemAdminHandler(session)
            return await handler.sources.get_by_id(source_id)

    def test_ingest_from_existing_datasource(self) -> None:
        """Upload a PDF to create a DataSource, then ingest it using ingest_file."""

        # Step 1: Upload the PDF to MinIO and create a DataSource (reuses upload_file workflow).
        upload_payload = FileUploadInput(
            file=self.SAMPLE_FILE.read_bytes(),
            filename=self.SAMPLE_FILE.name,
            source_type=UploadSourceType.FILE,
            created_by=self.CREATED_BY,
            additional_config={
                "category": "ingest-integration-test",
            },
        )

        upload_result = asyncio.run(
            start_upload_file(
                upload_payload,
                storage=self._storage,
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
                    {"status": "pending"},  # Reset to pending to allow ingestion
                )

        asyncio.run(_reset_datasource_status())

        # Step 2: Build a FileIngestionWorkflow with a lightweight Marker client.
        workspace_root = Path(settings.INGEST_FILE_WORKDIR)
        marker_client = self._TestMarkerClient(self.RAW_JSON)
        workflow = FileIngestionWorkflow(
            storage=self._storage,
            workspace_root=workspace_root,
            marker_client=marker_client,
            pg_client=self._pg_client,
        )

        ingest_input = FileIngestionInput(data_source_id=data_source_id)

        # Step 3: Run the ingest_file workflow end-to-end (except for real Marker).
        ingest_result = asyncio.run(workflow.run(ingest_input))

        # Basic result assertions.
        self.assertEqual(ingest_result.status, FileIngestionStatus.COMPLETE)
        self.assertEqual(ingest_result.data_source_id, data_source_id)
        self.assertIsNotNone(ingest_result.root_node)
        self.assertGreater(len(ingest_result.root_node.children), 0)

        # Step 4: Verify DataSource metadata has been updated in the database.
        datasource_after = asyncio.run(self._fetch_datasource(data_source_id))
        self.assertIsNotNone(datasource_after)
        self.assertEqual(datasource_after.status, "active")

        # Step 5: Save output to JSON files for inspection.
        self._save_test_output(ingest_result, data_source_id)

    def _save_test_output(self, ingest_result, data_source_id: UUID) -> None:
        """Save workflow output to JSON files for manual inspection.
        
        Saves the complete FileIngestionResult in the same format as the model definition.
        """
        # Create output directory if it doesn't exist
        self.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

        # Generate timestamp for unique filenames
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        base_filename = f"ingest_result_{timestamp}_{data_source_id.hex[:8]}"

        # Save complete FileIngestionResult in the same format as the model
        result_path = self.OUTPUT_DIR / f"{base_filename}.json"
        # Use model_dump_json() to serialize the entire FileIngestionResult object
        result_json = ingest_result.model_dump_json(
            indent=2,
            exclude_none=False
        )
        with open(result_path, "w", encoding="utf-8") as f:
            f.write(result_json)
        print(f"\n✅ Saved complete FileIngestionResult to: {result_path}")


if __name__ == "__main__":
    unittest.main()

