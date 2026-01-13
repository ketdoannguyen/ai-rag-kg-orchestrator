from __future__ import annotations

import asyncio
import unittest
from pathlib import Path
from uuid import UUID

from arango_client import ArangoClient, SystemAdminHandler, User

from workflows.dependencies import get_minio_client, init_arango_client
from workflows.upload_file import FileUploadInput, UploadSourceType, start_upload_file


class UploadFileWorkflowIntegrationTest(unittest.TestCase):
    """Integration test that uploads a real PDF into real storage + database."""

    SAMPLE_FILE = Path("data/input/SGK-Vat-li-12-KNTT-0-10.pdf")
    CREATED_BY = UUID("550e8400-e29b-41d4-a716-446655440000")

    @classmethod
    def setUpClass(cls) -> None:
        if not cls.SAMPLE_FILE.exists():
            raise FileNotFoundError(
                f"Sample file '{cls.SAMPLE_FILE}' is required for the integration test."
            )
        cls._storage = get_minio_client()
        cls._arango_client: ArangoClient = init_arango_client()
        cls._ensure_user_exists()

    @classmethod
    def tearDownClass(cls) -> None:
        if hasattr(cls, "_arango_client"):
            cls._arango_client.close()

    @classmethod
    def _ensure_user_exists(cls) -> None:
        """Create test user if not exists (sync operation)."""
        handler = SystemAdminHandler(cls._arango_client.db)
        existing = handler.users.get_by_id(cls.CREATED_BY)
        if existing is None:
            user = User(
                id=cls.CREATED_BY,
                username="upload_workflow_tester",
                hashed_password="temporary",
            )
            handler.users.create(user)

    def test_upload_pdf_end_to_end(self) -> None:
        payload = FileUploadInput(
            file=self.SAMPLE_FILE.read_bytes(),
            filename=self.SAMPLE_FILE.name,
            source_type=UploadSourceType.FILE,
            created_by=self.CREATED_BY,
            additional_config={
                "category": "integration-test",
                "department": "qa",
            },
        )

        result = asyncio.run(
            start_upload_file(
                payload,
                storage=self._storage,
                arango_client=self._arango_client,
                fail_on_duplicate=False,
            )
        )

        self.assertIn("data_source", result)
        self.assertIn("minio_info", result)

        data_source = result["data_source"]
        minio_info = result["minio_info"]

        self.assertEqual(data_source["name"], self.SAMPLE_FILE.name)
        self.assertEqual(data_source["config"]["original_filename"], self.SAMPLE_FILE.name)
        self.assertIn("minio_path", data_source["config"])
        self.assertTrue(minio_info["path"].endswith(self.SAMPLE_FILE.name))
        self.assertEqual(minio_info["md5_hash"], data_source["hash_content"])


if __name__ == "__main__":
    unittest.main()
