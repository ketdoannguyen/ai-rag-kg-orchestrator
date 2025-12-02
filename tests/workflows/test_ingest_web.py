from __future__ import annotations

import asyncio
import json
import unittest
from datetime import datetime
from pathlib import Path
from uuid import UUID

from workflows.dependencies import init_postgres_client
from workflows.ingest_web import WebIngestionInput, WebIngestionStatus, start_ingest_web
from postgres_client import PostgresClient
from postgres_client.handlers.system_admin_handler import SystemAdminHandler
from postgres_client.models.system_admin import User


class IngestWebWorkflowIntegrationTest(unittest.TestCase):
    """Integration test for ingest_web workflow using real database and real crawling.

    This test performs actual web crawling without mocking, interacting with:
    - Real PostgreSQL database
    - Real web URLs (crawl4ai)
    - Real content extraction and knowledge tree building

    Test URLs:
    - https://tuyensinh.vku.udn.vn/cntt-ks
    - https://tuyensinh.vku.udn.vn/pt-tuyenthang
    """

    TEST_URLS = [
        "https://tuyensinh.vku.udn.vn/cntt-ks",
        "https://tuyensinh.vku.udn.vn/pt-tuyenthang",
    ]
    CREATED_BY = UUID("550e8400-e29b-41d4-a716-446655440000")
    OUTPUT_DIR = Path("tests/workflows/output")

    @classmethod
    def setUpClass(cls) -> None:
        """Set up test environment: initialize database client and ensure test user exists."""
        cls._pg_client: PostgresClient = init_postgres_client()
        cls._sessionmaker = cls._pg_client.sessionmaker
        asyncio.run(cls._ensure_user_exists())

    @classmethod
    def tearDownClass(cls) -> None:
        """Clean up: close database connection."""
        if hasattr(cls, "_pg_client"):
            asyncio.run(cls._pg_client.close())

    @classmethod
    async def _ensure_user_exists(cls) -> None:
        """Ensure test user exists in database."""
        async with cls._sessionmaker() as session:
            handler = SystemAdminHandler(session)
            existing = await handler.users.get_by_id(cls.CREATED_BY)
            if existing is None:
                user = User(
                    id=cls.CREATED_BY,
                    username="ingest_web_workflow_tester",
                    hashed_password="temporary",
                )
                await handler.users.create(user)
                print(f"✓ Created test user: {cls.CREATED_BY}")

    async def _fetch_datasource(self, source_id: UUID):
        """Fetch DataSource from database by ID."""
        async with self._sessionmaker() as session:
            handler = SystemAdminHandler(session)
            return await handler.sources.get_by_id(source_id)

    def test_ingest_web_urls_real_crawl(self) -> None:
        """Test web ingestion workflow with real URLs and real crawling.

        This test:
        1. Creates WebIngestionInput with test URLs
        2. Runs the complete workflow (validate, crawl, extract, build tree, save)
        3. Verifies results for each URL
        4. Checks DataSource records in database
        5. Saves output to JSON files for inspection
        """
        # Step 1: Create input payload
        payload = WebIngestionInput(
            urls=self.TEST_URLS,
            created_by=self.CREATED_BY,
            crawl_config={"respect_robots_txt": True},
            max_concurrent_urls=2,  # Process both URLs concurrently
            max_url_retries=2,
            per_url_timeout_seconds=120.0,
        )

        print(f"\n{'='*80}")
        print(f"Testing web ingestion with {len(self.TEST_URLS)} URLs")
        print(f"{'='*80}")

        # Step 2: Run the workflow
        result = asyncio.run(start_ingest_web(payload, pg_client=self._pg_client))

        # Step 3: Basic assertions
        self.assertIsNotNone(result)
        self.assertIn("results", result)
        self.assertEqual(len(result["results"]), len(self.TEST_URLS))

        # Step 4: Verify each URL result
        data_source_ids = []
        for i, url_result in enumerate(result["results"]):
            url = self.TEST_URLS[i]
            print(f"\n--- Result for URL {i+1}: {url} ---")
            print(f"Status: {url_result['status']}")

            if url_result["status"] == WebIngestionStatus.SUCCESS:
                self.assertIsNotNone(url_result["data_source"])
                self.assertIsNotNone(url_result["root_node"])
                self.assertIsNotNone(url_result["statistics"])

                data_source_id = UUID(url_result["data_source"]["id"])
                data_source_ids.append(data_source_id)

                # Verify DataSource in database
                datasource = asyncio.run(self._fetch_datasource(data_source_id))
                self.assertIsNotNone(datasource)
                # type can be enum or string depending on serialization, check case-insensitive
                type_value = datasource.type.value if hasattr(datasource.type, "value") else datasource.type
                self.assertEqual(type_value.upper() if isinstance(type_value, str) else str(type_value).upper(), "WEB")
                self.assertEqual(datasource.status, "processing")
                self.assertEqual(datasource.created_by, self.CREATED_BY)

                # Verify root node structure
                root_node = url_result["root_node"]
                node_type = root_node.get("node_type", "").upper()
                self.assertEqual(node_type, "STRUCTURE")
                self.assertGreater(len(root_node.get("children", [])), 0)

                # Verify statistics
                stats = url_result["statistics"]
                self.assertGreater(stats.get("total_nodes", 0), 0)
                self.assertIn("nodes_by_level", stats)
                self.assertIn("crawl_duration_seconds", stats)

                print(f"  ✓ DataSource ID: {data_source_id}")
                print(f"  ✓ Total nodes: {stats.get('total_nodes', 0)}")
                print(f"  ✓ Crawl duration: {stats.get('crawl_duration_seconds', 0):.2f}s")

            elif url_result["status"] == WebIngestionStatus.ERROR:
                print(f"  ✗ Error: {url_result.get('error_message', 'Unknown error')}")
                # Even on error, DataSource might be created
                if url_result.get("data_source"):
                    data_source_id = UUID(url_result["data_source"]["id"])
                    datasource = asyncio.run(self._fetch_datasource(data_source_id))
                    if datasource:
                        self.assertEqual(datasource.status, "error")

            elif url_result["status"] == WebIngestionStatus.SKIPPED:
                print(f"  ⊘ Skipped: {url_result.get('error_message', 'Unknown reason')}")

        # Step 5: Verify at least one URL succeeded
        succeeded_count = sum(
            1 for r in result["results"] if r["status"] == WebIngestionStatus.SUCCESS
        )
        self.assertGreater(
            succeeded_count,
            0,
            "At least one URL should be successfully ingested",
        )

        # Step 6: Save output to JSON files for inspection
        self._save_test_output(result, data_source_ids)

        print(f"\n{'='*80}")
        print(f"Test completed: {succeeded_count}/{len(self.TEST_URLS)} URLs succeeded")
        print(f"{'='*80}\n")

    def _save_test_output(self, result: dict, data_source_ids: list[UUID]) -> None:
        """Save workflow output to JSON files for manual inspection.

        Saves the complete WebIngestionResult in the same format as the model definition.
        """
        # Create output directory if it doesn't exist
        self.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

        # Generate timestamp for unique filenames
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        ids_str = "_".join([ds_id.hex[:8] for ds_id in data_source_ids[:3]])
        base_filename = f"ingest_web_result_{timestamp}_{ids_str}"

        # Save complete WebIngestionResult
        result_path = self.OUTPUT_DIR / f"{base_filename}.json"
        with open(result_path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        print(f"\n✅ Saved complete WebIngestionResult to: {result_path}")

        # Save individual results for each URL
        for i, url_result in enumerate(result["results"]):
            url = self.TEST_URLS[i]
            url_safe = url.replace("https://", "").replace("/", "_").replace("?", "_")[:50]
            url_result_path = self.OUTPUT_DIR / f"{base_filename}_{url_safe}.json"
            with open(url_result_path, "w", encoding="utf-8") as f:
                json.dump(url_result, f, indent=2, ensure_ascii=False)
            print(f"✅ Saved result for {url} to: {url_result_path}")


if __name__ == "__main__":
    unittest.main()

