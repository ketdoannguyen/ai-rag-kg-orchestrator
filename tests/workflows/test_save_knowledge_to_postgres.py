"""Integration test for save_knowledge_to_postgres workflow with real database.

This test:
- Loads enriched tree from enrich_knowledge_tree workflow output JSON file
- Uses real PostgreSQL database
- Saves knowledge nodes recursively
- Verifies nodes are saved correctly in database
"""

from __future__ import annotations

import asyncio
import json
import unittest
from pathlib import Path
from uuid import UUID

from postgres_client import PostgresClient, parse_node_from_dict
from postgres_client.handlers.knowledge_handler import KnowledgeNodeHandler
from postgres_client.handlers.system_admin_handler import SystemAdminHandler
from postgres_client.models.enums import NodeType
from workflows.dependencies import init_postgres_client
from workflows.save_knowledge_to_postgres import (
    SaveKnowledgeInput,
    start_save_knowledge_to_postgres,
)
class SaveKnowledgeToPostgresIntegrationTest(unittest.TestCase):
    """Integration test for save_knowledge_to_postgres workflow."""

    OUTPUT_DIR = Path("tests/workflows/output")
    CREATED_BY = UUID("550e8400-e29b-41d4-a716-446655440000")
    ENRICH_OUTPUT_FILE = Path("tests/workflows/output/enrich_tree_result_20251204_075350_4ec95e5b_tree.json")

    @classmethod
    def setUpClass(cls) -> None:
        """Set up test environment."""
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
                from postgres_client.models.system_admin import User

                user = User(
                    id=cls.CREATED_BY,
                    username="save_knowledge_workflow_tester",
                    hashed_password="temporary",
                )
                await handler.users.create(user)

    def _find_enrich_output_file(self) -> Path | None:
        """Find the most recent enrich_tree_result JSON file."""
        if not self.OUTPUT_DIR.exists():
            return None

        enrich_files = sorted(
            self.OUTPUT_DIR.glob("enrich_tree_result_*_tree.json"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )

        return enrich_files[0] if enrich_files else None

    async def _ensure_datasource_exists(self, source_id: UUID) -> None:
        """Ensure DataSource exists in database, create if not exists."""
        async with self._sessionmaker() as session:
            handler = SystemAdminHandler(session)
            existing = await handler.sources.get_by_id(source_id)
            if existing is None:
                # Create a minimal DataSource if it doesn't exist
                from postgres_client.models.system_admin import DataSource
                from postgres_client.models.enums import SourceType

                datasource = DataSource(
                    id=source_id,
                    name="Test DataSource from enrich output",
                    type=SourceType.PDF,
                    config={"source": "enrich_knowledge_tree_test"},
                    status="processing",
                    created_by=self.CREATED_BY,
                )
                await handler.sources.create(datasource)
                print(f"  ✓ Created DataSource: {source_id}")
            else:
                print(f"  ✓ DataSource already exists: {source_id}")

    async def _verify_nodes_in_database(
        self, root_node_id: UUID, expected_count: int
    ) -> None:
        """Verify nodes are saved correctly in database."""
        async with self._sessionmaker() as session:
            handler = KnowledgeNodeHandler(session)

            # Get root node
            root_node = await handler.get_by_id(root_node_id)
            self.assertIsNotNone(root_node, "Root node should be saved")
            self.assertEqual(root_node.node_type, NodeType.STRUCTURE)

            # Get all nodes by source_id
            all_nodes = await handler.get_by_source(root_node.source_id)
            self.assertGreaterEqual(
                len(all_nodes),
                expected_count,
                f"Should have at least {expected_count} nodes",
            )

            # Verify tree structure
            structure_nodes = [n for n in all_nodes if n.node_type == NodeType.STRUCTURE]
            detail_nodes = [
                n
                for n in all_nodes
                if n.node_type in (NodeType.PARAGRAPH, NodeType.IMAGE, NodeType.TABLE)
            ]

            print(f"  ✓ Total nodes in database: {len(all_nodes)}")
            print(f"  ✓ Structure nodes: {len(structure_nodes)}")
            print(f"  ✓ Detail nodes: {len(detail_nodes)}")

    def test_save_knowledge_from_enrich_output(self) -> None:
        """Test saving knowledge tree from enrich_knowledge_tree workflow output."""

        # Step 1: Use specified enrich output file or find the most recent one
        enrich_file = self.ENRICH_OUTPUT_FILE if self.ENRICH_OUTPUT_FILE.exists() else self._find_enrich_output_file()
        if enrich_file is None or not enrich_file.exists():
            self.skipTest(
                f"Enrich output file not found: {self.ENRICH_OUTPUT_FILE} or in {self.OUTPUT_DIR}. "
                "Please run enrich_knowledge_tree workflow first."
            )

        print(f"\n{'='*80}")
        print(f"Loading enriched tree from: {enrich_file.name}")
        print(f"{'='*80}")

        # Step 2: Load enriched tree from JSON
        tree_data = json.loads(enrich_file.read_text(encoding="utf-8"))
        enriched_tree = parse_node_from_dict(tree_data)

        self.assertIsNotNone(enriched_tree)
        self.assertEqual(enriched_tree.node_type, NodeType.STRUCTURE)

        # Step 3: Extract source_id from enriched tree
        data_source_id = enriched_tree.source_id
        if data_source_id is None:
            self.fail("Enriched tree does not have source_id. Cannot proceed with test.")

        print(f"  ✓ Extracted source_id from tree: {data_source_id}")

        # Count nodes before saving
        def count_nodes(node):
            count = 1
            for child in node.children:
                count += count_nodes(child)
            return count

        total_nodes = count_nodes(enriched_tree)
        print(f"  ✓ Loaded tree with {total_nodes} nodes")

        # Step 4: Ensure DataSource exists in database
        asyncio.run(self._ensure_datasource_exists(data_source_id))

        # Step 5: Run save_knowledge_to_postgres workflow
        save_input = SaveKnowledgeInput(
            enriched_tree=enriched_tree,
            data_source_id=data_source_id,
        )

        result = asyncio.run(
            start_save_knowledge_to_postgres(
                save_input,
                pg_client=self._pg_client,
            )
        )

        # Step 6: Verify results
        self.assertIsNotNone(result)
        self.assertIn("root_node_id", result)
        self.assertIn("total_nodes_saved", result)
        self.assertIn("structure_nodes_count", result)
        self.assertIn("detail_nodes_count", result)
        self.assertIn("saved_nodes", result)
        self.assertIn("failed_nodes", result)

        root_node_id = UUID(result["root_node_id"])
        total_saved = result["total_nodes_saved"]
        structure_count = result["structure_nodes_count"]
        detail_count = result["detail_nodes_count"]
        failed_count = len(result["failed_nodes"])

        print(f"\n--- Save Knowledge Results ---")
        print(f"  Root node ID: {root_node_id}")
        print(f"  Total nodes saved: {total_saved}")
        print(f"  Structure nodes: {structure_count}")
        print(f"  Detail nodes: {detail_count}")
        print(f"  Failed nodes: {failed_count}")

        # Print first few failed nodes for debugging
        if failed_count > 0:
            print(f"\n--- First Failed Node Error ---")
            first_failed = result["failed_nodes"][0]
            print(f"  Node ID: {first_failed.get('node', {}).get('id', 'N/A')}")
            print(f"  Error: {first_failed.get('error', 'N/A')}")
            if isinstance(first_failed.get('node'), dict):
                print(f"  Node Type: {first_failed.get('node', {}).get('node_type', 'N/A')}")

        # Assertions
        self.assertGreater(total_saved, 0, "Should save at least some nodes")
        self.assertGreater(structure_count, 0, "Should save structure nodes")
        self.assertGreater(detail_count, 0, "Should save detail nodes")

        # Step 7: Verify nodes in database
        asyncio.run(self._verify_nodes_in_database(root_node_id, total_saved))

        # Step 8: Verify saved_nodes list
        saved_nodes = result["saved_nodes"]
        self.assertEqual(len(saved_nodes), total_saved)
        self.assertEqual(saved_nodes[0]["id"], str(root_node_id))

        # Step 9: Save test output
        self._save_test_output(result, root_node_id, enrich_file.name)

        print(f"\n{'='*80}")
        print(f"✅ Test completed successfully")
        print(f"{'='*80}\n")

    def _save_test_output(
        self, result: dict, root_node_id: UUID, source_file: str
    ) -> None:
        """Save test output to JSON file."""
        self.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

        from datetime import datetime

        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        filename = f"save_knowledge_result_{timestamp}_{root_node_id.hex[:8]}.json"

        output_path = self.OUTPUT_DIR / filename
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False, default=str)

        print(f"✅ Saved test output to: {output_path}")


if __name__ == "__main__":
    unittest.main()

