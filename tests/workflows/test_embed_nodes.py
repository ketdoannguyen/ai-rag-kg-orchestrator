"""Integration test for embed_nodes workflow with real LLM and database.

This test:
- Loads enriched tree from enrich_knowledge_tree workflow output JSON file
- Uses real LLM Gateway for embedding generation
- Creates embeddings for structure and detail nodes
- Verifies embeddings are created correctly
"""

from __future__ import annotations

import asyncio
import json
import unittest
from pathlib import Path
from typing import List
from uuid import UUID

from postgres_client import PostgresClient, parse_node_from_dict
from postgres_client.models.enums import NodeType
from workflows.dependencies import init_llm_client, init_postgres_client
from workflows.embed_nodes import EmbedNodesInput, start_embed_nodes
from workflows.save_knowledge_to_postgres import (
    SaveKnowledgeInput,
    start_save_knowledge_to_postgres,
)
class EmbedNodesIntegrationTest(unittest.TestCase):
    """Integration test for embed_nodes workflow."""

    OUTPUT_DIR = Path("tests/workflows/output")
    CREATED_BY = UUID("550e8400-e29b-41d4-a716-446655440000")
    ENRICH_OUTPUT_FILE = Path("tests/workflows/output/enrich_tree_result_20251204_075350_4ec95e5b_tree.json")

    @classmethod
    def setUpClass(cls) -> None:
        """Set up test environment."""
        cls._pg_client: PostgresClient = init_postgres_client()
        cls._llm_client = init_llm_client()
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
            from postgres_client.handlers.system_admin_handler import SystemAdminHandler
            from postgres_client.models.system_admin import User

            handler = SystemAdminHandler(session)
            existing = await handler.users.get_by_id(cls.CREATED_BY)
            if existing is None:
                user = User(
                    id=cls.CREATED_BY,
                    username="embed_nodes_workflow_tester",
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
            from postgres_client.handlers.system_admin_handler import SystemAdminHandler
            from postgres_client.models.system_admin import DataSource
            from postgres_client.models.enums import SourceType

            handler = SystemAdminHandler(session)
            existing = await handler.sources.get_by_id(source_id)
            if existing is None:
                # Create a minimal DataSource if it doesn't exist
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

    async def _save_knowledge_tree(
        self, enriched_tree, data_source_id: UUID
    ) -> tuple[UUID, List]:
        """Save knowledge tree and return root_node_id and saved nodes from database."""
        from workflows.save_knowledge_to_postgres import SaveKnowledgeInput

        save_input = SaveKnowledgeInput(
            enriched_tree=enriched_tree,
            data_source_id=data_source_id,
        )

        result = await start_save_knowledge_to_postgres(
            save_input,
            pg_client=self._pg_client,
        )

        root_node_id = UUID(result["root_node_id"])

        # Query nodes from database to get the actual saved nodes with IDs
        async with self._sessionmaker() as session:
            from postgres_client.handlers.knowledge_handler import KnowledgeNodeHandler

            handler = KnowledgeNodeHandler(session)
            saved_nodes = await handler.get_by_source(data_source_id)

        return root_node_id, saved_nodes

    def test_embed_nodes_from_enrich_output(self) -> None:
        """Test embedding nodes from enrich_knowledge_tree workflow output."""

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

        # Step 4: Ensure DataSource exists in database
        asyncio.run(self._ensure_datasource_exists(data_source_id))

        # Step 5: Save knowledge tree to database
        root_node_id, saved_nodes = asyncio.run(
            self._save_knowledge_tree(enriched_tree, data_source_id)
        )
        print(f"  ✓ Saved {len(saved_nodes)} nodes to database")
        print(f"  ✓ Root node ID: {root_node_id}")

        # Step 6: Prepare nodes for embedding
        # Filter nodes that have embeddable text
        from workflows.embed_nodes.helper import extract_text_for_embedding

        embeddable_nodes = [
            node for node in saved_nodes if extract_text_for_embedding(node) is not None
        ]

        print(f"  ✓ Found {len(embeddable_nodes)} nodes with embeddable text")

        if len(embeddable_nodes) == 0:
            self.skipTest("No nodes with embeddable text found")

        # Step 7: Run embed_nodes workflow
        embed_input = EmbedNodesInput(
            nodes=embeddable_nodes,
            batch_size=5,  # Small batch size for testing
        )

        print(f"\n--- Starting Embedding Workflow ---")
        print(f"  Nodes to embed: {len(embeddable_nodes)}")
        print(f"  Batch size: {embed_input.batch_size}")

        result = asyncio.run(
            start_embed_nodes(
                embed_input,
                llm_client=self._llm_client,
            )
        )

        # Step 8: Verify results
        self.assertIsNotNone(result)
        self.assertIn("structure_embeddings", result)
        self.assertIn("content_embeddings", result)
        self.assertIn("total_embeddings_created", result)
        self.assertIn("structure_embeddings_count", result)
        self.assertIn("content_embeddings_count", result)
        self.assertIn("failed_nodes", result)

        structure_embeddings = result["structure_embeddings"]
        content_embeddings = result["content_embeddings"]
        total_created = result["total_embeddings_created"]
        structure_count = result["structure_embeddings_count"]
        content_count = result["content_embeddings_count"]
        failed_count = len(result["failed_nodes"])

        print(f"\n--- Embedding Results ---")
        print(f"  Total embeddings created: {total_created}")
        print(f"  Structure embeddings: {structure_count}")
        print(f"  Content embeddings: {content_count}")
        print(f"  Failed nodes: {failed_count}")

        # Assertions
        self.assertGreater(total_created, 0, "Should create at least some embeddings")
        self.assertEqual(
            len(structure_embeddings),
            structure_count,
            "Structure embeddings count should match",
        )
        self.assertEqual(
            len(content_embeddings),
            content_count,
            "Content embeddings count should match",
        )

        # Verify embedding structure
        if structure_embeddings:
            struct_emb = structure_embeddings[0]
            self.assertIn("node_id", struct_emb)
            self.assertIn("summary_text", struct_emb)
            self.assertIn("vector", struct_emb)
            self.assertIsNotNone(struct_emb["vector"])
            self.assertGreater(len(struct_emb["vector"]), 0, "Vector should not be empty")
            print(f"  ✓ Structure embedding vector dimension: {len(struct_emb['vector'])}")

        if content_embeddings:
            content_emb = content_embeddings[0]
            self.assertIn("node_id", content_emb)
            self.assertIn("chunk_text", content_emb)
            self.assertIn("vector", content_emb)
            self.assertIsNotNone(content_emb["vector"])
            self.assertGreater(len(content_emb["vector"]), 0, "Vector should not be empty")
            print(f"  ✓ Content embedding vector dimension: {len(content_emb['vector'])}")

        # Step 9: Save test output
        self._save_test_output(result, enrich_file.name)

        print(f"\n{'='*80}")
        print(f"✅ Test completed successfully")
        print(f"{'='*80}\n")

    def _save_test_output(self, result: dict, source_file: str) -> None:
        """Save test output to JSON file."""
        self.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

        from datetime import datetime

        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        filename = f"embed_nodes_result_{timestamp}.json"

        output_path = self.OUTPUT_DIR / filename

        # Truncate vectors only for first few embeddings to reduce file size (for display)
        # Keep full vectors for all others as they're needed for save_embeddings_to_postgres test
        output_result = result.copy()
        if output_result.get("structure_embeddings"):
            for i, emb in enumerate(output_result["structure_embeddings"]):
                if "vector" in emb and emb["vector"]:
                    # Only truncate first 3 for display, keep full vectors for rest
                    if i < 3:
                        emb["vector"] = emb["vector"][:10]  # Keep only first 10 dimensions for display
                    # For others, keep full vector (needed for save_embeddings_to_postgres test)
        if output_result.get("content_embeddings"):
            for i, emb in enumerate(output_result["content_embeddings"]):
                if "vector" in emb and emb["vector"]:
                    # Only truncate first 3 for display, keep full vectors for rest
                    if i < 3:
                        emb["vector"] = emb["vector"][:10]  # Keep only first 10 dimensions for display
                    # For others, keep full vector (needed for save_embeddings_to_postgres test)

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(output_result, f, indent=2, ensure_ascii=False, default=str)

        print(f"✅ Saved test output to: {output_path}")


if __name__ == "__main__":
    unittest.main()

