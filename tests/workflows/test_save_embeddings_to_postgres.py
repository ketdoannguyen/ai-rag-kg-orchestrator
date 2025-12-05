"""Integration test for save_embeddings_to_postgres workflow with real database.

This test:
- Loads embeddings from embed_nodes workflow output JSON file
- Uses real PostgreSQL database
- Saves embeddings to embeddings_structure and embeddings_content tables
- Verifies embeddings are saved correctly in database
"""

from __future__ import annotations

import asyncio
import json
import unittest
from pathlib import Path
from uuid import UUID

from postgres_client import PostgresClient
from postgres_client.handlers.embedding_handler import EmbeddingHandler
from workflows.dependencies import init_postgres_client
from workflows.save_embeddings_to_postgres import (
    SaveEmbeddingsInput,
    start_save_embeddings_to_postgres,
)


class SaveEmbeddingsToPostgresIntegrationTest(unittest.TestCase):
    """Integration test for save_embeddings_to_postgres workflow."""

    OUTPUT_DIR = Path("tests/workflows/output")
    CREATED_BY = UUID("550e8400-e29b-41d4-a716-446655440000")

    @classmethod
    def setUpClass(cls) -> None:
        """Set up test environment."""
        cls._pg_client: PostgresClient = init_postgres_client()
        cls._sessionmaker = cls._pg_client.sessionmaker

    @classmethod
    def tearDownClass(cls) -> None:
        """Clean up: close database connection."""
        if hasattr(cls, "_pg_client"):
            asyncio.run(cls._pg_client.close())

    def _find_embed_output_file(self) -> Path | None:
        """Find the most recent embed_nodes_result JSON file."""
        if not self.OUTPUT_DIR.exists():
            return None

        embed_files = sorted(
            self.OUTPUT_DIR.glob("embed_nodes_result_*.json"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )

        return embed_files[0] if embed_files else None

    async def _verify_embeddings_in_database(
        self, structure_count: int, content_count: int
    ) -> None:
        """Verify embeddings are saved correctly in database."""
        async with self._sessionmaker() as session:
            handler = EmbeddingHandler(session)

            # Get structure embeddings with vectors
            structure_embeddings = await handler.structures.get_with_vectors()
            self.assertGreaterEqual(
                len(structure_embeddings),
                structure_count,
                f"Should have at least {structure_count} structure embeddings",
            )

            # Get content embeddings with vectors
            content_embeddings = await handler.contents.get_with_vectors()
            self.assertGreaterEqual(
                len(content_embeddings),
                content_count,
                f"Should have at least {content_count} content embeddings",
            )

            print(f"  ✓ Structure embeddings in database: {len(structure_embeddings)}")
            print(f"  ✓ Content embeddings in database: {len(content_embeddings)}")

            # Verify embedding structure
            if structure_embeddings:
                struct_emb = structure_embeddings[0]
                self.assertIsNotNone(struct_emb.node_id)
                self.assertIsNotNone(struct_emb.summary_text)
                self.assertIsNotNone(struct_emb.vector)
                self.assertGreater(len(struct_emb.vector), 0)

            if content_embeddings:
                content_emb = content_embeddings[0]
                self.assertIsNotNone(content_emb.node_id)
                self.assertIsNotNone(content_emb.chunk_text)
                self.assertIsNotNone(content_emb.vector)
                self.assertGreater(len(content_emb.vector), 0)

    def test_save_embeddings_from_embed_output(self) -> None:
        """Test saving embeddings from embed_nodes workflow output."""

        # Step 1: Find embed output file
        embed_file = self._find_embed_output_file()
        if embed_file is None:
            self.skipTest(
                f"No embed_nodes_result JSON file found in {self.OUTPUT_DIR}. "
                "Please run embed_nodes workflow first."
            )

        print(f"\n{'='*80}")
        print(f"Loading embeddings from: {embed_file.name}")
        print(f"{'='*80}")

        # Step 2: Load embeddings from JSON
        embed_data = json.loads(embed_file.read_text(encoding="utf-8"))

        structure_embeddings_data = embed_data.get("structure_embeddings", [])
        content_embeddings_data = embed_data.get("content_embeddings", [])

        if not structure_embeddings_data and not content_embeddings_data:
            self.skipTest("No embeddings found in embed output file")

        print(f"  ✓ Loaded {len(structure_embeddings_data)} structure embeddings")
        print(f"  ✓ Loaded {len(content_embeddings_data)} content embeddings")

        # Step 3: Parse embeddings to Pydantic models
        from postgres_client.models.embeddings import (
            EmbeddingStructure,
            EmbeddingContent,
        )

        # Parse all embeddings (workflow will normalize vector dimensions)
        structure_embeddings = [
            EmbeddingStructure(**emb) for emb in structure_embeddings_data if emb.get("vector")
        ]
        content_embeddings = [
            EmbeddingContent(**emb) for emb in content_embeddings_data if emb.get("vector")
        ]
        
        # Log vector dimensions info
        if structure_embeddings:
            dims = [len(emb.vector) for emb in structure_embeddings if emb.vector]
            if dims:
                print(f"  ✓ Structure embedding vector dimensions: min={min(dims)}, max={max(dims)}, unique={set(dims)}")
        if content_embeddings:
            dims = [len(emb.vector) for emb in content_embeddings if emb.vector]
            if dims:
                print(f"  ✓ Content embedding vector dimensions: min={min(dims)}, max={max(dims)}, unique={set(dims)}")

        # Step 4: Run save_embeddings_to_postgres workflow
        save_input = SaveEmbeddingsInput(
            structure_embeddings=structure_embeddings,
            content_embeddings=content_embeddings,
        )

        print(f"\n--- Starting Save Embeddings Workflow ---")
        print(f"  Structure embeddings: {len(structure_embeddings)}")
        print(f"  Content embeddings: {len(content_embeddings)}")

        result = asyncio.run(
            start_save_embeddings_to_postgres(
                save_input,
                pg_client=self._pg_client,
            )
        )

        # Step 5: Verify results
        self.assertIsNotNone(result)
        self.assertIn("structures_saved", result)
        self.assertIn("contents_saved", result)
        self.assertIn("total_saved", result)
        self.assertIn("failed_nodes", result)

        structures_saved = result["structures_saved"]
        contents_saved = result["contents_saved"]
        total_saved = result["total_saved"]
        failed_nodes = result["failed_nodes"]

        print(f"\n--- Save Embeddings Results ---")
        print(f"  Structures saved: {structures_saved}")
        print(f"  Contents saved: {contents_saved}")
        print(f"  Total saved: {total_saved}")
        print(f"  Failed nodes: {len(failed_nodes)}")

        # Assertions
        self.assertGreater(total_saved, 0, "Should save at least some embeddings")
        self.assertEqual(
            structures_saved,
            len(structure_embeddings),
            "Should save all structure embeddings",
        )
        self.assertEqual(
            contents_saved,
            len(content_embeddings),
            "Should save all content embeddings",
        )

        # Step 6: Verify embeddings in database
        asyncio.run(
            self._verify_embeddings_in_database(structures_saved, contents_saved)
        )

        # Step 7: Test update scenario (save same embeddings again)
        print(f"\n--- Testing Update Scenario ---")
        result2 = asyncio.run(
            start_save_embeddings_to_postgres(
                save_input,
                pg_client=self._pg_client,
            )
        )

        # Should still save successfully (update existing)
        self.assertGreater(result2["total_saved"], 0)
        print(f"  ✓ Updated {result2['total_saved']} embeddings")

        # Step 8: Save test output
        self._save_test_output(result, embed_file.name)

        print(f"\n{'='*80}")
        print(f"✅ Test completed successfully")
        print(f"{'='*80}\n")

    def _save_test_output(self, result: dict, source_file: str) -> None:
        """Save test output to JSON file."""
        self.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

        from datetime import datetime

        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        filename = f"save_embeddings_result_{timestamp}.json"

        output_path = self.OUTPUT_DIR / filename
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False, default=str)

        print(f"✅ Saved test output to: {output_path}")


if __name__ == "__main__":
    unittest.main()

