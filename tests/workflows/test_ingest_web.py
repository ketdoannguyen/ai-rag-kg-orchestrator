"""Integration tests for web ingestion pipeline.

Full pipeline: ingest_web → enrich_knowledge_tree → save_knowledge_to_arango → embed_nodes → save_embeddings

Output JSON files for manual inspection and pipeline efficiency evaluation.
All output files are organized in timestamped folders for easy management.
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
    parse_node_from_dict,
)
from loguru import logger

from workflows.dependencies import init_arango_client, init_llm_client
from workflows.embed_nodes import start_embed_nodes
from workflows.embed_nodes.helper import extract_text_for_embedding
from workflows.embed_nodes.models import EmbedNodesInput
from workflows.enrich_knowledge_tree import start_enrich_knowledge_tree
from workflows.enrich_knowledge_tree.models import EnrichKnowledgeTreeInput
from workflows.ingest_web import WebIngestionInput, WebIngestionStatus, start_ingest_web
from workflows.save_embeddings_to_arango import start_save_embeddings_to_arango
from workflows.save_embeddings_to_arango.models import SaveEmbeddingsInput
from workflows.save_knowledge_to_arango import start_save_knowledge_to_arango
from workflows.save_knowledge_to_arango.models import SaveKnowledgeInput


class IngestWebPipelineTest(unittest.TestCase):
    """Full pipeline integration test for web ingestion.

    Pipeline: ingest_web → enrich → save → embed → save_embeddings

    Output files are organized in timestamped folders for easy management.
    """

    # Test configuration
    TEST_URL = "https://vietnamnet.vn/giao-vien-tphcm-se-nhan-hang-chuc-trieu-dong-dip-tet-cao-nhat-gan-100-trieu-2479008.html"
    CREATED_BY = UUID("550e8400-e29b-41d4-a716-446655440000")
    BASE_OUTPUT_DIR = Path("tests/workflows/output")

    @classmethod
    def setUpClass(cls) -> None:
        """Set up test fixtures."""
        cls._arango_client: ArangoClient = init_arango_client()
        cls._llm_client = init_llm_client()

    @classmethod
    def tearDownClass(cls) -> None:
        """Clean up resources."""
        if hasattr(cls, "_arango_client"):
            cls._arango_client.close()

    def _create_output_folder(self) -> Path:
        """Create timestamped output folder for this test run."""
        timestamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
        output_dir = self.BASE_OUTPUT_DIR / f"web_{timestamp}"
        output_dir.mkdir(parents=True, exist_ok=True)
        return output_dir

    def _save_json(self, data: dict, filepath: Path) -> None:
        """Save data to JSON file."""
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False, default=str)
        logger.info(f"Saved: {filepath}")

    def _count_tree_nodes(self, node: dict) -> dict:
        """Count nodes in tree by type."""
        from collections import Counter

        counts = Counter()

        def _count(n):
            counts[n.get("node_type", "unknown")] += 1
            for child in n.get("children", []):
                _count(child)

        _count(node)
        return dict(counts)

    def _cleanup_nodes_by_source(self, data_source_id: UUID) -> tuple[int, int]:
        """Clean up nodes from data source using direct AQL. Returns (nodes_deleted, edges_deleted)."""
        old_nodes = self._arango_client.db.aql.execute(
            "FOR n IN knowledge_nodes FILTER n.source_id == @source_id RETURN n._key",
            bind_vars={"source_id": str(data_source_id)},
        )
        old_keys = list(old_nodes)

        if not old_keys:
            return 0, 0

        # Delete edges first
        edge_result = self._arango_client.db.aql.execute(
            """
            FOR e IN knowledge_edges
                LET from_key = PARSE_IDENTIFIER(e._from).key
                LET to_key = PARSE_IDENTIFIER(e._to).key
                FILTER from_key IN @keys OR to_key IN @keys
                REMOVE e IN knowledge_edges
                RETURN 1
            """,
            bind_vars={"keys": old_keys},
        )
        edges_deleted = len(list(edge_result))

        # Delete nodes
        node_result = self._arango_client.db.aql.execute(
            "FOR n IN knowledge_nodes FILTER n.source_id == @source_id REMOVE n IN knowledge_nodes RETURN 1",
            bind_vars={"source_id": str(data_source_id)},
        )
        nodes_deleted = len(list(node_result))

        return nodes_deleted, edges_deleted

    def test_full_pipeline(self) -> None:
        """Test full pipeline: ingest_web → enrich → save → embed → save_embeddings.

        All output files are saved to a timestamped folder for easy inspection.
        """
        output_dir = self._create_output_folder()

        print(f"\n{'=' * 80}")
        print("Full Web Ingestion Pipeline Test")
        print(f"URL: {self.TEST_URL}")
        print(f"Output folder: {output_dir}")
        print(f"{'=' * 80}")

        # ============ Stage 1: Ingest Web ============
        print("\n[Stage 1/6] Ingesting web content...")
        ingest_result = asyncio.run(
            start_ingest_web(
                WebIngestionInput(
                    urls=[self.TEST_URL],
                    created_by=self.CREATED_BY,
                    crawl_config={"respect_robots_txt": False},
                    max_url_retries=2,
                    per_url_timeout_seconds=60.0,
                ),
                arango_client=self._arango_client,
            )
        )

        self._save_json(ingest_result, output_dir / "1_ingest_result.json")

        url_result = ingest_result["results"][0]
        self.assertEqual(url_result["status"], WebIngestionStatus.SUCCESS)

        root_node_dict = url_result["root_node"]
        data_source_id = UUID(url_result["data_source"]["id"])

        # Save raw tree
        self._save_json(root_node_dict, output_dir / "1_ingest_tree.json")

        ingest_tree_counts = self._count_tree_nodes(root_node_dict)
        ingest_total = sum(ingest_tree_counts.values())
        print(f"  ✓ DataSource ID: {data_source_id}")
        print(f"  ✓ Parsed: {ingest_total} nodes ({ingest_tree_counts})")

        # Parse for next stage
        root_node = parse_node_from_dict(root_node_dict)

        # ============ Stage 2: Enrich Knowledge Tree ============
        print("\n[Stage 2/6] Enriching knowledge tree...")
        from config.domains import get_current_domain

        domain_schema = get_current_domain()

        enrich_result = asyncio.run(
            start_enrich_knowledge_tree(
                EnrichKnowledgeTreeInput(
                    knowledge_tree=root_node,
                    domain_schema=domain_schema,
                ),
                llm_client=self._llm_client,
            )
        )

        self._save_json(enrich_result, output_dir / "2_enrich_result.json")
        self._save_json(enrich_result["enriched_tree"], output_dir / "2_enriched_tree.json")

        enrich_stats = enrich_result["statistics"]
        enriched_tree_counts = self._count_tree_nodes(enrich_result["enriched_tree"])
        enriched_total = sum(enriched_tree_counts.values())

        print(
            f"  ✓ Paragraphs: {enrich_stats['paragraphs_before']} → {enrich_stats['paragraphs_after']}"
        )
        print(f"  ✓ Structures enriched: {enrich_stats['structures_enriched']}")
        print(f"  ✓ Enriched tree: {enriched_total} nodes ({enriched_tree_counts})")
        print(
            f"  ✓ LLM calls: detail={enrich_stats['detail_llm_calls']}, structure={enrich_stats['structure_llm_calls']}"
        )

        # Parse enriched tree for next stage
        enriched_tree = parse_node_from_dict(enrich_result["enriched_tree"])

        # ============ Stage 3: Save to ArangoDB ============
        print("\n[Stage 3/6] Saving to ArangoDB...")

        # Cleanup old nodes if any
        old_nodes_deleted, old_edges_deleted = self._cleanup_nodes_by_source(data_source_id)
        if old_nodes_deleted > 0:
            print(f"  → Cleaned up: {old_nodes_deleted} old nodes, {old_edges_deleted} edges")

        save_result = asyncio.run(
            start_save_knowledge_to_arango(
                SaveKnowledgeInput(
                    enriched_tree=enriched_tree,
                    data_source_id=data_source_id,
                ),
                arango_client=self._arango_client,
            )
        )

        self._save_json(save_result, output_dir / "3_save_result.json")

        relationships = save_result.get("relationships_created", {})
        total_relationships = sum(relationships.values())
        print(f"  ✓ Nodes saved: {save_result['total_nodes_saved']}")
        print(f"  ✓ Relationships: {relationships} (total: {total_relationships})")

        # Verify: enriched_tree nodes == saved nodes
        self.assertEqual(
            save_result["total_nodes_saved"],
            enriched_total,
            f"Mismatch: enriched tree has {enriched_total} nodes but saved {save_result['total_nodes_saved']}",
        )

        # ============ Stage 4: Embed Nodes ============
        print("\n[Stage 4/6] Embedding nodes...")
        node_handler = KnowledgeNodeHandler(self._arango_client.db)
        saved_nodes = node_handler.get_by_source(data_source_id)

        # Verify: DB nodes == saved nodes
        self.assertEqual(
            len(saved_nodes),
            save_result["total_nodes_saved"],
            f"Mismatch: DB has {len(saved_nodes)} nodes but save result says {save_result['total_nodes_saved']}",
        )

        embeddable = [n for n in saved_nodes if extract_text_for_embedding(n)]
        print(f"  → Embeddable nodes: {len(embeddable)}/{len(saved_nodes)}")

        embed_result = None
        embed_count = 0
        structure_embs = []
        content_embs = []

        if embeddable:
            embed_result = asyncio.run(
                start_embed_nodes(
                    EmbedNodesInput(nodes=embeddable, batch_size=10),
                    llm_client=self._llm_client,
                )
            )

            self._save_json(embed_result, output_dir / "4_embed_result.json")

            structure_embs = embed_result.get("structure_embeddings", [])
            content_embs = embed_result.get("content_embeddings", [])
            embed_count = len(structure_embs) + len(content_embs)
            print(
                f"  ✓ Embeddings created: {embed_count} (structure={len(structure_embs)}, content={len(content_embs)})"
            )

            # Verify: embeddings == embeddable nodes (1:1)
            self.assertEqual(
                embed_count,
                len(embeddable),
                f"Mismatch: {len(embeddable)} embeddable nodes but created {embed_count} embeddings",
            )

            # ============ Stage 5: Save Embeddings ============
            print("\n[Stage 5/6] Saving embeddings to ArangoDB...")
            emb_save_result = asyncio.run(
                start_save_embeddings_to_arango(
                    SaveEmbeddingsInput(
                        structure_embeddings=structure_embs,
                        content_embeddings=content_embs,
                    ),
                    arango_client=self._arango_client,
                )
            )

            self._save_json(emb_save_result, output_dir / "5_save_embeddings_result.json")

            emb_saved_total = (
                emb_save_result["structures_saved"] + emb_save_result["contents_saved"]
            )
            print(
                f"  ✓ Saved: {emb_saved_total} (structure={emb_save_result['structures_saved']}, content={emb_save_result['contents_saved']})"
            )

            # Verify: saved embeddings == created embeddings
            self.assertEqual(
                emb_saved_total,
                embed_count,
                f"Mismatch: created {embed_count} embeddings but saved {emb_saved_total}",
            )
        else:
            print("  ⚠ No embeddable nodes found, skipping embedding stages")

        # ============ Stage 6: Final Verification ============
        print("\n[Stage 6/6] Verifying database state...")

        final_nodes = node_handler.get_by_source(data_source_id)
        from collections import Counter

        final_types = Counter(
            str(n.node_type) if hasattr(n.node_type, "value") else n.node_type for n in final_nodes
        )

        print(f"  ✓ Final DB state: {len(final_nodes)} nodes ({dict(final_types)})")

        # Final verification
        self.assertEqual(
            len(final_nodes),
            enriched_total,
            f"Final verification failed: expected {enriched_total} nodes, found {len(final_nodes)}",
        )

        # ============ Final Summary ============
        summary = {
            "timestamp": datetime.now(UTC).isoformat(),
            "url": self.TEST_URL,
            "data_source_id": str(data_source_id),
            "output_folder": str(output_dir),
            "pipeline_stages": {
                "1_ingest": {
                    "status": "success",
                    "parsed_nodes": ingest_total,
                    "by_type": ingest_tree_counts,
                },
                "2_enrich": {
                    "status": "success",
                    "paragraphs_before": enrich_stats["paragraphs_before"],
                    "paragraphs_after": enrich_stats["paragraphs_after"],
                    "structures_enriched": enrich_stats["structures_enriched"],
                    "enriched_nodes": enriched_total,
                    "by_type": enriched_tree_counts,
                    "llm_calls": enrich_stats["detail_llm_calls"]
                    + enrich_stats["structure_llm_calls"],
                },
                "3_save": {
                    "status": "success",
                    "nodes_saved": save_result["total_nodes_saved"],
                    "relationships": relationships,
                },
                "4_embed": {
                    "status": "success" if embed_count > 0 else "skipped",
                    "embeddable_nodes": len(embeddable),
                    "embeddings_created": embed_count,
                    "structure": len(structure_embs),
                    "content": len(content_embs),
                },
                "5_save_embeddings": {
                    "status": "success" if embed_count > 0 else "skipped",
                    "saved": emb_saved_total if embed_count > 0 else 0,
                },
                "6_verification": {
                    "status": "success",
                    "final_node_count": len(final_nodes),
                    "expected": enriched_total,
                    "match": len(final_nodes) == enriched_total,
                },
            },
        }

        self._save_json(summary, output_dir / "pipeline_summary.json")

        print(f"\n{'=' * 80}")
        print("Pipeline Summary")
        print(f"{'=' * 80}")
        print(f"  • URL: {self.TEST_URL}")
        print(f"  • Data Source ID: {data_source_id}")
        print(f"  • Ingest: {ingest_total} nodes → Enrich: {enriched_total} nodes")
        print(
            f"  • Save: {save_result['total_nodes_saved']} nodes + {total_relationships} relationships"
        )
        print(f"  • Embed: {embed_count} embeddings")
        print("  • Verification: ✓ All counts match!")
        print(f"  • Output: {output_dir}")
        print(f"{'=' * 80}\n")

        # Final assertions
        self.assertGreater(save_result["total_nodes_saved"], 0)


if __name__ == "__main__":
    unittest.main()
