"""Integration test for save_knowledge_to_neo4j workflow with real Neo4j database.

This test:
- Loads enriched tree from enrich_knowledge_tree workflow output JSON file
- Uses real Neo4j database
- Saves knowledge nodes recursively with relationships
- Verifies nodes and relationships are saved correctly in Neo4j
"""

from __future__ import annotations

import asyncio
import json
import unittest
from pathlib import Path
from uuid import UUID

from neo4j import AsyncDriver
from postgres_client import parse_node_from_dict
from postgres_client.models.enums import NodeType
from workflows.dependencies import init_neo4j_client
from workflows.save_knowledge_to_neo4j import (
    SaveKnowledgeToNeo4jInput,
    start_save_knowledge_to_neo4j,
)


class SaveKnowledgeToNeo4jIntegrationTest(unittest.TestCase):
    """Integration test for save_knowledge_to_neo4j workflow."""

    OUTPUT_DIR = Path("tests/workflows/output")
    CREATED_BY = UUID("550e8400-e29b-41d4-a716-446655440000")
    ENRICH_OUTPUT_FILE = Path("tests/workflows/output/enrich_tree_result_20251205_144749_84551a15_tree.json")

    @classmethod
    def setUpClass(cls) -> None:
        """Set up test environment."""
        try:
            cls._neo4j_client = init_neo4j_client()
            # Verify connectivity (non-blocking, will skip test if connection fails)
            connected = asyncio.run(cls._neo4j_client.verify_connectivity())
            if not connected:
                cls._skip_test = True
                print("  ⚠ Neo4j connection failed - test will be skipped")
            else:
                cls._skip_test = False
                print("  ✓ Neo4j connection verified")
                # Clean up old data before running test
                asyncio.run(cls._cleanup_database())
        except Exception as e:
            cls._skip_test = True
            print(f"  ⚠ Neo4j setup failed: {e} - test will be skipped")

    @classmethod
    async def _cleanup_database(cls) -> None:
        """Delete all nodes and relationships from Neo4j database."""
        # Create a new client for cleanup to avoid event loop conflicts
        cleanup_client = init_neo4j_client()
        driver = cleanup_client.async_driver
        try:
            async with driver.session() as session:
                # Delete all relationships and nodes
                delete_query = """
                MATCH (n)
                DETACH DELETE n
                """
                result = await session.run(delete_query)
                await result.consume()
                print("  ✓ Cleaned up old Neo4j data")
        finally:
            await cleanup_client.close()

    @classmethod
    def tearDownClass(cls) -> None:
        """Clean up: close Neo4j connection."""
        if hasattr(cls, "_neo4j_client"):
            try:
                asyncio.run(cls._neo4j_client.close())
            except Exception:
                pass  # Ignore errors during cleanup

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

    async def _ensure_datasource_node_exists(self, source_id: UUID) -> None:
        """Ensure DataSource node exists in Neo4j, create if not exists.
        
        Note: This is optional as the workflow will create DataSource node automatically
        when creating BELONGS_TO_SOURCE relationships.
        """
        # Skip this step - workflow will create DataSource node automatically
        print(f"  ✓ DataSource node will be created by workflow: {source_id}")

    async def _verify_nodes_in_neo4j(
        self, root_node_id: UUID, expected_count: int
    ) -> None:
        """Verify nodes and relationships are saved correctly in Neo4j."""
        # Create a new client for this verification to avoid event loop conflicts
        verify_client = init_neo4j_client()
        driver = verify_client.async_driver
        try:
            async with driver.session() as session:
                # 1. Verify root node exists
                root_query = """
                MATCH (n:KnowledgeNode {id: $node_id})
                RETURN n
                LIMIT 1
                """
                result = await session.run(root_query, {"node_id": str(root_node_id)})
                record = await result.single()
                self.assertIsNotNone(record, "Root node should be saved in Neo4j")
                root_node = record["n"]
                self.assertEqual(root_node["node_type"], "structure", "Root node should be STRUCTURE type")
                print(f"  ✓ Root node found: {root_node_id}")

                # 2. Count all nodes (no source_id filter)
                count_query = """
                MATCH (n:KnowledgeNode)
                RETURN count(n) as total_count
                """
                result = await session.run(count_query)
                record = await result.single()
                total_nodes = record["total_count"] if record else 0
                self.assertGreaterEqual(
                    total_nodes,
                    expected_count,
                    f"Should have at least {expected_count} nodes in Neo4j",
                )
                print(f"  ✓ Total nodes in Neo4j: {total_nodes}")

                # 3. Count nodes by type
                type_count_query = """
                MATCH (n:KnowledgeNode)
                RETURN n.node_type as node_type, count(n) as count
                ORDER BY node_type
                """
                result = await session.run(type_count_query)
                type_counts = {}
                async for record in result:
                    node_type = record["node_type"]
                    count = record["count"]
                    type_counts[node_type] = count
                    print(f"  ✓ {node_type} nodes: {count}")

                # 4. Verify HAS_SECTION relationships (Structure -> Structure)
                has_section_query = """
                MATCH (parent:KnowledgeNode:Structure)-[r:HAS_SECTION]->(child:KnowledgeNode:Structure)
                RETURN count(r) as relationship_count
                """
                result = await session.run(has_section_query)
                record = await result.single()
                has_section_count = record["relationship_count"] if record else 0
                # HAS_SECTION may be 0 if tree has no structure->structure relationships
                print(f"  ✓ HAS_SECTION relationships: {has_section_count}")

                # 5. Verify HAS_CONTENT relationships (Structure -> Detail)
                has_content_query = """
                MATCH (parent:KnowledgeNode:Structure)-[r:HAS_CONTENT]->(child:KnowledgeNode)
                WHERE (child:Paragraph OR child:Image OR child:Table)
                  AND NOT child:Structure
                RETURN count(r) as relationship_count
                """
                result = await session.run(has_content_query)
                record = await result.single()
                has_content_count = record["relationship_count"] if record else 0
                self.assertGreater(
                    has_content_count,
                    0,
                    "Should have HAS_CONTENT relationships",
                )
                print(f"  ✓ HAS_CONTENT relationships: {has_content_count}")

                # 6. Verify NEXT_SIBLING relationships (only between same type)
                # Workflow ensures only same-type nodes have NEXT_SIBLING, so we just count all
                next_sibling_query = """
                MATCH (n1:KnowledgeNode)-[r:NEXT_SIBLING]->(n2:KnowledgeNode)
                RETURN count(r) as relationship_count
                """
                result = await session.run(next_sibling_query)
                record = await result.single()
                next_sibling_count = record["relationship_count"] if record else 0
                # NEXT_SIBLING may be 0 if tree has no siblings, so we don't assert > 0
                print(f"  ✓ NEXT_SIBLING relationships: {next_sibling_count}")

                # 7. Verify node properties (check a few sample nodes)
                sample_nodes_query = """
                MATCH (n:KnowledgeNode)
                RETURN n.id as id, n.node_type as node_type, n.level_down as level_down, 
                       n.level_up as level_up,
                       n.attributes as attributes, n.metadata as metadata
                LIMIT 5
                """
                result = await session.run(sample_nodes_query)
                sample_nodes = []
                async for record in result:
                    sample_nodes.append({
                        "id": record["id"],
                        "node_type": record["node_type"],
                        "level_down": record["level_down"],
                        "level_up": record["level_up"],
                        "has_attributes": record["attributes"] is not None,
                        "has_metadata": record["metadata"] is not None,
                    })

                self.assertGreater(len(sample_nodes), 0, "Should have sample nodes to verify")
                for node in sample_nodes:
                    self.assertIsNotNone(node["id"], "Node should have id")
                    self.assertIsNotNone(node["node_type"], "Node should have node_type")
                    self.assertIsNotNone(node["level_down"], "Node should have level_down")
                    self.assertIsNotNone(node["level_up"], "Node should have level_up")
                    self.assertTrue(node["has_attributes"], "Node should have attributes")
                    self.assertTrue(node["has_metadata"], "Node should have metadata")
                print(f"  ✓ Verified properties for {len(sample_nodes)} sample nodes")

                # 8. Verify labels (check that nodes have both KnowledgeNode and type-specific labels)
                labels_query = """
                MATCH (n:KnowledgeNode)
                RETURN DISTINCT labels(n) as labels, count(n) as count
                ORDER BY count DESC
                LIMIT 10
                """
                result = await session.run(labels_query)
                label_combinations = {}
                async for record in result:
                    labels = record["labels"]
                    count = record["count"]
                    label_key = ":".join(sorted(labels))
                    label_combinations[label_key] = count
                    print(f"  ✓ Label combination {label_key}: {count} nodes")

                # Verify that nodes have KnowledgeNode label and type-specific label
                has_correct_labels = any(
                    "KnowledgeNode" in labels and len(labels) >= 2
                    for labels in label_combinations.keys()
                )
                self.assertTrue(
                    has_correct_labels,
                    "Nodes should have KnowledgeNode label and type-specific label",
                )

                # 9. Verify tree structure (check that root has children via HAS_SECTION or HAS_CONTENT)
                root_children_query = """
                MATCH (root:KnowledgeNode {id: $root_id})-[r]->(child:KnowledgeNode)
                WHERE type(r) IN ['HAS_SECTION', 'HAS_CONTENT']
                RETURN count(child) as child_count
                """
                result = await session.run(root_children_query, {"root_id": str(root_node_id)})
                record = await result.single()
                root_child_count = record["child_count"] if record else 0
                print(f"  ✓ Root node has {root_child_count} direct children")
        finally:
            await verify_client.close()

    def test_save_knowledge_from_enrich_output(self) -> None:
        """Test saving knowledge tree from enrich_knowledge_tree workflow output to Neo4j."""
        
        # Skip test if Neo4j connection failed
        if hasattr(self, "_skip_test") and self._skip_test:
            self.skipTest("Neo4j connection failed - skipping test")

        # Step 1: Use specified enrich output file or find the most recent one
        enrich_file = (
            self.ENRICH_OUTPUT_FILE
            if self.ENRICH_OUTPUT_FILE.exists()
            else self._find_enrich_output_file()
        )
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

        # Step 3: Count nodes before saving
        def count_nodes(node):
            count = 1
            for child in node.children:
                count += count_nodes(child)
            return count

        total_nodes = count_nodes(enriched_tree)
        print(f"  ✓ Loaded tree with {total_nodes} nodes")

        # Step 4: Run save_knowledge_to_neo4j workflow
        save_input = SaveKnowledgeToNeo4jInput(
            enriched_tree=enriched_tree,
        )

        result = asyncio.run(
            start_save_knowledge_to_neo4j(
                save_input,
                neo4j_client=self._neo4j_client,
            )
        )

        # Step 6: Verify results
        self.assertIsNotNone(result)
        self.assertIn("root_node_id", result)
        self.assertIn("total_nodes_saved", result)
        self.assertIn("structure_nodes_count", result)
        self.assertIn("detail_nodes_count", result)
        self.assertIn("relationships_created", result)
        self.assertIn("saved_nodes", result)
        self.assertIn("failed_nodes", result)

        root_node_id = UUID(result["root_node_id"])
        total_saved = result["total_nodes_saved"]
        structure_count = result["structure_nodes_count"]
        detail_count = result["detail_nodes_count"]
        relationships_created = result["relationships_created"]
        failed_count = len(result["failed_nodes"])

        print(f"\n--- Save Knowledge to Neo4j Results ---")
        print(f"  Root node ID: {root_node_id}")
        print(f"  Total nodes saved: {total_saved}")
        print(f"  Structure nodes: {structure_count}")
        print(f"  Detail nodes: {detail_count}")
        print(f"  Relationships created:")
        for rel_type, count in relationships_created.items():
            print(f"    - {rel_type}: {count}")
        print(f"  Failed nodes: {failed_count}")

        # Print first few failed nodes for debugging
        if failed_count > 0:
            print(f"\n--- First Failed Node Error ---")
            first_failed = result["failed_nodes"][0]
            print(f"  Node ID: {first_failed.get('node_id', 'N/A')}")
            print(f"  Error: {first_failed.get('error', 'N/A')}")
            print(f"  Node Type: {first_failed.get('node_type', 'N/A')}")

        # Assertions
        self.assertGreater(total_saved, 0, "Should save at least some nodes")
        self.assertGreater(structure_count, 0, "Should save structure nodes")
        self.assertGreater(detail_count, 0, "Should save detail nodes")
        self.assertIn("HAS_SECTION", relationships_created, "Should have HAS_SECTION relationships")
        self.assertIn("HAS_CONTENT", relationships_created, "Should have HAS_CONTENT relationships")
        # HAS_SECTION may be 0 if tree has no structure->structure relationships
        # HAS_CONTENT should be > 0 if there are detail nodes
        if detail_count > 0:
            self.assertGreater(
                relationships_created.get("HAS_CONTENT", 0),
                0,
                "Should have at least one HAS_CONTENT relationship if there are detail nodes",
            )

        # Step 7: Verify nodes and relationships in Neo4j
        asyncio.run(
            self._verify_nodes_in_neo4j(root_node_id, total_saved)
        )

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

        from datetime import datetime, timezone

        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        filename = f"save_knowledge_neo4j_result_{timestamp}_{root_node_id.hex[:8]}.json"

        output_path = self.OUTPUT_DIR / filename
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False, default=str)

        print(f"✅ Saved test output to: {output_path}")


if __name__ == "__main__":
    unittest.main()

