"""Cypher query templates for KnowledgeNode operations."""

from __future__ import annotations

# Relationship type constants for extensibility
RELATIONSHIP_TYPES = {
    "HAS_SECTION": "HAS_SECTION",  # Structure -> Structure (has mục)
    "HAS_CONTENT": "HAS_CONTENT",  # Structure -> Detail (paragraph/image/table)
    "NEXT_SIBLING": "NEXT_SIBLING",  # Same type nodes only
    "RELATED_TO": "RELATED_TO",  # For future semantic relationships
}

# Node label constants
NODE_LABELS = {
    "BASE": "KnowledgeNode",
    "ROOT": "Root",
    "STRUCTURE": "Structure",
    "PARAGRAPH": "Paragraph",
    "IMAGE": "Image",
    "TABLE": "Table",
}

# MERGE node query - labels are set dynamically based on node_type
# Note: Neo4j requires labels in MERGE clause, so caller constructs full label list
MERGE_NODE_QUERY = """
MERGE (n:KnowledgeNode {id: $id})
SET n.node_type = $node_type,
    n.attributes = $attributes,
    n.metadata = $metadata,
    n.level_down = $level_down,
    n.level_up = $level_up
RETURN n
"""

# Create HAS_SECTION relationship (Root/Structure -> Structure)
CREATE_HAS_SECTION_QUERY = """
MATCH (parent:KnowledgeNode {id: $parent_id})
WHERE parent:Structure OR parent:Root
MATCH (child:KnowledgeNode:Structure {id: $child_id})
MERGE (parent)-[r:HAS_SECTION]->(child)
RETURN r
"""

# Create HAS_CONTENT relationship (Root/Structure -> Detail nodes)
CREATE_HAS_CONTENT_QUERY = """
MATCH (parent:KnowledgeNode {id: $parent_id})
WHERE parent:Structure OR parent:Root
MATCH (child:KnowledgeNode {id: $child_id})
WHERE (child:Paragraph OR child:Image OR child:Table)
  AND NOT child:Structure AND NOT child:Root
MERGE (parent)-[r:HAS_CONTENT]->(child)
RETURN r
"""

# Create NEXT_SIBLING relationship (only between same type nodes)
# Note: Workflow ensures same type before calling this, so we don't need WHERE clause
CREATE_NEXT_SIBLING_QUERY = """
MATCH (sibling1:KnowledgeNode {id: $sibling1_id})
MATCH (sibling2:KnowledgeNode {id: $sibling2_id})
MERGE (sibling1)-[r:NEXT_SIBLING]->(sibling2)
RETURN r
"""

# Get node by ID
GET_NODE_BY_ID_QUERY = """
MATCH (n:KnowledgeNode {id: $node_id})
RETURN n
LIMIT 1
"""

# Delete subtree (for cleanup, optional)
DELETE_SUBTREE_QUERY = """
MATCH (root:KnowledgeNode {id: $node_id})
OPTIONAL MATCH (root)-[*]->(descendant:KnowledgeNode)
DETACH DELETE root, descendant
"""

# Get node with relationships
GET_NODE_WITH_RELATIONSHIPS_QUERY = """
MATCH (n:KnowledgeNode {id: $node_id})
OPTIONAL MATCH (n)-[r]->(related:KnowledgeNode)
RETURN n, collect({rel: type(r), node: related}) AS relationships
LIMIT 1
"""

