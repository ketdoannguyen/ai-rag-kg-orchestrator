"""Subgraph Extractor for Benchmark v4.0 — SEED + ENRICH + BOUND.

v4.0 key change: Relation-Coverage Subgraph Assembly to capture ≥80% Tier 2
relations (including cross-L5 which v3.2 missed entirely).
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field

from workflows.benchmark.models import (
    TIER2_TYPES,
    Subgraph,
    SubgraphNode,
    SubgraphRelation,
    SubgraphStructureNode,
)
from workflows.benchmark.query_allocator import classify_tier

MAX_SUBGRAPH_SIZE = 20


@dataclass
class _EnrichedNode:
    """Candidate cross-L5 node with connection metadata."""

    node_id: str
    text: str
    embedding: list[float] | None
    parent_section_key: str | None
    parent_section_title: str | None = None
    parent_section_level: int | None = None
    connection_count: int = 0
    relation_types: set[str] = field(default_factory=set)


class SubgraphExtractor:
    """Extract v4.0 subgraphs via SEED → ENRICH → BOUND pipeline.

    Given an L5 anchor section, assembles a subgraph that covers both
    same-L5 (core) nodes and cross-L5 1-hop neighbors (enriched nodes).
    """

    def __init__(self, client, max_size: int = MAX_SUBGRAPH_SIZE) -> None:
        self._client = client
        self._max_size = max_size

    # ------------------------------------------------------------------ #
    # Public API                                                           #
    # ------------------------------------------------------------------ #

    def extract(
        self,
        anchor_key: str,
        anchor_title: str,
        anchor_summary: str | None,
        breadcrumb: str,
    ) -> Subgraph | None:
        """Extract subgraph for the given L5 anchor section.

        Returns None if the anchor has no content nodes.
        """
        anchor_id = f"structure_nodes/{anchor_key}"

        # SEED: same-L5 content nodes (with actual parent info)
        core_nodes = self._seed(anchor_id)
        core_node_ids = [n["node_id"] for n in core_nodes]

        if not core_node_ids:
            return None

        # STRUCTURE: collect L6 children of anchor
        l6_nodes = self._collect_structure_nodes(anchor_id)

        # ENRICH: 1-hop cross-L5 neighbors
        enriched_candidates = self._enrich(core_node_ids, anchor_key)

        # BOUND: cap total at max_size
        enriched_nodes = self._bound(core_nodes, enriched_candidates)
        enriched_node_ids = [n.node_id for n in enriched_nodes]

        all_node_ids = core_node_ids + enriched_node_ids

        # Collect internal Tier 2 relations for bounded subgraph
        relations = self._collect_relations(all_node_ids)

        # Build structure_nodes list: anchor L5 + L6 children + enriched parents
        structure_nodes: list[SubgraphStructureNode] = [
            SubgraphStructureNode(
                section_key=anchor_key,
                title=anchor_title,
                summary=anchor_summary,
                level=5,
                parent_key=None,
            ),
        ]
        for l6 in l6_nodes:
            structure_nodes.append(
                SubgraphStructureNode(
                    section_key=l6["key"],
                    title=l6.get("title"),
                    summary=l6.get("summary"),
                    level=6,
                    parent_key=anchor_key,
                )
            )

        # Collect unique enriched parent sections as structure nodes
        seen_keys = {sn.section_key for sn in structure_nodes}
        for en in enriched_nodes:
            pkey = en.parent_section_key
            if pkey and pkey not in seen_keys:
                seen_keys.add(pkey)
                structure_nodes.append(
                    SubgraphStructureNode(
                        section_key=pkey,
                        title=en.parent_section_title,
                        summary=None,
                        level=en.parent_section_level or 5,
                        parent_key=None,
                    )
                )

        # Build SubgraphNode objects
        sg_nodes = [
            SubgraphNode(
                node_id=n["node_id"],
                text=n["text"],
                embedding=n.get("embedding"),
                parent_section_key=n.get("parent_section_key"),
                parent_section_title=n.get("parent_section_title"),
                parent_section_level=n.get("parent_section_level"),
                breadcrumb=breadcrumb,
            )
            for n in core_nodes
        ] + [
            SubgraphNode(
                node_id=n.node_id,
                text=n.text,
                embedding=n.embedding,
                parent_section_key=n.parent_section_key,
                parent_section_title=n.parent_section_title,
                parent_section_level=n.parent_section_level,
            )
            for n in enriched_nodes
        ]

        content_count = len(core_node_ids)
        tier = classify_tier(content_count)

        return Subgraph(
            subgraph_id=str(uuid.uuid4()),
            anchor_key=anchor_key,
            anchor_title=anchor_title,
            anchor_summary=anchor_summary,
            breadcrumb=breadcrumb,
            nodes=sg_nodes,
            core_node_ids=core_node_ids,
            enriched_node_ids=enriched_node_ids,
            relations=relations,
            structure_nodes=structure_nodes,
            content_count=content_count,
            relation_count=len(relations),
            tier=tier,
            query_allocation=0,  # Set by allocator
        )

    def compute_relation_coverage(self, subgraph: Subgraph) -> float:
        """Compute what fraction of all Tier 2 relations involving subgraph nodes
        are captured within the subgraph itself.

        coverage = internal_relations / total_related_relations
        """
        if not subgraph.nodes:
            return 0.0

        all_node_ids = [n.node_id for n in subgraph.nodes]

        # Count total Tier 2 edges involving any subgraph node
        query = """
        FOR e IN knowledge_edges
          FILTER e.type IN @tier2_types
          FILTER e._from IN @node_ids OR e._to IN @node_ids
          RETURN 1
        """
        total_result = self._client.execute_aql(
            query,
            bind_vars={"tier2_types": TIER2_TYPES, "node_ids": all_node_ids},
        )
        total = len(total_result)
        internal = len(subgraph.relations)

        if total == 0:
            return 0.0
        return internal / total

    # ------------------------------------------------------------------ #
    # Private pipeline steps                                               #
    # ------------------------------------------------------------------ #

    def _seed(self, anchor_id: str) -> list[dict]:
        """SEED: Extract all content nodes belonging to this L5 section."""
        query = """
        FOR v, e IN 1..2 OUTBOUND DOCUMENT(@anchor_id) knowledge_edges
          FILTER e.type IN ["HAS_SECTION", "HAS_CONTENT"]
          FILTER IS_SAME_COLLECTION("content_nodes", v)
          LET direct_parent = FIRST(
            FOR pp, pe IN 1..1 INBOUND v knowledge_edges
              FILTER pe.type == "HAS_CONTENT"
              FILTER IS_SAME_COLLECTION("structure_nodes", pp)
              RETURN {
                key: pp._key,
                title: pp.attributes.title,
                summary: pp.summary,
                level: pp.level
              }
          )
          RETURN {
            node_id: v._id,
            text: v.text,
            embedding: v.embedding,
            parent_section_key: direct_parent.key,
            parent_section_title: direct_parent.title,
            parent_section_level: direct_parent.level
          }
        """
        return self._client.execute_aql(query, bind_vars={"anchor_id": anchor_id})

    def _collect_structure_nodes(self, anchor_id: str) -> list[dict]:
        """Collect L6 sub-section children of the anchor L5 section."""
        query = """
        FOR v, e IN 1..1 OUTBOUND DOCUMENT(@anchor_id) knowledge_edges
          FILTER e.type == "HAS_SECTION"
          FILTER IS_SAME_COLLECTION("structure_nodes", v)
          FILTER v.level == 6
          RETURN {
            key: v._key,
            title: v.attributes.title,
            summary: v.summary,
            level: v.level
          }
        """
        return self._client.execute_aql(query, bind_vars={"anchor_id": anchor_id})

    def _enrich(self, core_node_ids: list[str], anchor_key: str) -> list[_EnrichedNode]:
        """ENRICH: Find 1-hop cross-L5 neighbors via Tier 2 relations."""
        if not core_node_ids:
            return []

        query = """
        FOR core_id IN @core_node_ids
          FOR v, e IN 1..1 ANY DOCUMENT(core_id) knowledge_edges
            FILTER e.type IN @tier2_types
            FILTER v._id NOT IN @core_node_ids
            FILTER IS_SAME_COLLECTION("content_nodes", v)
            LET v_parent_doc = FIRST(
              FOR p, pe IN 1..1 INBOUND v knowledge_edges
                FILTER pe.type == "HAS_CONTENT"
                FILTER IS_SAME_COLLECTION("structure_nodes", p)
                RETURN p
            )
            LET v_parent = v_parent_doc._key
            FILTER v_parent != @anchor_key
            RETURN {
              node_id: v._id,
              text: v.text,
              embedding: v.embedding,
              parent_section_key: v_parent,
              parent_section_title: v_parent_doc.attributes.title,
              parent_section_level: v_parent_doc.level,
              relation_type: e.type,
              connected_to: core_id
            }
        """
        rows = self._client.execute_aql(
            query,
            bind_vars={
                "core_node_ids": core_node_ids,
                "tier2_types": TIER2_TYPES,
                "anchor_key": anchor_key,
            },
        )

        # Aggregate by node_id: count connections and collect relation types
        seen: dict[str, _EnrichedNode] = {}
        for row in rows:
            nid = row["node_id"]
            if nid not in seen:
                seen[nid] = _EnrichedNode(
                    node_id=nid,
                    text=row["text"],
                    embedding=row.get("embedding"),
                    parent_section_key=row.get("parent_section_key"),
                    parent_section_title=row.get("parent_section_title"),
                    parent_section_level=row.get("parent_section_level"),
                )
            seen[nid].connection_count += 1
            seen[nid].relation_types.add(row["relation_type"])

        return list(seen.values())

    def _bound(
        self,
        core_nodes: list[dict],
        enriched_candidates: list[_EnrichedNode],
    ) -> list[_EnrichedNode]:
        """BOUND: Cap total subgraph at max_size nodes.

        When enriched candidates exceed available slots, rank by:
        1. connection_count (desc) — more connections to core = higher priority
        2. len(relation_types) (desc) — more relation type diversity = higher priority
        """
        available = self._max_size - len(core_nodes)
        if available <= 0:
            return []
        if len(enriched_candidates) <= available:
            return enriched_candidates

        sorted_candidates = sorted(
            enriched_candidates,
            key=lambda n: (n.connection_count, len(n.relation_types)),
            reverse=True,
        )
        return sorted_candidates[:available]

    def _collect_relations(self, all_node_ids: list[str]) -> list[SubgraphRelation]:
        """Collect all Tier 2 relations between nodes in the bounded subgraph."""
        if len(all_node_ids) < 2:
            return []

        query = """
        FOR e IN knowledge_edges
          FILTER e.type IN @tier2_types
          FILTER e._from IN @node_ids AND e._to IN @node_ids
          FILTER e._from != e._to
          RETURN DISTINCT {type: e.type, from: e._from, to: e._to}
        """
        rows = self._client.execute_aql(
            query,
            bind_vars={"tier2_types": TIER2_TYPES, "node_ids": all_node_ids},
        )
        return [
            SubgraphRelation(
                from_id=r["from"],
                to_id=r["to"],
                relation_type=r["type"],
            )
            for r in rows
        ]
