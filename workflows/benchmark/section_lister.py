"""Candidate Section Lister for Benchmark v4.0 (Task 9).

Lists all L5 sections from ArangoDB with content count, relation count,
summary availability, and source/chapter metadata for subgraph selection.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from workflows.benchmark.models import TIER2_TYPES


@dataclass
class CandidateSection:
    """An L5 section candidate for subgraph extraction."""

    section_id: str
    section_key: str
    title: str
    level: int
    content_count: int
    relation_count: int
    has_summary: bool
    summary: str | None
    parent_id: str | None
    parent_title: str | None
    source: str | None
    has_subsections: bool = False
    subsection_count: int = 0

    def to_dict(self) -> dict:
        return {
            "section_id": self.section_id,
            "section_key": self.section_key,
            "title": self.title,
            "level": self.level,
            "content_count": self.content_count,
            "relation_count": self.relation_count,
            "has_summary": self.has_summary,
            "summary": self.summary,
            "parent_id": self.parent_id,
            "parent_title": self.parent_title,
            "source": self.source,
            "has_subsections": self.has_subsections,
            "subsection_count": self.subsection_count,
        }


def list_candidate_sections(client) -> list[CandidateSection]:
    """Query ArangoDB for all L5 sections that have content nodes.

    Returns sections sorted by relation_count descending (richest first).
    """
    query = """
    FOR s IN structure_nodes
      FILTER s.level == 5
      LET content_children = (
        FOR v, e IN 1..2 OUTBOUND s knowledge_edges
          FILTER e.type IN ["HAS_SECTION", "HAS_CONTENT"]
          FILTER IS_SAME_COLLECTION("content_nodes", v)
          RETURN v._id
      )
      FILTER LENGTH(content_children) >= 1

      LET l6_sections = (
        FOR v, e IN 1..1 OUTBOUND s knowledge_edges
          FILTER e.type == "HAS_SECTION"
          FILTER IS_SAME_COLLECTION("structure_nodes", v)
          FILTER v.level == 6
          RETURN v._key
      )

      LET tier2_relations = (
        FOR c_id IN content_children
          FOR v, e IN 1..1 ANY DOCUMENT(c_id) knowledge_edges
            FILTER e.type IN @tier2_types
            RETURN DISTINCT e._key
      )

      LET parent = FIRST(
        FOR p, pe IN 1..1 INBOUND s knowledge_edges
          FILTER pe.type == "HAS_SECTION"
          FILTER IS_SAME_COLLECTION("structure_nodes", p)
          RETURN p
      )

      SORT LENGTH(tier2_relations) DESC
      RETURN {
        section_id: s._id,
        section_key: s._key,
        title: s.attributes.title,
        level: s.level,
        content_count: LENGTH(content_children),
        relation_count: LENGTH(tier2_relations),
        has_summary: s.summary != null AND s.summary != "",
        summary: s.summary,
        parent_id: parent != null ? parent._id : null,
        parent_title: parent != null ? parent.attributes.title : null,
        source: s.source_file,
        has_subsections: LENGTH(l6_sections) > 0,
        subsection_count: LENGTH(l6_sections)
      }
    """
    rows = client.execute_aql(query, bind_vars={"tier2_types": TIER2_TYPES})
    return [
        CandidateSection(
            section_id=r["section_id"],
            section_key=r["section_key"],
            title=r["title"],
            level=r["level"],
            content_count=r["content_count"],
            relation_count=r["relation_count"],
            has_summary=r["has_summary"],
            summary=r.get("summary"),
            parent_id=r.get("parent_id"),
            parent_title=r.get("parent_title"),
            source=r.get("source"),
            has_subsections=r.get("has_subsections", False),
            subsection_count=r.get("subsection_count", 0),
        )
        for r in rows
    ]


def export_candidates_json(
    candidates: list[CandidateSection],
    output_dir: str = "data/benchmark",
) -> str:
    """Export candidate sections to JSON for review.

    Returns path to the output file.
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    out_path = str(Path(output_dir) / "candidate_sections.json")

    data = {
        "total_candidates": len(candidates),
        "summary": {
            "with_summary": sum(1 for c in candidates if c.has_summary),
            "without_summary": sum(1 for c in candidates if not c.has_summary),
            "total_content_nodes": sum(c.content_count for c in candidates),
            "total_relations": sum(c.relation_count for c in candidates),
            "sources": list({c.source for c in candidates if c.source}),
        },
        "candidates": [c.to_dict() for c in candidates],
    }

    with open(out_path, "w") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    return out_path
