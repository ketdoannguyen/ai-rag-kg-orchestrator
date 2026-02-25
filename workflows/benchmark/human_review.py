"""Human Review Pipeline Utilities for Benchmark v4.0 (Task 22).

Export queries for external review (CSV/JSON), import review labels,
and apply corrections to the dataset.
"""

from __future__ import annotations

import csv
import json
import logging
from dataclasses import dataclass
from pathlib import Path

from workflows.benchmark.models import BenchmarkQuery

logger = logging.getLogger(__name__)


@dataclass
class ReviewLabel:
    """A single human review decision."""

    query_id: str
    decision: str  # approved, rejected, fix
    notes: str = ""
    grade_corrections: dict[str, int] | None = None  # node_id → corrected grade


@dataclass
class ReviewSummary:
    """Summary of review label import."""

    total_labels: int
    approved: int
    rejected: int
    fixed: int
    not_found: int


def export_for_review_csv(
    queries: list[BenchmarkQuery],
    output_path: str,
) -> str:
    """Export queries as CSV for Google Sheets review.

    Columns: query_id, query_text, subgraph_id, fact_nodes_summary,
    confidence_overall, direction, decision (empty), notes (empty)
    """
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "query_id",
                "query_text",
                "subgraph_id",
                "fact_nodes",
                "confidence_overall",
                "direction",
                "decision",
                "notes",
            ]
        )
        for q in queries:
            fact_summary = "; ".join(f"{fn.node_id}(g={fn.grade})" for fn in q.fact_nodes)
            writer.writerow(
                [
                    q.query_id,
                    q.query_text,
                    q.subgraph_id,
                    fact_summary,
                    q.confidence.overall,
                    q.direction or "",
                    "",  # decision — to be filled by reviewer
                    "",  # notes — to be filled by reviewer
                ]
            )

    logger.info("Exported %d queries for review to %s", len(queries), path)
    return str(path)


def export_for_review_json(
    queries: list[BenchmarkQuery],
    output_path: str,
) -> str:
    """Export queries as JSON for review."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    data = [
        {
            "query_id": q.query_id,
            "query_text": q.query_text,
            "subgraph_id": q.subgraph_id,
            "fact_nodes": [
                {"node_id": fn.node_id, "grade": fn.grade, "parent_section": fn.parent_section}
                for fn in q.fact_nodes
            ],
            "confidence_overall": q.confidence.overall,
            "direction": q.direction,
            "decision": None,
            "notes": "",
        }
        for q in queries
    ]

    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    logger.info("Exported %d queries for review to %s", len(queries), path)
    return str(path)


def import_review_labels_csv(csv_path: str) -> list[ReviewLabel]:
    """Import review labels from CSV.

    Expected columns: query_id, decision, notes (optional), grade_corrections (optional JSON)
    """
    labels = []
    with open(csv_path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            decision = row.get("decision", "").strip().lower()
            if not decision:
                continue

            grade_corrections = None
            raw_corrections = row.get("grade_corrections", "").strip()
            if raw_corrections:
                try:
                    grade_corrections = json.loads(raw_corrections)
                except json.JSONDecodeError:
                    logger.warning(
                        "Invalid grade_corrections JSON for %s: %s",
                        row.get("query_id"),
                        raw_corrections,
                    )

            labels.append(
                ReviewLabel(
                    query_id=row["query_id"],
                    decision=decision,
                    notes=row.get("notes", ""),
                    grade_corrections=grade_corrections,
                )
            )

    logger.info("Imported %d review labels from %s", len(labels), csv_path)
    return labels


def import_review_labels_json(json_path: str) -> list[ReviewLabel]:
    """Import review labels from JSON."""
    with open(json_path, encoding="utf-8") as f:
        data = json.load(f)

    labels = []
    for item in data:
        decision = (item.get("decision") or "").strip().lower()
        if not decision:
            continue
        labels.append(
            ReviewLabel(
                query_id=item["query_id"],
                decision=decision,
                notes=item.get("notes", ""),
                grade_corrections=item.get("grade_corrections"),
            )
        )

    logger.info("Imported %d review labels from %s", len(labels), json_path)
    return labels


def apply_review_labels(
    queries: list[BenchmarkQuery],
    labels: list[ReviewLabel],
) -> tuple[list[BenchmarkQuery], ReviewSummary]:
    """Apply review labels to query set.

    - approved: Mark as approved
    - rejected: Remove from dataset
    - fix: Apply grade corrections, mark as approved

    Returns:
        (updated_queries, summary)
    """
    label_map = {label.query_id: label for label in labels}
    result: list[BenchmarkQuery] = []
    approved = 0
    rejected = 0
    fixed = 0
    not_found = 0

    for query in queries:
        label = label_map.get(query.query_id)

        if label is None:
            # No label — keep as-is (not reviewed)
            result.append(query)
            continue

        if label.decision == "rejected":
            rejected += 1
            continue  # Remove from dataset

        if label.decision not in ("approved", "fix"):
            logger.warning(
                "Unknown review decision '%s' for query %s, skipping",
                label.decision,
                query.query_id,
            )
            result.append(query)
            continue

        if label.decision == "fix" and label.grade_corrections:
            if not isinstance(label.grade_corrections, dict):
                logger.warning(
                    "Invalid grade_corrections type for %s, skipping corrections",
                    query.query_id,
                )
                query = query.model_copy(update={"human_review_status": "approved"})
                approved += 1
                result.append(query)
                continue

            # Apply grade corrections
            corrected_nodes = []
            for fn in query.fact_nodes:
                new_grade = label.grade_corrections.get(fn.node_id)
                if new_grade is not None:
                    if new_grade not in (0, 1, 2):
                        logger.warning(
                            "Invalid grade %s for node %s in query %s, skipping",
                            new_grade,
                            fn.node_id,
                            query.query_id,
                        )
                        corrected_nodes.append(fn)
                    else:
                        corrected_nodes.append(fn.model_copy(update={"grade": new_grade}))
                else:
                    corrected_nodes.append(fn)
            query = query.model_copy(
                update={
                    "fact_nodes": corrected_nodes,
                    "human_review_status": "approved",
                }
            )
            fixed += 1
        else:
            # approved or fix without corrections
            query = query.model_copy(update={"human_review_status": "approved"})
            if label.decision == "fix":
                logger.warning(
                    "Query %s has decision='fix' but no grade_corrections, treating as approved",
                    query.query_id,
                )
            approved += 1

        result.append(query)

    not_found = sum(1 for label in labels if label.query_id not in {q.query_id for q in queries})

    summary = ReviewSummary(
        total_labels=len(labels),
        approved=approved,
        rejected=rejected,
        fixed=fixed,
        not_found=not_found,
    )

    logger.info(
        "Applied labels: %d approved, %d rejected, %d fixed, %d not found",
        approved,
        rejected,
        fixed,
        not_found,
    )

    return result, summary
