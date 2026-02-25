"""Full Benchmark Pipeline: Extract → Synthetic Generate → Evaluate → Report.

Runs the complete benchmark pipeline end-to-end:
  1. List L5 candidate sections from ArangoDB
  2. Select & extract subgraphs (SEED+ENRICH+BOUND)
  3. Allocate queries proportionally
  4. Generate synthetic queries from subgraph data (no LLM needed)
  5. Classify directions (rule-based)
  6. Run QA checks
  7. Inject distractors
  8. Simulate retrieval results for 14 methods
  9. Evaluate with IR metrics (NDCG@10, Recall@10, Precision@10, MRR)
  10. Statistical tests + hypothesis validation
  11. Generate JSON + MD reports

Usage: uv run python scripts/benchmark_full_evaluation.py
"""

from __future__ import annotations

import asyncio
import json
import logging
import random
import sys
import uuid
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(name)s | %(message)s")
logger = logging.getLogger("benchmark_full")

# --- Configuration ---
TARGET_QUERIES = 100
MAX_SUBGRAPHS = 30
DISTRACTOR_COUNT = 2
OUTPUT_DIR = Path("data/benchmark/full_evaluation")
K = 10


def step_1_connect():
    """Connect to ArangoDB."""
    from workflows.dependencies import init_arango_client

    logger.info("=== Step 1: Connect to ArangoDB ===")
    client = init_arango_client()
    for coll in ["structure_nodes", "content_nodes", "knowledge_edges"]:
        count = client.db.collection(coll).count()
        logger.info("  %s: %d docs", coll, count)
    return client


def step_2_list_sections(client):
    """List candidate L5 sections."""
    from workflows.benchmark.section_lister import (
        export_candidates_json,
        list_candidate_sections,
    )

    logger.info("=== Step 2: List L5 candidate sections ===")
    candidates = list_candidate_sections(client)
    logger.info("  Total candidates: %d", len(candidates))

    out_dir = str(OUTPUT_DIR)
    export_candidates_json(candidates, out_dir)

    for c in candidates[:10]:
        logger.info(
            "  [%s] %s — %d content, %d relations",
            c.section_key[:15],
            (c.title or "?")[:40],
            c.content_count,
            c.relation_count,
        )
    return candidates


def step_3_extract_subgraphs(client, candidates):
    """Extract subgraphs with diversity selection."""
    from workflows.benchmark.subgraph_selector import select_and_extract
    from workflows.benchmark.query_allocator import balance_allocations

    logger.info("=== Step 3: Select & extract subgraphs (max %d) ===", MAX_SUBGRAPHS)
    subgraphs, report = select_and_extract(client, candidates, max_subgraphs=MAX_SUBGRAPHS)

    logger.info("  Selected: %d subgraphs", report.selected_count)
    logger.info("  Tier distribution: %s", report.tier_distribution)
    logger.info("  Sources: %s", report.sources)
    logger.info("  Avg relation coverage: %.1f%%", report.avg_relation_coverage * 100)

    # Re-balance allocations to hit TARGET_QUERIES
    subgraphs = balance_allocations(subgraphs, target_total=TARGET_QUERIES)
    total_alloc = sum(sg.query_allocation for sg in subgraphs)
    logger.info("  Re-balanced allocation: %d (target=%d)", total_alloc, TARGET_QUERIES)

    return subgraphs, report


def step_4_generate_synthetic_queries(subgraphs):
    """Generate synthetic benchmark queries from subgraph data.

    Creates realistic queries by analyzing actual content nodes and relations
    in each subgraph. No LLM call needed — uses template-based generation.
    """
    from workflows.benchmark.models import (
        BenchmarkQuery,
        ConfidenceScore,
        FactNode,
    )

    logger.info("=== Step 4: Generate synthetic queries ===")
    rng = random.Random(42)
    all_queries = []

    # Vietnamese history question templates
    FACTUAL_TEMPLATES = [
        "Trình bày nội dung chính về {topic}?",
        "Nêu những đặc điểm quan trọng của {topic}?",
        "Hãy mô tả {topic} trong bối cảnh lịch sử?",
        "{topic} diễn ra như thế nào?",
        "Những yếu tố nào liên quan đến {topic}?",
    ]
    RELATION_TEMPLATES = [
        "Mối quan hệ giữa {topic_a} và {topic_b} như thế nào?",
        "Phân tích ảnh hưởng của {topic_a} đến {topic_b}?",
        "{topic_a} dẫn đến những thay đổi gì trong {topic_b}?",
        "So sánh {topic_a} với {topic_b}?",
    ]
    MULTI_NODE_TEMPLATES = [
        "Tổng hợp các sự kiện liên quan đến {topic} trong giai đoạn này?",
        "Đánh giá vai trò của {topic} trong tiến trình lịch sử?",
        "Phân tích các khía cạnh khác nhau của {topic}?",
        "Nhận xét về {topic} qua các nguồn tài liệu?",
    ]

    for sg in subgraphs:
        target = sg.query_allocation
        if target <= 0:
            continue

        nodes = sg.nodes
        relations = sg.relations
        core_ids = set(sg.core_node_ids)
        generated_count = 0

        for _ in range(target):
            if generated_count >= target:
                break

            # Strategy selection based on subgraph properties
            strategy = rng.choices(
                ["factual_single", "factual_multi", "relation", "cross_section"],
                weights=[30, 25, 25, 20],
                k=1,
            )[0]

            try:
                if strategy == "factual_single" and nodes:
                    # Single-node factual query
                    node = rng.choice(nodes)
                    topic = _extract_topic(node.text)
                    template = rng.choice(FACTUAL_TEMPLATES)
                    query_text = template.format(topic=topic)
                    fact_nodes = [
                        FactNode(
                            node_id=node.node_id,
                            grade=2,
                            parent_section=node.parent_section_title or sg.anchor_title,
                        )
                    ]
                    relations_used = []

                elif strategy == "factual_multi" and len(nodes) >= 2:
                    # Multi-node factual query (same section)
                    section_nodes = [n for n in nodes if n.node_id in core_ids]
                    if len(section_nodes) < 2:
                        section_nodes = nodes
                    chosen = rng.sample(section_nodes, min(3, len(section_nodes)))
                    topic = _extract_topic(chosen[0].text)
                    template = rng.choice(MULTI_NODE_TEMPLATES)
                    query_text = template.format(topic=topic)
                    fact_nodes = [
                        FactNode(
                            node_id=n.node_id,
                            grade=2 if i == 0 else 1,
                            parent_section=n.parent_section_title or sg.anchor_title,
                        )
                        for i, n in enumerate(chosen)
                    ]
                    relations_used = []

                elif strategy == "relation" and relations:
                    # Relation-based query
                    rel = rng.choice(relations)
                    from_node = next((n for n in nodes if n.node_id == rel.from_id), None)
                    to_node = next((n for n in nodes if n.node_id == rel.to_id), None)
                    if from_node and to_node:
                        topic_a = _extract_topic(from_node.text)
                        topic_b = _extract_topic(to_node.text)
                        template = rng.choice(RELATION_TEMPLATES)
                        query_text = template.format(topic_a=topic_a, topic_b=topic_b)
                        fact_nodes = [
                            FactNode(
                                node_id=from_node.node_id,
                                grade=2,
                                parent_section=from_node.parent_section_title or sg.anchor_title,
                            ),
                            FactNode(
                                node_id=to_node.node_id,
                                grade=2,
                                parent_section=to_node.parent_section_title or sg.anchor_title,
                            ),
                        ]
                        relations_used = [rel.relation_type]
                    else:
                        continue

                elif strategy == "cross_section" and len(sg.enriched_node_ids) > 0:
                    # Cross-section query using enriched nodes
                    enriched = [n for n in nodes if n.node_id in set(sg.enriched_node_ids)]
                    core = [n for n in nodes if n.node_id in core_ids]
                    if enriched and core:
                        e_node = rng.choice(enriched)
                        c_node = rng.choice(core)
                        topic_a = _extract_topic(c_node.text)
                        topic_b = _extract_topic(e_node.text)
                        template = rng.choice(RELATION_TEMPLATES)
                        query_text = template.format(topic_a=topic_a, topic_b=topic_b)
                        fact_nodes = [
                            FactNode(
                                node_id=c_node.node_id,
                                grade=2,
                                parent_section=c_node.parent_section_title or sg.anchor_title,
                            ),
                            FactNode(
                                node_id=e_node.node_id,
                                grade=1,
                                parent_section=e_node.parent_section_title or "related",
                            ),
                        ]
                        relations_used = []
                    else:
                        continue
                else:
                    # Fallback: simple single-node query
                    node = rng.choice(nodes)
                    topic = _extract_topic(node.text)
                    template = rng.choice(FACTUAL_TEMPLATES)
                    query_text = template.format(topic=topic)
                    fact_nodes = [
                        FactNode(
                            node_id=node.node_id,
                            grade=2,
                            parent_section=node.parent_section_title or sg.anchor_title,
                        )
                    ]
                    relations_used = []

                # Build confidence based on strategy
                clarity = rng.randint(7, 10)
                completeness = rng.randint(6, 9)
                precision = rng.randint(7, 10)
                certainty = rng.randint(6, 9)
                overall = int((clarity + completeness + precision + certainty) / 4 * 10)

                query = BenchmarkQuery(
                    query_id=str(uuid.uuid4()),
                    query_text=query_text,
                    subgraph_id=sg.subgraph_id,
                    fact_nodes=fact_nodes,
                    relations_used=relations_used,
                    confidence=ConfidenceScore(
                        question_clarity=clarity,
                        fact_node_completeness=completeness,
                        fact_node_precision=precision,
                        grade_certainty=certainty,
                        overall=overall,
                    ),
                    confidence_concerns=""
                    if overall >= 70
                    else "Synthetic query, may lack nuance",
                    reasoning=f"Generated via {strategy} strategy from subgraph {sg.subgraph_id[:8]}",
                )
                all_queries.append(query)
                generated_count += 1

            except (IndexError, StopIteration, ValueError):
                continue

    logger.info(
        "  Generated %d synthetic queries from %d subgraphs", len(all_queries), len(subgraphs)
    )
    return all_queries


def _extract_topic(text: str, max_len: int = 50) -> str:
    """Extract a meaningful topic phrase from content node text."""
    # Take first sentence or first max_len chars
    text = text.strip()
    # Try to find first sentence
    for sep in [".", "。", "\n"]:
        idx = text.find(sep)
        if 10 < idx < 150:
            return text[:idx].strip()
    # Fall back to first chunk
    if len(text) > max_len:
        # Try to break at word boundary
        cut = text[:max_len].rfind(" ")
        if cut > 20:
            return text[:cut].strip()
        return text[:max_len].strip()
    return text


def step_5_classify_directions(queries):
    """Classify query directions."""
    from workflows.benchmark.direction_classifier import classify_batch

    logger.info("=== Step 5: Classify directions ===")
    updated, distribution = classify_batch(queries)
    logger.info("  Distribution: %s", distribution)
    return updated, distribution


def step_6_inject_distractors(queries, subgraphs):
    """Inject grade-0 distractor nodes."""
    from workflows.benchmark.runner import inject_distractors_batch

    logger.info("=== Step 6: Inject distractors ===")
    sg_map = {sg.subgraph_id: sg for sg in subgraphs}
    augmented = []
    for q in queries:
        sg = sg_map.get(q.subgraph_id)
        if sg:
            aug = inject_distractors_batch([q], sg, count=DISTRACTOR_COUNT, seed=42)
            augmented.extend(aug)
        else:
            augmented.append(q)

    g0_count = sum(1 for q in augmented for fn in q.fact_nodes if fn.grade == 0)
    logger.info("  Injected %d total grade-0 nodes across %d queries", g0_count, len(augmented))
    return augmented


def step_7_qa_checks(queries, client):
    """Run QA checks."""
    from workflows.benchmark.qa_checks import run_qa_checks

    logger.info("=== Step 7: QA checks ===")

    result = client.execute_aql("FOR c IN content_nodes RETURN c._id")
    valid_node_ids = set(result)

    clean, flagged, qa_report = run_qa_checks(
        queries,
        valid_node_ids=valid_node_ids,
        min_per_direction=1,
    )

    logger.info(
        "  Input=%d, Clean=%d, Flagged=%d, Removed=%d",
        qa_report.total_input,
        len(clean),
        qa_report.total_flagged,
        qa_report.total_removed,
    )
    if qa_report.warnings:
        for w in qa_report.warnings:
            logger.warning("  Warning: %s", w)

    all_valid = clean + flagged
    return all_valid, qa_report


def step_8_simulate_retrieval(queries, client):
    """Simulate retrieval results for all 14 methods.

    Uses varying recall rates to model different method performance levels.
    Graph-enhanced methods (E*) get higher recall than baselines (B*).
    """
    from workflows.benchmark.run_evaluation import METHOD_CONFIGS

    logger.info("=== Step 8: Simulate retrieval results ===")

    all_content_ids = client.execute_aql("FOR c IN content_nodes RETURN c._id")
    logger.info("  Total content nodes in DB: %d", len(all_content_ids))

    rng = random.Random(42)
    all_results: dict[str, dict[str, list[str]]] = {}

    # Define per-method recall rates (realistic hierarchy)
    RECALL_RATES = {
        "B1": 0.30,
        "B2": 0.25,
        "B3": 0.40,  # Baselines
        "E1a": 0.50,
        "E1b": 0.55,  # Top-Down
        "E2a": 0.52,
        "E2b": 0.58,
        "E2c": 0.60,
        "E2d": 0.65,  # Horizontal
        "E3a": 0.45,
        "E3b": 0.48,
        "E3c": 0.52,
        "E3d": 0.55,
        "E3e": 0.58,  # Bottom-Up
    }

    for config in METHOD_CONFIGS:
        mid = config["id"]
        recall_rate = RECALL_RATES.get(mid, 0.40)
        results_map: dict[str, list[str]] = {}

        for q in queries:
            relevant_ids = [fn.node_id for fn in q.fact_nodes if fn.grade > 0]
            all_fact_ids = [fn.node_id for fn in q.fact_nodes]

            retrieved = _simulate_retrieval(
                relevant_ids,
                all_fact_ids,
                all_content_ids,
                recall_rate=recall_rate,
                k=K,
                rng=rng,
            )
            results_map[q.query_id] = retrieved

        all_results[mid] = results_map
        logger.info(
            "  Method %s (recall=%.0f%%): %d queries", mid, recall_rate * 100, len(results_map)
        )

    return all_results


def _simulate_retrieval(
    relevant_ids: list[str],
    all_fact_ids: list[str],
    all_content_ids: list[str],
    recall_rate: float,
    k: int,
    rng: random.Random,
) -> list[str]:
    """Simulate retrieval with controlled recall rate."""
    retrieved = []

    # Include relevant nodes based on recall_rate
    for nid in relevant_ids:
        if rng.random() < recall_rate:
            retrieved.append(nid)

    # Also grade-1 nodes occasionally
    for nid in all_fact_ids:
        if nid not in retrieved and rng.random() < recall_rate * 0.4:
            retrieved.append(nid)

    rng.shuffle(retrieved)

    # Fill with noise
    noise_pool = [nid for nid in all_content_ids if nid not in retrieved]
    remaining = k - len(retrieved)
    if remaining > 0 and noise_pool:
        noise = rng.sample(noise_pool, min(remaining, len(noise_pool)))
        retrieved.extend(noise)

    return retrieved[:k]


def step_9_evaluate(queries, all_results):
    """Run IR evaluation on all 14 methods."""
    from workflows.benchmark.run_evaluation import evaluate_all_methods_offline

    logger.info("=== Step 9: Evaluate all methods ===")
    summary = evaluate_all_methods_offline(queries, all_results, k=K)

    logger.info("  Methods: %d, Queries: %d", summary.total_methods, summary.total_queries)
    logger.info("")
    logger.info("  %-6s %-30s  NDCG    Recall  Prec    MRR", "ID", "Name")
    logger.info("  " + "-" * 76)
    for mr in summary.method_results:
        logger.info(
            "  %-6s %-30s  %.4f  %.4f  %.4f  %.4f",
            mr.method_id,
            mr.method_name,
            mr.avg_ndcg,
            mr.avg_recall,
            mr.avg_precision,
            mr.avg_mrr,
        )

    return summary


def step_10_statistics(summary, queries):
    """Run statistical tests."""
    from workflows.benchmark.statistics import (
        analyze_per_direction,
        run_friedman_test,
        run_pairwise_wilcoxon,
        test_hypotheses,
        StatisticsReport,
    )

    logger.info("=== Step 10: Statistical analysis ===")

    f_stat, f_pval = run_friedman_test(summary.method_results)
    significant = f_pval < 0.05
    logger.info(
        "  Friedman: χ²=%.4f, p=%.6f (%s)",
        f_stat,
        f_pval,
        "SIGNIFICANT" if significant else "not significant",
    )

    pairwise = []
    if significant:
        pairwise = run_pairwise_wilcoxon(summary.method_results)
        sig_count = sum(1 for p in pairwise if p.significant)
        logger.info("  Pairwise: %d/%d significant", sig_count, len(pairwise))

    hypotheses = test_hypotheses(pairwise, summary.method_results)
    for h in hypotheses:
        status = "SUPPORTED" if h.supported else "NOT SUPPORTED"
        logger.info("  %s: %s — %s", h.hypothesis_id, status, h.description)

    per_direction = analyze_per_direction(summary.method_results, queries)

    return StatisticsReport(
        friedman_statistic=f_stat,
        friedman_p_value=f_pval,
        friedman_significant=significant,
        pairwise_results=pairwise,
        hypothesis_results=hypotheses,
        per_direction=per_direction,
    )


def step_11_generate_reports(
    summary, stats, queries, qa_report, subgraphs, direction_dist, selection_report
):
    """Generate JSON + MD reports."""
    from workflows.benchmark.report import generate_report

    logger.info("=== Step 11: Generate reports ===")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Full evaluation results JSON
    eval_json_path = OUTPUT_DIR / "evaluation_results.json"
    eval_data = {
        "generated_at": datetime.now(UTC).isoformat(),
        "pipeline_version": "4.0",
        "note": "Synthetic queries (no LLM) + simulated retrieval for pipeline validation",
        "dataset": {
            "total_queries": len(queries),
            "subgraphs": len(subgraphs),
            "direction_distribution": direction_dist,
            "grade_distribution": dict(Counter(fn.grade for q in queries for fn in q.fact_nodes)),
            "avg_fact_nodes_per_query": round(
                sum(len(q.fact_nodes) for q in queries) / max(len(queries), 1), 2
            ),
            "avg_confidence": round(
                sum(q.confidence.overall for q in queries) / max(len(queries), 1), 1
            ),
        },
        "subgraph_stats": {
            "total_selected": selection_report.selected_count,
            "tier_distribution": selection_report.tier_distribution,
            "avg_relation_coverage": round(selection_report.avg_relation_coverage, 4),
            "sources": selection_report.sources,
        },
        "qa_report": qa_report.to_dict(),
        "evaluation_summary": summary.to_dict(),
        "statistics": stats.to_dict(),
        "sample_queries": [q.model_dump(mode="json") for q in queries[:20]],
    }
    with open(eval_json_path, "w") as f:
        json.dump(eval_data, f, indent=2, ensure_ascii=False)
    logger.info("  JSON: %s", eval_json_path)

    # 2. Markdown report
    md_path = str(OUTPUT_DIR / "evaluation_report.md")
    generate_report(summary, stats, queries, md_path)
    logger.info("  MD: %s", md_path)

    # 3. All queries dataset
    queries_path = OUTPUT_DIR / "benchmark_queries.json"
    with open(queries_path, "w") as f:
        json.dump(
            {
                "version": "4.0",
                "created_at": datetime.now(UTC).isoformat(),
                "total_queries": len(queries),
                "queries": [q.model_dump(mode="json") for q in queries],
            },
            f,
            indent=2,
            ensure_ascii=False,
        )
    logger.info("  Queries: %s", queries_path)

    # 4. Quick summary
    quick_path = OUTPUT_DIR / "quick_summary.json"
    best = max(summary.method_results, key=lambda mr: mr.avg_ndcg)
    worst = min(summary.method_results, key=lambda mr: mr.avg_ndcg)
    quick = {
        "generated_at": datetime.now(UTC).isoformat(),
        "total_queries": len(queries),
        "subgraphs": len(subgraphs),
        "direction_distribution": direction_dist,
        "qa": {
            "passed": qa_report.total_output,
            "removed": qa_report.total_removed,
            "flagged": qa_report.total_flagged,
        },
        "best_method": {
            "id": best.method_id,
            "name": best.method_name,
            "ndcg": round(best.avg_ndcg, 4),
        },
        "worst_method": {
            "id": worst.method_id,
            "name": worst.method_name,
            "ndcg": round(worst.avg_ndcg, 4),
        },
        "friedman_significant": stats.friedman_significant,
        "hypotheses_supported": sum(1 for h in stats.hypothesis_results if h.supported),
        "hypotheses_total": len(stats.hypothesis_results),
        "methods": {
            mr.method_id: {
                "name": mr.method_name,
                "ndcg": round(mr.avg_ndcg, 4),
                "recall": round(mr.avg_recall, 4),
                "precision": round(mr.avg_precision, 4),
                "mrr": round(mr.avg_mrr, 4),
            }
            for mr in summary.method_results
        },
    }
    with open(quick_path, "w") as f:
        json.dump(quick, f, indent=2, ensure_ascii=False)
    logger.info("  Summary: %s", quick_path)

    return eval_json_path, md_path


async def main():
    logger.info("=" * 60)
    logger.info("BENCHMARK FULL EVALUATION PIPELINE v4.0")
    logger.info("Target: %d queries, %d subgraphs", TARGET_QUERIES, MAX_SUBGRAPHS)
    logger.info("Mode: Synthetic queries + simulated retrieval")
    logger.info("=" * 60)

    client = step_1_connect()

    candidates = step_2_list_sections(client)
    if not candidates:
        logger.error("No candidates found!")
        return False

    subgraphs, selection_report = step_3_extract_subgraphs(client, candidates)
    if not subgraphs:
        logger.error("No subgraphs extracted!")
        return False

    raw_queries = step_4_generate_synthetic_queries(subgraphs)
    if not raw_queries:
        logger.error("No queries generated!")
        return False

    classified, direction_dist = step_5_classify_directions(raw_queries)

    augmented = step_6_inject_distractors(classified, subgraphs)

    qa_queries, qa_report = step_7_qa_checks(augmented, client)
    if not qa_queries:
        logger.error("All queries removed by QA!")
        return False

    all_results = step_8_simulate_retrieval(qa_queries, client)

    summary = step_9_evaluate(qa_queries, all_results)

    stats = step_10_statistics(summary, qa_queries)

    json_path, md_path = step_11_generate_reports(
        summary, stats, qa_queries, qa_report, subgraphs, direction_dist, selection_report
    )

    logger.info("\n" + "=" * 60)
    logger.info("PIPELINE COMPLETE")
    logger.info("=" * 60)
    logger.info("Queries:    %d generated → %d after QA", len(raw_queries), len(qa_queries))
    logger.info("Methods:    %d evaluated", summary.total_methods)
    logger.info("Reports:    %s/", OUTPUT_DIR)
    logger.info("  - evaluation_results.json  (full results)")
    logger.info("  - evaluation_report.md     (markdown report)")
    logger.info("  - benchmark_queries.json   (all queries)")
    logger.info("  - quick_summary.json       (dashboard)")

    return True


if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)
