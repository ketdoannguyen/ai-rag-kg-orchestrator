"""End-to-end smoke test: ArangoDB → extract subgraphs → LLM → queries.

Usage: .venv/bin/python scripts/benchmark_e2e_smoke.py
"""

from __future__ import annotations

import asyncio
import json
import logging
import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

# Number of subgraphs to test (keep small for smoke test)
MAX_SUBGRAPHS = 3
QUERIES_PER_SUBGRAPH = 2


def step_1_connect_arango():
    """Step 1: Connect to ArangoDB and verify collections exist."""
    from workflows.dependencies import init_arango_client

    logger.info("--- Step 1: Connecting to ArangoDB ---")
    client = init_arango_client()

    # Quick health check
    collections = [c["name"] for c in client.db.collections() if not c["name"].startswith("_")]
    logger.info("Collections: %s", collections)

    # Count docs in key collections
    for coll in ["structure_nodes", "content_nodes", "knowledge_edges"]:
        if coll in collections:
            count = client.db.collection(coll).count()
            logger.info("  %s: %d docs", coll, count)
        else:
            logger.warning("  %s: MISSING!", coll)

    return client


def step_2_list_candidates(client, limit: int = 10):
    """Step 2: List candidate L5 sections from ArangoDB."""
    from workflows.benchmark.section_lister import list_candidate_sections

    logger.info("--- Step 2: Listing candidate sections ---")
    candidates = list_candidate_sections(client)
    logger.info("Total candidates: %d", len(candidates))

    for c in candidates[:limit]:
        logger.info(
            "  [%s] %s — %d content nodes, %d relations, summary=%s",
            c.section_key[:20],
            (c.title or "untitled")[:40],
            c.content_count,
            c.relation_count,
            c.has_summary,
        )

    return candidates


def step_3_extract_subgraphs(client, candidates, max_sg: int = MAX_SUBGRAPHS):
    """Step 3: Extract subgraphs from top candidates."""
    from workflows.benchmark.subgraph_extractor import SubgraphExtractor

    logger.info("--- Step 3: Extracting %d subgraphs ---", max_sg)
    extractor = SubgraphExtractor(client)
    subgraphs = []

    for c in candidates[: max_sg * 2]:  # try more in case some are empty
        if len(subgraphs) >= max_sg:
            break

        title = c.title or "untitled"
        breadcrumb = (c.parent_title or "") + " > " + title if c.parent_title else title
        sg = extractor.extract(
            anchor_key=c.section_key,
            anchor_title=title,
            anchor_summary=c.summary,
            breadcrumb=breadcrumb,
        )
        if sg is None:
            logger.debug("  Skipping empty subgraph: %s", c.section_key)
            continue

        # Set a query allocation for testing
        sg.query_allocation = QUERIES_PER_SUBGRAPH

        logger.info(
            "  SG %s: %d nodes (%d core + %d enriched), %d relations",
            sg.subgraph_id[:8],
            len(sg.nodes),
            len(sg.core_node_ids),
            len(sg.enriched_node_ids),
            len(sg.relations),
        )
        subgraphs.append(sg)

    return subgraphs


def step_4_preview_prompt(subgraphs):
    """Step 4: Preview the prompt that will be sent to LLM."""
    from workflows.benchmark.prompts import build_prompt

    logger.info("--- Step 4: Prompt preview (first subgraph) ---")
    if not subgraphs:
        logger.error("No subgraphs to preview!")
        return

    system_prompt, user_prompt = build_prompt(subgraphs[0], QUERIES_PER_SUBGRAPH)
    logger.info("System prompt length: %d chars", len(system_prompt))
    logger.info("User prompt length:   %d chars", len(user_prompt))
    logger.info("User prompt (first 500 chars):\n%s", user_prompt[:500])


async def step_5_generate_queries(subgraphs):
    """Step 5: Call LLM to generate queries for each subgraph."""
    from workflows.benchmark.query_generator import (
        build_benchmark_llm_client,
        generate_queries_for_subgraph,
    )

    logger.info("--- Step 5: Calling LLM for %d subgraphs ---", len(subgraphs))
    llm_client = build_benchmark_llm_client()

    all_queries = []
    all_rejected = []

    for sg in subgraphs:
        logger.info("  Generating for SG %s (%s)...", sg.subgraph_id[:8], sg.anchor_title[:30])
        try:
            valid, rejected = await generate_queries_for_subgraph(
                llm_client,
                sg,
                QUERIES_PER_SUBGRAPH,
            )
            all_queries.extend(valid)
            all_rejected.extend(rejected)
            logger.info("    → %d valid, %d rejected", len(valid), len(rejected))

            for q in valid:
                logger.info(
                    "    Q: %s (facts=%d, confidence=%d)",
                    q.query_text[:80],
                    len(q.fact_nodes),
                    q.confidence.overall,
                )

        except Exception:
            logger.exception("    → FAILED for SG %s", sg.subgraph_id[:8])

    return all_queries, all_rejected


def step_6_save_results(all_queries, all_rejected, subgraphs):
    """Step 6: Save results to JSON."""
    output_dir = Path("data/benchmark/smoke_test")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "e2e_smoke_results.json"

    results = {
        "subgraphs_tested": len(subgraphs),
        "total_valid_queries": len(all_queries),
        "total_rejected": len(all_rejected),
        "queries": [q.model_dump(mode="json") for q in all_queries],
        "rejected": all_rejected,
    }
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    logger.info("--- Step 6: Results saved to %s ---", output_path)
    return output_path


async def main():
    logger.info("=== Benchmark E2E Smoke Test ===\n")

    # Step 1: Connect
    client = step_1_connect_arango()

    # Step 2: List candidates
    candidates = step_2_list_candidates(client)
    if not candidates:
        logger.error("No candidates found! Is the KG populated?")
        sys.exit(1)

    # Step 3: Extract subgraphs
    subgraphs = step_3_extract_subgraphs(client, candidates)
    if not subgraphs:
        logger.error("No subgraphs extracted!")
        sys.exit(1)

    # Step 4: Preview prompt
    step_4_preview_prompt(subgraphs)

    # Step 5: Call LLM
    all_queries, all_rejected = await step_5_generate_queries(subgraphs)

    # Step 6: Save
    output_path = step_6_save_results(all_queries, all_rejected, subgraphs)

    # Summary
    logger.info("\n=== SUMMARY ===")
    logger.info("Subgraphs tested:  %d", len(subgraphs))
    logger.info("Valid queries:     %d", len(all_queries))
    logger.info("Rejected:          %d", len(all_rejected))
    logger.info("Results:           %s", output_path)

    if all_queries:
        logger.info("\nSample query:")
        q = all_queries[0]
        logger.info("  Text: %s", q.query_text)
        logger.info("  Facts: %d nodes", len(q.fact_nodes))
        for fn in q.fact_nodes:
            logger.info("    - %s (grade=%d)", fn.node_id, fn.grade)

    return len(all_queries) > 0


if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)
