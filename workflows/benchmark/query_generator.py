"""LLM Query Generation Pipeline for Benchmark v4.0 (Task 14).

End-to-end: load subgraph → format prompt → call LLM → parse response →
validate → store. Resume-safe with per-subgraph checkpointing.

Uses dedicated BENCHMARK_LLM_* env vars (OpenAI-compatible endpoint)
via llm_gateway_client for consistency with the rest of the project.
"""

from __future__ import annotations

import json
import logging
import re
import uuid
from pathlib import Path
from typing import Any

from llm_gateway_client import AIGatewayClient, GatewayModelConfig
from llm_gateway_client.models.request import GatewayChatTurn, GatewayInvokeRequest
from tenacity import retry, stop_after_attempt, wait_exponential

from config.langfuse import configure_langfuse
from config.settings import get_settings
from workflows.benchmark.models import (
    BenchmarkQuery,
    ConfidenceScore,
    FactNode,
    Subgraph,
)
from workflows.benchmark.prompts import build_prompt

logger = logging.getLogger(__name__)


def build_benchmark_llm_client() -> AIGatewayClient:
    """Build an AIGatewayClient using BENCHMARK_LLM_* env vars."""
    cfg = get_settings()
    configure_langfuse(cfg)
    if not cfg.BENCHMARK_LLM_API_KEY or not cfg.BENCHMARK_LLM_API_BASE:
        raise ValueError("BENCHMARK_LLM_API_KEY and BENCHMARK_LLM_API_BASE must be set in .env")
    chat_config = GatewayModelConfig(
        model=f"openai/{cfg.BENCHMARK_LLM_MODEL}",
        api_key=cfg.BENCHMARK_LLM_API_KEY,
        api_base=cfg.BENCHMARK_LLM_API_BASE,
    )
    return AIGatewayClient(chat_models={"default": chat_config})


# ------------------------------------------------------------------ #
# JSON parsing helpers                                                 #
# ------------------------------------------------------------------ #


def _extract_json(text: str) -> dict:
    """Extract JSON from LLM response, handling markdown fences."""
    text = text.strip()

    # Try direct parse first
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Strip markdown code fences
    fence_match = re.search(r"```(?:json)?\s*\n?(.*?)\n?\s*```", text, re.DOTALL)
    if fence_match:
        try:
            return json.loads(fence_match.group(1).strip())
        except json.JSONDecodeError:
            pass

    # Try to find JSON object boundaries
    brace_start = text.find("{")
    brace_end = text.rfind("}")
    if brace_start != -1 and brace_end > brace_start:
        try:
            return json.loads(text[brace_start : brace_end + 1])
        except json.JSONDecodeError:
            pass

    raise ValueError(f"Could not extract valid JSON from LLM response: {text[:200]}...")


def _parse_fact_node(raw: dict, valid_node_ids: set[str]) -> FactNode:
    """Parse a single fact_node dict into FactNode model.

    Auto-prefixes 'content_nodes/' if the LLM returned a bare key
    that matches an existing node in the subgraph.
    """
    node_id = raw["node_id"]
    if "/" not in node_id:
        prefixed = f"content_nodes/{node_id}"
        if prefixed in valid_node_ids:
            node_id = prefixed
    return FactNode(
        node_id=node_id,
        grade=raw["grade"],
        parent_section=raw.get("parent_section", "unknown"),
    )


def _parse_confidence(raw: dict) -> ConfidenceScore:
    """Parse confidence dict into ConfidenceScore model."""
    return ConfidenceScore(
        question_clarity=raw.get("question_clarity", 5),
        fact_node_completeness=raw.get("fact_node_completeness", 5),
        fact_node_precision=raw.get("fact_node_precision", 5),
        grade_certainty=raw.get("grade_certainty", 5),
        overall=raw.get("overall", 50),
    )


def parse_llm_response(
    raw_text: str,
    subgraph: Subgraph,
) -> tuple[list[BenchmarkQuery], list[dict]]:
    """Parse LLM JSON response into BenchmarkQuery objects.

    Validates that all fact_node IDs exist in the subgraph.

    Returns:
        (valid_queries, rejected_items) where rejected_items contains
        dicts with reason and raw data for audit.
    """
    data = _extract_json(raw_text)
    if not isinstance(data, dict):
        raise ValueError(f"Expected JSON object from LLM, got {type(data).__name__}")
    raw_queries = data.get("queries", [])
    if not isinstance(raw_queries, list):
        raise ValueError(f"Expected 'queries' to be a list, got {type(raw_queries).__name__}")

    valid_node_ids = {n.node_id for n in subgraph.nodes}
    valid_queries: list[BenchmarkQuery] = []
    rejected: list[dict] = []

    for i, raw_q in enumerate(raw_queries):
        try:
            if not isinstance(raw_q, dict):
                rejected.append(
                    {"index": i, "reason": f"Expected dict, got {type(raw_q).__name__}"}
                )
                continue

            # Parse fact nodes
            fact_nodes = [
                _parse_fact_node(fn, valid_node_ids) for fn in raw_q.get("fact_nodes", [])
            ]

            # Validate node IDs against subgraph
            invalid_ids = [fn.node_id for fn in fact_nodes if fn.node_id not in valid_node_ids]
            if invalid_ids:
                rejected.append(
                    {
                        "index": i,
                        "reason": f"Invalid node_ids: {invalid_ids}",
                        "query_text": raw_q.get("query_text", ""),
                    }
                )
                continue

            # Must have at least one grade-2 node
            has_grade_2 = any(fn.grade == 2 for fn in fact_nodes)
            if not has_grade_2:
                rejected.append(
                    {
                        "index": i,
                        "reason": "No grade-2 fact_node",
                        "query_text": raw_q.get("query_text", ""),
                    }
                )
                continue

            confidence = _parse_confidence(raw_q.get("confidence", {}))

            query = BenchmarkQuery(
                query_id=str(uuid.uuid4()),
                query_text=raw_q["query_text"],
                subgraph_id=subgraph.subgraph_id,
                fact_nodes=fact_nodes,
                relations_used=raw_q.get("relations_used", []),
                confidence=confidence,
                confidence_concerns=raw_q.get("confidence_concerns", ""),
                reasoning=raw_q.get("reasoning", ""),
            )
            valid_queries.append(query)

        except (KeyError, ValueError, TypeError, AttributeError) as exc:
            rejected.append(
                {
                    "index": i,
                    "reason": str(exc),
                    "query_text": raw_q.get("query_text", "N/A") if isinstance(raw_q, dict) else "N/A",
                }
            )

    return valid_queries, rejected


# ------------------------------------------------------------------ #
# Generation pipeline                                                 #
# ------------------------------------------------------------------ #


async def _call_llm(
    llm_client: AIGatewayClient,
    system_prompt: str,
    user_prompt: str,
) -> str:
    """Call LLM with retry on transient failures.

    Uses tenacity retry: 3 attempts with exponential backoff.
    """

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=60),
        reraise=True,
    )
    async def _invoke():
        request = GatewayInvokeRequest(
            messages=[
                GatewayChatTurn(role="system", content=system_prompt),
                GatewayChatTurn(role="user", content=user_prompt),
            ],
            model="default",
            provider_options={"temperature": 0.7, "max_tokens": 8000},
        )
        response = await llm_client.invoke(request)
        return response.message.content if response.message else ""

    return await _invoke()


async def generate_queries_for_subgraph(
    llm_client: AIGatewayClient,
    subgraph: Subgraph,
    target_count: int,
) -> tuple[list[BenchmarkQuery], list[dict]]:
    """Generate benchmark queries for a single subgraph via LLM.

    Args:
        llm_client: AIGatewayClient configured with benchmark endpoint.
        subgraph: The subgraph to generate queries for.
        target_count: Number of queries to generate.

    Returns:
        (valid_queries, rejected_items)
    """
    system_prompt, user_prompt = build_prompt(subgraph, target_count)

    raw_text = await _call_llm(llm_client, system_prompt, user_prompt)

    if not raw_text:
        logger.error("LLM returned empty response for subgraph %s", subgraph.subgraph_id)
        return [], [{"reason": "Empty LLM response", "subgraph_id": subgraph.subgraph_id}]

    valid, rejected = parse_llm_response(raw_text, subgraph)

    logger.info(
        "Subgraph %s: generated %d valid, %d rejected queries (target=%d)",
        subgraph.subgraph_id[:8],
        len(valid),
        len(rejected),
        target_count,
    )

    return valid, rejected


async def generate_batch(
    llm_client: AIGatewayClient,
    subgraphs: list[Subgraph],
    output_dir: str = "data/benchmark/queries",
) -> dict[str, Any]:
    """Generate queries for multiple subgraphs with progress tracking.

    Resume-safe: skips subgraphs whose output file already exists.

    Returns:
        Summary stats dict.
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    total_valid = 0
    total_rejected = 0
    all_queries: list[dict] = []

    for i, sg in enumerate(subgraphs):
        checkpoint_path = Path(output_dir) / f"{sg.subgraph_id}.json"

        # Resume: skip if already processed
        if checkpoint_path.exists():
            logger.info("Skipping %s (already processed)", sg.subgraph_id[:8])
            with open(checkpoint_path) as f:
                existing = json.load(f)
            total_valid += len(existing.get("queries", []))
            total_rejected += existing.get("rejected_count", 0)
            all_queries.extend(existing.get("queries", []))
            continue

        target = sg.query_allocation
        if target <= 0:
            logger.debug("Skipping %s (allocation=0)", sg.subgraph_id[:8])
            continue

        try:
            valid, rejected = await generate_queries_for_subgraph(
                llm_client,
                sg,
                target,
            )
        except Exception:
            logger.exception("Failed to generate for subgraph %s", sg.subgraph_id[:8])
            continue

        total_valid += len(valid)
        total_rejected += len(rejected)

        # Save checkpoint
        checkpoint_data = {
            "subgraph_id": sg.subgraph_id,
            "target_count": target,
            "valid_count": len(valid),
            "rejected_count": len(rejected),
            "queries": [q.model_dump(mode="json") for q in valid],
            "rejected": rejected,
        }
        with open(checkpoint_path, "w") as f:
            json.dump(checkpoint_data, f, indent=2, ensure_ascii=False)

        all_queries.extend(checkpoint_data["queries"])

        if (i + 1) % 10 == 0:
            logger.info(
                "Progress: %d/%d subgraphs, %d queries so far",
                i + 1,
                len(subgraphs),
                total_valid,
            )

    return {
        "subgraphs_processed": len(subgraphs),
        "total_valid": total_valid,
        "total_rejected": total_rejected,
        "output_dir": output_dir,
    }
