"""LLM Query Generation Pipeline for Benchmark v4.0 (Task 14).

End-to-end: load subgraph → format prompt → call LLM → parse response →
validate → store. Resume-safe with per-subgraph checkpointing.

Uses dedicated BENCHMARK_LLM_* env vars (OpenAI-compatible endpoint)
via llm_gateway_client for consistency with the rest of the project.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import tempfile
import uuid
from pathlib import Path
from typing import Any

from litellm.exceptions import BadRequestError as LiteLLMBadRequestError
from llm_gateway_client import AIGatewayClient, GatewayModelConfig
from llm_gateway_client.models.request import GatewayChatTurn, GatewayInvokeRequest
from tenacity import retry, retry_if_not_exception_type, stop_after_attempt, wait_exponential

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

# Only 5 directions accepted from LLM (AMBIGUOUS is not a valid LLM output)
_LLM_VALID_DIRECTIONS = {"NT", "TD", "HZ", "BU", "CS"}


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


def _parse_fact_node(
    raw: dict,
    valid_node_ids: set[str],
    reverse_alias: dict[str, str] | None = None,
) -> FactNode:
    """Parse a single fact_node dict into FactNode model.

    Resolution order:
    1. If ``reverse_alias`` is provided and ``node_id`` is an alias key, resolve to full ID.
    2. Auto-prefix ``content_nodes/`` if the bare key matches an existing node.
    """
    node_id = raw["node_id"]
    if reverse_alias is not None and node_id in reverse_alias:
        node_id = reverse_alias[node_id]
    elif "/" not in node_id:
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
        overall=raw.get("overall", 50),
    )


def parse_llm_response(
    raw_text: str,
    subgraph: Subgraph,
    alias_map: dict[str, str] | None = None,
) -> tuple[list[BenchmarkQuery], list[dict]]:
    """Parse LLM JSON response into BenchmarkQuery objects.

    Validates that all fact_node IDs exist in the subgraph.

    Args:
        raw_text: Raw LLM response text (JSON).
        subgraph: Subgraph used to validate node IDs.
        alias_map: Optional alias → full_id mapping from ``build_prompt``.  When
            provided, LLM-returned aliases (``n1``, ``n2``, …) are resolved to
            full node IDs before validation.

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
    # Build reverse map: alias → full_id
    reverse_alias: dict[str, str] | None = (
        {alias: full_id for full_id, alias in alias_map.items()} if alias_map else None
    )
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
                _parse_fact_node(fn, valid_node_ids, reverse_alias)
                for fn in raw_q.get("fact_nodes", [])
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

            # Parse and validate direction (must be one of 5 valid directions from LLM)
            direction = raw_q.get("direction")
            if direction not in _LLM_VALID_DIRECTIONS:
                rejected.append(
                    {
                        "index": i,
                        "reason": f"Invalid or missing direction: {direction!r}",
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
                reasoning=raw_q.get("reasoning", ""),
                direction=direction,
            )
            valid_queries.append(query)

        except (KeyError, ValueError, TypeError, AttributeError) as exc:
            rejected.append(
                {
                    "index": i,
                    "reason": str(exc),
                    "query_text": raw_q.get("query_text", "N/A")
                    if isinstance(raw_q, dict)
                    else "N/A",
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
        retry=retry_if_not_exception_type(LiteLLMBadRequestError),
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
    direction_allocation: dict[str, int] | None = None,
    context_mode: str = "full",
) -> tuple[list[BenchmarkQuery], list[dict]]:
    """Generate benchmark queries for a single subgraph via LLM.

    Args:
        llm_client: AIGatewayClient configured with benchmark endpoint.
        subgraph: The subgraph to generate queries for.
        target_count: Number of queries to generate.
        direction_allocation: Per-direction quota, e.g. ``{"NT": 1, "TD": 2, "HZ": 3}``.
        context_mode: Context format mode passed to ``build_prompt``.

    Returns:
        (valid_queries, rejected_items)
    """
    system_prompt, user_prompt, alias_map = build_prompt(
        subgraph,
        target_count,
        direction_allocation,
        context_mode,  # type: ignore[arg-type]
    )

    raw_text = await _call_llm(llm_client, system_prompt, user_prompt)

    if not raw_text:
        logger.error("LLM returned empty response for subgraph %s", subgraph.subgraph_id)
        return [], [{"reason": "Empty LLM response", "subgraph_id": subgraph.subgraph_id}]

    valid, rejected = parse_llm_response(raw_text, subgraph, alias_map)

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
    per_subgraph_allocations: dict[str, dict[str, int]] | None = None,
    context_mode: str = "full",
    max_concurrency: int = 5,
) -> dict[str, Any]:
    """Generate queries for multiple subgraphs with parallel execution.

    Resume-safe: skips subgraphs whose output file already exists.
    Concurrency is controlled by ``max_concurrency`` (Semaphore).

    Args:
        llm_client: LLM client for generation.
        subgraphs: List of subgraphs to process.
        output_dir: Directory for checkpoint JSON files.
        per_subgraph_allocations: Mapping of subgraph_id → direction allocation dict.
            If None, ``generate_queries_for_subgraph`` uses its own default.
        context_mode: Context format mode for all subgraphs.
        max_concurrency: Maximum number of concurrent LLM calls (default 5).

    Returns:
        Summary stats dict including ``subgraphs_failed`` count.
    """
    max_concurrency = max(1, max_concurrency)
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    semaphore = asyncio.Semaphore(max_concurrency)

    async def _process_one(sg: Subgraph) -> dict[str, Any]:
        checkpoint_path = out_path / f"{sg.subgraph_id}.json"

        async with semaphore:
            # Resume check inside semaphore to avoid TOCTOU
            if checkpoint_path.exists():
                logger.info("Skipping %s (already processed)", sg.subgraph_id[:8])
                with open(checkpoint_path) as f:
                    existing = json.load(f)
                return {
                    "status": "skipped",
                    "valid": len(existing.get("queries", [])),
                    "rejected": existing.get("rejected_count", 0),
                    "queries": existing.get("queries", []),
                }

            target = sg.query_allocation
            if target <= 0:
                logger.debug("Skipping %s (allocation=0)", sg.subgraph_id[:8])
                return {"status": "skipped_zero", "valid": 0, "rejected": 0, "queries": []}

            allocation = (
                per_subgraph_allocations.get(sg.subgraph_id) if per_subgraph_allocations else None
            )
            valid, rejected = await generate_queries_for_subgraph(
                llm_client,
                sg,
                target,
                direction_allocation=allocation,
                context_mode=context_mode,
            )

            checkpoint_data = {
                "subgraph_id": sg.subgraph_id,
                "target_count": target,
                "valid_count": len(valid),
                "rejected_count": len(rejected),
                "queries": [q.model_dump(mode="json") for q in valid],
                "rejected": rejected,
            }

            # Atomic write: tmpfile → Path.replace()
            fd, tmp_path = tempfile.mkstemp(dir=out_path, suffix=".tmp")
            try:
                with os.fdopen(fd, "w") as f:
                    json.dump(checkpoint_data, f, indent=2, ensure_ascii=False)
                Path(tmp_path).replace(checkpoint_path)
            except BaseException:
                try:
                    Path(tmp_path).unlink()
                except OSError:
                    pass
                raise

            return {
                "status": "generated",
                "valid": len(valid),
                "rejected": len(rejected),
                "queries": checkpoint_data["queries"],
            }

    tasks = [asyncio.create_task(_process_one(sg)) for sg in subgraphs]

    total_valid = 0
    total_rejected = 0
    subgraphs_failed = 0
    all_queries: list[dict] = []

    for completed, coro in enumerate(asyncio.as_completed(tasks), start=1):
        try:
            result = await coro
            total_valid += result["valid"]
            total_rejected += result["rejected"]
            all_queries.extend(result["queries"])
        except Exception as exc:
            subgraphs_failed += 1
            logger.exception("Subgraph task failed: %s", exc)
        logger.info("Progress: %d/%d subgraphs done", completed, len(subgraphs))

    return {
        "subgraphs_processed": len(subgraphs),
        "subgraphs_failed": subgraphs_failed,
        "total_valid": total_valid,
        "total_rejected": total_rejected,
        "output_dir": output_dir,
    }
