"""Extraction debug logger for generating detailed MD reports.

This module generates markdown debug reports for each extraction run,
tracking the full decision-making process for debugging and auditing.

Reports include:
- Node statistics and similarity matrices
- Decision log for each node pair (merge/relation/separate)
- LLM batch tracking
- Summary statistics and performance metrics

Enhanced with deep debugging support:
- Phase-by-phase input/output tracking
- Similarity search results per node
- LLM prompt/response logging
- Decision flow tracing
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from .intra_batch_dedup import IntraBatchResult


@dataclass
class NodeInfo:
    """Information about a single node for debugging."""

    node_id: str
    node_type: str
    text_preview: str  # First 200 chars
    has_embedding: bool = True
    source_id: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class SimilarNodeResult:
    """Result of similarity search for a single candidate."""

    node_id: str
    node_type: str
    text_preview: str
    similarity_score: float
    is_reviewed: bool = True
    source_id: str | None = None


@dataclass
class SimilaritySearchLog:
    """Log of similarity search for a single node."""

    source_node: NodeInfo
    candidates_found: int
    candidates: list[SimilarNodeResult] = field(default_factory=list)
    best_match_decision: str = ""  # "MERGE", "LLM", "SEPARATE", "NO_MATCH"
    decision_reason: str = ""
    threshold_used: float = 0.0


@dataclass
class LLMCallLog:
    """Detailed log of a single LLM call."""

    call_index: int
    pair_count: int
    prompt_preview: str  # First 500 chars
    prompt_full: str | None = None  # Full prompt (optional)
    response_raw: str | None = None  # Raw LLM response
    parse_success: bool = True
    retry_count: int = 0
    results: list[dict[str, Any]] = field(default_factory=list)
    error: str | None = None
    duration_ms: float = 0.0


@dataclass
class PhaseLog:
    """Log for a single phase of processing."""

    phase_name: str
    start_time: datetime | None = None
    end_time: datetime | None = None
    input_count: int = 0
    output_count: int = 0
    input_nodes: list[NodeInfo] = field(default_factory=list)
    similarity_searches: list[SimilaritySearchLog] = field(default_factory=list)
    llm_calls: list[LLMCallLog] = field(default_factory=list)
    actions_taken: list[dict[str, Any]] = field(default_factory=list)
    summary: dict[str, Any] = field(default_factory=dict)


@dataclass
class PairDecision:
    """Record of a pair processing decision."""

    node_a_id: str
    node_a_type: str
    node_a_text: str
    node_b_id: str
    node_b_type: str
    node_b_text: str
    similarity: float
    decision: str  # "merge_auto", "merge_llm", "relation", "separate"
    relation_type: str | None = None
    llm_used: bool = False
    llm_reasoning: str | None = None
    phase: str = ""  # "intra_batch", "cross_batch", "phase_3"


@dataclass
class TierDebugLog:
    """Debug log for a single tier processing."""

    tier_name: str
    nodes_count: int = 0
    nodes_with_embeddings: int = 0
    similarity_matrix: np.ndarray | None = None
    node_ids: list[str] = field(default_factory=list)
    node_labels: list[str] = field(default_factory=list)
    intra_batch_components: int = 0
    intra_batch_merges: int = 0
    intra_batch_relation_candidates: int = 0
    cross_batch_processed: int = 0
    decisions: list[PairDecision] = field(default_factory=list)
    llm_batches: list[dict] = field(default_factory=list)

    # Enhanced debugging: phase-by-phase tracking
    input_nodes: list[NodeInfo] = field(default_factory=list)
    phases: dict[str, PhaseLog] = field(default_factory=dict)
    similarity_searches: list[SimilaritySearchLog] = field(default_factory=list)
    detailed_llm_calls: list[LLMCallLog] = field(default_factory=list)


@dataclass
class ExtractionDebugLog:
    """Full debug log for an extraction run."""

    run_id: str
    start_time: datetime
    end_time: datetime | None = None
    settings: dict = field(default_factory=dict)
    tier_logs: dict[str, TierDebugLog] = field(default_factory=dict)
    llm_calls_total: int = 0
    llm_pairs_total: int = 0

    # Enhanced debugging: global tracking
    deep_debug_enabled: bool = False
    database_state_before: dict[str, Any] = field(default_factory=dict)
    database_state_after: dict[str, Any] = field(default_factory=dict)

    def add_tier_log(self, tier_name: str) -> TierDebugLog:
        """Create and add a new tier log."""
        tier_log = TierDebugLog(tier_name=tier_name)
        self.tier_logs[tier_name] = tier_log
        return tier_log

    def get_tier_log(self, tier_name: str) -> TierDebugLog | None:
        """Get tier log by name."""
        return self.tier_logs.get(tier_name)


class ExtractionDebugLogger:
    """Logger for extraction debugging, generates MD reports.

    Enhanced with deep debugging support:
    - Phase-by-phase input/output tracking
    - Similarity search results per node
    - LLM prompt/response logging
    - Decision flow tracing
    """

    def __init__(
        self,
        output_dir: str | Path | None = None,
        deep_debug: bool = False,
    ) -> None:
        """Initialize debug logger.

        Args:
            output_dir: Directory for output files. Defaults to .claude/logs/extraction/
            deep_debug: If True, enables detailed logging including full LLM prompts
        """
        if output_dir is None:
            project_root = Path(__file__).parent.parent.parent
            output_dir = project_root / ".claude" / "logs" / "extraction"
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._current_log: ExtractionDebugLog | None = None
        self._deep_debug = deep_debug

    def start_run(
        self,
        settings: dict | None = None,
        deep_debug: bool | None = None,
    ) -> ExtractionDebugLog:
        """Start a new extraction run log."""
        run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self._current_log = ExtractionDebugLog(
            run_id=run_id,
            start_time=datetime.now(),
            settings=settings or {},
            deep_debug_enabled=deep_debug if deep_debug is not None else self._deep_debug,
        )
        return self._current_log

    @property
    def current_log(self) -> ExtractionDebugLog | None:
        """Get the current log."""
        return self._current_log

    def log_tier_start(
        self,
        tier_name: str,
        nodes_count: int,
        node_ids: list[str],
        node_labels: list[str],
    ) -> TierDebugLog:
        """Log the start of tier processing."""
        if self._current_log is None:
            raise RuntimeError("No active extraction run. Call start_run() first.")

        tier_log = self._current_log.add_tier_log(tier_name)
        tier_log.nodes_count = nodes_count
        tier_log.node_ids = node_ids
        tier_log.node_labels = node_labels
        return tier_log

    def log_intra_batch_result(
        self,
        tier_name: str,
        result: IntraBatchResult,
    ) -> None:
        """Log intra-batch deduplication result."""
        if self._current_log is None:
            return

        tier_log = self._current_log.get_tier_log(tier_name)
        if tier_log is None:
            return

        tier_log.similarity_matrix = result.similarity_matrix
        tier_log.nodes_with_embeddings = len(result.leader_indices) + len(result.merge_operations)
        tier_log.intra_batch_components = len(result.leader_indices)
        tier_log.intra_batch_merges = len(result.merge_operations)
        tier_log.intra_batch_relation_candidates = len(result.relation_candidates)

    def log_decision(
        self,
        tier_name: str,
        decision: PairDecision,
    ) -> None:
        """Log a pair processing decision."""
        if self._current_log is None:
            return

        tier_log = self._current_log.get_tier_log(tier_name)
        if tier_log is None:
            return

        tier_log.decisions.append(decision)

    def log_llm_batch(
        self,
        tier_name: str,
        pair_count: int,
        results: list[dict],
    ) -> None:
        """Log an LLM batch call."""
        if self._current_log is None:
            return

        tier_log = self._current_log.get_tier_log(tier_name)
        if tier_log is None:
            # Auto-create tier log if not exists
            tier_log = self._current_log.add_tier_log(tier_name)

        tier_log.llm_batches.append(
            {
                "pair_count": pair_count,
                "results": results,
            }
        )

        if self._current_log:
            self._current_log.llm_calls_total += 1
            self._current_log.llm_pairs_total += pair_count

    # =========================================================================
    # Enhanced Deep Debug Methods
    # =========================================================================

    def log_database_state(
        self,
        state: dict[str, Any],
        phase: str = "before",
    ) -> None:
        """Log database state (before/after extraction).

        Args:
            state: Dict with node counts, edge counts, etc.
            phase: "before" or "after"
        """
        if self._current_log is None:
            return

        if phase == "before":
            self._current_log.database_state_before = state
        else:
            self._current_log.database_state_after = state

    def log_input_nodes(
        self,
        tier_name: str,
        nodes: list[dict[str, Any]],
    ) -> None:
        """Log input nodes for a tier with detailed info.

        Args:
            tier_name: Tier name (tier_1_structure, tier_2_content)
            nodes: List of node dicts with id, type, text, embedding status
        """
        if self._current_log is None:
            return

        tier_log = self._current_log.get_tier_log(tier_name)
        if tier_log is None:
            tier_log = self._current_log.add_tier_log(tier_name)

        for node in nodes:
            tier_log.input_nodes.append(
                NodeInfo(
                    node_id=str(node.get("id", "")),
                    node_type=str(node.get("type", "")),
                    text_preview=str(node.get("text", ""))[:200],
                    has_embedding=node.get("has_embedding", True),
                    source_id=node.get("source_id"),
                    extra=node.get("extra", {}),
                )
            )

    def log_similarity_search(
        self,
        tier_name: str,
        source_node: dict[str, Any],
        candidates: list[dict[str, Any]],
        decision: str,
        decision_reason: str,
        threshold: float,
    ) -> None:
        """Log similarity search results for a single node.

        Args:
            tier_name: Tier name
            source_node: Source node being processed
            candidates: List of similar nodes found
            decision: Decision made (MERGE, LLM, SEPARATE, NO_MATCH)
            decision_reason: Human-readable reason
            threshold: Threshold used for decision
        """
        if self._current_log is None:
            return

        tier_log = self._current_log.get_tier_log(tier_name)
        if tier_log is None:
            tier_log = self._current_log.add_tier_log(tier_name)

        search_log = SimilaritySearchLog(
            source_node=NodeInfo(
                node_id=str(source_node.get("id", "")),
                node_type=str(source_node.get("type", "")),
                text_preview=str(source_node.get("text", ""))[:200],
                has_embedding=source_node.get("has_embedding", True),
                source_id=source_node.get("source_id"),
            ),
            candidates_found=len(candidates),
            best_match_decision=decision,
            decision_reason=decision_reason,
            threshold_used=threshold,
        )

        for cand in candidates:
            search_log.candidates.append(
                SimilarNodeResult(
                    node_id=str(cand.get("id", "")),
                    node_type=str(cand.get("type", "")),
                    text_preview=str(cand.get("text", ""))[:200],
                    similarity_score=float(cand.get("similarity", 0.0)),
                    is_reviewed=cand.get("is_reviewed", True),
                    source_id=cand.get("source_id"),
                )
            )

        tier_log.similarity_searches.append(search_log)

    def log_detailed_llm_call(
        self,
        tier_name: str,
        call_index: int,
        pair_count: int,
        prompt: str,
        response: str | None,
        results: list[dict[str, Any]],
        success: bool = True,
        retry_count: int = 0,
        error: str | None = None,
        duration_ms: float = 0.0,
    ) -> None:
        """Log detailed LLM call with full prompt and response.

        Args:
            tier_name: Tier name
            call_index: Index of this call in the batch sequence
            pair_count: Number of pairs in this call
            prompt: Full LLM prompt
            response: Raw LLM response
            results: Parsed results
            success: Whether parsing succeeded
            retry_count: Number of retries
            error: Error message if failed
            duration_ms: Call duration in milliseconds
        """
        if self._current_log is None:
            return

        tier_log = self._current_log.get_tier_log(tier_name)
        if tier_log is None:
            tier_log = self._current_log.add_tier_log(tier_name)

        # Only store full prompt if deep debug is enabled
        full_prompt = prompt if self._current_log.deep_debug_enabled else None

        llm_log = LLMCallLog(
            call_index=call_index,
            pair_count=pair_count,
            prompt_preview=prompt[:500] if prompt else "",
            prompt_full=full_prompt,
            response_raw=response,
            parse_success=success,
            retry_count=retry_count,
            results=results,
            error=error,
            duration_ms=duration_ms,
        )

        tier_log.detailed_llm_calls.append(llm_log)

    def log_action(
        self,
        tier_name: str,
        action_type: str,
        source_node_id: str,
        target_node_id: str | None = None,
        relation_type: str | None = None,
        similarity: float = 0.0,
        confidence: float = 0.0,
        method: str = "",
        success: bool = True,
        details: dict[str, Any] | None = None,
    ) -> None:
        """Log an action taken (merge, relation creation, mark reviewed).

        Args:
            tier_name: Tier name
            action_type: Type of action (merge, relation, mark_reviewed, keep_separate)
            source_node_id: Source node ID
            target_node_id: Target node ID (for merge/relation)
            relation_type: Relation type (for relation action)
            similarity: Similarity score
            confidence: Confidence score
            method: Detection method
            success: Whether action succeeded
            details: Additional details
        """
        if self._current_log is None:
            return

        tier_log = self._current_log.get_tier_log(tier_name)
        if tier_log is None:
            tier_log = self._current_log.add_tier_log(tier_name)

        # Get current phase from phases dict
        if tier_log.phases:
            # Get the most recent phase that's still in progress
            for _, phase in tier_log.phases.items():
                if phase.end_time is None:
                    phase.actions_taken.append(
                        {
                            "action_type": action_type,
                            "source_node_id": source_node_id,
                            "target_node_id": target_node_id,
                            "relation_type": relation_type,
                            "similarity": similarity,
                            "confidence": confidence,
                            "method": method,
                            "success": success,
                            "details": details or {},
                        }
                    )
                    break

    def start_phase(self, tier_name: str, phase_name: str) -> PhaseLog:
        """Start logging a new phase.

        Args:
            tier_name: Tier name
            phase_name: Phase name (e.g., "similarity_search", "llm_batch", "post_process")

        Returns:
            PhaseLog for the new phase
        """
        if self._current_log is None:
            raise RuntimeError("No active extraction run. Call start_run() first.")

        tier_log = self._current_log.get_tier_log(tier_name)
        if tier_log is None:
            tier_log = self._current_log.add_tier_log(tier_name)

        phase_log = PhaseLog(phase_name=phase_name, start_time=datetime.now())
        tier_log.phases[phase_name] = phase_log
        return phase_log

    def end_phase(
        self,
        tier_name: str,
        phase_name: str,
        output_count: int = 0,
        summary: dict[str, Any] | None = None,
    ) -> None:
        """End logging a phase.

        Args:
            tier_name: Tier name
            phase_name: Phase name
            output_count: Number of outputs from this phase
            summary: Summary statistics for this phase
        """
        if self._current_log is None:
            return

        tier_log = self._current_log.get_tier_log(tier_name)
        if tier_log is None:
            return

        phase_log = tier_log.phases.get(phase_name)
        if phase_log is None:
            return

        phase_log.end_time = datetime.now()
        phase_log.output_count = output_count
        phase_log.summary = summary or {}

    def finish_run(self) -> Path | None:
        """Finish the current run and generate report."""
        if self._current_log is None:
            return None

        self._current_log.end_time = datetime.now()
        report_path = self._generate_report(self._current_log)
        self._current_log = None
        return report_path

    def _generate_report(self, log: ExtractionDebugLog) -> Path:
        """Generate markdown report from log."""
        import json

        filename = f"extraction_{log.run_id}.md"
        report_path = self.output_dir / filename

        lines: list[str] = []

        # Header
        lines.append(f"# Extraction Debug Report - {log.run_id}\n")
        lines.append(f"**Start Time:** {log.start_time.isoformat()}\n")
        if log.end_time:
            duration = (log.end_time - log.start_time).total_seconds()
            lines.append(f"**End Time:** {log.end_time.isoformat()}\n")
            lines.append(f"**Duration:** {duration:.2f} seconds\n")
        lines.append(f"**Deep Debug:** {'Enabled' if log.deep_debug_enabled else 'Disabled'}\n")

        # Settings
        lines.append("\n## Settings\n")
        lines.append("```json")
        lines.append(json.dumps(log.settings, indent=2, default=str))
        lines.append("```\n")

        # Database state comparison (if available)
        if log.database_state_before or log.database_state_after:
            lines.append("\n## Database State\n")

            if log.database_state_before:
                lines.append("### Before Extraction\n")
                lines.append("```json")
                lines.append(json.dumps(log.database_state_before, indent=2, default=str))
                lines.append("```\n")

            if log.database_state_after:
                lines.append("### After Extraction\n")
                lines.append("```json")
                lines.append(json.dumps(log.database_state_after, indent=2, default=str))
                lines.append("```\n")

            # Calculate diff
            if log.database_state_before and log.database_state_after:
                lines.append("### Changes\n")
                before = log.database_state_before
                after = log.database_state_after

                # Node count changes
                if "nodes" in before and "nodes" in after:
                    for coll, before_counts in before.get("nodes", {}).items():
                        after_counts = after.get("nodes", {}).get(coll, before_counts)
                        if isinstance(before_counts, dict) and isinstance(after_counts, dict):
                            total_diff = after_counts.get("total", 0) - before_counts.get(
                                "total", 0
                            )
                            reviewed_diff = after_counts.get("reviewed", 0) - before_counts.get(
                                "reviewed", 0
                            )
                            lines.append(
                                f"- **{coll}:** Total {total_diff:+d}, Reviewed {reviewed_diff:+d}"
                            )

                # Edge count changes
                if "edges" in before and "edges" in after:
                    edge_diff = after.get("edges", {}).get("total", 0) - before.get(
                        "edges", {}
                    ).get("total", 0)
                    lines.append(f"- **Edges:** {edge_diff:+d}")

                lines.append("")

        # Summary
        lines.append("\n## Summary\n")
        lines.append(f"- **LLM Calls Total:** {log.llm_calls_total}")
        lines.append(f"- **LLM Pairs Processed:** {log.llm_pairs_total}")
        if log.llm_calls_total > 0:
            avg = log.llm_pairs_total / log.llm_calls_total
            lines.append(f"- **Avg Pairs per Call:** {avg:.2f}")
        lines.append("")

        # Per-tier details
        for tier_name, tier_log in log.tier_logs.items():
            lines.extend(self._generate_tier_section(tier_name, tier_log))

        report_path.write_text("\n".join(lines), encoding="utf-8")
        return report_path

    def _generate_tier_section(
        self,
        tier_name: str,
        tier_log: TierDebugLog,
    ) -> list[str]:
        """Generate tier section of report."""
        import json

        lines: list[str] = []
        display_name = (
            "Tier 1: Structure" if "structure" in tier_name.lower() else "Tier 2: Content"
        )

        lines.append(f"\n## {display_name}\n")

        # Node statistics
        lines.append("### Node Statistics\n")
        lines.append(f"- **Total Nodes:** {tier_log.nodes_count}")
        lines.append(f"- **Nodes with Embeddings:** {tier_log.nodes_with_embeddings}")
        lines.append("")

        # Enhanced: Input nodes detail (if available)
        if tier_log.input_nodes:
            lines.append("### Input Nodes (Unreviewed)\n")
            lines.append("| # | Node ID | Type | Text Preview | Has Embedding |")
            lines.append("|---|---------|------|--------------|---------------|")
            for i, node in enumerate(tier_log.input_nodes[:50]):  # Limit to 50
                node_id_short = node.node_id[:12] if len(node.node_id) > 12 else node.node_id
                text_preview = node.text_preview[:50].replace("|", "\\|").replace("\n", " ")
                emb_status = "✓" if node.has_embedding else "✗"
                lines.append(
                    f"| {i + 1} | {node_id_short} | {node.node_type} | {text_preview}... | {emb_status} |"
                )
            if len(tier_log.input_nodes) > 50:
                lines.append(f"\n*... and {len(tier_log.input_nodes) - 50} more nodes*")
            lines.append("")

        # Intra-batch results
        lines.append("### Phase 1: Intra-Batch Deduplication\n")
        lines.append(f"- **Connected Components:** {tier_log.intra_batch_components}")
        lines.append(f"- **Merge Operations:** {tier_log.intra_batch_merges}")
        lines.append(f"- **Relation Candidates:** {tier_log.intra_batch_relation_candidates}")
        lines.append("")

        # Similarity matrix visualization (if small enough)
        if tier_log.similarity_matrix is not None and len(tier_log.similarity_matrix) <= 20:
            lines.extend(
                self._generate_similarity_matrix_table(
                    tier_log.similarity_matrix,
                    tier_log.node_labels[: len(tier_log.similarity_matrix)],
                )
            )

        # Enhanced: Similarity search details (per node)
        if tier_log.similarity_searches:
            lines.append("### Phase 2: Similarity Search Results\n")
            lines.append("Detailed per-node similarity search with candidates and decisions:\n")

            for i, search in enumerate(tier_log.similarity_searches[:30]):  # Limit to 30
                source_id = search.source_node.node_id[:12]
                source_text = search.source_node.text_preview[:80].replace("\n", " ")

                lines.append(f"#### Node {i + 1}: `{source_id}`")
                lines.append(f"**Text:** {source_text}...")
                lines.append(f"**Candidates Found:** {search.candidates_found}")
                lines.append(
                    f"**Decision:** `{search.best_match_decision}` - {search.decision_reason}"
                )
                lines.append(f"**Threshold:** {search.threshold_used:.2f}")

                if search.candidates:
                    lines.append("\n| Rank | Candidate ID | Similarity | Type | Reviewed | Text |")
                    lines.append("|------|--------------|------------|------|----------|------|")
                    for j, cand in enumerate(search.candidates[:5]):  # Top 5 candidates
                        cand_id = cand.node_id[:12]
                        cand_text = cand.text_preview[:40].replace("|", "\\|").replace("\n", " ")
                        reviewed = "✓" if cand.is_reviewed else "✗"
                        sim_fmt = (
                            f"**{cand.similarity_score:.3f}**"
                            if cand.similarity_score >= 0.72
                            else (
                                f"*{cand.similarity_score:.3f}*"
                                if cand.similarity_score >= 0.40
                                else f"{cand.similarity_score:.3f}"
                            )
                        )
                        lines.append(
                            f"| {j + 1} | {cand_id} | {sim_fmt} | {cand.node_type} | {reviewed} | {cand_text}... |"
                        )
                lines.append("")

            if len(tier_log.similarity_searches) > 30:
                lines.append(f"*... and {len(tier_log.similarity_searches) - 30} more searches*\n")

        # Decision log
        if tier_log.decisions:
            lines.append("### Decision Log\n")
            lines.append("| Phase | Node A | Node B | Similarity | Decision | Relation | LLM |")
            lines.append("|-------|--------|--------|------------|----------|----------|-----|")

            for d in tier_log.decisions:
                node_a_short = d.node_a_id[:8] if len(d.node_a_id) > 8 else d.node_a_id
                node_b_short = d.node_b_id[:8] if len(d.node_b_id) > 8 else d.node_b_id
                llm_mark = "Yes" if d.llm_used else "-"
                relation = d.relation_type or "-"
                lines.append(
                    f"| {d.phase} | {node_a_short} | {node_b_short} | "
                    f"{d.similarity:.3f} | {d.decision} | {relation} | {llm_mark} |"
                )
            lines.append("")

        # Enhanced: Detailed LLM Calls
        if tier_log.detailed_llm_calls:
            lines.append("### Phase 3: LLM Batch Processing (Detailed)\n")

            for llm_call in tier_log.detailed_llm_calls:
                lines.append(f"#### LLM Call {llm_call.call_index + 1}")
                lines.append(f"- **Pairs:** {llm_call.pair_count}")
                lines.append(f"- **Duration:** {llm_call.duration_ms:.0f}ms")
                lines.append(f"- **Parse Success:** {'✓' if llm_call.parse_success else '✗'}")
                if llm_call.retry_count > 0:
                    lines.append(f"- **Retries:** {llm_call.retry_count}")
                if llm_call.error:
                    lines.append(f"- **Error:** {llm_call.error}")

                # Prompt preview
                lines.append("\n**Prompt Preview:**")
                lines.append("```")
                lines.append(llm_call.prompt_preview)
                lines.append("```")

                # Full prompt (if deep debug)
                if llm_call.prompt_full:
                    lines.append("\n<details>")
                    lines.append("<summary>Full Prompt (click to expand)</summary>")
                    lines.append("\n```")
                    lines.append(llm_call.prompt_full)
                    lines.append("```")
                    lines.append("</details>")

                # Raw response
                if llm_call.response_raw:
                    lines.append("\n**Raw Response:**")
                    lines.append("```json")
                    # Truncate long responses
                    response_preview = llm_call.response_raw[:2000]
                    if len(llm_call.response_raw) > 2000:
                        response_preview += "\n... (truncated)"
                    lines.append(response_preview)
                    lines.append("```")

                # Parsed results
                if llm_call.results:
                    lines.append("\n**Parsed Results:**")
                    lines.append("| # | Merge? | Relation | Confidence | Reasoning |")
                    lines.append("|---|--------|----------|------------|-----------|")
                    for j, r in enumerate(llm_call.results):
                        merge = "✓" if r.get("should_merge") else "✗"
                        rel = r.get("relation_type") or "-"
                        conf = r.get("confidence", 0)
                        reasoning = str(r.get("reasoning", ""))[:50].replace("|", "\\|")
                        lines.append(
                            f"| {j + 1} | {merge} | {rel} | {conf:.2f} | {reasoning}... |"
                        )

                lines.append("")

        # Legacy LLM batch details (fallback)
        elif tier_log.llm_batches:
            lines.append("### LLM Batch Calls\n")
            for i, batch in enumerate(tier_log.llm_batches):
                lines.append(f"**Batch {i + 1}:** {batch['pair_count']} pairs")
                if batch.get("results"):
                    lines.append("```json")
                    lines.append(json.dumps(batch["results"], indent=2, default=str))
                    lines.append("```")
            lines.append("")

        # Enhanced: Phase summaries
        if tier_log.phases:
            lines.append("### Phase Summaries\n")
            for phase_name, phase in tier_log.phases.items():
                duration = 0.0
                if phase.start_time and phase.end_time:
                    duration = (phase.end_time - phase.start_time).total_seconds()

                lines.append(f"#### {phase_name}")
                lines.append(f"- **Input Count:** {phase.input_count}")
                lines.append(f"- **Output Count:** {phase.output_count}")
                lines.append(f"- **Duration:** {duration:.2f}s")
                lines.append(f"- **Actions Taken:** {len(phase.actions_taken)}")

                if phase.summary:
                    lines.append("- **Summary:**")
                    lines.append("```json")
                    lines.append(json.dumps(phase.summary, indent=2, default=str))
                    lines.append("```")

                # Actions detail
                if phase.actions_taken:
                    lines.append("\n**Actions:**")
                    lines.append("| Action | Source | Target | Relation | Similarity | Method |")
                    lines.append("|--------|--------|--------|----------|------------|--------|")
                    for action in phase.actions_taken[:20]:  # Limit to 20
                        src = str(action.get("source_node_id", ""))[:8]
                        tgt = (
                            str(action.get("target_node_id", ""))[:8]
                            if action.get("target_node_id")
                            else "-"
                        )
                        rel = action.get("relation_type") or "-"
                        sim = action.get("similarity", 0)
                        method = action.get("method", "")[:15]
                        lines.append(
                            f"| {action.get('action_type', '')} | {src} | {tgt} | {rel} | {sim:.3f} | {method} |"
                        )
                    if len(phase.actions_taken) > 20:
                        lines.append(f"\n*... and {len(phase.actions_taken) - 20} more actions*")

                lines.append("")

        return lines

    def _generate_similarity_matrix_table(
        self,
        matrix: np.ndarray,
        labels: list[str],
    ) -> list[str]:
        """Generate markdown table for similarity matrix."""
        lines: list[str] = []
        lines.append("### Similarity Matrix\n")

        if len(matrix) == 0:
            lines.append("*No similarity data available*\n")
            return lines

        n = len(matrix)
        short_labels = [lbl[:12] if len(lbl) > 12 else lbl for lbl in labels]

        # Header row
        header = "| |" + "|".join(short_labels) + "|"
        lines.append(header)
        separator = "|" + "|".join(["---"] * (n + 1)) + "|"
        lines.append(separator)

        # Data rows with color coding
        for i in range(n):
            row_cells = [short_labels[i]]
            for j in range(n):
                val = matrix[i, j]
                if i == j:
                    row_cells.append("1.00")
                elif val >= 0.72:
                    row_cells.append(f"**{val:.2f}**")  # Bold for merge threshold
                elif val >= 0.40:
                    row_cells.append(f"*{val:.2f}*")  # Italic for LLM threshold
                else:
                    row_cells.append(f"{val:.2f}")
            lines.append("|" + "|".join(row_cells) + "|")

        lines.append("")
        lines.append(
            "*Legend: **Bold** = merge threshold (>=0.72), *Italic* = LLM threshold (>=0.40)*\n"
        )

        return lines


# Module-level singleton instance
_debug_logger: ExtractionDebugLogger | None = None


def get_debug_logger(
    output_dir: str | Path | None = None,
    deep_debug: bool = False,
) -> ExtractionDebugLogger:
    """Get or create the debug logger singleton.

    Args:
        output_dir: Directory for output files
        deep_debug: If True, enables detailed logging including full LLM prompts

    Returns:
        ExtractionDebugLogger instance
    """
    global _debug_logger  # noqa: PLW0603
    if _debug_logger is None:
        _debug_logger = ExtractionDebugLogger(output_dir, deep_debug=deep_debug)
    elif deep_debug and not _debug_logger._deep_debug:
        # Upgrade to deep debug if requested
        _debug_logger._deep_debug = deep_debug
    return _debug_logger


def reset_debug_logger() -> None:
    """Reset the debug logger singleton (for testing)."""
    global _debug_logger  # noqa: PLW0603
    _debug_logger = None
