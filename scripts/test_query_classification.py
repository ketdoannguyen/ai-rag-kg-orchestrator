#!/usr/bin/env python3
"""Test script for Query Classification with real LLM.

Evaluates the QueryRouter's ability to:
1. Classify query types correctly
2. Select appropriate strategies
3. Determine relation usage

Dataset: 50 queries with difficulty levels (easy/medium/hard)

Usage:
    uv run python scripts/test_query_classification.py
"""

from __future__ import annotations

import asyncio
import json
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from loguru import logger

from workflows.dependencies import init_llm_client
from workflows.query_retrieval.models import (
    QueryType,
    RelationCandidate,
    RetrievalStrategy,
)
from workflows.query_retrieval.router import QueryRouter

# =============================================================================
# Test Dataset - 50 Vietnamese History Queries
# Difficulty: EASY (clear keywords), MEDIUM (some ambiguity), HARD (tricky)
# =============================================================================


@dataclass
class TestCase:
    """Test case with expected classification."""

    query: str
    expected_type: QueryType
    expected_use_relations: bool
    difficulty: str = "medium"


TEST_DATASET: list[TestCase] = [
    # ==========================================================================
    # FACTUAL - 8 queries (asking for specific facts)
    # ==========================================================================
    # EASY
    TestCase(
        query="Chiến thắng Điện Biên Phủ diễn ra năm nào?",
        expected_type=QueryType.FACTUAL,
        expected_use_relations=False,
        difficulty="easy",
    ),
    TestCase(
        query="Nhà Lý được thành lập vào năm nào?",
        expected_type=QueryType.FACTUAL,
        expected_use_relations=False,
        difficulty="easy",
    ),
    TestCase(
        query="Thủ đô của nước Đại Việt thời Lý là gì?",
        expected_type=QueryType.FACTUAL,
        expected_use_relations=False,
        difficulty="easy",
    ),
    # MEDIUM
    TestCase(
        query="Ngô Quyền đánh bại quân Nam Hán ở đâu?",
        expected_type=QueryType.FACTUAL,
        expected_use_relations=False,
        difficulty="medium",
    ),
    TestCase(
        query="Ai là người viết Bình Ngô đại cáo?",
        expected_type=QueryType.FACTUAL,
        expected_use_relations=False,
        difficulty="medium",
    ),
    TestCase(
        query="Cuộc khởi nghĩa Lam Sơn kéo dài bao nhiêu năm?",
        expected_type=QueryType.FACTUAL,
        expected_use_relations=False,
        difficulty="medium",
    ),
    # HARD
    TestCase(
        query="Vua nào đã dời đô từ Hoa Lư về Thăng Long?",
        expected_type=QueryType.FACTUAL,
        expected_use_relations=False,
        difficulty="hard",
    ),
    TestCase(
        query="Trận Bạch Đằng năm 938 do ai chỉ huy?",
        expected_type=QueryType.FACTUAL,
        expected_use_relations=False,
        difficulty="hard",
    ),
    # ==========================================================================
    # CONCEPT - 8 queries (asking for definitions/characteristics)
    # ==========================================================================
    # EASY
    TestCase(
        query="Chế độ quân chủ chuyên chế là gì?",
        expected_type=QueryType.CONCEPT,
        expected_use_relations=True,
        difficulty="easy",
    ),
    TestCase(
        query="Khoa cử là gì?",
        expected_type=QueryType.CONCEPT,
        expected_use_relations=True,
        difficulty="easy",
    ),
    TestCase(
        query="Thế nào là chế độ phong kiến?",
        expected_type=QueryType.CONCEPT,
        expected_use_relations=True,
        difficulty="easy",
    ),
    # MEDIUM
    TestCase(
        query="Văn hóa thời kỳ Bắc thuộc có đặc điểm gì?",
        expected_type=QueryType.CONCEPT,
        expected_use_relations=True,
        difficulty="medium",
    ),
    TestCase(
        query="Chính sách ngoại giao của nhà Nguyễn?",
        expected_type=QueryType.CONCEPT,
        expected_use_relations=True,
        difficulty="medium",
    ),
    TestCase(
        query="Những cải cách của nhà Lê sơ là gì?",
        expected_type=QueryType.CONCEPT,
        expected_use_relations=True,
        difficulty="medium",
    ),
    # HARD
    TestCase(
        query="Tình hình kinh tế thời Trần như thế nào?",
        expected_type=QueryType.CONCEPT,
        expected_use_relations=True,
        difficulty="hard",
    ),
    TestCase(
        query="Xã hội Việt Nam thế kỷ 19 có những biến đổi gì?",
        expected_type=QueryType.CONCEPT,
        expected_use_relations=True,
        difficulty="hard",
    ),
    # ==========================================================================
    # COMPARISON - 6 queries (comparing entities/events)
    # ==========================================================================
    # EASY
    TestCase(
        query="So sánh cải cách của Hồ Quý Ly và Lê Thánh Tông?",
        expected_type=QueryType.COMPARISON,
        expected_use_relations=True,
        difficulty="easy",
    ),
    TestCase(
        query="So sánh hai cuộc kháng chiến chống Pháp và chống Mỹ?",
        expected_type=QueryType.COMPARISON,
        expected_use_relations=True,
        difficulty="easy",
    ),
    # MEDIUM
    TestCase(
        query="Điểm giống và khác nhau giữa phong trào Đông Du và Duy Tân?",
        expected_type=QueryType.COMPARISON,
        expected_use_relations=True,
        difficulty="medium",
    ),
    TestCase(
        query="Sự khác biệt giữa kháng chiến chống Pháp và chống Mỹ?",
        expected_type=QueryType.COMPARISON,
        expected_use_relations=True,
        difficulty="medium",
    ),
    # HARD
    TestCase(
        query="Nhà Lý và nhà Trần có gì khác nhau về chính sách đối ngoại?",
        expected_type=QueryType.COMPARISON,
        expected_use_relations=True,
        difficulty="hard",
    ),
    TestCase(
        query="Khởi nghĩa Lam Sơn và khởi nghĩa Tây Sơn có điểm gì chung?",
        expected_type=QueryType.COMPARISON,
        expected_use_relations=True,
        difficulty="hard",
    ),
    # ==========================================================================
    # CAUSAL - 8 queries (cause/effect/significance)
    # ==========================================================================
    # EASY
    TestCase(
        query="Nguyên nhân thất bại của phong trào Cần Vương là gì?",
        expected_type=QueryType.CAUSAL,
        expected_use_relations=True,
        difficulty="easy",
    ),
    TestCase(
        query="Tại sao nhà Hồ lại sụp đổ?",
        expected_type=QueryType.CAUSAL,
        expected_use_relations=True,
        difficulty="easy",
    ),
    TestCase(
        query="Hậu quả của cuộc kháng chiến chống Nguyên Mông?",
        expected_type=QueryType.CAUSAL,
        expected_use_relations=True,
        difficulty="easy",
    ),
    # MEDIUM
    TestCase(
        query="Ý nghĩa lịch sử của Cách mạng tháng Tám?",
        expected_type=QueryType.CAUSAL,
        expected_use_relations=True,
        difficulty="medium",
    ),
    TestCase(
        query="Chiến thắng Điện Biên Phủ có ý nghĩa như thế nào?",
        expected_type=QueryType.CAUSAL,
        expected_use_relations=True,
        difficulty="medium",
    ),
    TestCase(
        query="Vì sao cuộc khởi nghĩa Hai Bà Trưng thất bại?",
        expected_type=QueryType.CAUSAL,
        expected_use_relations=True,
        difficulty="medium",
    ),
    # HARD
    TestCase(
        query="Điều gì dẫn đến sự sụp đổ của nhà Trần?",
        expected_type=QueryType.CAUSAL,
        expected_use_relations=True,
        difficulty="hard",
    ),
    TestCase(
        query="Những yếu tố nào quyết định thắng lợi của khởi nghĩa Lam Sơn?",
        expected_type=QueryType.CAUSAL,
        expected_use_relations=True,
        difficulty="hard",
    ),
    # ==========================================================================
    # FIGURE - 6 queries (about historical figures)
    # ==========================================================================
    # EASY
    TestCase(
        query="Trần Hưng Đạo có những đóng góp gì cho lịch sử dân tộc?",
        expected_type=QueryType.FIGURE,
        expected_use_relations=True,
        difficulty="easy",
    ),
    TestCase(
        query="Cuộc đời và sự nghiệp của Lê Lợi?",
        expected_type=QueryType.FIGURE,
        expected_use_relations=True,
        difficulty="easy",
    ),
    # MEDIUM
    TestCase(
        query="Hồ Chí Minh đã làm gì trong Cách mạng tháng Tám?",
        expected_type=QueryType.FIGURE,
        expected_use_relations=True,
        difficulty="medium",
    ),
    TestCase(
        query="Vai trò của Phan Bội Châu trong phong trào yêu nước?",
        expected_type=QueryType.FIGURE,
        expected_use_relations=True,
        difficulty="medium",
    ),
    # HARD
    TestCase(
        query="Nguyễn Trãi đã đóng góp gì cho văn học Việt Nam?",
        expected_type=QueryType.FIGURE,
        expected_use_relations=True,
        difficulty="hard",
    ),
    TestCase(
        query="Quang Trung đã làm gì để đánh bại quân Thanh?",
        expected_type=QueryType.FIGURE,
        expected_use_relations=True,
        difficulty="hard",
    ),
    # ==========================================================================
    # LESSON_SYNTHESIS - 4 queries (synthesizing lessons)
    # ==========================================================================
    # EASY
    TestCase(
        query="Bài học rút ra từ các cuộc kháng chiến chống ngoại xâm?",
        expected_type=QueryType.LESSON_SYNTHESIS,
        expected_use_relations=True,
        difficulty="easy",
    ),
    TestCase(
        query="Truyền thống yêu nước của dân tộc Việt Nam qua các thời kỳ?",
        expected_type=QueryType.LESSON_SYNTHESIS,
        expected_use_relations=True,
        difficulty="easy",
    ),
    # MEDIUM
    TestCase(
        query="Tinh thần đoàn kết dân tộc thể hiện như thế nào trong lịch sử?",
        expected_type=QueryType.LESSON_SYNTHESIS,
        expected_use_relations=True,
        difficulty="medium",
    ),
    # HARD
    TestCase(
        query="Những giá trị lịch sử từ các cuộc đấu tranh giành độc lập?",
        expected_type=QueryType.LESSON_SYNTHESIS,
        expected_use_relations=True,
        difficulty="hard",
    ),
    # ==========================================================================
    # STRUCTURAL - 4 queries (about document structure)
    # ==========================================================================
    # EASY
    TestCase(
        query="Bài 15 nói về những nội dung gì?",
        expected_type=QueryType.STRUCTURAL,
        expected_use_relations=False,
        difficulty="easy",
    ),
    TestCase(
        query="Chương 3 có bao nhiêu bài?",
        expected_type=QueryType.STRUCTURAL,
        expected_use_relations=False,
        difficulty="easy",
    ),
    # MEDIUM
    TestCase(
        query="Phần mở đầu của sách nói gì?",
        expected_type=QueryType.STRUCTURAL,
        expected_use_relations=False,
        difficulty="medium",
    ),
    TestCase(
        query="Mục 2 của bài 10 trình bày vấn đề gì?",
        expected_type=QueryType.STRUCTURAL,
        expected_use_relations=False,
        difficulty="medium",
    ),
    # ==========================================================================
    # OTHER - 6 queries (not related to history)
    # ==========================================================================
    # EASY
    TestCase(
        query="Xin chào, bạn là ai?",
        expected_type=QueryType.OTHER,
        expected_use_relations=False,
        difficulty="easy",
    ),
    TestCase(
        query="1 + 1 bằng mấy?",
        expected_type=QueryType.OTHER,
        expected_use_relations=False,
        difficulty="easy",
    ),
    TestCase(
        query="Hôm nay thời tiết thế nào?",
        expected_type=QueryType.OTHER,
        expected_use_relations=False,
        difficulty="easy",
    ),
    # MEDIUM
    TestCase(
        query="Bạn có thể giúp tôi viết code Python không?",
        expected_type=QueryType.OTHER,
        expected_use_relations=False,
        difficulty="medium",
    ),
    TestCase(
        query="Cho tôi hỏi về cách nấu phở?",
        expected_type=QueryType.OTHER,
        expected_use_relations=False,
        difficulty="medium",
    ),
    # HARD - mentions history but not Vietnamese
    TestCase(
        query="Lịch sử nước Pháp thế kỷ 18 như thế nào?",
        expected_type=QueryType.OTHER,
        expected_use_relations=False,
        difficulty="hard",
    ),
]


# =============================================================================
# Mock Relation Candidates (English descriptions)
# =============================================================================

MOCK_RELATION_CANDIDATES: list[RelationCandidate] = [
    RelationCandidate("CAUSED_BY", "Event A was caused by event B", 0.85),
    RelationCandidate("RESULTED_IN", "Event A resulted in event B", 0.82),
    RelationCandidate("PARTICIPATED_IN", "Figure participated in event", 0.80),
    RelationCandidate("LED_TO", "Event A led to event B", 0.78),
    RelationCandidate("DEFINES", "Defines a concept", 0.75),
    RelationCandidate("CONTRASTS_WITH", "Contrasts with another", 0.72),
    RelationCandidate("CONTRIBUTED_TO", "Contributed to event", 0.70),
    RelationCandidate("SIMILAR_TO", "Similar to another", 0.68),
    RelationCandidate("PART_OF", "Is part of", 0.65),
    RelationCandidate("ABOUT", "About same topic", 0.62),
]


# =============================================================================
# Evaluation Result
# =============================================================================


@dataclass
class EvaluationResult:
    """Single query evaluation result."""

    query: str
    expected_type: QueryType
    predicted_type: QueryType
    type_correct: bool

    expected_use_relations: bool
    predicted_use_relations: bool
    relations_correct: bool

    predicted_strategy: RetrievalStrategy
    confidence: float
    reasoning: str
    latency_ms: float
    difficulty: str

    def __str__(self) -> str:
        status = "✓" if self.type_correct else "✗"
        rel_status = "✓" if self.relations_correct else "✗"
        return (
            f"{status} Type: {self.expected_type.value} → {self.predicted_type.value} | "
            f"{rel_status} Rel | {self.latency_ms:.0f}ms"
        )


@dataclass
class EvaluationReport:
    """Full evaluation report."""

    results: list[EvaluationResult]
    total_queries: int
    type_accuracy: float
    relations_accuracy: float
    avg_latency_ms: float
    total_time_s: float

    type_breakdown: dict[str, dict[str, float]] = field(default_factory=dict)
    difficulty_breakdown: dict[str, dict[str, float]] = field(default_factory=dict)

    def to_markdown(self) -> str:
        """Generate markdown report."""
        lines = [
            "# Query Classification Evaluation Report",
            f"\n**Date**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"\n**Total Queries**: {self.total_queries}",
            f"\n**Total Time**: {self.total_time_s:.1f}s",
            "",
            "## Summary Metrics",
            "",
            "| Metric | Score |",
            "|--------|-------|",
            f"| **Query Type Accuracy** | {self.type_accuracy:.1%} |",
            f"| **Use Relations Accuracy** | {self.relations_accuracy:.1%} |",
            f"| **Average Latency** | {self.avg_latency_ms:.0f}ms |",
            "",
            "## Per-Type Breakdown",
            "",
            "| Query Type | Total | Correct | Accuracy |",
            "|------------|-------|---------|----------|",
        ]

        for qtype, stats in self.type_breakdown.items():
            lines.append(
                f"| {qtype} | {stats['total']:.0f} | {stats['correct']:.0f} | {stats['accuracy']:.1%} |"
            )

        lines.extend(
            [
                "",
                "## Per-Difficulty Breakdown",
                "",
                "| Difficulty | Total | Correct | Accuracy |",
                "|------------|-------|---------|----------|",
            ]
        )

        for diff, stats in self.difficulty_breakdown.items():
            lines.append(
                f"| {diff} | {stats['total']:.0f} | {stats['correct']:.0f} | {stats['accuracy']:.1%} |"
            )

        lines.extend(
            [
                "",
                "## Error Analysis",
                "",
            ]
        )

        errors = [r for r in self.results if not r.type_correct]
        if errors:
            lines.append("### Misclassified Queries")
            lines.append("")
            for r in errors:
                lines.append(f"- **Query**: {r.query}")
                lines.append(
                    f"  - Expected: `{r.expected_type.value}` → Got: `{r.predicted_type.value}`"
                )
                lines.append(f"  - Difficulty: {r.difficulty}")
                lines.append(f"  - Reasoning: {r.reasoning}")
                lines.append("")
        else:
            lines.append("*No misclassifications!*")

        return "\n".join(lines)


# =============================================================================
# Evaluator
# =============================================================================


class QueryClassificationEvaluator:
    """Evaluates QueryRouter performance."""

    def __init__(self, router: QueryRouter):
        self.router = router
        self._log = logger.bind(component="evaluator")

    async def evaluate_single(
        self, test_case: TestCase, candidates: list[RelationCandidate]
    ) -> EvaluationResult:
        """Evaluate a single test case."""
        import time

        start = time.time()
        classification = await self.router.route(test_case.query, candidates)
        latency_ms = (time.time() - start) * 1000

        type_correct = classification.query_type == test_case.expected_type
        relations_correct = classification.use_relations == test_case.expected_use_relations

        return EvaluationResult(
            query=test_case.query,
            expected_type=test_case.expected_type,
            predicted_type=classification.query_type,
            type_correct=type_correct,
            expected_use_relations=test_case.expected_use_relations,
            predicted_use_relations=classification.use_relations,
            relations_correct=relations_correct,
            predicted_strategy=classification.strategy,
            confidence=classification.confidence,
            reasoning=classification.reasoning,
            latency_ms=latency_ms,
            difficulty=test_case.difficulty,
        )

    async def evaluate_all(
        self, test_cases: list[TestCase], candidates: list[RelationCandidate]
    ) -> EvaluationReport:
        """Evaluate all test cases."""
        import time

        total_start = time.time()
        results: list[EvaluationResult] = []

        for i, tc in enumerate(test_cases, 1):
            self._log.info(f"Evaluating {i}/{len(test_cases)}: {tc.query[:40]}...")
            result = await self.evaluate_single(tc, candidates)
            results.append(result)
            print(f"  {result}")

        total_time_s = time.time() - total_start

        type_correct = sum(1 for r in results if r.type_correct)
        relations_correct = sum(1 for r in results if r.relations_correct)
        latency_sum = sum(r.latency_ms for r in results)

        # Per-type breakdown
        type_breakdown: dict[str, dict[str, float]] = {}
        for qtype in QueryType:
            type_results = [r for r in results if r.expected_type == qtype]
            if type_results:
                correct = sum(1 for r in type_results if r.type_correct)
                type_breakdown[qtype.value] = {
                    "total": len(type_results),
                    "correct": correct,
                    "accuracy": correct / len(type_results),
                }

        # Per-difficulty breakdown
        difficulty_breakdown: dict[str, dict[str, float]] = {}
        for diff in ["easy", "medium", "hard"]:
            diff_results = [r for r in results if r.difficulty == diff]
            if diff_results:
                correct = sum(1 for r in diff_results if r.type_correct)
                difficulty_breakdown[diff] = {
                    "total": len(diff_results),
                    "correct": correct,
                    "accuracy": correct / len(diff_results),
                }

        return EvaluationReport(
            results=results,
            total_queries=len(results),
            type_accuracy=type_correct / len(results),
            relations_accuracy=relations_correct / len(results),
            avg_latency_ms=latency_sum / len(results),
            total_time_s=total_time_s,
            type_breakdown=type_breakdown,
            difficulty_breakdown=difficulty_breakdown,
        )


# =============================================================================
# Main
# =============================================================================


async def main():
    """Run the evaluation."""
    print("=" * 70)
    print("Query Classification Evaluation (50 queries)")
    print("=" * 70)
    print()

    print("Initializing LLM client...")
    try:
        llm_client = init_llm_client()
        print("  ✓ LLM client initialized")
    except Exception as e:
        print(f"  ✗ Failed to initialize LLM client: {e}")
        print("\nMake sure environment variables are set:")
        print("  - LLM_API_KEY")
        print("  - LLM_API_BASE")
        print("  - LLM_CHAT_MODEL")
        return

    router = QueryRouter(llm_client)
    print("  ✓ QueryRouter created")

    evaluator = QueryClassificationEvaluator(router)
    print("  ✓ Evaluator ready")
    print()

    print(f"Running evaluation on {len(TEST_DATASET)} queries...")
    print("-" * 70)

    report = await evaluator.evaluate_all(TEST_DATASET, MOCK_RELATION_CANDIDATES)

    print("-" * 70)
    print()
    print("=" * 70)
    print("EVALUATION SUMMARY")
    print("=" * 70)
    print(
        f"  Query Type Accuracy:    {report.type_accuracy:.1%} ({int(report.type_accuracy * report.total_queries)}/{report.total_queries})"
    )
    print(f"  Use Relations Accuracy: {report.relations_accuracy:.1%}")
    print(f"  Average Latency:        {report.avg_latency_ms:.0f}ms")
    print(f"  Total Time:             {report.total_time_s:.1f}s")
    print()

    print("Per-Type Breakdown:")
    for qtype, stats in report.type_breakdown.items():
        print(
            f"  {qtype:20s}: {stats['correct']:.0f}/{stats['total']:.0f} ({stats['accuracy']:.1%})"
        )
    print()

    print("Per-Difficulty Breakdown:")
    for diff, stats in report.difficulty_breakdown.items():
        print(
            f"  {diff:20s}: {stats['correct']:.0f}/{stats['total']:.0f} ({stats['accuracy']:.1%})"
        )
    print()

    # Save report
    report_path = project_root / "research" / "query_classification_report.md"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(report.to_markdown())
    print(f"Report saved to: {report_path}")

    # Save JSON
    json_path = project_root / "research" / "query_classification_results.json"
    json_data = {
        "summary": {
            "total_queries": report.total_queries,
            "type_accuracy": report.type_accuracy,
            "relations_accuracy": report.relations_accuracy,
            "avg_latency_ms": report.avg_latency_ms,
            "total_time_s": report.total_time_s,
        },
        "type_breakdown": report.type_breakdown,
        "difficulty_breakdown": report.difficulty_breakdown,
        "results": [
            {
                "query": r.query,
                "expected_type": r.expected_type.value,
                "predicted_type": r.predicted_type.value,
                "type_correct": r.type_correct,
                "relations_correct": r.relations_correct,
                "confidence": r.confidence,
                "reasoning": r.reasoning,
                "latency_ms": r.latency_ms,
                "difficulty": r.difficulty,
            }
            for r in report.results
        ],
    }
    json_path.write_text(json.dumps(json_data, ensure_ascii=False, indent=2))
    print(f"JSON results saved to: {json_path}")


if __name__ == "__main__":
    asyncio.run(main())
