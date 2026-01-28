"""Query router with LLM-based classification.

This module handles:
1. Keyword pre-classification for high-confidence patterns
2. LLM-based query classification for ambiguous cases
3. Relation selection from candidates
4. Smart fallback when LLM fails

Optimized for weak/cheap LLMs with clear English prompts.
Note: Filter extraction removed - BM25 hybrid retrieval handles keyword matching.
"""

from __future__ import annotations

import json
import re
import time
from typing import TYPE_CHECKING

from loguru import logger

from .models import (
    QueryClassification,
    QueryType,
    RelationCandidate,
    RetrievalStrategy,
)

if TYPE_CHECKING:
    from llm_gateway_client.client import AIGatewayClient


# =============================================================================
# Keyword Rules for Pre-Classification
# =============================================================================

# High-confidence patterns - can skip LLM for these (strict matching)
KEYWORD_RULES: dict[str, list[str]] = {
    "comparison": [
        "so sánh",
        "khác nhau",
        "giống nhau",
        "điểm khác",
        "điểm giống",
        "khác biệt",
        "tương đồng",
    ],
    "causal": [
        "tại sao",
        "vì sao",
        "nguyên nhân",
        "hậu quả",
        "kết quả",
        "ý nghĩa lịch sử",
        "dẫn đến",
        "gây ra",
    ],
    "lesson_synthesis": [
        "bài học",
        "truyền thống",
        "qua các thời kỳ",
        "qua các giai đoạn",
        "xuyên suốt",
        "tổng kết",
    ],
    "structural": [
        "bài số",
        "chương số",
        "mục số",
        "phần mở đầu",
        "nội dung bài",
        "nội dung chương",
        "bao nhiêu bài",
        "bao nhiêu chương",
    ],
    "other": [
        "xin chào",
        "hello",
        "hi ",
        "thời tiết",
        "bằng mấy",
        "cộng",
        "trừ",
        "nhân",
        "chia",
        "bạn là ai",
        "tên bạn",
    ],
}

# Looser keyword rules for fallback matching (used when LLM fails)
FALLBACK_KEYWORD_RULES: dict[str, list[str]] = {
    "comparison": ["so sánh", "khác", "giống"],
    "causal": ["tại sao", "nguyên nhân", "hậu quả", "ý nghĩa"],
    "lesson_synthesis": ["bài học", "truyền thống", "qua các"],
    "concept": ["là gì", "thế nào"],
    "figure": ["đóng góp", "cuộc đời", "sự nghiệp"],
}


# =============================================================================
# Prompt Builder
# =============================================================================


def build_router_prompt(query: str, candidate_relations: str) -> str:
    """Build router prompt for query classification.

    Args:
        query: User query
        candidate_relations: Formatted relation candidates

    Returns:
        Complete prompt for LLM
    """
    return f"""Classify Vietnamese History query.

## Query
{query}

## Relations
{candidate_relations}

## Types
| Type | Description | use_relations |
|------|-------------|---------------|
| FACTUAL | Specific fact: name, date, number | false |
| CONCEPT | Definition, characteristics ("là gì", "như thế nào") | true |
| COMPARISON | Compare 2+ entities ("so sánh", "khác") | true |
| CAUSAL | Cause/effect ("tại sao", "nguyên nhân") | true |
| LESSON_SYNTHESIS | Lessons across events ("bài học", "qua các thời kỳ") | true |
| STRUCTURAL | Document structure: chapter, section | false |
| FIGURE | Person's life/contributions/actions | true |
| OTHER | Not Vietnamese history | false |

## Key Rules
- "[Person] đã làm gì/đóng góp gì" → FIGURE (not FACTUAL)
- "Tình hình/Chính sách [X] thời [Y]" → CONCEPT (not FACTUAL)
- "Ai là người..." asking NAME → FACTUAL

## Output JSON only:
```json
{{
  "query_type": "factual",
  "confidence": 0.9,
  "use_relations": false,
  "selected_relations": [],
  "strategy": "content_only",
  "reasoning": "Brief"
}}
```"""


class QueryRouter:
    """LLM-based query router with keyword pre-classification.

    Workflow:
    1. Try keyword pre-classification for high-confidence patterns
    2. If ambiguous, call LLM to classify query + select relations
    3. Use smart fallback if LLM fails
    4. Return QueryClassification

    Optimized for weak/cheap LLMs with reduced token usage.
    Note: Filter extraction removed - BM25 handles keyword matching implicitly.
    """

    def __init__(
        self,
        llm_client: AIGatewayClient,
        model: str | None = None,
    ):
        self.llm = llm_client
        self.model = model
        self._log = logger.bind(component="query_router")

    async def route(
        self,
        query: str,
        candidate_relations: list[RelationCandidate],
    ) -> QueryClassification:
        """Route query and select relations.

        Args:
            query: User query
            candidate_relations: Pre-filtered relation candidates

        Returns:
            QueryClassification with routing info
        """
        start_time = time.time()

        # Step 1: Try keyword pre-classification for high-confidence cases
        preclassified = self._keyword_preclassify(query)
        if preclassified is not None:
            elapsed_ms = (time.time() - start_time) * 1000
            self._log.info(
                "Query pre-classified by keywords",
                query_type=preclassified.value,
                elapsed_ms=f"{elapsed_ms:.0f}",
            )
            return self._build_classification_from_type(
                query_type=preclassified,
                candidates_count=len(candidate_relations),
                candidate_relations=candidate_relations,
            )

        # Step 2: Call LLM for ambiguous cases
        candidates_text = self._format_candidates(candidate_relations)
        prompt = build_router_prompt(
            query=query,
            candidate_relations=candidates_text,
        )

        from llm_gateway_client.models.request import (
            GatewayChatTurn,
            GatewayInvokeRequest,
        )

        request = GatewayInvokeRequest(
            model=self.model or self._get_default_model(),
            messages=[
                GatewayChatTurn(
                    role="system",
                    content="You are a query router for Vietnamese History RAG. Output JSON only.",
                ),
                GatewayChatTurn(role="user", content=prompt),
            ],
            provider_options={"temperature": 0.1},
        )

        response = await self.llm.invoke(request)
        elapsed_ms = (time.time() - start_time) * 1000

        # Step 3: Parse response or use smart fallback
        if not response.message or not response.message.content:
            self._log.warning("Empty LLM response, using smart fallback")
            return self._smart_fallback(query, len(candidate_relations))

        try:
            data = self._extract_json(response.message.content)
            classification = self._parse_classification(data, len(candidate_relations))
            self._log.info(
                "Query classified by LLM",
                query_type=classification.query_type.value,
                use_relations=classification.use_relations,
                elapsed_ms=f"{elapsed_ms:.0f}",
            )
            return classification
        except Exception as e:
            self._log.warning(f"Failed to parse router response: {e}")
            return self._smart_fallback(query, len(candidate_relations))

    def _keyword_preclassify(self, query: str) -> QueryType | None:
        """Pre-classify query using keyword patterns.

        Returns QueryType if high-confidence match found, None otherwise.
        """
        query_lower = query.lower()

        # Check each keyword rule
        for query_type_str, keywords in KEYWORD_RULES.items():
            for keyword in keywords:
                if keyword in query_lower:
                    try:
                        return QueryType(query_type_str)
                    except ValueError:
                        continue

        return None

    def _build_classification_from_type(
        self,
        query_type: QueryType,
        candidates_count: int,
        candidate_relations: list[RelationCandidate] | None = None,
    ) -> QueryClassification:
        """Build classification from pre-classified type."""
        # Determine use_relations and strategy based on type
        use_relations = query_type in [
            QueryType.COMPARISON,
            QueryType.CAUSAL,
            QueryType.LESSON_SYNTHESIS,
            QueryType.CONCEPT,
            QueryType.FIGURE,
        ]

        if query_type == QueryType.OTHER:
            strategy = RetrievalStrategy.NO_RETRIEVAL
        elif query_type == QueryType.STRUCTURAL:
            strategy = RetrievalStrategy.STRUCTURE_FIRST
        elif use_relations:
            strategy = RetrievalStrategy.CONTENT_WITH_RELATIONS
        else:
            strategy = RetrievalStrategy.CONTENT_ONLY

        # Select relations if needed
        selected_relations: list[str] = []
        if use_relations and candidate_relations:
            # Take top 3 most relevant relations
            selected_relations = [r.relation_type for r in candidate_relations[:3]]

        return QueryClassification(
            query_type=query_type,
            confidence=0.85,  # High confidence for keyword matches
            strategy=strategy,
            use_relations=use_relations,
            selected_relations=selected_relations,
            requires_structure_tier=query_type == QueryType.STRUCTURAL,
            requires_content_tier=query_type != QueryType.OTHER,
            min_sources=1,
            max_sources=5,
            reasoning=f"Pre-classified by keyword match as {query_type.value}",
            candidate_relations_considered=candidates_count,
        )

    def _smart_fallback(
        self,
        query: str,
        candidates_count: int,
    ) -> QueryClassification:
        """Smart fallback using keyword detection instead of always returning FACTUAL."""
        detected_type = _match_fallback_keywords(query)

        return self._build_classification_from_type(
            query_type=detected_type,
            candidates_count=candidates_count,
        )


def _match_fallback_keywords(query: str) -> QueryType:
    """Match query against fallback keyword rules (looser matching).

    Args:
        query: User query text

    Returns:
        Detected QueryType, defaults to FACTUAL if no match
    """
    query_lower = query.lower()

    for query_type_str, keywords in FALLBACK_KEYWORD_RULES.items():
        if any(kw in query_lower for kw in keywords):
            try:
                return QueryType(query_type_str)
            except ValueError:
                continue

    return QueryType.FACTUAL

    def _format_candidates(self, candidates: list[RelationCandidate]) -> str:
        """Format relation candidates for LLM prompt."""
        if not candidates:
            return "(No relation candidates available)"

        lines = []
        for i, c in enumerate(candidates, 1):
            lines.append(f"{i}. {c.relation_type} ({c.similarity:.2f}): {c.description}")

        return "\n".join(lines)

    def _extract_json(self, text: str) -> dict:
        """Extract JSON from LLM response."""
        json_match = re.search(r"\{[\s\S]*\}", text)
        if json_match:
            return json.loads(json_match.group())
        raise ValueError("No JSON found in response")

    def _parse_classification(
        self,
        data: dict,
        candidates_count: int,
    ) -> QueryClassification:
        """Parse dict to QueryClassification."""
        # Parse query type
        query_type_str = data.get("query_type", "factual").lower()
        try:
            query_type = QueryType(query_type_str)
        except ValueError:
            query_type = QueryType.FACTUAL

        # Parse strategy
        strategy_str = data.get("strategy", "content_only")
        try:
            strategy = RetrievalStrategy(strategy_str)
        except ValueError:
            strategy = RetrievalStrategy.CONTENT_ONLY

        return QueryClassification(
            query_type=query_type,
            confidence=data.get("confidence", 0.8),
            strategy=strategy,
            use_relations=data.get("use_relations", False),
            selected_relations=data.get("selected_relations", []),
            requires_structure_tier=data.get("requires_structure_tier", False),
            requires_content_tier=query_type != QueryType.OTHER,
            min_sources=data.get("min_sources", 1),
            max_sources=data.get("max_sources", 5),
            reasoning=data.get("reasoning", ""),
            candidate_relations_considered=candidates_count,
        )

    def _get_default_model(self) -> str:
        """Get default weak model for routing."""
        from config.settings import get_settings

        settings = get_settings()
        return getattr(settings, "LLM_WEAK_MODEL", None) or settings.LLM_CHAT_MODEL
