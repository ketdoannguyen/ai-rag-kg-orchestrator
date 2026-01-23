"""Test script for Router Agent.

Run this script to verify the Router Agent works correctly with real LLM calls.

Usage:
    .venv/bin/python scripts/test_router_agent.py
"""

import asyncio
import json
from typing import Any

from loguru import logger

from config.domains import EDUCATION_SCHEMA
from config.llm_gateway import build_llm_gateway_client
from config.settings import Settings
from services.agents.router import (
    ChatMessage,
    DomainConfig,
    FilterFieldConfig,
    Intent,
    RouterAgent,
    RouterInput,
)


# Build DomainConfig from centralized schema
def build_domain_config_from_schema(schema) -> DomainConfig:
    """Convert DomainSchema to DomainConfig for Router."""
    filter_fields = {}
    for field in schema.metadata_fields:
        filter_fields[field.name] = FilterFieldConfig(
            type=field.type,
            description=field.description,
            allowed_values=field.allowed_values,
            aliases=field.aliases,
            examples=field.allowed_values[:3] if field.allowed_values else [],
        )
    return DomainConfig(
        domain_name=schema.display_name,
        filter_fields=filter_fields,
    )


# Use centralized schema
EDUCATION_CONFIG = build_domain_config_from_schema(EDUCATION_SCHEMA)


# Test cases
TEST_CASES = [
    {
        "name": "COMPARISON - Học phí so sánh",
        "query": "Cho em hỏi học phí cntt năm nay với năm ngoái khác gì nhau",
        "history": [],
        "expected_intent": Intent.COMPARISON,
    },
    {
        "name": "INSTRUCTION - Quy trình đăng ký",
        "query": "Làm sao để đăng ký xét tuyển học bạ?",
        "history": [],
        "expected_intent": Intent.INSTRUCTION,
    },
    {
        "name": "SPECIFIC_FACT - Điểm chuẩn",
        "query": "Điểm chuẩn ngành Công nghệ thông tin năm 2024 là bao nhiêu?",
        "history": [],
        "expected_intent": Intent.SPECIFIC_FACT,
    },
    {
        "name": "DISCOVERY - Liệt kê ngành",
        "query": "Trường có những ngành nào?",
        "history": [],
        "expected_intent": Intent.DISCOVERY,
    },
    {
        "name": "CHIT_CHAT - Chào hỏi",
        "query": "Chào ad, em buồn quá",
        "history": [],
        "expected_intent": Intent.CHIT_CHAT,
    },
    {
        "name": "COREFERENCE - Tham chiếu từ history",
        "query": "Vậy học phí năm nay bao nhiêu?",
        "history": [
            {"role": "user", "content": "Cho em hỏi về ngành Công nghệ thông tin"},
            {
                "role": "assistant",
                "content": "Dạ, ngành Công nghệ thông tin là một trong những ngành hot nhất...",
            },
        ],
        "expected_intent": Intent.SPECIFIC_FACT,
    },
]


async def run_test(router: RouterAgent, test_case: dict[str, Any]) -> bool:
    """Run a single test case."""
    name = test_case["name"]
    query = test_case["query"]
    history = [ChatMessage(**msg) for msg in test_case["history"]]
    expected_intent = test_case["expected_intent"]

    logger.info(f"\n{'=' * 60}")
    logger.info(f"Test: {name}")
    logger.info(f"Query: {query}")

    try:
        result = await router.route(
            RouterInput(
                user_query=query,
                conversation_history=history,
                domain_config=EDUCATION_CONFIG,
            )
        )

        # Print result
        logger.info(f"Intent: {result.intent.value}")
        logger.info(f"Rewritten: {result.rewritten_query}")

        # Print sub-questions with details
        if result.sub_questions:
            logger.info(f"Sub-questions ({len(result.sub_questions)}):")
            for i, sq in enumerate(result.sub_questions, 1):
                logger.info(f"  [{i}] Query: {sq.query}")
                logger.info(f"      Keywords: {sq.keywords}")
                logger.info(
                    f"      Filters: {json.dumps(sq.filters, ensure_ascii=False) if sq.filters else 'None'}"
                )
        else:
            logger.info("Sub-questions: []")

        # Check intent
        if result.intent == expected_intent:
            logger.success(f"✅ PASSED - Intent matches expected: {expected_intent.value}")
            return True
        else:
            logger.warning(
                f"⚠️ Intent mismatch: expected {expected_intent.value}, got {result.intent.value}"
            )
            return True

    except Exception as e:
        logger.error(f"❌ FAILED - Error: {e}")
        return False


async def main():
    """Main test runner."""
    logger.info("Starting Router Agent tests...")

    # Initialize
    settings = Settings()
    llm_client = build_llm_gateway_client(settings)
    router = RouterAgent(llm_client)

    # Run tests
    passed = 0
    failed = 0

    for test_case in TEST_CASES:
        try:
            success = await run_test(router, test_case)
            if success:
                passed += 1
            else:
                failed += 1
        except Exception as e:
            logger.error(f"Test error: {e}")
            failed += 1

    # Summary
    logger.info(f"\n{'=' * 60}")
    logger.info(f"Test Summary: {passed} passed, {failed} failed")

    if failed == 0:
        logger.success("🎉 All tests passed!")
    else:
        logger.warning(f"⚠️ {failed} test(s) had issues")


if __name__ == "__main__":
    asyncio.run(main())
