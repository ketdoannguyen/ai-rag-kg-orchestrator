"""Integration test for new relation validation flow.

Tests the complete flow with REAL Strong LLM calls:
1. Mock weak LLM output → new relation proposals
2. Strong LLM (gemini-3-pro-preview) verifies + extracts full info
3. Embedding + similarity check
4. Store to DB

Run with: python -m pytest tests/integration/test_new_relation_flow_integration.py -v -s
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass

from loguru import logger

# Test samples: Each represents a weak LLM output proposing a NEW relation
TEST_SAMPLES = [
    # Kinship relations
    {
        "id": 1,
        "node_a_text": "Lý Thường Kiệt sinh năm 1019, là con trai của Lý Công Uẩn.",
        "node_b_text": "Lý Công Uẩn (974-1028) là vị vua sáng lập nhà Lý.",
        "proposed_relation": "FATHER_OF",
        "summary": "Quan hệ cha con trong hoàng tộc",
        "expected_families": ["kinship"],
    },
    {
        "id": 2,
        "node_a_text": "Trần Thủ Độ kết hôn với Trần Thị Dung, người trước đây là hoàng hậu nhà Lý.",
        "node_b_text": "Trần Thị Dung là vợ cả của Trần Thủ Độ.",
        "proposed_relation": "MARRIED_TO",
        "summary": "Quan hệ vợ chồng",
        "expected_families": ["kinship"],
    },
    # Geographic relations
    {
        "id": 3,
        "node_a_text": "Thăng Long là kinh đô của nhà Lý, nằm ở phía Bắc.",
        "node_b_text": "Đại Việt là quốc gia do nhà Lý cai trị từ năm 1010.",
        "proposed_relation": "CAPITAL_OF",
        "summary": "Thăng Long là thủ đô của Đại Việt",
        "expected_families": ["geographic"],
    },
    {
        "id": 4,
        "node_a_text": "Chiêm Thành nằm ở phía Nam Đại Việt, tiếp giáp vùng Thanh Hóa.",
        "node_b_text": "Đại Việt mở rộng lãnh thổ về phía Nam qua các cuộc chiến với Chiêm Thành.",
        "proposed_relation": "BORDERS",
        "summary": "Hai quốc gia có chung đường biên giới",
        "expected_families": ["geographic"],
    },
    # Military/Political relations
    {
        "id": 5,
        "node_a_text": "Quân Tống thất bại hoàn toàn tại trận Như Nguyệt năm 1077.",
        "node_b_text": "Lý Thường Kiệt chỉ huy quân Đại Việt đánh tan quân Tống.",
        "proposed_relation": "DEFEATED_BY",
        "summary": "Quân Tống bị đánh bại bởi Lý Thường Kiệt",
        "expected_families": ["participatory", "causal"],
    },
    {
        "id": 6,
        "node_a_text": "Trần Hưng Đạo được phong làm Quốc Công Tiết Chế, tổng chỉ huy quân đội.",
        "node_b_text": "Nhà Trần giao quyền điều binh cho Trần Hưng Đạo chống Nguyên Mông.",
        "proposed_relation": "COMMANDED_BY",
        "summary": "Quân đội nhà Trần do Trần Hưng Đạo chỉ huy",
        "expected_families": ["participatory", "hierarchical"],
    },
    # Temporal/Succession relations
    {
        "id": 7,
        "node_a_text": "Nhà Trần thay thế nhà Lý vào năm 1225 sau khi Lý Chiêu Hoàng nhường ngôi.",
        "node_b_text": "Nhà Lý kết thúc sau 216 năm cai trị (1009-1225).",
        "proposed_relation": "SUCCEEDED_BY",
        "summary": "Nhà Trần kế thừa nhà Lý",
        "expected_families": ["temporal", "hierarchical"],
    },
    # Cultural/Religious relations
    {
        "id": 8,
        "node_a_text": "Chùa Một Cột được xây dựng dưới thời vua Lý Thái Tông năm 1049.",
        "node_b_text": "Vua Lý Thái Tông mộng thấy Phật Quan Âm nên cho xây chùa.",
        "proposed_relation": "FOUNDED_BY",
        "summary": "Chùa Một Cột do Lý Thái Tông sáng lập",
        "expected_families": ["participatory", "causal"],
    },
    # Scholarly/Institutional relations
    {
        "id": 9,
        "node_a_text": "Quốc Tử Giám là trường đại học đầu tiên của Việt Nam, thành lập năm 1076.",
        "node_b_text": "Vua Lý Nhân Tông lập Quốc Tử Giám để đào tạo nhân tài cho triều đình.",
        "proposed_relation": "ESTABLISHED_BY",
        "summary": "Quốc Tử Giám được thành lập bởi Lý Nhân Tông",
        "expected_families": ["participatory", "causal"],
    },
    # Lineage/Dynasty relations
    {
        "id": 10,
        "node_a_text": "Lý Thái Tổ là vị vua đầu tiên của nhà Lý, húy là Lý Công Uẩn.",
        "node_b_text": "Nhà Lý (1009-1225) là triều đại phong kiến thứ ba của Việt Nam.",
        "proposed_relation": "FOUNDER_OF",
        "summary": "Lý Thái Tổ là người sáng lập nhà Lý",
        "expected_families": ["participatory", "hierarchical"],
    },
]


@dataclass
class TestResult:
    """Result from testing one sample."""

    sample_id: int
    proposed_relation: str
    strong_accepted: bool
    strong_reasoning: str
    final_relation: str
    final_description: str
    families: list[str]
    is_symmetric: bool
    is_transitive: bool
    keywords: list[str]
    similarity_merge: bool
    similarity_score: float | None
    stored_to_db: bool
    error: str | None = None


async def run_single_test(
    sample: dict,
    validator,
    reset_db: bool = False,
) -> TestResult:
    """Run a single test sample through the validation flow."""
    from workflows.extract_relations.relation_validator import NodeInfo

    node_a = NodeInfo(
        node_id=f"node_a_{sample['id']}",
        node_type="paragraph",
        text=sample["node_a_text"],
    )
    node_b = NodeInfo(
        node_id=f"node_b_{sample['id']}",
        node_type="paragraph",
        text=sample["node_b_text"],
    )

    try:
        result = await validator.validate_new_relation(
            node_a=node_a,
            node_b=node_b,
            proposed_relation=sample["proposed_relation"],
            summary=sample["summary"],
            tier="tier_2_content",
        )

        return TestResult(
            sample_id=sample["id"],
            proposed_relation=sample["proposed_relation"],
            strong_accepted=not result.merged_with_existing,  # If not merged, Strong accepted/proposed
            strong_reasoning=result.reasoning,
            final_relation=result.final_relation_type,
            final_description=result.final_description or "",
            families=result.families or [],
            is_symmetric=False,  # From Strong output
            is_transitive=False,  # From Strong output
            keywords=[],  # From Strong output
            similarity_merge=result.merged_with_existing,
            similarity_score=result.similarity_score,
            stored_to_db=result.is_valid and not result.merged_with_existing,
            error=None,
        )
    except Exception as e:
        logger.error(f"Test {sample['id']} failed: {e}")
        return TestResult(
            sample_id=sample["id"],
            proposed_relation=sample["proposed_relation"],
            strong_accepted=False,
            strong_reasoning="",
            final_relation="",
            final_description="",
            families=[],
            is_symmetric=False,
            is_transitive=False,
            keywords=[],
            similarity_merge=False,
            similarity_score=None,
            stored_to_db=False,
            error=str(e),
        )


async def run_all_tests(clear_db: bool = True) -> list[TestResult]:
    """Run all 10 test samples."""
    from config.extraction import get_extraction_settings
    from config.langfuse import configure_langfuse
    from config.llm_gateway import build_llm_gateway_client
    from config.settings import get_settings

    from workflows.extract_relations.relation_validator import NewRelationValidator

    settings = get_settings()
    extraction_settings = get_extraction_settings()

    # Configure langfuse for tracing
    configure_langfuse(settings)

    # Build LLM client
    llm_client = build_llm_gateway_client(settings)

    # Setup DB connection
    db = None
    ontology_handler = None

    try:
        from arango import ArangoClient
        from arango_client.handlers import OntologyHandler

        client = ArangoClient(hosts=settings.ARANGO_URL)
        db = client.db(
            settings.ARANGO_DATABASE,
            username=settings.ARANGO_USER,
            password=settings.ARANGO_PASSWORD,
        )

        # Clear LLM-proposed relations if requested
        if clear_db and db.has_collection("ontology_relations"):
            # Only delete non-builtin relations
            db.aql.execute(
                """
                FOR r IN ontology_relations
                    FILTER r.is_builtin == false
                    REMOVE r IN ontology_relations
                """
            )
            logger.info("Cleared LLM-proposed relations from DB")

        ontology_handler = OntologyHandler(db)

    except Exception as e:
        logger.warning(f"Could not connect to DB: {e}")
        logger.info("Running tests without DB (no storage)")

    # Create validator
    validator = NewRelationValidator(
        settings=extraction_settings,
        db=db,
        ontology_handler=ontology_handler,
        llm_client=llm_client,
    )

    logger.info(f"Strong LLM model: {extraction_settings.EXTRACTION_STRONG_MODEL}")
    logger.info(f"Merge threshold: {extraction_settings.EXTRACTION_MERGE_THRESHOLD}")
    logger.info(f"Running {len(TEST_SAMPLES)} test samples...")

    results = []
    for sample in TEST_SAMPLES:
        logger.info(f"\n{'=' * 60}")
        logger.info(f"Test {sample['id']}: {sample['proposed_relation']}")
        logger.info(f"{'=' * 60}")

        result = await run_single_test(sample, validator)
        results.append(result)

        # Log result
        if result.error:
            logger.error(f"  ERROR: {result.error}")
        else:
            logger.info(f"  Proposed: {result.proposed_relation}")
            logger.info(f"  Final: {result.final_relation}")
            logger.info(f"  Families: {result.families}")
            logger.info(f"  Reasoning: {result.strong_reasoning[:100]}...")
            if result.similarity_merge:
                logger.info(f"  MERGED with score: {result.similarity_score:.2f}")
            else:
                logger.info(f"  STORED to DB: {result.stored_to_db}")

        # Small delay between tests
        await asyncio.sleep(1)

    return results


def print_summary(results: list[TestResult]) -> None:
    """Print summary of all test results."""
    print("\n" + "=" * 80)
    print("SUMMARY: New Relation Validation Flow Test")
    print("=" * 80)

    total = len(results)
    errors = sum(1 for r in results if r.error)
    merged = sum(1 for r in results if r.similarity_merge and not r.error)
    stored = sum(1 for r in results if r.stored_to_db and not r.error)

    print(f"\nTotal samples: {total}")
    print(f"Errors: {errors}")
    print(f"Merged with existing: {merged}")
    print(f"Stored as new: {stored}")

    print("\n" + "-" * 80)
    print(f"{'ID':<4} {'Proposed':<15} {'Final':<15} {'Families':<25} {'Status':<15}")
    print("-" * 80)

    for r in results:
        if r.error:
            status = "ERROR"
        elif r.similarity_merge:
            status = f"MERGED ({r.similarity_score:.2f})"
        else:
            status = "NEW"

        families_str = ", ".join(r.families[:3]) if r.families else "-"
        print(
            f"{r.sample_id:<4} {r.proposed_relation:<15} {r.final_relation:<15} {families_str:<25} {status:<15}"
        )

    print("-" * 80)

    # Detailed output for each result
    print("\n" + "=" * 80)
    print("DETAILED RESULTS")
    print("=" * 80)

    for r in results:
        print(f"\n[Sample {r.sample_id}] {r.proposed_relation}")
        print(f"  Final relation: {r.final_relation}")
        print(
            f"  Description: {r.final_description[:100]}..."
            if r.final_description
            else "  Description: -"
        )
        print(f"  Families: {r.families}")
        print(
            f"  Reasoning: {r.strong_reasoning[:150]}..."
            if len(r.strong_reasoning) > 150
            else f"  Reasoning: {r.strong_reasoning}"
        )
        if r.similarity_merge:
            print(f"  → MERGED (similarity: {r.similarity_score:.2f})")
        elif r.stored_to_db:
            print(f"  → STORED as new relation")
        elif r.error:
            print(f"  → ERROR: {r.error}")


async def main():
    """Main entry point."""
    logger.info("Starting New Relation Validation Flow Integration Test")
    logger.info("Using REAL Strong LLM calls (google/gemini-3-pro-preview)")

    results = await run_all_tests(clear_db=True)
    print_summary(results)

    # Return success/failure
    errors = sum(1 for r in results if r.error)
    return errors == 0


if __name__ == "__main__":
    success = asyncio.run(main())
    exit(0 if success else 1)
