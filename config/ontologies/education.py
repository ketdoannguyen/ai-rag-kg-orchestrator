"""Education domain ontology with learning-specific relationships.

Extends the base ontology with relationships specific to educational materials:
- Learning flow: PREREQUISITE_OF, BUILDS_ON, LEADS_TO
- Content types: DEFINES, EXPLAINS, PROVES, DERIVES, APPLIES
- Examples & Exercises: ILLUSTRATES, SOLVES, TESTS, REQUIRES
- Cross-reference: CITES, SEE_ALSO
"""

from __future__ import annotations

from enum import Enum

from .base import (
    BASE_RELATIONS,
    CONTENT_TYPES,
    IMAGE,
    PARAGRAPH,
    STRUCTURE,
    TABLE,
    TIER_2_CONTENT,
    BaseOntology,
    RelationConstraint,
)


class EducationRelationType(str, Enum):
    """Education-specific relationship types.

    Extends RelationType with domain-specific relationships for
    educational materials, textbooks, and learning resources.

    NOTE: DEFINES is already in base RelationType - use that instead.
    """

    # Learning flow (CRITICAL for education)
    PREREQUISITE_OF = "PREREQUISITE_OF"  # Must learn before
    BUILDS_ON = "BUILDS_ON"  # Extends/develops from
    LEADS_TO = "LEADS_TO"  # Natural progression to

    # Content types (DEFINES is in base, don't duplicate here)
    EXPLAINS = "EXPLAINS"  # Detailed explanation
    PROVES = "PROVES"  # Mathematical/logical proof
    DERIVES = "DERIVES"  # Derivation from
    APPLIES = "APPLIES"  # Application of concept

    # Examples & Exercises
    ILLUSTRATES = "ILLUSTRATES"  # Example/illustration
    SOLVES = "SOLVES"  # Solution for problem
    TESTS = "TESTS"  # Question tests concept
    REQUIRES = "REQUIRES"  # Exercise requires knowledge

    # Cross-reference
    CITES = "CITES"  # Citation/reference
    SEE_ALSO = "SEE_ALSO"  # Related content


# Education-specific relation constraints
# NOTE: DEFINES constraint uses base RelationType, not duplicated here
EDUCATION_RELATION_CONSTRAINTS: list[RelationConstraint] = [
    # Learning flow relationships
    RelationConstraint(
        relation_type=EducationRelationType.PREREQUISITE_OF,
        source_node_types=[PARAGRAPH, STRUCTURE],
        target_node_types=[PARAGRAPH, STRUCTURE],
        is_transitive=True,
        description="Nội dung này cần học trước khi học nội dung kia",
        tiers=[TIER_2_CONTENT],
    ),
    RelationConstraint(
        relation_type=EducationRelationType.BUILDS_ON,
        source_node_types=[PARAGRAPH, STRUCTURE],
        target_node_types=[PARAGRAPH, STRUCTURE],
        is_transitive=True,
        description="Mở rộng/phát triển từ kiến thức trước",
        tiers=[TIER_2_CONTENT],
    ),
    RelationConstraint(
        relation_type=EducationRelationType.LEADS_TO,
        source_node_types=[PARAGRAPH, STRUCTURE],
        target_node_types=[PARAGRAPH, STRUCTURE],
        is_transitive=True,
        description="Dẫn đến chủ đề/khái niệm tiếp theo",
        tiers=[TIER_2_CONTENT],
    ),
    # Content type relationships (DEFINES already in base, only add education-specific)
    RelationConstraint(
        relation_type=EducationRelationType.EXPLAINS,
        source_node_types=[PARAGRAPH],
        target_node_types=CONTENT_TYPES,
        description="Giải thích chi tiết cho nội dung",
        tiers=[TIER_2_CONTENT],
    ),
    RelationConstraint(
        relation_type=EducationRelationType.PROVES,
        source_node_types=[PARAGRAPH],
        target_node_types=[PARAGRAPH],
        description="Chứng minh toán học/logic",
        tiers=[TIER_2_CONTENT],
    ),
    RelationConstraint(
        relation_type=EducationRelationType.DERIVES,
        source_node_types=[PARAGRAPH],
        target_node_types=[PARAGRAPH],
        is_transitive=True,
        description="Suy ra/chứng minh từ",
        tiers=[TIER_2_CONTENT],
    ),
    RelationConstraint(
        relation_type=EducationRelationType.APPLIES,
        source_node_types=CONTENT_TYPES,
        target_node_types=[PARAGRAPH],
        description="Ứng dụng của khái niệm/công thức",
        tiers=[TIER_2_CONTENT],
    ),
    # Examples & Exercises relationships
    RelationConstraint(
        relation_type=EducationRelationType.ILLUSTRATES,
        source_node_types=[IMAGE, TABLE, PARAGRAPH],
        target_node_types=[PARAGRAPH],
        description="Hình ảnh/bảng/ví dụ minh họa cho nội dung",
        tiers=[TIER_2_CONTENT],
    ),
    RelationConstraint(
        relation_type=EducationRelationType.SOLVES,
        source_node_types=[PARAGRAPH],
        target_node_types=[PARAGRAPH],
        description="Lời giải cho bài tập/câu hỏi",
        tiers=[TIER_2_CONTENT],
    ),
    RelationConstraint(
        relation_type=EducationRelationType.TESTS,
        source_node_types=[PARAGRAPH],
        target_node_types=[PARAGRAPH],
        description="Câu hỏi/bài tập kiểm tra khái niệm này",
        tiers=[TIER_2_CONTENT],
    ),
    RelationConstraint(
        relation_type=EducationRelationType.REQUIRES,
        source_node_types=[PARAGRAPH],
        target_node_types=[PARAGRAPH, STRUCTURE],
        description="Bài tập yêu cầu kiến thức này",
        tiers=[TIER_2_CONTENT],
    ),
    # Cross-reference relationships
    RelationConstraint(
        relation_type=EducationRelationType.CITES,
        source_node_types=CONTENT_TYPES,
        target_node_types=CONTENT_TYPES,
        description="Trích dẫn nguồn/tài liệu",
        tiers=[TIER_2_CONTENT],
    ),
    RelationConstraint(
        relation_type=EducationRelationType.SEE_ALSO,
        source_node_types=CONTENT_TYPES,
        target_node_types=CONTENT_TYPES,
        is_symmetric=True,
        description="Xem thêm nội dung liên quan",
        tiers=[TIER_2_CONTENT],
    ),
]


# Education ontology - inherits all base relations + education-specific
EDUCATION_ONTOLOGY = BaseOntology(
    name="education",
    display_name="Tài liệu Học tập Đại học",
    relations=BASE_RELATIONS + EDUCATION_RELATION_CONSTRAINTS,
)
