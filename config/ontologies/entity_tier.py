"""Entity tier ontology with entity-level relationships.

Extends the base ontology with relationships specific to Tier 3 entities:
- Hierarchy: LÀ_MỘT, LÀ_VÍ_DỤ_CỦA, CÓ_PHẦN_LÀ
- Dependencies: PHỤ_THUỘC_VÀO
- Content links: MENTIONED_IN, DEFINES

Note: This is a scaffold. Full implementation will be added
when Tier 3 entity extraction is implemented.
"""

from __future__ import annotations

from enum import Enum

from .base import (
    CONTENT_TYPES,
    PARAGRAPH,
    STRUCTURE,
    TIER_3_ENTITY,
    BaseOntology,
    RelationConstraint,
)


class EntityRelationType(str, Enum):
    """Entity-level relationship types (Tier 3).

    These relationships connect entities to each other
    and to content/structure nodes.
    """

    # Hierarchy relationships (Entity -> Entity)
    IS_A = "LÀ_MỘT"  # Concept is a type of another
    IS_EXAMPLE_OF = "LÀ_VÍ_DỤ_CỦA"  # Concrete example of abstract
    HAS_PART = "CÓ_PHẦN_LÀ"  # Composition relationship

    # Dependency relationships (Entity -> Entity)
    DEPENDS_ON = "PHỤ_THUỘC_VÀO"  # Prerequisite knowledge

    # Formula relationships (Entity -> Entity)
    HAS_FORMULA = "CÓ_CÔNG_THỨC_LÀ"  # Concept has formula
    REPRESENTS = "BIỂU_DIỄN_CHO"  # Formula represents concept

    # Image relationships (Entity -> Entity)
    ILLUSTRATED_BY = "ĐƯỢC_MINH_HỌA_BỞI"  # Concept illustrated by image
    ILLUSTRATES = "MINH_HỌA_CHO"  # Image illustrates concept

    # Person relationships (Entity -> Entity)
    INVENTED_BY = "ĐƯỢC_PHÁT_MINH_BỞI"  # Concept invented by person
    INVENTED = "PHÁT_MINH_RA"  # Person invented concept

    # Content links (Entity <-> Content)
    MENTIONED_IN = "MENTIONED_IN"  # Entity mentioned in content
    DEFINES = "DEFINES_ENTITY"  # Content defines entity
    EXTRACTED_FROM = "EXTRACTED_FROM"  # Entity extracted from content


# Entity type constants (match EntityType enum values)
CONCEPT = "concept"
FORMULA = "formula"
IMAGE_ENTITY = "image_entity"
PERSON = "person"
ORGANIZATION = "organization"
DATE = "date"
LOCATION = "location"

ALL_ENTITY_TYPES = [CONCEPT, FORMULA, IMAGE_ENTITY, PERSON, ORGANIZATION, DATE, LOCATION]


# Entity-specific relation constraints
ENTITY_RELATION_CONSTRAINTS: list[RelationConstraint] = [
    # Hierarchy relationships
    RelationConstraint(
        relation_type=EntityRelationType.IS_A,
        source_node_types=[CONCEPT],
        target_node_types=[CONCEPT],
        is_transitive=True,
        description="Khái niệm A là một loại của khái niệm B",
        tiers=[TIER_3_ENTITY],
    ),
    RelationConstraint(
        relation_type=EntityRelationType.IS_EXAMPLE_OF,
        source_node_types=[CONCEPT],
        target_node_types=[CONCEPT],
        description="Khái niệm A là ví dụ cụ thể của khái niệm B",
        tiers=[TIER_3_ENTITY],
    ),
    RelationConstraint(
        relation_type=EntityRelationType.HAS_PART,
        source_node_types=[CONCEPT],
        target_node_types=[CONCEPT],
        is_transitive=True,
        description="Khái niệm A có thành phần là khái niệm B",
        tiers=[TIER_3_ENTITY],
    ),
    # Dependency relationships
    RelationConstraint(
        relation_type=EntityRelationType.DEPENDS_ON,
        source_node_types=[CONCEPT],
        target_node_types=[CONCEPT],
        is_transitive=True,
        description="Cần học khái niệm B trước khi học khái niệm A (tiên quyết)",
        tiers=[TIER_3_ENTITY],
    ),
    # Formula relationships
    RelationConstraint(
        relation_type=EntityRelationType.HAS_FORMULA,
        source_node_types=[CONCEPT],
        target_node_types=[FORMULA],
        description="Khái niệm có công thức liên quan",
        tiers=[TIER_3_ENTITY],
    ),
    RelationConstraint(
        relation_type=EntityRelationType.REPRESENTS,
        source_node_types=[FORMULA],
        target_node_types=[CONCEPT],
        description="Công thức biểu diễn cho khái niệm",
        tiers=[TIER_3_ENTITY],
    ),
    # Image relationships
    RelationConstraint(
        relation_type=EntityRelationType.ILLUSTRATED_BY,
        source_node_types=[CONCEPT],
        target_node_types=[IMAGE_ENTITY],
        description="Khái niệm được minh họa bởi hình ảnh",
        tiers=[TIER_3_ENTITY],
    ),
    RelationConstraint(
        relation_type=EntityRelationType.ILLUSTRATES,
        source_node_types=[IMAGE_ENTITY],
        target_node_types=[CONCEPT],
        description="Hình ảnh minh họa cho khái niệm",
        tiers=[TIER_3_ENTITY],
    ),
    # Person relationships
    RelationConstraint(
        relation_type=EntityRelationType.INVENTED_BY,
        source_node_types=[CONCEPT, FORMULA],
        target_node_types=[PERSON],
        description="Được phát minh/phát hiện bởi nhân vật",
        tiers=[TIER_3_ENTITY],
    ),
    RelationConstraint(
        relation_type=EntityRelationType.INVENTED,
        source_node_types=[PERSON],
        target_node_types=[CONCEPT, FORMULA],
        description="Nhân vật phát minh/phát hiện",
        tiers=[TIER_3_ENTITY],
    ),
    # Content links (Entity <-> Tier 2)
    RelationConstraint(
        relation_type=EntityRelationType.MENTIONED_IN,
        source_node_types=ALL_ENTITY_TYPES,
        target_node_types=CONTENT_TYPES,
        description="Thực thể được nhắc đến trong nội dung",
        tiers=[TIER_3_ENTITY],
    ),
    RelationConstraint(
        relation_type=EntityRelationType.DEFINES,
        source_node_types=[PARAGRAPH],
        target_node_types=ALL_ENTITY_TYPES,
        description="Nội dung định nghĩa cho thực thể",
        tiers=[TIER_3_ENTITY],
    ),
    RelationConstraint(
        relation_type=EntityRelationType.EXTRACTED_FROM,
        source_node_types=ALL_ENTITY_TYPES,
        target_node_types=CONTENT_TYPES + [STRUCTURE],
        description="Thực thể được trích xuất từ nội dung",
        tiers=[TIER_3_ENTITY],
    ),
]


# Entity tier ontology (standalone, not merged with base)
ENTITY_ONTOLOGY = BaseOntology(
    name="entity",
    display_name="Entity Tier Ontology (Tier 3)",
    relations=ENTITY_RELATION_CONSTRAINTS,
)
