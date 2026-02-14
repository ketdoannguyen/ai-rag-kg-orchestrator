"""Base ontology classes with universal relationship types.

This module provides:
- RelationType: Enum of domain-agnostic relationship types
- RelationConstraint: Schema for relationship constraints
- BaseOntology: Base class for domain-specific ontologies

DESIGN PRINCIPLES:
1. Relations are between PARAGRAPHS (300-700 words), NOT entities
2. Tier 1 = Structure (hierarchy, navigation)
3. Tier 2 = Content (semantic relationships between paragraphs)
4. No duplicates - each relation has ONE clear purpose
5. Families must be accurate for query-relation matching
"""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field


class RelationType(str, Enum):
    """Base relationship types (domain-agnostic).

    Organized by tier and category:

    TIER 1 - STRUCTURE (6):
    - Hierarchical: HAS_SECTION, BELONGS_TO_SECTION
    - Sequential: NEXT_SIBLING, PREVIOUS_SIBLING
    - Containment: HAS_CONTENT
    - Equivalence: STRUCTURALLY_EQUIVALENT

    TIER 1+2 - SHARED (4):
    - Similarity: SIMILAR_TO
    - Reference: REFERS_TO
    - Duplication: DUPLICATE_OF
    - Versioning: SUPERSEDES

    TIER 2 - CONTENT (16):
    - Elaboration: ELABORATES, SUMMARIZES, EXEMPLIFIES, DEFINES
    - Logical: SUPPORTS, CONTRADICTS, CONTRASTS
    - Sequential: CONTINUES
    - Discourse: INTRODUCES, CONCLUDES, CONTEXTUALIZES
    - Cross-modal: VISUALIZES, HAS_ILLUSTRATION, RELATED_TO
    - Educational: PREREQUISITE_OF, BUILDS_UPON
    """

    # =========================================================================
    # TIER 1 - STRUCTURE ONLY (6 relations)
    # =========================================================================
    # Hierarchical
    HAS_SECTION = "HAS_SECTION"  # Parent structure → Child structure
    BELONGS_TO_SECTION = "BELONGS_TO_SECTION"  # Child → Parent (inverse)

    # Sequential
    NEXT_SIBLING = "NEXT_SIBLING"  # A → B (B comes after A in reading order)
    PREVIOUS_SIBLING = "PREVIOUS_SIBLING"  # A → B (B comes before A)

    # Containment
    HAS_CONTENT = "HAS_CONTENT"  # Structure → Content

    # Equivalence
    STRUCTURALLY_EQUIVALENT = "STRUCTURALLY_EQUIVALENT"  # Same structure from different sources

    # =========================================================================
    # TIER 1+2 - SHARED (4 relations)
    # =========================================================================
    SIMILAR_TO = "SIMILAR_TO"  # Semantic similarity (symmetric)
    REFERS_TO = "REFERS_TO"  # Explicit reference/citation
    DUPLICATE_OF = "DUPLICATE_OF"  # Exact/near copy (symmetric)
    SUPERSEDES = "SUPERSEDES"  # A replaces/updates B

    # =========================================================================
    # TIER 2 - CONTENT ONLY (14 relations)
    # =========================================================================
    # Elaboration (4)
    ELABORATES = "ELABORATES"  # A provides detail for B
    SUMMARIZES = "SUMMARIZES"  # A condenses B
    EXEMPLIFIES = "EXEMPLIFIES"  # A is example of B
    DEFINES = "DEFINES"  # A defines concept in B

    # Logical (3)
    SUPPORTS = "SUPPORTS"  # A provides evidence for B
    CONTRADICTS = "CONTRADICTS"  # A opposes B (symmetric)
    CONTRASTS = "CONTRASTS"  # A highlights differences with B (symmetric)

    # Sequential (1)
    CONTINUES = "CONTINUES"  # A continues split content from B

    # Discourse (3)
    INTRODUCES = "INTRODUCES"  # A introduces topic developed in B
    CONCLUDES = "CONCLUDES"  # A concludes content from B
    CONTEXTUALIZES = "CONTEXTUALIZES"  # A provides background for B

    # Cross-modal (3)
    VISUALIZES = "VISUALIZES"  # Image/Table visualizes Paragraph
    HAS_ILLUSTRATION = "HAS_ILLUSTRATION"  # Paragraph references Image/Table
    RELATED_TO = "RELATED_TO"  # General cross-reference relationship

    # Educational (2)
    PREREQUISITE_OF = "PREREQUISITE_OF"  # Must understand A before B
    BUILDS_UPON = "BUILDS_UPON"  # A extends knowledge from B


class RelationConstraint(BaseModel):
    """Constraint for a relationship type.

    Defines which node types can participate as source/target
    and the semantic properties of the relationship.
    """

    relation_type: RelationType | str = Field(..., description="The relationship type")
    source_node_types: list[str] = Field(
        ..., description="Valid source node types (NodeType values)"
    )
    target_node_types: list[str] = Field(
        ..., description="Valid target node types (NodeType values)"
    )
    is_symmetric: bool = Field(default=False, description="A->B implies B->A")
    is_transitive: bool = Field(default=False, description="A->B, B->C implies A->C")
    description: str = Field(default="", description="Human-readable description")
    summary: str = Field(default="", description="Short description (<100 chars)")
    families: list[str] = Field(
        default_factory=list,
        description="Family names: causal, temporal, comparative, elaborative, participation, structural",
    )
    keywords: list[str] = Field(
        default_factory=list, description="Vietnamese keywords for query matching"
    )
    tiers: list[str] = Field(
        default_factory=lambda: [TIER_2_CONTENT],
        description="Processing tiers: tier_1_structure, tier_2_content",
    )

    def applies_to_tier(self, tier: str) -> bool:
        """Check if constraint applies to a specific tier."""
        return tier in self.tiers

    def applies_to_any_tier(self, tiers: list[str]) -> bool:
        """Check if constraint applies to any of the given tiers."""
        return any(t in self.tiers for t in tiers)

    def is_valid_pair(self, source_type: str, target_type: str) -> bool:
        """Check if source/target pair is valid for this constraint."""
        return source_type in self.source_node_types and target_type in self.target_node_types


class BaseOntology(BaseModel):
    """Base ontology with common relationships.

    Provides the foundation for domain-specific ontologies.
    Domain ontologies inherit base relations and add domain-specific ones.
    """

    name: str = Field(default="base", description="Ontology identifier")
    display_name: str = Field(default="Base Ontology", description="Human-readable name")
    relations: list[RelationConstraint] = Field(
        default_factory=list, description="Relationship definitions"
    )

    def get_relation(self, relation_type: str) -> RelationConstraint | None:
        """Get a relation constraint by type."""
        for r in self.relations:
            if str(r.relation_type) == relation_type or (
                hasattr(r.relation_type, "value") and r.relation_type.value == relation_type
            ):
                return r
        return None

    def get_valid_relations(self, source_type: str, target_type: str) -> list[RelationType | str]:
        """Get valid relation types between two node types."""
        return [
            r.relation_type for r in self.relations if r.is_valid_pair(source_type, target_type)
        ]

    def get_relations_for_tier(self, tier: str) -> list[RelationConstraint]:
        """Get all relations that apply to a specific tier."""
        return [r for r in self.relations if r.applies_to_tier(tier)]

    def get_relations_for_tiers(self, tiers: list[str]) -> list[RelationConstraint]:
        """Get all relations that apply to any of the given tiers."""
        return [r for r in self.relations if r.applies_to_any_tier(tiers)]

    def to_llm_prompt_section(self, tier: str | None = None) -> str:
        """Build prompt section for LLM relationship extraction."""
        relations = self.get_relations_for_tier(tier) if tier else self.relations

        if not relations:
            return "(No relationships defined)"

        lines = ["## Available Relationship Types\n"]
        lines.append("Only use relationships from this list:\n")

        for r in relations:
            rel_type = (
                r.relation_type.value
                if hasattr(r.relation_type, "value")
                else str(r.relation_type)
            )
            line = f"- **{rel_type}**: {r.summary}"
            if r.is_symmetric:
                line += " (symmetric)"
            if r.is_transitive:
                line += " (transitive)"
            lines.append(line)

        return "\n".join(lines)

    def validate_relation(
        self,
        relation_type: str,
        source_type: str,
        target_type: str,
    ) -> tuple[bool, str]:
        """Validate if a relation is allowed by this ontology."""
        constraint = self.get_relation(relation_type)
        if constraint is None:
            return False, f"Unknown relation type: {relation_type}"

        if not constraint.is_valid_pair(source_type, target_type):
            return False, (
                f"Invalid node types for {relation_type}: {source_type} -> {target_type}"
            )

        return True, ""


# =============================================================================
# NODE TYPE CONSTANTS
# =============================================================================
PARAGRAPH = "paragraph"
IMAGE = "image"
TABLE = "table"
STRUCTURE = "structure"

CONTENT_TYPES = [PARAGRAPH, IMAGE, TABLE]
STRUCTURE_TYPES = [STRUCTURE]
ALL_TYPES = CONTENT_TYPES + STRUCTURE_TYPES

# =============================================================================
# TIER CONSTANTS
# =============================================================================
TIER_1_STRUCTURE = "tier_1_structure"
TIER_2_CONTENT = "tier_2_content"
TIER_3_ENTITY = "tier_3_entity"  # For entity-level relations (used by entity_tier.py)

# =============================================================================
# FAMILY CONSTANTS (imported from families.py for consistency)
# =============================================================================
STRUCTURAL = "structural"
TEMPORAL = "temporal"
COMPARATIVE = "comparative"
ELABORATIVE = "elaborative"
CAUSAL = "causal"
PARTICIPATION = "participation"


# =============================================================================
# BASE RELATIONS - 24 domain-agnostic relations
# =============================================================================
BASE_RELATIONS: list[RelationConstraint] = [
    # =========================================================================
    # TIER 1 ONLY - STRUCTURE (6 relations)
    # =========================================================================
    RelationConstraint(
        relation_type=RelationType.HAS_SECTION,
        source_node_types=STRUCTURE_TYPES,
        target_node_types=STRUCTURE_TYPES,
        summary="Cấu trúc cha chứa cấu trúc con",
        description="""Quan hệ phân cấp: Node cấu trúc cha chứa node cấu trúc con.

Đặc điểm:
- Chỉ áp dụng cho Tier 1 (Structure)
- Thể hiện quan hệ cha-con trong cây cấu trúc tài liệu
- Hướng: Parent → Child

Ví dụ:
- "Chương 1: Lịch sử cổ đại" HAS_SECTION "Mục 1.1: Thời Hùng Vương"
- "Phần I: Bắc thuộc" HAS_SECTION "Chương 1: Khởi nghĩa Hai Bà Trưng"

Phân biệt với:
- BELONGS_TO_SECTION: Inverse (child → parent)
- HAS_CONTENT: Cấu trúc chứa nội dung""",
        families=[STRUCTURAL],
        keywords=["chương", "mục", "phần", "bao gồm", "gồm có"],
        tiers=[TIER_1_STRUCTURE],
    ),
    RelationConstraint(
        relation_type=RelationType.BELONGS_TO_SECTION,
        source_node_types=STRUCTURE_TYPES,
        target_node_types=STRUCTURE_TYPES,
        summary="Cấu trúc con thuộc về cấu trúc cha",
        description="""Quan hệ phân cấp ngược: Node con thuộc về node cha.

Đặc điểm:
- Inverse của HAS_SECTION
- Dùng để traverse ngược lên cây cấu trúc
- Hướng: Child → Parent

Ví dụ:
- "Mục 1.1: Thời Hùng Vương" BELONGS_TO_SECTION "Chương 1"
- "Bài 3: Khởi nghĩa Lam Sơn" BELONGS_TO_SECTION "Phần II"

Phân biệt với:
- HAS_SECTION: Inverse (parent → child)
- PART_OF: Quan hệ nội dung, không phải cấu trúc""",
        families=[STRUCTURAL],
        keywords=["thuộc về", "nằm trong", "là phần của"],
        tiers=[TIER_1_STRUCTURE],
    ),
    RelationConstraint(
        relation_type=RelationType.NEXT_SIBLING,
        source_node_types=ALL_TYPES,
        target_node_types=ALL_TYPES,
        summary="Phần tử đứng ngay trước trong thứ tự đọc",
        description="""Quan hệ thứ tự đọc: A đứng ngay trước B trong văn bản.

Đặc điểm:
- Thể hiện thứ tự đọc tự nhiên của tài liệu
- Hướng: A → B (đọc A rồi đọc B)

Ví dụ:
- "Mục 1.1" NEXT_SIBLING "Mục 1.2"
- "Đoạn văn 1" NEXT_SIBLING "Đoạn văn 2"

Phân biệt với:
- PREVIOUS_SIBLING: Ngược hướng
- CONTINUES: Nội dung bị cắt tiếp tục""",
        families=[STRUCTURAL, TEMPORAL],
        keywords=["tiếp theo", "kế tiếp", "sau đó"],
        tiers=[TIER_1_STRUCTURE],
    ),
    RelationConstraint(
        relation_type=RelationType.PREVIOUS_SIBLING,
        source_node_types=ALL_TYPES,
        target_node_types=ALL_TYPES,
        summary="Phần tử đứng ngay sau trong thứ tự đọc",
        description="""Quan hệ thứ tự đọc ngược: A đứng ngay sau B trong văn bản.

Đặc điểm:
- Inverse của NEXT_SIBLING
- Dùng để navigate ngược

Ví dụ:
- "Mục 1.2" PREVIOUS_SIBLING "Mục 1.1"

Phân biệt với:
- NEXT_SIBLING: Ngược hướng""",
        families=[STRUCTURAL, TEMPORAL],
        keywords=["trước đó", "phần trước", "đứng trước"],
        tiers=[TIER_1_STRUCTURE],
    ),
    RelationConstraint(
        relation_type=RelationType.HAS_CONTENT,
        source_node_types=STRUCTURE_TYPES,
        target_node_types=CONTENT_TYPES,
        summary="Cấu trúc chứa nội dung (đoạn văn, hình, bảng)",
        description="""Quan hệ cấu trúc-nội dung: Node cấu trúc chứa node nội dung.

Đặc điểm:
- Liên kết structure với paragraphs, images, tables
- Hướng: Structure → Content

Ví dụ:
- "Mục 1.1" HAS_CONTENT "Đoạn văn mô tả nguyên nhân"
- "Chương 2" HAS_CONTENT "Bảng thống kê trận đánh"

Phân biệt với:
- HAS_SECTION: Cấu trúc chứa cấu trúc""",
        families=[STRUCTURAL],
        keywords=["chứa", "bao gồm", "nội dung"],
        tiers=[TIER_1_STRUCTURE],
    ),
    RelationConstraint(
        relation_type=RelationType.STRUCTURALLY_EQUIVALENT,
        source_node_types=STRUCTURE_TYPES,
        target_node_types=STRUCTURE_TYPES,
        is_symmetric=True,
        summary="Hai cấu trúc tương đương từ nguồn khác nhau",
        description="""Quan hệ tương đương cấu trúc: Hai node có vị trí/nội dung tương đương.

Đặc điểm:
- Symmetric - A ↔ B
- Xảy ra khi import từ nhiều nguồn

Ví dụ:
- "Chương 1 từ sách A" STRUCTURALLY_EQUIVALENT "Chương 1 từ sách B"

Phân biệt với:
- DUPLICATE_OF: Trùng lặp nội dung
- SIMILAR_TO: Tương đồng ngữ nghĩa""",
        families=[STRUCTURAL],
        keywords=["tương đương", "giống nhau", "cùng cấu trúc"],
        tiers=[TIER_1_STRUCTURE],
    ),
    # =========================================================================
    # TIER 1+2 - SHARED (4 relations)
    # =========================================================================
    RelationConstraint(
        relation_type=RelationType.SIMILAR_TO,
        source_node_types=ALL_TYPES,
        target_node_types=ALL_TYPES,
        is_symmetric=True,
        summary="Hai đoạn có nội dung tương đồng về ngữ nghĩa",
        description="""Quan hệ tương đồng ngữ nghĩa dựa trên embedding similarity.

Đặc điểm:
- Symmetric - A ↔ B
- Dựa trên vector similarity
- Áp dụng cả Tier 1 và Tier 2

Ví dụ:
- "Quân Trần đánh tan quân Nguyên 1288" SIMILAR_TO "Chiến thắng Bạch Đằng"
- "Nguyên nhân thất bại nhà Hồ" SIMILAR_TO "Lý do sụp đổ triều Hồ"

Phân biệt với:
- DUPLICATE_OF: Giống hệt hoàn toàn""",
        families=[COMPARATIVE],
        keywords=["tương tự", "giống", "như nhau", "tương đồng"],
        tiers=[TIER_1_STRUCTURE, TIER_2_CONTENT],
    ),
    RelationConstraint(
        relation_type=RelationType.REFERS_TO,
        source_node_types=ALL_TYPES,
        target_node_types=ALL_TYPES,
        summary="Đoạn A tham chiếu rõ ràng đến đoạn B",
        description="""Quan hệ tham chiếu rõ ràng: A đề cập, trích dẫn đến B.

Đặc điểm:
- Tham chiếu phải explicit, không ngầm định
- Hướng: A → B

Ví dụ:
- "Như đã nói ở phần trước..." REFERS_TO "Phần giới thiệu"
- "Xem thêm Chương 3" REFERS_TO "Chương 3"

Phân biệt với:
- ELABORATES: Mở rộng nội dung""",
        families=[STRUCTURAL, ELABORATIVE],
        keywords=["tham chiếu", "xem thêm", "như đã nói", "trích dẫn"],
        tiers=[TIER_1_STRUCTURE, TIER_2_CONTENT],
    ),
    RelationConstraint(
        relation_type=RelationType.DUPLICATE_OF,
        source_node_types=ALL_TYPES,
        target_node_types=ALL_TYPES,
        is_symmetric=True,
        is_transitive=True,
        summary="Hai đoạn là bản sao trùng lặp hoàn toàn",
        description="""Quan hệ trùng lặp: Nội dung giống hệt hoặc gần giống hệt.

Đặc điểm:
- Symmetric và Transitive
- Thường từ import nhiều nguồn

Ví dụ:
- "Văn bản nguồn 1" DUPLICATE_OF "Văn bản giống hệt nguồn 2"

Phân biệt với:
- SIMILAR_TO: Tương đồng nhưng không giống hệt""",
        families=[COMPARATIVE],
        keywords=["trùng lặp", "giống hệt", "bản sao"],
        tiers=[TIER_1_STRUCTURE, TIER_2_CONTENT],
    ),
    RelationConstraint(
        relation_type=RelationType.SUPERSEDES,
        source_node_types=CONTENT_TYPES,
        target_node_types=CONTENT_TYPES,
        is_transitive=True,
        summary="Đoạn A thay thế/cập nhật đoạn B cũ hơn",
        description="""Quan hệ thay thế: A là phiên bản mới hơn của B.

Đặc điểm:
- Transitive
- A thay thế B trong hệ thống

Ví dụ:
- "Theo nghiên cứu 2024..." SUPERSEDES "Quan điểm trước đây..."
- "SGK 2023" SUPERSEDES "SGK 2020"

Phân biệt với:
- BUILDS_UPON: Phát triển từ, không thay thế""",
        families=[TEMPORAL],
        keywords=["thay thế", "cập nhật", "phiên bản mới"],
        tiers=[TIER_2_CONTENT],
    ),
    # =========================================================================
    # TIER 2 ONLY - ELABORATION (4 relations)
    # =========================================================================
    RelationConstraint(
        relation_type=RelationType.ELABORATES,
        source_node_types=CONTENT_TYPES,
        target_node_types=CONTENT_TYPES,
        is_transitive=True,
        summary="Đoạn A mở rộng/giải thích chi tiết đoạn B",
        description="""Quan hệ mở rộng: A cung cấp thông tin chi tiết cho B.

Đặc điểm:
- A chi tiết hơn B
- Transitive
- Thường A đi sau B

Ví dụ:
- "Cụ thể, chiến thuật gồm 3 bước..." ELABORATES "Ngô Quyền dùng mai phục"
- "Các yếu tố: địa hình, thời tiết..." ELABORATES "Nhiều yếu tố góp phần"

Phân biệt với:
- SUMMARIZES: Ngược lại
- EXEMPLIFIES: Ví dụ cụ thể""",
        families=[ELABORATIVE],
        keywords=["cụ thể", "chi tiết", "giải thích", "mở rộng", "nói rõ"],
        tiers=[TIER_2_CONTENT],
    ),
    RelationConstraint(
        relation_type=RelationType.SUMMARIZES,
        source_node_types=[PARAGRAPH],
        target_node_types=CONTENT_TYPES,
        is_transitive=True,
        summary="Đoạn A tóm tắt nội dung đoạn B",
        description="""Quan hệ tóm tắt: A rút gọn nội dung B.

Đặc điểm:
- A ngắn gọn hơn B
- A chứa ý chính của B

Ví dụ:
- "Tóm lại, nguyên nhân chính là mâu thuẫn giai cấp" SUMMARIZES "Phân tích chi tiết"
- "Kết luận: ĐBP mang ý nghĩa lịch sử" SUMMARIZES "Toàn bộ phân tích"

Phân biệt với:
- ELABORATES: Ngược lại
- CONCLUDES: Kết luận, không chỉ tóm tắt""",
        families=[ELABORATIVE],
        keywords=["tóm tắt", "kết luận", "tóm lại", "nói chung", "ngắn gọn"],
        tiers=[TIER_2_CONTENT],
    ),
    RelationConstraint(
        relation_type=RelationType.EXEMPLIFIES,
        source_node_types=CONTENT_TYPES,
        target_node_types=[PARAGRAPH],
        summary="Đoạn A là ví dụ minh họa cho quan điểm trong B",
        description="""Quan hệ ví dụ: A minh họa cho quan điểm/khái niệm B.

Đặc điểm:
- A là ví dụ cụ thể
- B là quan điểm tổng quát

Ví dụ:
- "Trận Bạch Đằng 938 với cọc nhọn" EXEMPLIFIES "Chiến thuật đánh thủy quân"
- "Khởi nghĩa Hai Bà Trưng 40" EXEMPLIFIES "Tinh thần đấu tranh dân tộc"

Phân biệt với:
- ELABORATES: Giải thích, không phải ví dụ""",
        families=[ELABORATIVE, STRUCTURAL],
        keywords=["ví dụ", "minh họa", "chẳng hạn", "cụ thể là", "như"],
        tiers=[TIER_2_CONTENT],
    ),
    RelationConstraint(
        relation_type=RelationType.DEFINES,
        source_node_types=[PARAGRAPH],
        target_node_types=CONTENT_TYPES,
        summary="Đoạn A định nghĩa khái niệm/thuật ngữ trong B",
        description="""Quan hệ định nghĩa: A cung cấp định nghĩa cho B.

Đặc điểm:
- Thường cho thuật ngữ chuyên môn
- A là định nghĩa, B sử dụng khái niệm

Ví dụ:
- "Chế độ quân chủ là..." DEFINES "Dưới chế độ quân chủ phong kiến VN"
- "Công xã nông thôn là đơn vị..." DEFINES "Vai trò của công xã nông thôn"

Phân biệt với:
- ELABORATES: Giải thích, không định nghĩa""",
        families=[ELABORATIVE],
        keywords=["định nghĩa", "là gì", "được hiểu là", "nghĩa là"],
        tiers=[TIER_2_CONTENT],
    ),
    # =========================================================================
    # TIER 2 ONLY - LOGICAL (3 relations)
    # =========================================================================
    RelationConstraint(
        relation_type=RelationType.SUPPORTS,
        source_node_types=CONTENT_TYPES,
        target_node_types=CONTENT_TYPES,
        is_transitive=True,
        summary="Đoạn A cung cấp bằng chứng/lập luận ủng hộ B",
        description="""Quan hệ ủng hộ: A cung cấp bằng chứng cho B.

Đặc điểm:
- A làm tăng độ tin cậy của B
- Transitive

Ví dụ:
- "Số liệu cho thấy sản lượng tăng 20%" SUPPORTS "Cải cách hiệu quả"
- "Theo sử sách, Lý Thường Kiệt..." SUPPORTS "LTK là danh tướng"

Phân biệt với:
- ELABORATES: Giải thích, không ủng hộ
- CAUSED_BY: Nhân quả""",
        families=[CAUSAL, ELABORATIVE],
        keywords=["bằng chứng", "chứng minh", "ủng hộ", "xác nhận", "theo"],
        tiers=[TIER_2_CONTENT],
    ),
    RelationConstraint(
        relation_type=RelationType.CONTRADICTS,
        source_node_types=CONTENT_TYPES,
        target_node_types=CONTENT_TYPES,
        is_symmetric=True,
        summary="Đoạn A mâu thuẫn/phản bác trực tiếp B",
        description="""Quan hệ mâu thuẫn: A và B không thể cùng đúng.

Đặc điểm:
- Symmetric
- Thể hiện xung đột quan điểm

Ví dụ:
- "Nguyên nhân là ngoại xâm" CONTRADICTS "Nguyên nhân là mâu thuẫn nội bộ"
- "Sử Trung Quốc ghi khác..." CONTRADICTS "Theo sử Việt..."

Phân biệt với:
- CONTRASTS: Khác biệt, không mâu thuẫn
- OPPOSED_BY: Phản đối hành động""",
        families=[COMPARATIVE],
        keywords=["mâu thuẫn", "phản bác", "trái ngược", "không đồng ý", "bác bỏ"],
        tiers=[TIER_2_CONTENT],
    ),
    RelationConstraint(
        relation_type=RelationType.CONTRASTS,
        source_node_types=CONTENT_TYPES,
        target_node_types=CONTENT_TYPES,
        is_symmetric=True,
        summary="Đoạn A so sánh, làm nổi bật sự khác biệt với B",
        description="""Quan hệ đối lập: A và B có điểm khác biệt đáng chú ý.

Đặc điểm:
- Symmetric
- Không nhất thiết mâu thuẫn

Ví dụ:
- "Miền Bắc phong trào mạnh" CONTRASTS "Miền Nam còn hạn chế"
- "Chính sách nhà Lý" CONTRASTS "Chính sách nhà Trần"

Phân biệt với:
- CONTRADICTS: Mâu thuẫn trực tiếp
- SIMILAR_TO: Tương đồng""",
        families=[COMPARATIVE],
        keywords=["so sánh", "khác biệt", "trong khi", "ngược lại", "trái lại"],
        tiers=[TIER_2_CONTENT],
    ),
    # =========================================================================
    # TIER 2 ONLY - SEQUENTIAL (1 relation)
    # =========================================================================
    RelationConstraint(
        relation_type=RelationType.CONTINUES,
        source_node_types=CONTENT_TYPES,
        target_node_types=CONTENT_TYPES,
        is_transitive=True,
        summary="Đoạn A tiếp tục nội dung bị cắt từ B",
        description="""Quan hệ tiếp nối: A tiếp tục nội dung B bị chia cắt.

Đặc điểm:
- A và B vốn là một đoạn
- Nội dung liền mạch
- Transitive

Ví dụ:
- "...quân ta đã anh dũng chiến đấu" CONTINUES "Trong trận đánh này..."
- "...và cuối cùng thắng lợi" CONTINUES "Sau nhiều ngày cầm cự..."

Phân biệt với:
- ELABORATES: Mở rộng nội dung mới""",
        families=[TEMPORAL, ELABORATIVE],
        keywords=["tiếp tục", "tiếp theo", "và", "cùng với"],
        tiers=[TIER_2_CONTENT],
    ),
    # =========================================================================
    # TIER 2 ONLY - DISCOURSE (3 relations) - NEW!
    # =========================================================================
    RelationConstraint(
        relation_type=RelationType.INTRODUCES,
        source_node_types=[PARAGRAPH],
        target_node_types=CONTENT_TYPES,
        summary="Đoạn A giới thiệu chủ đề được phát triển trong B",
        description="""Quan hệ giới thiệu: A mở đầu, giới thiệu chủ đề cho B.

Đặc điểm:
- A thường ngắn gọn, khái quát
- B phát triển chi tiết chủ đề A đề cập

Ví dụ:
- "Sau đây là các nguyên nhân..." INTRODUCES "Phân tích chi tiết nguyên nhân"
- "Cuộc khởi nghĩa có ý nghĩa..." INTRODUCES "Ý nghĩa lịch sử chi tiết"

Phân biệt với:
- ELABORATES: Mở rộng, không giới thiệu
- SUMMARIZES: Tóm tắt, không giới thiệu""",
        families=[ELABORATIVE, STRUCTURAL],
        keywords=["giới thiệu", "mở đầu", "sau đây", "tiếp theo sẽ"],
        tiers=[TIER_2_CONTENT],
    ),
    RelationConstraint(
        relation_type=RelationType.CONCLUDES,
        source_node_types=[PARAGRAPH],
        target_node_types=CONTENT_TYPES,
        summary="Đoạn A kết luận/đúc kết nội dung từ B",
        description="""Quan hệ kết luận: A đúc kết, rút ra bài học từ B.

Đặc điểm:
- A thường ở cuối
- A rút ra insight/bài học từ B

Ví dụ:
- "Từ những phân tích trên, có thể thấy..." CONCLUDES "Toàn bộ phân tích"
- "Bài học lịch sử cho thấy..." CONCLUDES "Mô tả sự kiện"

Phân biệt với:
- SUMMARIZES: Tóm tắt nội dung
- ELABORATES: Mở rộng""",
        families=[ELABORATIVE],
        keywords=["kết luận", "bài học", "từ đó", "như vậy", "do đó"],
        tiers=[TIER_2_CONTENT],
    ),
    RelationConstraint(
        relation_type=RelationType.CONTEXTUALIZES,
        source_node_types=CONTENT_TYPES,
        target_node_types=CONTENT_TYPES,
        summary="Đoạn A cung cấp bối cảnh/background cho B",
        description="""Quan hệ bối cảnh: A cung cấp thông tin nền cho B.

Đặc điểm:
- A giúp hiểu B trong context phù hợp
- Thường A là thông tin lịch sử/xã hội

Ví dụ:
- "Trong bối cảnh thực dân xâm lược..." CONTEXTUALIZES "Phong trào yêu nước"
- "Sau chiến tranh thế giới..." CONTEXTUALIZES "Tình hình Việt Nam"

Phân biệt với:
- ELABORATES: Giải thích chi tiết
- INTRODUCES: Giới thiệu chủ đề""",
        families=[ELABORATIVE, TEMPORAL],
        keywords=["bối cảnh", "trong hoàn cảnh", "lúc đó", "khi ấy"],
        tiers=[TIER_2_CONTENT],
    ),
    # =========================================================================
    # TIER 2 ONLY - CROSS-MODAL (3 relations)
    # =========================================================================
    RelationConstraint(
        relation_type=RelationType.VISUALIZES,
        source_node_types=[IMAGE, TABLE],
        target_node_types=[PARAGRAPH],
        summary="Hình ảnh/bảng A minh họa trực quan cho đoạn văn B",
        description="""Quan hệ trực quan hóa: Image/Table minh họa cho Paragraph.

Đặc điểm:
- Source là IMAGE hoặc TABLE
- Target là PARAGRAPH

Ví dụ:
- "Bản đồ trận Bạch Đằng" VISUALIZES "Mô tả diễn biến trận đánh"
- "Bảng thống kê khởi nghĩa" VISUALIZES "Tổng quan phong trào"

Phân biệt với:
- ELABORATES: Mở rộng bằng văn bản
- EXEMPLIFIES: Ví dụ bằng văn bản""",
        families=[ELABORATIVE],
        keywords=["minh họa", "hình ảnh", "bảng", "biểu đồ", "sơ đồ"],
        tiers=[TIER_2_CONTENT],
    ),
    RelationConstraint(
        relation_type=RelationType.HAS_ILLUSTRATION,
        source_node_types=[PARAGRAPH],
        target_node_types=[IMAGE, TABLE],
        summary="Đoạn văn A có hình ảnh/bảng B minh họa",
        description="""Quan hệ tham chiếu trực quan: Paragraph tham chiếu đến Image/Table.

Đặc điểm:
- Inverse của VISUALIZES
- Source là PARAGRAPH
- Target là IMAGE hoặc TABLE
- Dùng khi paragraph đề cập "xem hình X", "bảng Y cho thấy"

Ví dụ:
- "Như hình 1 cho thấy, diễn biến trận đánh..." HAS_ILLUSTRATION "Hình 1: Bản đồ trận Bạch Đằng"
- "Bảng dưới đây thống kê các cuộc khởi nghĩa" HAS_ILLUSTRATION "Bảng 2: Thống kê"

Phân biệt với:
- VISUALIZES: Từ hình/bảng → đoạn văn (inverse)
- REFERS_TO: Tham chiếu chung, không phải trực quan""",
        families=[ELABORATIVE, STRUCTURAL],
        keywords=["xem hình", "hình dưới", "bảng sau", "như hình", "trong hình"],
        tiers=[TIER_2_CONTENT],
    ),
    RelationConstraint(
        relation_type=RelationType.RELATED_TO,
        source_node_types=ALL_TYPES,
        target_node_types=ALL_TYPES,
        is_symmetric=True,
        summary="Hai đoạn có liên quan chung về chủ đề/nội dung",
        description="""Quan hệ liên quan chung: A và B có liên quan nhưng không thuộc loại cụ thể.

Đặc điểm:
- Symmetric - A ↔ B
- Fallback relationship khi không xác định được quan hệ cụ thể hơn
- Thể hiện có sự liên quan về chủ đề nhưng không phải elaboration, contrast, etc.

Ví dụ:
- "Chính sách thuế nhà Nguyễn" RELATED_TO "Đời sống nông dân thế kỷ 19"
- "Văn hóa thời Lý" RELATED_TO "Phật giáo Đại Việt"

Phân biệt với:
- SIMILAR_TO: Tương đồng cao về ngữ nghĩa
- REFERS_TO: Tham chiếu trực tiếp
- ELABORATES: Mở rộng chi tiết""",
        families=[COMPARATIVE, ELABORATIVE],
        keywords=["liên quan", "có liên hệ", "liên quan đến", "gắn với"],
        tiers=[TIER_2_CONTENT],
    ),
    # =========================================================================
    # TIER 2 ONLY - EDUCATIONAL (2 relations) - NEW!
    # =========================================================================
    RelationConstraint(
        relation_type=RelationType.PREREQUISITE_OF,
        source_node_types=CONTENT_TYPES,
        target_node_types=CONTENT_TYPES,
        is_transitive=True,
        summary="Cần hiểu đoạn A trước khi hiểu đoạn B",
        description="""Quan hệ tiên quyết: Phải hiểu A trước để hiểu B.

Đặc điểm:
- Quan trọng cho educational content
- Transitive
- Xác định learning path

Ví dụ:
- "Khái niệm chế độ phong kiến" PREREQUISITE_OF "Phân tích xã hội Đại Việt"
- "Nguyên nhân khởi nghĩa" PREREQUISITE_OF "Diễn biến khởi nghĩa"

Phân biệt với:
- BUILDS_UPON: Phát triển từ, không bắt buộc""",
        families=[STRUCTURAL, ELABORATIVE],
        keywords=["trước tiên", "cần hiểu", "điều kiện", "tiên quyết"],
        tiers=[TIER_2_CONTENT],
    ),
    RelationConstraint(
        relation_type=RelationType.BUILDS_UPON,
        source_node_types=CONTENT_TYPES,
        target_node_types=CONTENT_TYPES,
        is_transitive=True,
        summary="Đoạn A mở rộng/phát triển kiến thức từ B",
        description="""Quan hệ phát triển: A xây dựng dựa trên kiến thức B.

Đặc điểm:
- A mở rộng, nâng cao từ B
- Không bắt buộc như PREREQUISITE_OF
- Transitive

Ví dụ:
- "Phân tích chuyên sâu về chiến thuật" BUILDS_UPON "Giới thiệu chiến thuật cơ bản"
- "Đánh giá tổng hợp" BUILDS_UPON "Mô tả từng sự kiện riêng lẻ"

Phân biệt với:
- PREREQUISITE_OF: Bắt buộc phải hiểu trước
- ELABORATES: Giải thích chi tiết""",
        families=[ELABORATIVE, STRUCTURAL],
        keywords=["phát triển từ", "mở rộng từ", "dựa trên", "nâng cao"],
        tiers=[TIER_2_CONTENT],
    ),
]


# Default base ontology instance
BASE_ONTOLOGY = BaseOntology(
    name="base",
    display_name="Base Ontology",
    relations=BASE_RELATIONS,
)
