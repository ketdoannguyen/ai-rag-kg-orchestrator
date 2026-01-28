"""History domain ontology for Vietnamese History textbooks.

Provides domain-specific relation types for content tier (paragraphs 300-700 words).
These relations describe relationships BETWEEN PARAGRAPHS, not between entities.

DESIGN PRINCIPLES:
1. Only add relations NOT in base.py
2. No duplicates with base relations
3. History-specific: Causal, Temporal, Participation relations
4. All families must be accurate for query matching

Relations are organized into 4 groups:
1. CAUSAL - Cause and effect relationships (history-specific)
2. TEMPORAL - Time sequence relationships (history-specific)
3. PARTICIPATION - Historical figures and forces
4. TAXONOMIC - Classification (history-specific)
"""

from __future__ import annotations

from enum import Enum

from .base import (
    CONTENT_TYPES,
    PARAGRAPH,
    TIER_2_CONTENT,
    BaseOntology,
    RelationConstraint,
)
from .families import CAUSAL, PARTICIPATION, STRUCTURAL, TEMPORAL


class HistoryRelationType(str, Enum):
    """History-specific relationship types.

    Only includes relations NOT covered by base.py.
    Organized into 4 groups for Content Tier (paragraphs).
    """

    # =========================================================================
    # Group 1: Causal Relations (4) - History-specific causation
    # =========================================================================
    CAUSED_BY = "CAUSED_BY"  # Event A was caused by event B
    LED_TO = "LED_TO"  # Event A led to event B
    RESULTED_IN = "RESULTED_IN"  # Event A resulted in event B
    CONSEQUENCE_OF = "CONSEQUENCE_OF"  # A is consequence of B

    # =========================================================================
    # Group 2: Temporal Relations (4) - History-specific time
    # =========================================================================
    PRECEDED_BY = "PRECEDED_BY"  # A happened after B (B preceded A)
    FOLLOWED_BY = "FOLLOWED_BY"  # A happened before B (B followed A)
    DURING = "DURING"  # A happened during period B
    CONTEMPORARY_WITH = "CONTEMPORARY_WITH"  # A and B happened at same time

    # =========================================================================
    # Group 3: Participation Relations (4) - Historical figures
    # =========================================================================
    LED_BY = "LED_BY"  # Event A was led by figure B
    PARTICIPATED_IN = "PARTICIPATED_IN"  # Figure A participated in event B
    CONTRIBUTED_TO = "CONTRIBUTED_TO"  # Factor A contributed to B
    OPPOSED_BY = "OPPOSED_BY"  # A was opposed by force B

    # =========================================================================
    # Group 4: Taxonomic Relations (2) - Classification
    # =========================================================================
    IS_A = "IS_A"  # A is a type of B
    PART_OF = "PART_OF"  # A is part of B


# History-specific relation constraints (14 relations)
HISTORY_RELATION_CONSTRAINTS: list[RelationConstraint] = [
    # =========================================================================
    # Group 1: Causal Relations (4)
    # =========================================================================
    RelationConstraint(
        relation_type=HistoryRelationType.CAUSED_BY,
        source_node_types=CONTENT_TYPES,
        target_node_types=CONTENT_TYPES,
        summary="Đoạn A mô tả sự kiện bị gây ra bởi sự kiện trong đoạn B",
        description="""Mối quan hệ nhân quả ngược: A là kết quả, B là nguyên nhân.

Đặc điểm:
- B xảy ra TRƯỚC A
- B là nguyên nhân trực tiếp/gián tiếp của A
- Hướng: A ← B

Ví dụ:
- "Nhà Hồ sụp đổ nhanh" CAUSED_BY "Cải cách quá nhanh gây bất mãn"
- "Quân Minh thất bại Chi Lăng" CAUSED_BY "Mai phục nghĩa quân Lam Sơn"

Phân biệt với:
- LED_TO: Ngược hướng (A → B)
- CONSEQUENCE_OF: Nhấn mạnh hậu quả hơn""",
        families=[CAUSAL],
        keywords=["bị gây ra bởi", "do", "vì", "xuất phát từ", "bắt nguồn từ", "nguyên nhân"],
        tiers=[TIER_2_CONTENT],
    ),
    RelationConstraint(
        relation_type=HistoryRelationType.LED_TO,
        source_node_types=CONTENT_TYPES,
        target_node_types=CONTENT_TYPES,
        is_transitive=True,
        summary="Đoạn A mô tả sự kiện dẫn đến sự kiện trong đoạn B",
        description="""Mối quan hệ nhân quả xuôi: A là nguyên nhân, B là kết quả.

Đặc điểm:
- A xảy ra TRƯỚC B
- Transitive
- Hướng: A → B

Ví dụ:
- "Quân Pháp đổ bộ Đà Nẵng 1858" LED_TO "Ký hiệp ước Nhâm Tuất 1862"
- "Phong trào Cần Vương thất bại" LED_TO "Phong trào yêu nước mới đầu TK XX"

Phân biệt với:
- CAUSED_BY: Ngược hướng
- RESULTED_IN: Kết quả trực tiếp hơn""",
        families=[CAUSAL, TEMPORAL],
        keywords=["dẫn đến", "dẫn tới", "gây ra", "tạo ra", "làm cho", "khiến", "kéo theo"],
        tiers=[TIER_2_CONTENT],
    ),
    RelationConstraint(
        relation_type=HistoryRelationType.RESULTED_IN,
        source_node_types=CONTENT_TYPES,
        target_node_types=CONTENT_TYPES,
        is_transitive=True,
        summary="Đoạn A mô tả sự kiện có kết quả trực tiếp trong đoạn B",
        description="""Mối quan hệ kết quả trực tiếp: B là kết quả TRỰC TIẾP của A.

Đặc điểm:
- Nhân-quả rõ ràng, không qua trung gian
- Transitive

Ví dụ:
- "Chiến thắng Điện Biên Phủ 1954" RESULTED_IN "Ký Hiệp định Genève"
- "Tổng tiến công Xuân 1975" RESULTED_IN "Đất nước thống nhất"

Phân biệt với:
- LED_TO: Có thể gián tiếp
- CONSEQUENCE_OF: Ngược hướng""",
        families=[CAUSAL],
        keywords=["kết quả là", "mang lại", "đem lại", "thu được", "đạt được", "thành quả"],
        tiers=[TIER_2_CONTENT],
    ),
    RelationConstraint(
        relation_type=HistoryRelationType.CONSEQUENCE_OF,
        source_node_types=CONTENT_TYPES,
        target_node_types=CONTENT_TYPES,
        summary="Đoạn A mô tả hậu quả của sự kiện trong đoạn B",
        description="""Mối quan hệ hậu quả: A là hậu quả (thường tiêu cực/dài hạn) của B.

Đặc điểm:
- Nhấn mạnh tác động, ảnh hưởng
- Hướng: A ← B

Ví dụ:
- "Hàng triệu người thiệt mạng" CONSEQUENCE_OF "Kháng chiến chống Mỹ 21 năm"
- "Kinh tế kiệt quệ" CONSEQUENCE_OF "Chính sách vơ vét thực dân"

Phân biệt với:
- CAUSED_BY: Nhấn mạnh nguyên nhân
- RESULTED_IN: Kết quả tích cực/trung tính""",
        families=[CAUSAL],
        keywords=["hậu quả", "hệ quả", "tác động", "ảnh hưởng", "thiệt hại", "tổn thất"],
        tiers=[TIER_2_CONTENT],
    ),
    # =========================================================================
    # Group 2: Temporal Relations (4)
    # =========================================================================
    RelationConstraint(
        relation_type=HistoryRelationType.PRECEDED_BY,
        source_node_types=CONTENT_TYPES,
        target_node_types=CONTENT_TYPES,
        summary="Sự kiện trong đoạn A xảy ra SAU sự kiện trong đoạn B",
        description="""Mối quan hệ thời gian: B xảy ra TRƯỚC A.

Đặc điểm:
- Không nhất thiết có quan hệ nhân quả
- Hướng: A ← B (B đi trước A)

Ví dụ:
- "Khởi nghĩa Lam Sơn 1418" PRECEDED_BY "KN Trần Quý Khoáng 1407-1413"
- "Cách mạng tháng Tám 1945" PRECEDED_BY "Xô viết Nghệ Tĩnh 1930"

Phân biệt với:
- FOLLOWED_BY: Ngược hướng
- CAUSED_BY: Có quan hệ nhân quả""",
        families=[TEMPORAL],
        keywords=["trước đó", "trước khi", "đã từng", "trước năm", "tiền thân"],
        tiers=[TIER_2_CONTENT],
    ),
    RelationConstraint(
        relation_type=HistoryRelationType.FOLLOWED_BY,
        source_node_types=CONTENT_TYPES,
        target_node_types=CONTENT_TYPES,
        summary="Sự kiện trong đoạn A xảy ra TRƯỚC sự kiện trong đoạn B",
        description="""Mối quan hệ thời gian: A xảy ra TRƯỚC B.

Đặc điểm:
- Không nhất thiết có quan hệ nhân quả
- Hướng: A → B (B theo sau A)

Ví dụ:
- "Triều Lý thành lập 1009" FOLLOWED_BY "Triều Trần 1225"
- "CTTG I kết thúc 1918" FOLLOWED_BY "Phong trào công nhân VN bùng nổ"

Phân biệt với:
- PRECEDED_BY: Ngược hướng
- LED_TO: Có quan hệ nhân quả""",
        families=[TEMPORAL],
        keywords=["sau đó", "tiếp theo", "kế tiếp", "sau năm", "về sau"],
        tiers=[TIER_2_CONTENT],
    ),
    RelationConstraint(
        relation_type=HistoryRelationType.DURING,
        source_node_types=CONTENT_TYPES,
        target_node_types=CONTENT_TYPES,
        summary="Sự kiện trong đoạn A xảy ra TRONG thời kỳ mô tả ở đoạn B",
        description="""Mối quan hệ thời gian bao hàm: A nằm trong khoảng thời gian B.

Đặc điểm:
- A là sự kiện cụ thể
- B là giai đoạn/thời kỳ rộng hơn

Ví dụ:
- "Trận Bạch Đằng 938" DURING "Bắc thuộc lần ba 905-938"
- "Phong trào Đông Du 1905-1909" DURING "Giai đoạn đầu TK XX"

Phân biệt với:
- CONTEMPORARY_WITH: Hai sự kiện cùng cấp độ
- PART_OF: Quan hệ cấu trúc""",
        families=[TEMPORAL],
        keywords=["trong thời kỳ", "trong giai đoạn", "dưới thời", "thời", "trong suốt"],
        tiers=[TIER_2_CONTENT],
    ),
    RelationConstraint(
        relation_type=HistoryRelationType.CONTEMPORARY_WITH,
        source_node_types=CONTENT_TYPES,
        target_node_types=CONTENT_TYPES,
        is_symmetric=True,
        summary="Đoạn A và đoạn B mô tả sự kiện xảy ra CÙNG thời điểm",
        description="""Mối quan hệ đồng thời: A và B xảy ra cùng khoảng thời gian.

Đặc điểm:
- Symmetric
- Không có quan hệ nhân quả trực tiếp

Ví dụ:
- "Cần Vương ở Bắc Kỳ" CONTEMPORARY_WITH "KN Ba Đình Thanh Hóa"
- "CTTG II 1939-1945" CONTEMPORARY_WITH "Cao trào kháng Nhật VN"

Phân biệt với:
- DURING: Một sự kiện nằm trong giai đoạn kia
- SIMILAR_TO: Tương đồng nội dung""",
        families=[TEMPORAL],
        keywords=["cùng thời", "đồng thời", "song song", "cùng lúc", "trong khi đó"],
        tiers=[TIER_2_CONTENT],
    ),
    # =========================================================================
    # Group 3: Participation Relations (4)
    # =========================================================================
    RelationConstraint(
        relation_type=HistoryRelationType.LED_BY,
        source_node_types=CONTENT_TYPES,
        target_node_types=[PARAGRAPH],
        summary="Sự kiện trong đoạn A được LÃNH ĐẠO bởi nhân vật trong đoạn B",
        description="""Mối quan hệ lãnh đạo: B là người lãnh đạo sự kiện A.

Đặc điểm:
- Nhấn mạnh vai trò đầu tàu
- Hướng: A ← B

Ví dụ:
- "Khởi nghĩa Lam Sơn 1418-1427" LED_BY "Lê Lợi - anh hùng dân tộc"
- "Cách mạng tháng Tám 1945" LED_BY "Hồ Chí Minh và ĐCSVN"

Phân biệt với:
- PARTICIPATED_IN: Chỉ tham gia, không lãnh đạo
- CONTRIBUTED_TO: Đóng góp, không lãnh đạo""",
        families=[PARTICIPATION],
        keywords=[
            "lãnh đạo bởi",
            "do ai lãnh đạo",
            "người đứng đầu",
            "chỉ huy",
            "dưới sự lãnh đạo",
        ],
        tiers=[TIER_2_CONTENT],
    ),
    RelationConstraint(
        relation_type=HistoryRelationType.PARTICIPATED_IN,
        source_node_types=[PARAGRAPH],
        target_node_types=CONTENT_TYPES,
        summary="Nhân vật trong đoạn A THAM GIA vào sự kiện trong đoạn B",
        description="""Mối quan hệ tham gia: A tham gia vào sự kiện B.

Đặc điểm:
- Không nhất thiết vai trò lãnh đạo
- Hướng: A → B

Ví dụ:
- "Nguyễn Trãi - quân sự ngoại giao" PARTICIPATED_IN "Kháng chiến chống Minh"
- "Nhân dân Nghệ An - Hà Tĩnh" PARTICIPATED_IN "Xô viết Nghệ Tĩnh 1930"

Phân biệt với:
- LED_BY: Vai trò lãnh đạo (ngược hướng)
- CONTRIBUTED_TO: Nhấn mạnh đóng góp""",
        families=[PARTICIPATION],
        keywords=["tham gia", "tham dự", "có mặt", "góp phần", "trong sự kiện"],
        tiers=[TIER_2_CONTENT],
    ),
    RelationConstraint(
        relation_type=HistoryRelationType.CONTRIBUTED_TO,
        source_node_types=CONTENT_TYPES,
        target_node_types=CONTENT_TYPES,
        summary="Nhân vật/yếu tố trong đoạn A ĐÓNG GÓP vào kết quả trong đoạn B",
        description="""Mối quan hệ đóng góp: A đóng góp vào thành công B.

Đặc điểm:
- Có thể trực tiếp hoặc gián tiếp
- Hướng: A → B

Ví dụ:
- "Tinh thần yêu nước của nhân dân" CONTRIBUTED_TO "Chiến thắng ĐBP 1954"
- "Viện trợ Liên Xô, Trung Quốc" CONTRIBUTED_TO "Thắng lợi kháng chiến chống Pháp"

Phân biệt với:
- CAUSED_BY/LED_TO: Nhân quả trực tiếp
- PARTICIPATED_IN: Sự tham gia""",
        families=[PARTICIPATION, CAUSAL],
        keywords=["đóng góp", "góp phần", "giúp cho", "hỗ trợ", "tạo điều kiện"],
        tiers=[TIER_2_CONTENT],
    ),
    RelationConstraint(
        relation_type=HistoryRelationType.OPPOSED_BY,
        source_node_types=CONTENT_TYPES,
        target_node_types=CONTENT_TYPES,
        summary="Sự kiện/chính sách trong đoạn A bị PHẢN ĐỐI bởi lực lượng trong đoạn B",
        description="""Mối quan hệ đối lập: B chống lại, phản đối A.

Đặc điểm:
- Thể hiện xung đột, mâu thuẫn
- Hướng: A ← B

Ví dụ:
- "Chính sách khai thác thuộc địa Pháp" OPPOSED_BY "Phong trào yêu nước đầu TK XX"
- "Âm mưu chia cắt của Mỹ" OPPOSED_BY "Kháng chiến nhân dân miền Nam"

Phân biệt với:
- CONTRASTS: So sánh, không xung đột
- CONTRADICTS: Mâu thuẫn quan điểm""",
        families=[PARTICIPATION],
        keywords=["bị phản đối", "bị chống lại", "đấu tranh chống", "kháng cự", "nổi dậy chống"],
        tiers=[TIER_2_CONTENT],
    ),
    # =========================================================================
    # Group 4: Taxonomic Relations (2)
    # =========================================================================
    RelationConstraint(
        relation_type=HistoryRelationType.IS_A,
        source_node_types=CONTENT_TYPES,
        target_node_types=CONTENT_TYPES,
        is_transitive=True,
        summary="Sự kiện trong đoạn A là MỘT LOẠI của khái niệm trong đoạn B",
        description="""Mối quan hệ phân loại: A là loại cụ thể của B.

Đặc điểm:
- B là danh mục rộng hơn
- Transitive

Ví dụ:
- "Khởi nghĩa Hai Bà Trưng 40" IS_A "Các cuộc KN chống Bắc thuộc"
- "Phong trào Cần Vương" IS_A "Phong trào yêu nước cuối TK XIX"

Phân biệt với:
- PART_OF: Bộ phận-tổng thể
- EXEMPLIFIES: Ví dụ minh họa""",
        families=[STRUCTURAL],
        keywords=["là một loại", "thuộc loại", "được xếp vào", "nằm trong nhóm"],
        tiers=[TIER_2_CONTENT],
    ),
    RelationConstraint(
        relation_type=HistoryRelationType.PART_OF,
        source_node_types=CONTENT_TYPES,
        target_node_types=CONTENT_TYPES,
        is_transitive=True,
        summary="Sự kiện trong đoạn A là MỘT PHẦN của sự kiện lớn hơn trong đoạn B",
        description="""Mối quan hệ bộ phận-tổng thể: A là một phần của B.

Đặc điểm:
- B là tổng thể lớn hơn
- Transitive

Ví dụ:
- "Trận Bạch Đằng 1288" PART_OF "Kháng chiến chống Nguyên-Mông lần 3"
- "Chiến dịch Hồ Chí Minh" PART_OF "Tổng tiến công Xuân 1975"

Phân biệt với:
- IS_A: Phân loại
- DURING: Quan hệ thời gian""",
        families=[STRUCTURAL],
        keywords=["là một phần", "thuộc về", "nằm trong", "bao gồm", "gồm có"],
        tiers=[TIER_2_CONTENT],
    ),
]


def _build_history_ontology() -> BaseOntology:
    """Build history ontology with base + history-specific relations."""
    from .base import BASE_RELATIONS, TIER_1_STRUCTURE

    # Get all base relations for Tier 1 + Tier 2
    base_relations = [
        r for r in BASE_RELATIONS if TIER_2_CONTENT in r.tiers or TIER_1_STRUCTURE in r.tiers
    ]

    # Combine with history-specific relations
    all_relations = base_relations + HISTORY_RELATION_CONSTRAINTS

    return BaseOntology(
        name="history",
        display_name="Lịch sử Việt Nam Ontology",
        relations=all_relations,
    )


# Singleton instance
HISTORY_ONTOLOGY = _build_history_ontology()
