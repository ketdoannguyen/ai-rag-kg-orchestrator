"""Benchmark samples for EXTRACTION_MERGE_THRESHOLD evaluation.

70 sample pairs of relations with ground truth labels.
Domain: Vietnamese History (Lịch sử Việt Nam)

FOCUS: Testing BOUNDARY CASES between merge and no-merge.
- All pairs have DIFFERENT names (exact name match has separate logic)
- Format matches ontology description format from history.py/base.py

Each sample contains:
- relation_a: First relation (name + summary + description)
- relation_b: Second relation (name + summary + description)
- should_merge: Ground truth - True if semantically equivalent, False if distinct
- category: Type of comparison
- reasoning: Explanation for the ground truth

Categories:
1. SEMANTIC_SYNONYMS (should_merge=True): Same meaning, different names
2. DIRECTIONAL_INVERSES (should_merge=False): Same concept but opposite direction - CRITICAL
3. GRANULARITY_VARIANTS (boundary cases): Same concept, different specificity
4. RELATED_DISTINCT (should_merge=False): Same family, different concepts
5. DOMAIN_OVERLAPS (boundary cases): History-specific overlapping concepts
6. CROSS_FAMILY (should_merge=False): Similar wording, different families
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class MergeCategory(str, Enum):
    """Categories of relation pairs for analysis."""

    SEMANTIC_SYNONYMS = "semantic_synonyms"  # Same meaning, different names -> MERGE
    DIRECTIONAL_INVERSES = "directional_inverses"  # Opposite directions -> NO MERGE
    GRANULARITY_VARIANTS = "granularity_variants"  # Different specificity -> DEPENDS
    RELATED_DISTINCT = "related_distinct"  # Same family, different concepts -> NO MERGE
    DOMAIN_OVERLAPS = "domain_overlaps"  # History-specific overlaps -> DEPENDS
    CROSS_FAMILY = "cross_family"  # Similar wording, different families -> NO MERGE


@dataclass
class RelationSample:
    """A sample pair of relations for benchmark.

    Format matches ontology description from history.py:
    - name: SCREAMING_SNAKE_CASE
    - summary: 10-20 words Vietnamese
    - description: Detailed with examples
    """

    id: int
    category: MergeCategory

    # Relation A
    relation_a_name: str
    relation_a_summary: str
    relation_a_description: str

    # Relation B
    relation_b_name: str
    relation_b_summary: str
    relation_b_description: str

    # Ground truth
    should_merge: bool
    reasoning: str

    # For analysis
    is_boundary_case: bool = False  # Near threshold - critical for evaluation


# =============================================================================
# CATEGORY 1: SEMANTIC SYNONYMS (should_merge=True) - 12 samples
# Các relation khác tên nhưng CÓ CÙNG Ý NGHĨA ngữ nghĩa
# =============================================================================

SEMANTIC_SYNONYM_SAMPLES = [
    RelationSample(
        id=1,
        category=MergeCategory.SEMANTIC_SYNONYMS,
        relation_a_name="LED_TO",
        relation_a_summary="Đoạn A mô tả sự kiện dẫn đến sự kiện trong đoạn B",
        relation_a_description="""Mối quan hệ nhân quả xuôi: A là nguyên nhân, B là kết quả.

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
        relation_b_name="RESULTED_IN",
        relation_b_summary="Đoạn A mô tả sự kiện có kết quả trực tiếp trong đoạn B",
        relation_b_description="""Mối quan hệ kết quả trực tiếp: B là kết quả TRỰC TIẾP của A.

Đặc điểm:
- Nhân-quả rõ ràng, không qua trung gian
- Transitive
- Hướng: A → B

Ví dụ:
- "Chiến thắng Điện Biên Phủ 1954" RESULTED_IN "Ký Hiệp định Genève"
- "Tổng tiến công Xuân 1975" RESULTED_IN "Đất nước thống nhất"

Phân biệt với:
- LED_TO: Có thể gián tiếp
- CONSEQUENCE_OF: Ngược hướng""",
        should_merge=True,
        reasoning="Cả hai đều là nhân quả xuôi A→B. LED_TO có thể gián tiếp, RESULTED_IN trực tiếp hơn, "
        "nhưng về bản chất semantic trong knowledge graph là tương đương.",
        is_boundary_case=True,
    ),
    RelationSample(
        id=2,
        category=MergeCategory.SEMANTIC_SYNONYMS,
        relation_a_name="CAUSED_BY",
        relation_a_summary="Đoạn A mô tả sự kiện bị gây ra bởi sự kiện trong đoạn B",
        relation_a_description="""Mối quan hệ nhân quả ngược: A là kết quả, B là nguyên nhân.

Đặc điểm:
- B xảy ra TRƯỚC A
- Hướng: A ← B

Ví dụ:
- "Nhà Hồ sụp đổ nhanh" CAUSED_BY "Cải cách quá nhanh gây bất mãn"
- "Quân Minh thất bại Chi Lăng" CAUSED_BY "Mai phục nghĩa quân Lam Sơn"

Phân biệt với:
- LED_TO: Ngược hướng (A → B)""",
        relation_b_name="CONSEQUENCE_OF",
        relation_b_summary="Đoạn A mô tả hậu quả của sự kiện trong đoạn B",
        relation_b_description="""Mối quan hệ hậu quả: A là hậu quả (thường tiêu cực/dài hạn) của B.

Đặc điểm:
- Nhấn mạnh tác động, ảnh hưởng
- Hướng: A ← B

Ví dụ:
- "Hàng triệu người thiệt mạng" CONSEQUENCE_OF "Kháng chiến chống Mỹ 21 năm"
- "Kinh tế kiệt quệ" CONSEQUENCE_OF "Chính sách vơ vét thực dân"

Phân biệt với:
- CAUSED_BY: Nhấn mạnh nguyên nhân
- RESULTED_IN: Kết quả tích cực/trung tính""",
        should_merge=True,
        reasoning="Cả hai đều là nhân quả ngược A←B. CAUSED_BY nhấn mạnh nguyên nhân, "
        "CONSEQUENCE_OF nhấn mạnh hậu quả, nhưng cùng hướng semantic.",
        is_boundary_case=True,
    ),
    RelationSample(
        id=3,
        category=MergeCategory.SEMANTIC_SYNONYMS,
        relation_a_name="PRECEDED_BY",
        relation_a_summary="Sự kiện trong đoạn A xảy ra SAU sự kiện trong đoạn B",
        relation_a_description="""Mối quan hệ thời gian: B xảy ra TRƯỚC A.

Đặc điểm:
- Không nhất thiết có quan hệ nhân quả
- Hướng: A ← B (B đi trước A)

Ví dụ:
- "Khởi nghĩa Lam Sơn 1418" PRECEDED_BY "KN Trần Quý Khoáng 1407-1413"

Phân biệt với:
- FOLLOWED_BY: Ngược hướng""",
        relation_b_name="COMES_AFTER",
        relation_b_summary="Sự kiện trong đoạn A diễn ra sau sự kiện trong đoạn B",
        relation_b_description="""Mối quan hệ thời gian: A xảy ra sau B trong dòng lịch sử.

Đặc điểm:
- Thuần túy thời gian, không có nhân quả
- Hướng: A ← B (A đến sau B)

Ví dụ:
- "Triều Trần 1225" COMES_AFTER "Triều Lý 1009"

Phân biệt với:
- COMES_BEFORE: Ngược hướng""",
        should_merge=True,
        reasoning="Cả hai đều mô tả A xảy ra sau B trong thời gian. PRECEDED_BY và COMES_AFTER "
        "là đồng nghĩa hoàn toàn.",
        is_boundary_case=False,
    ),
    RelationSample(
        id=4,
        category=MergeCategory.SEMANTIC_SYNONYMS,
        relation_a_name="FOLLOWED_BY",
        relation_a_summary="Sự kiện trong đoạn A xảy ra TRƯỚC sự kiện trong đoạn B",
        relation_a_description="""Mối quan hệ thời gian: A xảy ra TRƯỚC B.

Đặc điểm:
- Không nhất thiết có quan hệ nhân quả
- Hướng: A → B (B theo sau A)

Ví dụ:
- "Triều Lý thành lập 1009" FOLLOWED_BY "Triều Trần 1225"

Phân biệt với:
- PRECEDED_BY: Ngược hướng""",
        relation_b_name="COMES_BEFORE",
        relation_b_summary="Sự kiện trong đoạn A diễn ra trước sự kiện trong đoạn B",
        relation_b_description="""Mối quan hệ thời gian: A xảy ra trước B.

Đặc điểm:
- Thuần túy thời gian
- Hướng: A → B (A đến trước B)

Ví dụ:
- "Triều Lý 1009" COMES_BEFORE "Triều Trần 1225"

Phân biệt với:
- COMES_AFTER: Ngược hướng""",
        should_merge=True,
        reasoning="Cả hai đều mô tả A xảy ra trước B. FOLLOWED_BY và COMES_BEFORE là đồng nghĩa.",
        is_boundary_case=False,
    ),
    RelationSample(
        id=5,
        category=MergeCategory.SEMANTIC_SYNONYMS,
        relation_a_name="OPPOSED_BY",
        relation_a_summary="Sự kiện/chính sách trong đoạn A bị PHẢN ĐỐI bởi lực lượng trong đoạn B",
        relation_a_description="""Mối quan hệ đối lập: B chống lại, phản đối A.

Đặc điểm:
- Thể hiện xung đột, mâu thuẫn
- Hướng: A ← B

Ví dụ:
- "Chính sách khai thác thuộc địa Pháp" OPPOSED_BY "Phong trào yêu nước đầu TK XX"

Phân biệt với:
- CONTRASTS: So sánh, không xung đột""",
        relation_b_name="RESISTED_BY",
        relation_b_summary="Sự kiện/chính sách A bị kháng cự bởi lực lượng B",
        relation_b_description="""Mối quan hệ kháng cự: B chống lại A.

Đặc điểm:
- Thể hiện sự phản kháng
- Hướng: A ← B

Ví dụ:
- "Ách đô hộ phong kiến phương Bắc" RESISTED_BY "Các cuộc khởi nghĩa chống Bắc thuộc"

Phân biệt với:
- SUPPORTS: Ủng hộ, không chống""",
        should_merge=True,
        reasoning="OPPOSED_BY và RESISTED_BY đều mô tả B chống lại A. "
        "Resisted có thể mạnh hơn nhưng semantic tương đương.",
        is_boundary_case=False,
    ),
    RelationSample(
        id=6,
        category=MergeCategory.SEMANTIC_SYNONYMS,
        relation_a_name="LED_BY",
        relation_a_summary="Sự kiện trong đoạn A được LÃNH ĐẠO bởi nhân vật trong đoạn B",
        relation_a_description="""Mối quan hệ lãnh đạo: B là người lãnh đạo sự kiện A.

Đặc điểm:
- Nhấn mạnh vai trò đầu tàu
- Hướng: A ← B

Ví dụ:
- "Khởi nghĩa Lam Sơn 1418-1427" LED_BY "Lê Lợi - anh hùng dân tộc"

Phân biệt với:
- PARTICIPATED_IN: Chỉ tham gia, không lãnh đạo""",
        relation_b_name="COMMANDED_BY",
        relation_b_summary="Sự kiện/chiến dịch A được chỉ huy bởi nhân vật B",
        relation_b_description="""Mối quan hệ chỉ huy: B là người chỉ huy sự kiện A.

Đặc điểm:
- Nhấn mạnh vai trò điều khiển
- Thường dùng cho quân sự
- Hướng: A ← B

Ví dụ:
- "Chiến dịch Điện Biên Phủ" COMMANDED_BY "Đại tướng Võ Nguyên Giáp"

Phân biệt với:
- PARTICIPATED_IN: Chỉ tham gia""",
        should_merge=True,
        reasoning="LED_BY và COMMANDED_BY đều mô tả B lãnh đạo/chỉ huy A. "
        "COMMANDED_BY thiên về quân sự nhưng semantic tương đương.",
        is_boundary_case=True,
    ),
    RelationSample(
        id=7,
        category=MergeCategory.SEMANTIC_SYNONYMS,
        relation_a_name="CONTRIBUTED_TO",
        relation_a_summary="Nhân vật/yếu tố trong đoạn A ĐÓNG GÓP vào kết quả trong đoạn B",
        relation_a_description="""Mối quan hệ đóng góp: A đóng góp vào thành công B.

Đặc điểm:
- Có thể trực tiếp hoặc gián tiếp
- Hướng: A → B

Ví dụ:
- "Tinh thần yêu nước của nhân dân" CONTRIBUTED_TO "Chiến thắng ĐBP 1954"

Phân biệt với:
- CAUSED_BY/LED_TO: Nhân quả trực tiếp""",
        relation_b_name="HELPED_ACHIEVE",
        relation_b_summary="Yếu tố A giúp đạt được kết quả B",
        relation_b_description="""Mối quan hệ hỗ trợ: A giúp B thành công.

Đặc điểm:
- Là một phần nguyên nhân
- Hướng: A → B

Ví dụ:
- "Viện trợ quốc tế" HELPED_ACHIEVE "Thắng lợi kháng chiến"

Phân biệt với:
- LED_TO: Nguyên nhân chính""",
        should_merge=True,
        reasoning="CONTRIBUTED_TO và HELPED_ACHIEVE đều mô tả A góp phần vào B. "
        "Semantic tương đương.",
        is_boundary_case=False,
    ),
    RelationSample(
        id=8,
        category=MergeCategory.SEMANTIC_SYNONYMS,
        relation_a_name="ELABORATES",
        relation_a_summary="Đoạn A mở rộng/giải thích chi tiết đoạn B",
        relation_a_description="""Quan hệ mở rộng: A cung cấp thông tin chi tiết cho B.

Đặc điểm:
- A chi tiết hơn B
- Transitive

Ví dụ:
- "Cụ thể, chiến thuật gồm 3 bước..." ELABORATES "Ngô Quyền dùng mai phục"

Phân biệt với:
- SUMMARIZES: Ngược lại""",
        relation_b_name="EXPANDS_ON",
        relation_b_summary="Đoạn A mở rộng nội dung từ đoạn B",
        relation_b_description="""Quan hệ mở rộng: A phát triển chi tiết từ B.

Đặc điểm:
- A thêm thông tin cho B
- Hướng: A → B

Ví dụ:
- "Chi tiết về các chiến thuật..." EXPANDS_ON "Tổng quan chiến lược"

Phân biệt với:
- SUMMARIZES: Rút gọn""",
        should_merge=True,
        reasoning="ELABORATES và EXPANDS_ON đều mô tả A mở rộng chi tiết cho B. Đồng nghĩa.",
        is_boundary_case=False,
    ),
    RelationSample(
        id=9,
        category=MergeCategory.SEMANTIC_SYNONYMS,
        relation_a_name="CONTEMPORARY_WITH",
        relation_a_summary="Đoạn A và đoạn B mô tả sự kiện xảy ra CÙNG thời điểm",
        relation_a_description="""Mối quan hệ đồng thời: A và B xảy ra cùng khoảng thời gian.

Đặc điểm:
- Symmetric
- Không có quan hệ nhân quả trực tiếp

Ví dụ:
- "Cần Vương ở Bắc Kỳ" CONTEMPORARY_WITH "KN Ba Đình Thanh Hóa"

Phân biệt với:
- DURING: Một sự kiện nằm trong giai đoạn kia""",
        relation_b_name="CONCURRENT_WITH",
        relation_b_summary="Sự kiện A xảy ra đồng thời với sự kiện B",
        relation_b_description="""Mối quan hệ song song: A và B diễn ra cùng lúc.

Đặc điểm:
- Symmetric
- Cùng thời điểm lịch sử

Ví dụ:
- "Phong trào công nhân" CONCURRENT_WITH "Phong trào nông dân"

Phân biệt với:
- DURING: Bao hàm thời gian""",
        should_merge=True,
        reasoning="CONTEMPORARY_WITH và CONCURRENT_WITH đều mô tả A và B xảy ra cùng thời. Đồng nghĩa.",
        is_boundary_case=False,
    ),
    RelationSample(
        id=10,
        category=MergeCategory.SEMANTIC_SYNONYMS,
        relation_a_name="PART_OF",
        relation_a_summary="Sự kiện trong đoạn A là MỘT PHẦN của sự kiện lớn hơn trong đoạn B",
        relation_a_description="""Mối quan hệ bộ phận-tổng thể: A là một phần của B.

Đặc điểm:
- B là tổng thể lớn hơn
- Transitive

Ví dụ:
- "Trận Bạch Đằng 1288" PART_OF "Kháng chiến chống Nguyên-Mông lần 3"

Phân biệt với:
- IS_A: Phân loại""",
        relation_b_name="BELONGS_TO",
        relation_b_summary="Sự kiện A thuộc về/nằm trong sự kiện lớn hơn B",
        relation_b_description="""Mối quan hệ thuộc về: A nằm trong B.

Đặc điểm:
- A là thành phần của B
- Transitive

Ví dụ:
- "Chiến dịch Hồ Chí Minh" BELONGS_TO "Tổng tiến công Xuân 1975"

Phân biệt với:
- IS_A: Phân loại taxonomy""",
        should_merge=True,
        reasoning="PART_OF và BELONGS_TO đều mô tả A là bộ phận của B. Đồng nghĩa trong context này.",
        is_boundary_case=False,
    ),
    RelationSample(
        id=11,
        category=MergeCategory.SEMANTIC_SYNONYMS,
        relation_a_name="SUPPORTS",
        relation_a_summary="Đoạn A cung cấp bằng chứng/lập luận ủng hộ B",
        relation_a_description="""Quan hệ ủng hộ: A cung cấp bằng chứng cho B.

Đặc điểm:
- A làm tăng độ tin cậy của B
- Transitive

Ví dụ:
- "Số liệu cho thấy sản lượng tăng 20%" SUPPORTS "Cải cách hiệu quả"

Phân biệt với:
- ELABORATES: Giải thích, không ủng hộ""",
        relation_b_name="PROVIDES_EVIDENCE_FOR",
        relation_b_summary="Đoạn A cung cấp bằng chứng cho luận điểm trong đoạn B",
        relation_b_description="""Quan hệ chứng cứ: A là evidence cho B.

Đặc điểm:
- A chứng minh B
- Hướng: A → B

Ví dụ:
- "Tài liệu lưu trữ" PROVIDES_EVIDENCE_FOR "Luận điểm sử học"

Phân biệt với:
- ELABORATES: Giải thích không chứng minh""",
        should_merge=True,
        reasoning="SUPPORTS và PROVIDES_EVIDENCE_FOR đều mô tả A là bằng chứng cho B. Đồng nghĩa.",
        is_boundary_case=False,
    ),
    RelationSample(
        id=12,
        category=MergeCategory.SEMANTIC_SYNONYMS,
        relation_a_name="SUMMARIZES",
        relation_a_summary="Đoạn A tóm tắt nội dung đoạn B",
        relation_a_description="""Quan hệ tóm tắt: A rút gọn nội dung B.

Đặc điểm:
- A ngắn gọn hơn B
- A chứa ý chính của B

Ví dụ:
- "Tóm lại, nguyên nhân chính là mâu thuẫn giai cấp" SUMMARIZES "Phân tích chi tiết"

Phân biệt với:
- ELABORATES: Ngược lại""",
        relation_b_name="CONDENSES",
        relation_b_summary="Đoạn A rút gọn nội dung từ đoạn B",
        relation_b_description="""Quan hệ rút gọn: A là phiên bản ngắn của B.

Đặc điểm:
- A ngắn hơn B
- Giữ ý chính

Ví dụ:
- "Kết luận ngắn gọn" CONDENSES "Phân tích dài"

Phân biệt với:
- EXPANDS: Ngược lại""",
        should_merge=True,
        reasoning="SUMMARIZES và CONDENSES đều mô tả A là bản rút gọn của B. Đồng nghĩa.",
        is_boundary_case=False,
    ),
]


# =============================================================================
# CATEGORY 2: DIRECTIONAL INVERSES (should_merge=False) - 10 samples
# Cùng concept nhưng NGƯỢC HƯỚNG - KHÔNG ĐƯỢC MERGE
# =============================================================================

DIRECTIONAL_INVERSE_SAMPLES = [
    RelationSample(
        id=13,
        category=MergeCategory.DIRECTIONAL_INVERSES,
        relation_a_name="LED_TO",
        relation_a_summary="Đoạn A mô tả sự kiện dẫn đến sự kiện trong đoạn B",
        relation_a_description="""Mối quan hệ nhân quả xuôi: A là nguyên nhân, B là kết quả.
Hướng: A → B

Ví dụ: "Phong trào Cần Vương thất bại" LED_TO "Phong trào yêu nước mới đầu TK XX"
""",
        relation_b_name="CAUSED_BY",
        relation_b_summary="Đoạn A mô tả sự kiện bị gây ra bởi sự kiện trong đoạn B",
        relation_b_description="""Mối quan hệ nhân quả ngược: A là kết quả, B là nguyên nhân.
Hướng: A ← B

Ví dụ: "Nhà Hồ sụp đổ nhanh" CAUSED_BY "Cải cách quá nhanh gây bất mãn"
""",
        should_merge=False,
        reasoning="CRITICAL: Ngược hướng! LED_TO: A→B, CAUSED_BY: B→A. "
        "Merge sẽ đảo ngược quan hệ nhân quả trong knowledge graph.",
        is_boundary_case=True,
    ),
    RelationSample(
        id=14,
        category=MergeCategory.DIRECTIONAL_INVERSES,
        relation_a_name="PRECEDED_BY",
        relation_a_summary="Sự kiện trong đoạn A xảy ra SAU sự kiện trong đoạn B",
        relation_a_description="""Mối quan hệ thời gian: B xảy ra TRƯỚC A.
Hướng: A ← B (B đi trước A)

Ví dụ: "Khởi nghĩa Lam Sơn 1418" PRECEDED_BY "KN Trần Quý Khoáng 1407-1413"
""",
        relation_b_name="FOLLOWED_BY",
        relation_b_summary="Sự kiện trong đoạn A xảy ra TRƯỚC sự kiện trong đoạn B",
        relation_b_description="""Mối quan hệ thời gian: A xảy ra TRƯỚC B.
Hướng: A → B (B theo sau A)

Ví dụ: "Triều Lý thành lập 1009" FOLLOWED_BY "Triều Trần 1225"
""",
        should_merge=False,
        reasoning="CRITICAL: Ngược hướng thời gian! PRECEDED_BY: B→A, FOLLOWED_BY: A→B. "
        "Merge sẽ đảo ngược trình tự thời gian.",
        is_boundary_case=True,
    ),
    RelationSample(
        id=15,
        category=MergeCategory.DIRECTIONAL_INVERSES,
        relation_a_name="RESULTED_IN",
        relation_a_summary="Đoạn A mô tả sự kiện có kết quả trực tiếp trong đoạn B",
        relation_a_description="""Mối quan hệ kết quả: B là kết quả TRỰC TIẾP của A.
Hướng: A → B

Ví dụ: "Chiến thắng ĐBP 1954" RESULTED_IN "Ký Hiệp định Genève"
""",
        relation_b_name="CONSEQUENCE_OF",
        relation_b_summary="Đoạn A mô tả hậu quả của sự kiện trong đoạn B",
        relation_b_description="""Mối quan hệ hậu quả: A là hậu quả của B.
Hướng: A ← B

Ví dụ: "Hàng triệu người thiệt mạng" CONSEQUENCE_OF "Kháng chiến 21 năm"
""",
        should_merge=False,
        reasoning="Ngược hướng! RESULTED_IN: A→B (A gây ra B), CONSEQUENCE_OF: A←B (A là hậu quả của B). "
        "Mặc dù cùng family causal nhưng hướng khác nhau hoàn toàn.",
        is_boundary_case=True,
    ),
    RelationSample(
        id=16,
        category=MergeCategory.DIRECTIONAL_INVERSES,
        relation_a_name="LED_BY",
        relation_a_summary="Sự kiện A được LÃNH ĐẠO bởi nhân vật B",
        relation_a_description="""Mối quan hệ lãnh đạo: B là người lãnh đạo sự kiện A.
Hướng: A ← B (B lãnh đạo A)

Ví dụ: "Khởi nghĩa Lam Sơn" LED_BY "Lê Lợi"
""",
        relation_b_name="PARTICIPATED_IN",
        relation_b_summary="Nhân vật A THAM GIA vào sự kiện B",
        relation_b_description="""Mối quan hệ tham gia: A tham gia vào sự kiện B.
Hướng: A → B (A tham gia B)

Ví dụ: "Nguyễn Trãi" PARTICIPATED_IN "Kháng chiến chống Minh"
""",
        should_merge=False,
        reasoning="Khác hướng VÀ khác vai trò! LED_BY: B→A (B lãnh đạo A), "
        "PARTICIPATED_IN: A→B (A tham gia B). Lãnh đạo ≠ tham gia.",
        is_boundary_case=True,
    ),
    RelationSample(
        id=17,
        category=MergeCategory.DIRECTIONAL_INVERSES,
        relation_a_name="ELABORATES",
        relation_a_summary="Đoạn A mở rộng/giải thích chi tiết đoạn B",
        relation_a_description="""Quan hệ mở rộng: A cung cấp thông tin chi tiết cho B.
A chi tiết hơn B.
Hướng: A → B (A giải thích B)

Ví dụ: "Chi tiết chiến thuật..." ELABORATES "Tổng quan chiến lược"
""",
        relation_b_name="SUMMARIZES",
        relation_b_summary="Đoạn A tóm tắt nội dung đoạn B",
        relation_b_description="""Quan hệ tóm tắt: A rút gọn nội dung B.
A ngắn gọn hơn B.
Hướng: A → B (A tóm tắt B)

Ví dụ: "Kết luận ngắn" SUMMARIZES "Phân tích chi tiết"
""",
        should_merge=False,
        reasoning="Ngược logic! ELABORATES: A chi tiết hơn B, SUMMARIZES: A ngắn hơn B. "
        "Mặc dù cùng hướng A→B nhưng semantic ngược nhau hoàn toàn.",
        is_boundary_case=False,
    ),
    RelationSample(
        id=18,
        category=MergeCategory.DIRECTIONAL_INVERSES,
        relation_a_name="HAS_SECTION",
        relation_a_summary="Cấu trúc cha chứa cấu trúc con",
        relation_a_description="""Quan hệ phân cấp: Node cấu trúc cha chứa node cấu trúc con.
Hướng: Parent → Child

Ví dụ: "Chương 1" HAS_SECTION "Mục 1.1"
""",
        relation_b_name="BELONGS_TO_SECTION",
        relation_b_summary="Cấu trúc con thuộc về cấu trúc cha",
        relation_b_description="""Quan hệ phân cấp ngược: Node con thuộc về node cha.
Hướng: Child → Parent

Ví dụ: "Mục 1.1" BELONGS_TO_SECTION "Chương 1"
""",
        should_merge=False,
        reasoning="Inverse relation! HAS_SECTION: Parent→Child, BELONGS_TO_SECTION: Child→Parent. "
        "Merge sẽ phá vỡ cấu trúc phân cấp.",
        is_boundary_case=False,
    ),
    RelationSample(
        id=19,
        category=MergeCategory.DIRECTIONAL_INVERSES,
        relation_a_name="INTRODUCES",
        relation_a_summary="Đoạn A giới thiệu chủ đề được phát triển trong B",
        relation_a_description="""Quan hệ giới thiệu: A mở đầu, giới thiệu chủ đề cho B.
A thường ngắn gọn, B phát triển chi tiết.

Ví dụ: "Sau đây là các nguyên nhân..." INTRODUCES "Phân tích chi tiết"
""",
        relation_b_name="CONCLUDES",
        relation_b_summary="Đoạn A kết luận/đúc kết nội dung từ B",
        relation_b_description="""Quan hệ kết luận: A đúc kết, rút ra bài học từ B.
A thường ở cuối.

Ví dụ: "Từ những phân tích trên..." CONCLUDES "Toàn bộ phân tích"
""",
        should_merge=False,
        reasoning="Ngược vị trí discourse! INTRODUCES: A ở đầu, CONCLUDES: A ở cuối. "
        "Semantic hoàn toàn khác nhau.",
        is_boundary_case=False,
    ),
    RelationSample(
        id=20,
        category=MergeCategory.DIRECTIONAL_INVERSES,
        relation_a_name="PREREQUISITE_OF",
        relation_a_summary="Cần hiểu đoạn A trước khi hiểu đoạn B",
        relation_a_description="""Quan hệ tiên quyết: Phải hiểu A trước để hiểu B.
Hướng: A → B (A cần trước B)

Ví dụ: "Khái niệm phong kiến" PREREQUISITE_OF "Phân tích xã hội Đại Việt"
""",
        relation_b_name="BUILDS_UPON",
        relation_b_summary="Đoạn A mở rộng/phát triển kiến thức từ B",
        relation_b_description="""Quan hệ phát triển: A xây dựng dựa trên kiến thức B.
Hướng: A → B (A phát triển từ B)

Ví dụ: "Phân tích chuyên sâu" BUILDS_UPON "Giới thiệu cơ bản"
""",
        should_merge=False,
        reasoning="Khác mức độ! PREREQUISITE_OF: A bắt buộc trước B, "
        "BUILDS_UPON: A không bắt buộc, chỉ mở rộng từ B. Semantic khác.",
        is_boundary_case=True,
    ),
    RelationSample(
        id=21,
        category=MergeCategory.DIRECTIONAL_INVERSES,
        relation_a_name="SUPPORTS",
        relation_a_summary="Đoạn A cung cấp bằng chứng ủng hộ B",
        relation_a_description="""Quan hệ ủng hộ: A cung cấp evidence cho B.
A làm tăng độ tin cậy của B.

Ví dụ: "Số liệu thống kê" SUPPORTS "Luận điểm chính"
""",
        relation_b_name="CONTRADICTS",
        relation_b_summary="Đoạn A mâu thuẫn/phản bác trực tiếp B",
        relation_b_description="""Quan hệ mâu thuẫn: A và B không thể cùng đúng.
Symmetric.

Ví dụ: "Nguyên nhân là ngoại xâm" CONTRADICTS "Nguyên nhân là nội bộ"
""",
        should_merge=False,
        reasoning="Ngược hoàn toàn! SUPPORTS: A ủng hộ B, CONTRADICTS: A phản bác B. "
        "Merge sẽ phá vỡ logic luận điểm.",
        is_boundary_case=False,
    ),
    RelationSample(
        id=22,
        category=MergeCategory.DIRECTIONAL_INVERSES,
        relation_a_name="NEXT_SIBLING",
        relation_a_summary="Phần tử A đứng ngay trước B trong thứ tự đọc",
        relation_a_description="""Quan hệ thứ tự đọc: A đứng ngay trước B.
Hướng: A → B (đọc A rồi đọc B)

Ví dụ: "Mục 1.1" NEXT_SIBLING "Mục 1.2"
""",
        relation_b_name="PREVIOUS_SIBLING",
        relation_b_summary="Phần tử A đứng ngay sau B trong thứ tự đọc",
        relation_b_description="""Quan hệ thứ tự đọc ngược: A đứng ngay sau B.
Hướng: A → B (B đứng trước A)

Ví dụ: "Mục 1.2" PREVIOUS_SIBLING "Mục 1.1"
""",
        should_merge=False,
        reasoning="Inverse relation! NEXT_SIBLING: A→B, PREVIOUS_SIBLING: B→A về thứ tự. "
        "Merge sẽ phá vỡ navigation structure.",
        is_boundary_case=False,
    ),
]


# =============================================================================
# CATEGORY 3: GRANULARITY VARIANTS (boundary cases) - 12 samples
# Cùng concept nhưng mức độ cụ thể khác nhau - CẦN XÉT KỸ
# =============================================================================

GRANULARITY_VARIANT_SAMPLES = [
    RelationSample(
        id=23,
        category=MergeCategory.GRANULARITY_VARIANTS,
        relation_a_name="LED_TO",
        relation_a_summary="Sự kiện A dẫn đến sự kiện B (trực tiếp hoặc gián tiếp)",
        relation_a_description="""Quan hệ nhân quả tổng quát: A là nguyên nhân của B.
Có thể qua nhiều bước trung gian.

Ví dụ: "Khai thác thuộc địa" LED_TO "Phong trào công nhân"
""",
        relation_b_name="DIRECTLY_CAUSED",
        relation_b_summary="Sự kiện A trực tiếp gây ra sự kiện B không qua trung gian",
        relation_b_description="""Quan hệ nhân quả TRỰC TIẾP: B xảy ra ngay sau A.
Không có bước trung gian.

Ví dụ: "Đánh chiếm Sài Gòn 30/4" DIRECTLY_CAUSED "Chế độ Sài Gòn sụp đổ"
""",
        should_merge=True,
        reasoning="DIRECTLY_CAUSED là subset của LED_TO. Trong knowledge graph, "
        "có thể coi là cùng relation với attribute khác về directness.",
        is_boundary_case=True,
    ),
    RelationSample(
        id=24,
        category=MergeCategory.GRANULARITY_VARIANTS,
        relation_a_name="LED_TO",
        relation_a_summary="Sự kiện A dẫn đến sự kiện B",
        relation_a_description="""Quan hệ nhân quả: A là nguyên nhân của B.

Ví dụ: "Xâm lược" LED_TO "Kháng chiến"
""",
        relation_b_name="INDIRECTLY_LED_TO",
        relation_b_summary="Sự kiện A dẫn đến sự kiện B thông qua chuỗi sự kiện trung gian",
        relation_b_description="""Quan hệ nhân quả GIÁN TIẾP: A gây ra B qua nhiều bước.
Có chuỗi sự kiện trung gian quan trọng.

Ví dụ: "CTTG I" INDIRECTLY_LED_TO "Cách mạng tháng Tám" (qua nhiều giai đoạn)
""",
        should_merge=False,
        reasoning="INDIRECTLY_LED_TO nhấn mạnh tính gián tiếp quan trọng cho phân tích lịch sử. "
        "Sự phân biệt này có ý nghĩa học thuật, không nên merge.",
        is_boundary_case=True,
    ),
    RelationSample(
        id=25,
        category=MergeCategory.GRANULARITY_VARIANTS,
        relation_a_name="CONTRIBUTED_TO",
        relation_a_summary="Yếu tố A đóng góp vào kết quả B",
        relation_a_description="""Quan hệ đóng góp: A góp phần vào B.

Ví dụ: "Tinh thần yêu nước" CONTRIBUTED_TO "Chiến thắng"
""",
        relation_b_name="MAJOR_FACTOR_IN",
        relation_b_summary="Yếu tố A là nguyên nhân CHÍNH của kết quả B",
        relation_b_description="""Quan hệ yếu tố chính: A là nguyên nhân quan trọng nhất.
Phân biệt với các yếu tố phụ.

Ví dụ: "Chiến lược đúng đắn" MAJOR_FACTOR_IN "Chiến thắng Điện Biên Phủ"
""",
        should_merge=False,
        reasoning="MAJOR_FACTOR_IN chỉ rõ mức độ quan trọng, khác với CONTRIBUTED_TO tổng quát. "
        "Trong phân tích lịch sử, sự phân biệt này quan trọng.",
        is_boundary_case=True,
    ),
    RelationSample(
        id=26,
        category=MergeCategory.GRANULARITY_VARIANTS,
        relation_a_name="OPPOSED_BY",
        relation_a_summary="Chính sách A bị phản đối bởi lực lượng B",
        relation_a_description="""Quan hệ đối lập: B chống lại A.

Ví dụ: "Chính sách thuộc địa" OPPOSED_BY "Phong trào yêu nước"
""",
        relation_b_name="STRONGLY_OPPOSED_BY",
        relation_b_summary="Chính sách A bị phản đối MẠNH MẼ bởi lực lượng B",
        relation_b_description="""Quan hệ đối lập mạnh: B kịch liệt chống lại A.
Mức độ phản đối cao.

Ví dụ: "Luật Gia Long" STRONGLY_OPPOSED_BY "Phong trào nông dân"
""",
        should_merge=True,
        reasoning="STRONGLY_OPPOSED_BY chỉ thêm mức độ, nhưng semantic giống OPPOSED_BY. "
        "Có thể merge và dùng attribute cho intensity.",
        is_boundary_case=True,
    ),
    RelationSample(
        id=27,
        category=MergeCategory.GRANULARITY_VARIANTS,
        relation_a_name="DURING",
        relation_a_summary="Sự kiện A xảy ra TRONG thời kỳ B",
        relation_a_description="""Quan hệ thời gian bao hàm: A nằm trong khoảng thời gian B.

Ví dụ: "Trận Bạch Đằng 938" DURING "Bắc thuộc lần ba"
""",
        relation_b_name="THROUGHOUT",
        relation_b_summary="Sự kiện A diễn ra SUỐT thời kỳ B",
        relation_b_description="""Quan hệ thời gian kéo dài: A diễn ra trong toàn bộ B.

Ví dụ: "Đấu tranh gìn giữ văn hóa" THROUGHOUT "Ngàn năm Bắc thuộc"
""",
        should_merge=True,
        reasoning="THROUGHOUT là trường hợp đặc biệt của DURING (toàn bộ thay vì một phần). "
        "Có thể merge, dùng attribute cho duration.",
        is_boundary_case=True,
    ),
    RelationSample(
        id=28,
        category=MergeCategory.GRANULARITY_VARIANTS,
        relation_a_name="SIMILAR_TO",
        relation_a_summary="Hai đoạn có nội dung tương đồng về ngữ nghĩa",
        relation_a_description="""Quan hệ tương đồng: A và B có nội dung giống nhau.
Dựa trên embedding similarity.

Ví dụ: "Nguyên nhân thất bại nhà Hồ" SIMILAR_TO "Lý do sụp đổ triều Hồ"
""",
        relation_b_name="DUPLICATE_OF",
        relation_b_summary="Hai đoạn là bản sao trùng lặp hoàn toàn",
        relation_b_description="""Quan hệ trùng lặp: Nội dung giống hệt hoặc gần hệt.

Ví dụ: "Văn bản nguồn 1" DUPLICATE_OF "Văn bản giống hệt nguồn 2"
""",
        should_merge=False,
        reasoning="DUPLICATE_OF yêu cầu giống hệt, SIMILAR_TO chỉ cần tương đồng. "
        "Sự khác biệt quan trọng cho data quality.",
        is_boundary_case=False,
    ),
    RelationSample(
        id=29,
        category=MergeCategory.GRANULARITY_VARIANTS,
        relation_a_name="PARTICIPAGED_IN",
        relation_a_summary="Nhân vật A tham gia vào sự kiện B",
        relation_a_description="""Quan hệ tham gia: A có mặt trong sự kiện B.

Ví dụ: "Nhân dân miền Nam" PARTICIPATED_IN "Kháng chiến chống Mỹ"
""",
        relation_b_name="ACTIVELY_PARTICIPATED_IN",
        relation_b_summary="Nhân vật A tích cực tham gia vào sự kiện B",
        relation_b_description="""Quan hệ tham gia tích cực: A có vai trò chủ động trong B.

Ví dụ: "Thanh niên" ACTIVELY_PARTICIPATED_IN "Phong trào Đông Du"
""",
        should_merge=True,
        reasoning="ACTIVELY_PARTICIPATED_IN thêm mức độ cho PARTICIPATED_IN. "
        "Có thể merge, dùng attribute cho activity level.",
        is_boundary_case=True,
    ),
    RelationSample(
        id=30,
        category=MergeCategory.GRANULARITY_VARIANTS,
        relation_a_name="CONTEMPORARY_WITH",
        relation_a_summary="Hai sự kiện xảy ra cùng thời điểm",
        relation_a_description="""Quan hệ đồng thời: A và B xảy ra cùng khoảng thời gian.

Ví dụ: "Phong trào Bắc Kỳ" CONTEMPORARY_WITH "Phong trào Nam Kỳ"
""",
        relation_b_name="EXACTLY_CONTEMPORARY_WITH",
        relation_b_summary="Hai sự kiện xảy ra CHÍNH XÁC cùng thời điểm",
        relation_b_description="""Quan hệ đồng thời chính xác: A và B xảy ra đúng cùng lúc.

Ví dụ: "Đánh Bắc" EXACTLY_CONTEMPORARY_WITH "Kháng chiến ở Nam"
""",
        should_merge=True,
        reasoning="EXACTLY_CONTEMPORARY_WITH là trường hợp chính xác của CONTEMPORARY_WITH. "
        "Có thể merge, dùng attribute cho precision.",
        is_boundary_case=False,
    ),
    RelationSample(
        id=31,
        category=MergeCategory.GRANULARITY_VARIANTS,
        relation_a_name="REFERS_TO",
        relation_a_summary="Đoạn A tham chiếu rõ ràng đến đoạn B",
        relation_a_description="""Quan hệ tham chiếu: A đề cập đến B.

Ví dụ: "Như đã nói ở phần trước..." REFERS_TO "Phần giới thiệu"
""",
        relation_b_name="EXPLICITLY_CITES",
        relation_b_summary="Đoạn A trích dẫn rõ ràng nội dung từ đoạn B",
        relation_b_description="""Quan hệ trích dẫn: A trích nguyên văn từ B.

Ví dụ: "Theo sử sách ghi: '...'" EXPLICITLY_CITES "Văn bản gốc"
""",
        should_merge=False,
        reasoning="EXPLICITLY_CITES yêu cầu trích nguyên văn, REFERS_TO chỉ cần đề cập. "
        "Sự khác biệt quan trọng cho học thuật.",
        is_boundary_case=True,
    ),
    RelationSample(
        id=32,
        category=MergeCategory.GRANULARITY_VARIANTS,
        relation_a_name="IS_A",
        relation_a_summary="Sự kiện A là MỘT LOẠI của khái niệm B",
        relation_a_description="""Quan hệ phân loại: A thuộc loại B.

Ví dụ: "KN Hai Bà Trưng" IS_A "Cuộc khởi nghĩa chống Bắc thuộc"
""",
        relation_b_name="IS_SUBTYPE_OF",
        relation_b_summary="Khái niệm A là phân loại con của khái niệm B",
        relation_b_description="""Quan hệ phân loại con: A là loại cụ thể hơn của B.
Dùng cho taxonomy chi tiết.

Ví dụ: "Khởi nghĩa nông dân" IS_SUBTYPE_OF "Phong trào đấu tranh"
""",
        should_merge=True,
        reasoning="IS_SUBTYPE_OF và IS_A về bản chất giống nhau trong taxonomy. Có thể merge.",
        is_boundary_case=False,
    ),
    RelationSample(
        id=33,
        category=MergeCategory.GRANULARITY_VARIANTS,
        relation_a_name="CONTEXTUALIZES",
        relation_a_summary="Đoạn A cung cấp bối cảnh cho đoạn B",
        relation_a_description="""Quan hệ bối cảnh: A giúp hiểu B trong context.

Ví dụ: "Trong bối cảnh thực dân..." CONTEXTUALIZES "Phong trào yêu nước"
""",
        relation_b_name="PROVIDES_HISTORICAL_CONTEXT",
        relation_b_summary="Đoạn A cung cấp bối cảnh lịch sử cho đoạn B",
        relation_b_description="""Quan hệ bối cảnh lịch sử: A là background sử học cho B.

Ví dụ: "Tình hình thế giới 1914" PROVIDES_HISTORICAL_CONTEXT "VN đầu TK XX"
""",
        should_merge=True,
        reasoning="PROVIDES_HISTORICAL_CONTEXT là trường hợp cụ thể của CONTEXTUALIZES. "
        "Trong domain lịch sử, có thể merge.",
        is_boundary_case=True,
    ),
    RelationSample(
        id=34,
        category=MergeCategory.GRANULARITY_VARIANTS,
        relation_a_name="SUPERSEDES",
        relation_a_summary="Đoạn A thay thế/cập nhật đoạn B cũ hơn",
        relation_a_description="""Quan hệ thay thế: A là phiên bản mới của B.

Ví dụ: "SGK 2023" SUPERSEDES "SGK 2020"
""",
        relation_b_name="COMPLETELY_REPLACES",
        relation_b_summary="Đoạn A thay thế HOÀN TOÀN đoạn B",
        relation_b_description="""Quan hệ thay thế hoàn toàn: B không còn valid sau A.

Ví dụ: "Hiến pháp 2013" COMPLETELY_REPLACES "Hiến pháp 1992"
""",
        should_merge=True,
        reasoning="COMPLETELY_REPLACES nhấn mạnh mức độ nhưng semantic giống SUPERSEDES. "
        "Có thể merge.",
        is_boundary_case=False,
    ),
]


# =============================================================================
# CATEGORY 4: RELATED BUT DISTINCT (should_merge=False) - 14 samples
# Cùng family nhưng KHÁC CONCEPT - KHÔNG MERGE
# =============================================================================

RELATED_DISTINCT_SAMPLES = [
    RelationSample(
        id=35,
        category=MergeCategory.RELATED_DISTINCT,
        relation_a_name="DURING",
        relation_a_summary="Sự kiện A xảy ra TRONG thời kỳ B",
        relation_a_description="""Quan hệ thời gian bao hàm: A nằm trong B.

Ví dụ: "Trận Bạch Đằng 938" DURING "Bắc thuộc lần ba"
""",
        relation_b_name="CONTEMPORARY_WITH",
        relation_b_summary="Sự kiện A và B xảy ra CÙNG thời điểm",
        relation_b_description="""Quan hệ đồng thời: A và B ngang hàng về thời gian.
Symmetric.

Ví dụ: "Phong trào Bắc" CONTEMPORARY_WITH "Phong trào Nam"
""",
        should_merge=False,
        reasoning="DURING: A nằm trong B (bất đối xứng), CONTEMPORARY_WITH: A và B ngang hàng (đối xứng). "
        "Khác cấu trúc quan hệ.",
        is_boundary_case=True,
    ),
    RelationSample(
        id=36,
        category=MergeCategory.RELATED_DISTINCT,
        relation_a_name="IS_A",
        relation_a_summary="A là một loại của B (phân loại)",
        relation_a_description="""Quan hệ phân loại taxonomy.

Ví dụ: "KN Hai Bà Trưng" IS_A "Cuộc khởi nghĩa"
""",
        relation_b_name="PART_OF",
        relation_b_summary="A là một phần của B (bộ phận-tổng thể)",
        relation_b_description="""Quan hệ mereology.

Ví dụ: "Trận Bạch Đằng" PART_OF "Kháng chiến chống Nguyên"
""",
        should_merge=False,
        reasoning="IS_A: taxonomy (A thuộc loại B), PART_OF: mereology (A là bộ phận của B). "
        "Hai loại quan hệ khác nhau về bản chất.",
        is_boundary_case=True,
    ),
    RelationSample(
        id=37,
        category=MergeCategory.RELATED_DISTINCT,
        relation_a_name="CONTRASTS",
        relation_a_summary="Đoạn A so sánh, làm nổi bật sự khác biệt với B",
        relation_a_description="""Quan hệ đối lập: A và B có điểm khác biệt.
Không nhất thiết mâu thuẫn.

Ví dụ: "Miền Bắc phong trào mạnh" CONTRASTS "Miền Nam còn hạn chế"
""",
        relation_b_name="CONTRADICTS",
        relation_b_summary="Đoạn A mâu thuẫn/phản bác trực tiếp B",
        relation_b_description="""Quan hệ mâu thuẫn: A và B không thể cùng đúng.

Ví dụ: "Nguyên nhân là ngoại xâm" CONTRADICTS "Nguyên nhân là nội bộ"
""",
        should_merge=False,
        reasoning="CONTRASTS: khác biệt nhưng có thể cùng đúng, CONTRADICTS: không thể cùng đúng. "
        "Khác mức độ logic quan trọng.",
        is_boundary_case=True,
    ),
    RelationSample(
        id=38,
        category=MergeCategory.RELATED_DISTINCT,
        relation_a_name="INTRODUCES",
        relation_a_summary="Đoạn A giới thiệu chủ đề trong B",
        relation_a_description="""Quan hệ giới thiệu: A mở đầu cho B.

Ví dụ: "Sau đây là nguyên nhân..." INTRODUCES "Phân tích nguyên nhân"
""",
        relation_b_name="CONTEXTUALIZES",
        relation_b_summary="Đoạn A cung cấp bối cảnh cho B",
        relation_b_description="""Quan hệ bối cảnh: A là background cho B.

Ví dụ: "Trong bối cảnh thế giới..." CONTEXTUALIZES "Tình hình VN"
""",
        should_merge=False,
        reasoning="INTRODUCES: giới thiệu chủ đề sẽ nói, CONTEXTUALIZES: cung cấp background. "
        "Chức năng discourse khác nhau.",
        is_boundary_case=True,
    ),
    RelationSample(
        id=39,
        category=MergeCategory.RELATED_DISTINCT,
        relation_a_name="DEFINES",
        relation_a_summary="Đoạn A định nghĩa khái niệm trong B",
        relation_a_description="""Quan hệ định nghĩa: A cung cấp definition cho B.

Ví dụ: "Phong kiến là..." DEFINES "Phân tích xã hội phong kiến"
""",
        relation_b_name="ELABORATES",
        relation_b_summary="Đoạn A mở rộng/giải thích chi tiết B",
        relation_b_description="""Quan hệ mở rộng: A thêm chi tiết cho B.

Ví dụ: "Cụ thể, chiến thuật gồm..." ELABORATES "Tổng quan chiến lược"
""",
        should_merge=False,
        reasoning="DEFINES: cung cấp định nghĩa chính thức, ELABORATES: mở rộng chi tiết. "
        "Chức năng khác nhau.",
        is_boundary_case=False,
    ),
    RelationSample(
        id=40,
        category=MergeCategory.RELATED_DISTINCT,
        relation_a_name="EXEMPLIFIES",
        relation_a_summary="Đoạn A là ví dụ minh họa cho B",
        relation_a_description="""Quan hệ ví dụ: A minh họa cho quan điểm B.

Ví dụ: "Trận Bạch Đằng 938" EXEMPLIFIES "Chiến thuật thủy quân"
""",
        relation_b_name="ELABORATES",
        relation_b_summary="Đoạn A mở rộng/giải thích chi tiết B",
        relation_b_description="""Quan hệ mở rộng: A thêm chi tiết cho B.

Ví dụ: "Chi tiết về chiến thuật..." ELABORATES "Tổng quan"
""",
        should_merge=False,
        reasoning="EXEMPLIFIES: ví dụ cụ thể minh họa, ELABORATES: giải thích thêm. "
        "Khác chức năng (example vs expansion).",
        is_boundary_case=True,
    ),
    RelationSample(
        id=41,
        category=MergeCategory.RELATED_DISTINCT,
        relation_a_name="VISUALIZES",
        relation_a_summary="Hình ảnh/bảng A minh họa cho đoạn văn B",
        relation_a_description="""Quan hệ trực quan hóa: Image/Table minh họa Paragraph.

Ví dụ: "Bản đồ trận Bạch Đằng" VISUALIZES "Mô tả diễn biến"
""",
        relation_b_name="EXEMPLIFIES",
        relation_b_summary="Đoạn A là ví dụ minh họa cho B",
        relation_b_description="""Quan hệ ví dụ: A minh họa bằng văn bản.

Ví dụ: "Trường hợp KN Hai Bà Trưng" EXEMPLIFIES "Tinh thần yêu nước"
""",
        should_merge=False,
        reasoning="VISUALIZES: minh họa bằng hình ảnh/bảng, EXEMPLIFIES: minh họa bằng văn bản. "
        "Khác modality (visual vs textual).",
        is_boundary_case=False,
    ),
    RelationSample(
        id=42,
        category=MergeCategory.RELATED_DISTINCT,
        relation_a_name="CONTINUES",
        relation_a_summary="Đoạn A tiếp tục nội dung bị cắt từ B",
        relation_a_description="""Quan hệ tiếp nối: A và B vốn là một đoạn bị chia.

Ví dụ: "...quân ta anh dũng chiến đấu" CONTINUES "Trong trận này..."
""",
        relation_b_name="ELABORATES",
        relation_b_summary="Đoạn A mở rộng nội dung B",
        relation_b_description="""Quan hệ mở rộng: A thêm thông tin cho B.

Ví dụ: "Chi tiết thêm..." ELABORATES "Ý chính"
""",
        should_merge=False,
        reasoning="CONTINUES: nội dung bị chia cắt, ELABORATES: thêm thông tin mới. "
        "Khác bản chất (split vs expansion).",
        is_boundary_case=False,
    ),
    RelationSample(
        id=43,
        category=MergeCategory.RELATED_DISTINCT,
        relation_a_name="BUILDS_UPON",
        relation_a_summary="Đoạn A phát triển kiến thức từ B",
        relation_a_description="""Quan hệ phát triển: A mở rộng từ nền tảng B.

Ví dụ: "Phân tích chuyên sâu" BUILDS_UPON "Giới thiệu cơ bản"
""",
        relation_b_name="SUPERSEDES",
        relation_b_summary="Đoạn A thay thế/cập nhật B cũ hơn",
        relation_b_description="""Quan hệ thay thế: A thay thế B, B không còn valid.

Ví dụ: "SGK mới" SUPERSEDES "SGK cũ"
""",
        should_merge=False,
        reasoning="BUILDS_UPON: B vẫn valid, A mở rộng từ B. SUPERSEDES: B không còn valid. "
        "Khác về validity của B.",
        is_boundary_case=True,
    ),
    RelationSample(
        id=44,
        category=MergeCategory.RELATED_DISTINCT,
        relation_a_name="LED_BY",
        relation_a_summary="Sự kiện A được lãnh đạo bởi B",
        relation_a_description="""Quan hệ lãnh đạo: B là người đứng đầu A.

Ví dụ: "KN Lam Sơn" LED_BY "Lê Lợi"
""",
        relation_b_name="CONTRIBUTED_TO",
        relation_b_summary="Yếu tố A đóng góp vào B",
        relation_b_description="""Quan hệ đóng góp: A góp phần vào B.

Ví dụ: "Tinh thần yêu nước" CONTRIBUTED_TO "Chiến thắng"
""",
        should_merge=False,
        reasoning="LED_BY: vai trò lãnh đạo cụ thể, CONTRIBUTED_TO: đóng góp chung. "
        "Khác về vai trò và mức độ.",
        is_boundary_case=False,
    ),
    RelationSample(
        id=45,
        category=MergeCategory.RELATED_DISTINCT,
        relation_a_name="OPPOSED_BY",
        relation_a_summary="Chính sách A bị phản đối bởi B",
        relation_a_description="""Quan hệ đối lập: B chống lại A về hành động.

Ví dụ: "Chính sách thuộc địa" OPPOSED_BY "Phong trào yêu nước"
""",
        relation_b_name="CONTRADICTS",
        relation_b_summary="Đoạn A mâu thuẫn với đoạn B về quan điểm",
        relation_b_description="""Quan hệ mâu thuẫn: A và B không thể cùng đúng.

Ví dụ: "Luận điểm A" CONTRADICTS "Luận điểm B"
""",
        should_merge=False,
        reasoning="OPPOSED_BY: phản đối hành động/chính sách, CONTRADICTS: mâu thuẫn quan điểm. "
        "Khác đối tượng (action vs viewpoint).",
        is_boundary_case=True,
    ),
    RelationSample(
        id=46,
        category=MergeCategory.RELATED_DISTINCT,
        relation_a_name="REFERS_TO",
        relation_a_summary="Đoạn A tham chiếu đến đoạn B",
        relation_a_description="""Quan hệ tham chiếu: A đề cập đến B.

Ví dụ: "Như đã nói ở phần trước..." REFERS_TO "Phần giới thiệu"
""",
        relation_b_name="SIMILAR_TO",
        relation_b_summary="Hai đoạn có nội dung tương đồng",
        relation_b_description="""Quan hệ tương đồng: A và B giống nhau về nội dung.

Ví dụ: "Đoạn từ nguồn A" SIMILAR_TO "Đoạn từ nguồn B"
""",
        should_merge=False,
        reasoning="REFERS_TO: tham chiếu explicit, SIMILAR_TO: tương đồng nội dung. "
        "Khác cơ chế (explicit mention vs content similarity).",
        is_boundary_case=False,
    ),
    RelationSample(
        id=47,
        category=MergeCategory.RELATED_DISTINCT,
        relation_a_name="SUMMARIZES",
        relation_a_summary="Đoạn A tóm tắt nội dung B",
        relation_a_description="""Quan hệ tóm tắt: A rút gọn B.

Ví dụ: "Tóm lại..." SUMMARIZES "Phân tích chi tiết"
""",
        relation_b_name="CONCLUDES",
        relation_b_summary="Đoạn A kết luận từ B",
        relation_b_description="""Quan hệ kết luận: A rút ra insight từ B.

Ví dụ: "Bài học lịch sử cho thấy..." CONCLUDES "Phân tích sự kiện"
""",
        should_merge=False,
        reasoning="SUMMARIZES: rút gọn nội dung, CONCLUDES: rút ra kết luận/bài học. "
        "Khác chức năng (compression vs inference).",
        is_boundary_case=True,
    ),
    RelationSample(
        id=48,
        category=MergeCategory.RELATED_DISTINCT,
        relation_a_name="PRECEDED_BY",
        relation_a_summary="A xảy ra sau B (thời gian)",
        relation_a_description="""Quan hệ thời gian: B xảy ra trước A.

Ví dụ: "Triều Trần" PRECEDED_BY "Triều Lý"
""",
        relation_b_name="CAUSED_BY",
        relation_b_summary="A bị gây ra bởi B (nhân quả)",
        relation_b_description="""Quan hệ nhân quả: B là nguyên nhân của A.

Ví dụ: "Thất bại" CAUSED_BY "Sai lầm chiến lược"
""",
        should_merge=False,
        reasoning="PRECEDED_BY: chỉ thời gian, không có nhân quả. CAUSED_BY: có nhân quả. "
        "Khác về causal implication.",
        is_boundary_case=True,
    ),
]


# =============================================================================
# CATEGORY 5: DOMAIN OVERLAPS (boundary cases) - 12 samples
# Các relation đặc thù lịch sử có overlap - CẦN XÉT KỸ
# =============================================================================

DOMAIN_OVERLAP_SAMPLES = [
    RelationSample(
        id=49,
        category=MergeCategory.DOMAIN_OVERLAPS,
        relation_a_name="MILITARY_VICTORY_OVER",
        relation_a_summary="Quân đội A giành chiến thắng quân sự trước B",
        relation_a_description="""Quan hệ chiến thắng quân sự: A đánh bại B trong chiến tranh.

Ví dụ: "Quân Trần" MILITARY_VICTORY_OVER "Quân Nguyên-Mông"
""",
        relation_b_name="DEFEATED",
        relation_b_summary="Lực lượng A đánh bại lực lượng B",
        relation_b_description="""Quan hệ đánh bại: A thắng B.

Ví dụ: "Nghĩa quân Lam Sơn" DEFEATED "Quân Minh"
""",
        should_merge=True,
        reasoning="MILITARY_VICTORY_OVER và DEFEATED đều mô tả A thắng B trong chiến tranh. "
        "MILITARY_VICTORY_OVER cụ thể hơn nhưng semantic tương đương.",
        is_boundary_case=True,
    ),
    RelationSample(
        id=50,
        category=MergeCategory.DOMAIN_OVERLAPS,
        relation_a_name="SIGNED_TREATY_WITH",
        relation_a_summary="Bên A ký hiệp ước với bên B",
        relation_a_description="""Quan hệ ký kết: A và B ký văn bản chính thức.

Ví dụ: "Nhà Nguyễn" SIGNED_TREATY_WITH "Pháp" (Hiệp ước Nhâm Tuất)
""",
        relation_b_name="NEGOTIATED_WITH",
        relation_b_summary="Bên A đàm phán với bên B",
        relation_b_description="""Quan hệ đàm phán: A và B thương lượng.

Ví dụ: "Việt Nam" NEGOTIATED_WITH "Pháp" (Hội nghị Fontainebleau)
""",
        should_merge=False,
        reasoning="SIGNED_TREATY_WITH: kết quả ký kết, NEGOTIATED_WITH: quá trình đàm phán. "
        "Đàm phán không nhất thiết dẫn đến ký kết.",
        is_boundary_case=True,
    ),
    RelationSample(
        id=51,
        category=MergeCategory.DOMAIN_OVERLAPS,
        relation_a_name="COLONIZED_BY",
        relation_a_summary="Vùng A bị thực dân B đô hộ",
        relation_a_description="""Quan hệ thuộc địa: A là thuộc địa của B.

Ví dụ: "Việt Nam" COLONIZED_BY "Pháp" (1858-1945)
""",
        relation_b_name="RULED_BY",
        relation_b_summary="Vùng A bị cai trị bởi B",
        relation_b_description="""Quan hệ cai trị: B cai trị A.

Ví dụ: "Giao Chỉ" RULED_BY "Nhà Hán"
""",
        should_merge=False,
        reasoning="COLONIZED_BY: đô hộ thực dân (khai thác, bóc lột), RULED_BY: cai trị tổng quát. "
        "COLONIZED_BY có connotation tiêu cực đặc thù.",
        is_boundary_case=True,
    ),
    RelationSample(
        id=52,
        category=MergeCategory.DOMAIN_OVERLAPS,
        relation_a_name="UPRISING_AGAINST",
        relation_a_summary="Lực lượng A khởi nghĩa chống lại B",
        relation_a_description="""Quan hệ khởi nghĩa: A nổi dậy chống B.

Ví dụ: "Hai Bà Trưng" UPRISING_AGAINST "Nhà Hán"
""",
        relation_b_name="OPPOSED_BY",
        relation_b_summary="Chính sách A bị phản đối bởi B",
        relation_b_description="""Quan hệ phản đối: B chống lại A.

Ví dụ: "Chính sách thuộc địa" OPPOSED_BY "Phong trào yêu nước"
""",
        should_merge=True,
        reasoning="UPRISING_AGAINST là trường hợp đặc biệt (vũ trang) của OPPOSED_BY. "
        "Trong knowledge graph lịch sử, có thể merge.",
        is_boundary_case=True,
    ),
    RelationSample(
        id=53,
        category=MergeCategory.DOMAIN_OVERLAPS,
        relation_a_name="SUCCEEDED_DYNASTY",
        relation_a_summary="Triều đại A kế thừa triều đại B",
        relation_a_description="""Quan hệ kế thừa triều đại: A thay thế B làm vua.

Ví dụ: "Nhà Trần" SUCCEEDED_DYNASTY "Nhà Lý"
""",
        relation_b_name="FOLLOWED_BY",
        relation_b_summary="Sự kiện A được theo sau bởi sự kiện B",
        relation_b_description="""Quan hệ thời gian: B xảy ra sau A.

Ví dụ: "Triều Lý" FOLLOWED_BY "Triều Trần"
""",
        should_merge=False,
        reasoning="SUCCEEDED_DYNASTY: kế thừa quyền lực chính thức, FOLLOWED_BY: chỉ thời gian. "
        "SUCCEEDED có ý nghĩa chính trị, không chỉ temporal.",
        is_boundary_case=True,
    ),
    RelationSample(
        id=54,
        category=MergeCategory.DOMAIN_OVERLAPS,
        relation_a_name="ALLIED_WITH",
        relation_a_summary="Lực lượng A liên minh với lực lượng B",
        relation_a_description="""Quan hệ liên minh: A và B hợp tác chống kẻ thù chung.

Ví dụ: "Việt Nam" ALLIED_WITH "Liên Xô" (kháng chiến)
""",
        relation_b_name="SUPPORTED_BY",
        relation_b_summary="Lực lượng A được hỗ trợ bởi B",
        relation_b_description="""Quan hệ hỗ trợ: B giúp đỡ A.

Ví dụ: "Kháng chiến VN" SUPPORTED_BY "Viện trợ quốc tế"
""",
        should_merge=False,
        reasoning="ALLIED_WITH: quan hệ ngang hàng, SUPPORTED_BY: B hỗ trợ A (không ngang hàng). "
        "Khác về cấu trúc quan hệ.",
        is_boundary_case=True,
    ),
    RelationSample(
        id=55,
        category=MergeCategory.DOMAIN_OVERLAPS,
        relation_a_name="FOUNDED_BY",
        relation_a_summary="Tổ chức/triều đại A được sáng lập bởi B",
        relation_a_description="""Quan hệ sáng lập: B thành lập A.

Ví dụ: "Nhà Lý" FOUNDED_BY "Lý Công Uẩn"
""",
        relation_b_name="LED_BY",
        relation_b_summary="Tổ chức A được lãnh đạo bởi B",
        relation_b_description="""Quan hệ lãnh đạo: B đứng đầu A.

Ví dụ: "KN Lam Sơn" LED_BY "Lê Lợi"
""",
        should_merge=False,
        reasoning="FOUNDED_BY: sáng lập ban đầu, LED_BY: lãnh đạo (có thể không phải người sáng lập). "
        "Founder ≠ Leader.",
        is_boundary_case=False,
    ),
    RelationSample(
        id=56,
        category=MergeCategory.DOMAIN_OVERLAPS,
        relation_a_name="CONQUERED_BY",
        relation_a_summary="Vùng A bị chinh phục bởi B",
        relation_a_description="""Quan hệ chinh phục: B đánh chiếm A.

Ví dụ: "Champa" CONQUERED_BY "Đại Việt" (Nam tiến)
""",
        relation_b_name="ANNEXED_BY",
        relation_b_summary="Vùng A bị sáp nhập vào B",
        relation_b_description="""Quan hệ sáp nhập: A trở thành phần của B.

Ví dụ: "Nam Kỳ" ANNEXED_BY "Đông Dương thuộc Pháp"
""",
        should_merge=True,
        reasoning="CONQUERED_BY và ANNEXED_BY đều mô tả B chiếm lấy A. "
        "CONQUERED nhấn mạnh quân sự, ANNEXED nhấn mạnh chính trị, nhưng kết quả tương tự.",
        is_boundary_case=True,
    ),
    RelationSample(
        id=57,
        category=MergeCategory.DOMAIN_OVERLAPS,
        relation_a_name="INFLUENCED_BY",
        relation_a_summary="Sự kiện/tư tưởng A chịu ảnh hưởng từ B",
        relation_a_description="""Quan hệ ảnh hưởng: B tác động đến A.

Ví dụ: "Phong trào Duy Tân" INFLUENCED_BY "Tư tưởng cải cách Nhật Bản"
""",
        relation_b_name="INSPIRED_BY",
        relation_b_summary="Phong trào A được truyền cảm hứng từ B",
        relation_b_description="""Quan hệ cảm hứng: B truyền cảm hứng cho A.

Ví dụ: "Cách mạng VN" INSPIRED_BY "Cách mạng Nga 1917"
""",
        should_merge=True,
        reasoning="INFLUENCED_BY và INSPIRED_BY đều mô tả B tác động tư tưởng đến A. "
        "INSPIRED có connotation tích cực hơn nhưng semantic tương đương.",
        is_boundary_case=True,
    ),
    RelationSample(
        id=58,
        category=MergeCategory.DOMAIN_OVERLAPS,
        relation_a_name="REFORMED_BY",
        relation_a_summary="Hệ thống A được cải cách bởi B",
        relation_a_description="""Quan hệ cải cách: B thay đổi A.

Ví dụ: "Chế độ ruộng đất" REFORMED_BY "Hồ Quý Ly"
""",
        relation_b_name="CHANGED_BY",
        relation_b_summary="Hệ thống A bị thay đổi bởi B",
        relation_b_description="""Quan hệ thay đổi: B làm A khác đi.

Ví dụ: "Xã hội VN" CHANGED_BY "Chính sách thực dân"
""",
        should_merge=True,
        reasoning="REFORMED_BY và CHANGED_BY đều mô tả B thay đổi A. "
        "REFORMED có connotation có chủ đích, CHANGED tổng quát hơn, nhưng có thể merge.",
        is_boundary_case=True,
    ),
    RelationSample(
        id=59,
        category=MergeCategory.DOMAIN_OVERLAPS,
        relation_a_name="EXECUTED_BY",
        relation_a_summary="Nhân vật A bị xử tử bởi B",
        relation_a_description="""Quan hệ xử tử: B giết A theo pháp luật.

Ví dụ: "Nguyễn Trãi" EXECUTED_BY "Nhà Lê" (oan án Lệ Chi Viên)
""",
        relation_b_name="KILLED_BY",
        relation_b_summary="Nhân vật A bị giết bởi B",
        relation_b_description="""Quan hệ giết: B gây ra cái chết của A.

Ví dụ: "Tổng đốc Hoàng Diệu" KILLED_BY "Quân Pháp"
""",
        should_merge=False,
        reasoning="EXECUTED_BY: xử tử theo pháp luật, KILLED_BY: giết tổng quát. "
        "Sự phân biệt quan trọng cho phân tích lịch sử pháp lý.",
        is_boundary_case=False,
    ),
    RelationSample(
        id=60,
        category=MergeCategory.DOMAIN_OVERLAPS,
        relation_a_name="EXILED_TO",
        relation_a_summary="Nhân vật A bị đày đến vùng B",
        relation_a_description="""Quan hệ lưu đày: A bị buộc rời khỏi quê hương đến B.

Ví dụ: "Phan Bội Châu" EXILED_TO "Nhà tù Côn Đảo"
""",
        relation_b_name="IMPRISONED_AT",
        relation_b_summary="Nhân vật A bị giam tại B",
        relation_b_description="""Quan hệ giam giữ: A bị nhốt tại B.

Ví dụ: "Nguyễn Ái Quốc" IMPRISONED_AT "Nhà tù Victoria, Hồng Kông"
""",
        should_merge=False,
        reasoning="EXILED_TO: bị đày (có thể không bị giam), IMPRISONED_AT: bị giam giữ. "
        "Exile và imprisonment là hai hình thức khác nhau.",
        is_boundary_case=False,
    ),
]


# =============================================================================
# CATEGORY 6: CROSS FAMILY (should_merge=False) - 10 samples
# Tương tự về ngôn ngữ nhưng KHÁC FAMILY - KHÔNG MERGE
# =============================================================================

CROSS_FAMILY_SAMPLES = [
    RelationSample(
        id=61,
        category=MergeCategory.CROSS_FAMILY,
        relation_a_name="FOLLOWED_BY",
        relation_a_summary="Sự kiện A xảy ra TRƯỚC sự kiện B (temporal)",
        relation_a_description="""Quan hệ thời gian: A xảy ra trước B.
Family: TEMPORAL

Ví dụ: "Triều Lý" FOLLOWED_BY "Triều Trần"
""",
        relation_b_name="RESULTED_IN",
        relation_b_summary="Sự kiện A có kết quả là B (causal)",
        relation_b_description="""Quan hệ nhân quả: A gây ra B.
Family: CAUSAL

Ví dụ: "Chiến thắng" RESULTED_IN "Ký hiệp ước"
""",
        should_merge=False,
        reasoning="FOLLOWED_BY: temporal (không có nhân quả), RESULTED_IN: causal (có nhân quả). "
        "Khác family hoàn toàn mặc dù cùng hướng A→B.",
        is_boundary_case=True,
    ),
    RelationSample(
        id=62,
        category=MergeCategory.CROSS_FAMILY,
        relation_a_name="PART_OF",
        relation_a_summary="A là một phần của B (structural)",
        relation_a_description="""Quan hệ cấu trúc: A nằm trong B.
Family: STRUCTURAL

Ví dụ: "Trận Bạch Đằng" PART_OF "Kháng chiến chống Nguyên"
""",
        relation_b_name="DURING",
        relation_b_summary="A xảy ra trong thời kỳ B (temporal)",
        relation_b_description="""Quan hệ thời gian: A nằm trong khoảng thời gian B.
Family: TEMPORAL

Ví dụ: "Trận Bạch Đằng" DURING "Kháng chiến chống Nguyên"
""",
        should_merge=False,
        reasoning="PART_OF: quan hệ cấu trúc logic, DURING: quan hệ thời gian. "
        "Mặc dù có thể cùng đúng nhưng mô tả khía cạnh khác nhau.",
        is_boundary_case=True,
    ),
    RelationSample(
        id=63,
        category=MergeCategory.CROSS_FAMILY,
        relation_a_name="CONTRIBUTED_TO",
        relation_a_summary="A đóng góp vào B (participation)",
        relation_a_description="""Quan hệ tham gia: A góp phần vào B.
Family: PARTICIPATION

Ví dụ: "Tinh thần yêu nước" CONTRIBUTED_TO "Chiến thắng"
""",
        relation_b_name="LED_TO",
        relation_b_summary="A dẫn đến B (causal)",
        relation_b_description="""Quan hệ nhân quả: A là nguyên nhân của B.
Family: CAUSAL

Ví dụ: "Thất bại ngoại giao" LED_TO "Chiến tranh"
""",
        should_merge=False,
        reasoning="CONTRIBUTED_TO: đóng góp (partial, participation), LED_TO: nguyên nhân (causal). "
        "Khác về mức độ và family.",
        is_boundary_case=True,
    ),
    RelationSample(
        id=64,
        category=MergeCategory.CROSS_FAMILY,
        relation_a_name="CONTEXTUALIZES",
        relation_a_summary="A cung cấp bối cảnh cho B (elaborative)",
        relation_a_description="""Quan hệ mở rộng: A là background cho B.
Family: ELABORATIVE

Ví dụ: "Bối cảnh quốc tế" CONTEXTUALIZES "Phong trào VN"
""",
        relation_b_name="CAUSED_BY",
        relation_b_summary="A bị gây ra bởi B (causal)",
        relation_b_description="""Quan hệ nhân quả: B là nguyên nhân của A.
Family: CAUSAL

Ví dụ: "Phong trào" CAUSED_BY "Bối cảnh"
""",
        should_merge=False,
        reasoning="CONTEXTUALIZES: cung cấp background (elaborative), CAUSED_BY: nhân quả. "
        "Bối cảnh không nhất thiết là nguyên nhân.",
        is_boundary_case=True,
    ),
    RelationSample(
        id=65,
        category=MergeCategory.CROSS_FAMILY,
        relation_a_name="SUPPORTS",
        relation_a_summary="A cung cấp bằng chứng cho B (elaborative)",
        relation_a_description="""Quan hệ hỗ trợ luận điểm: A chứng minh B.
Family: ELABORATIVE

Ví dụ: "Số liệu" SUPPORTS "Luận điểm"
""",
        relation_b_name="CAUSED_BY",
        relation_b_summary="A bị gây ra bởi B (causal)",
        relation_b_description="""Quan hệ nhân quả.
Family: CAUSAL

Ví dụ: "Kết quả" CAUSED_BY "Nguyên nhân"
""",
        should_merge=False,
        reasoning="SUPPORTS: bằng chứng (epistemological), CAUSED_BY: nhân quả (ontological). "
        "Khác hoàn toàn về bản chất.",
        is_boundary_case=False,
    ),
    RelationSample(
        id=66,
        category=MergeCategory.CROSS_FAMILY,
        relation_a_name="SIMILAR_TO",
        relation_a_summary="A và B tương đồng về nội dung (comparative)",
        relation_a_description="""Quan hệ so sánh: A giống B.
Family: COMPARATIVE

Ví dụ: "KN này" SIMILAR_TO "KN kia"
""",
        relation_b_name="CONTEMPORARY_WITH",
        relation_b_summary="A và B xảy ra cùng thời (temporal)",
        relation_b_description="""Quan hệ thời gian.
Family: TEMPORAL

Ví dụ: "Sự kiện A" CONTEMPORARY_WITH "Sự kiện B"
""",
        should_merge=False,
        reasoning="SIMILAR_TO: tương đồng nội dung, CONTEMPORARY_WITH: cùng thời gian. "
        "Hai sự kiện có thể cùng thời nhưng không tương đồng.",
        is_boundary_case=False,
    ),
    RelationSample(
        id=67,
        category=MergeCategory.CROSS_FAMILY,
        relation_a_name="PRECEDED_BY",
        relation_a_summary="A xảy ra sau B (temporal)",
        relation_a_description="""Quan hệ thời gian.
Family: TEMPORAL

Ví dụ: "Triều Trần" PRECEDED_BY "Triều Lý"
""",
        relation_b_name="BUILDS_UPON",
        relation_b_summary="A phát triển từ B (educational)",
        relation_b_description="""Quan hệ giáo dục.
Family: EDUCATIONAL

Ví dụ: "Bài nâng cao" BUILDS_UPON "Bài cơ bản"
""",
        should_merge=False,
        reasoning="PRECEDED_BY: thuần thời gian, BUILDS_UPON: phát triển kiến thức. "
        "Xảy ra sau không có nghĩa là phát triển từ.",
        is_boundary_case=False,
    ),
    RelationSample(
        id=68,
        category=MergeCategory.CROSS_FAMILY,
        relation_a_name="IS_A",
        relation_a_summary="A là một loại của B (structural)",
        relation_a_description="""Quan hệ phân loại taxonomy.
Family: STRUCTURAL

Ví dụ: "KN Hai Bà Trưng" IS_A "Cuộc khởi nghĩa"
""",
        relation_b_name="EXEMPLIFIES",
        relation_b_summary="A là ví dụ của B (elaborative)",
        relation_b_description="""Quan hệ minh họa.
Family: ELABORATIVE

Ví dụ: "KN Hai Bà Trưng" EXEMPLIFIES "Tinh thần yêu nước"
""",
        should_merge=False,
        reasoning="IS_A: phân loại (A thuộc class B), EXEMPLIFIES: minh họa (A là example của concept B). "
        "Taxonomy vs illustration.",
        is_boundary_case=True,
    ),
    RelationSample(
        id=69,
        category=MergeCategory.CROSS_FAMILY,
        relation_a_name="REFERS_TO",
        relation_a_summary="A tham chiếu đến B (structural)",
        relation_a_description="""Quan hệ tham chiếu explicit.
Family: STRUCTURAL

Ví dụ: "Xem phần trước" REFERS_TO "Phần giới thiệu"
""",
        relation_b_name="ELABORATES",
        relation_b_summary="A mở rộng B (elaborative)",
        relation_b_description="""Quan hệ mở rộng nội dung.
Family: ELABORATIVE

Ví dụ: "Chi tiết" ELABORATES "Tổng quan"
""",
        should_merge=False,
        reasoning="REFERS_TO: chỉ tham chiếu (không thêm nội dung), ELABORATES: mở rộng (thêm nội dung). "
        "Reference vs expansion.",
        is_boundary_case=False,
    ),
    RelationSample(
        id=70,
        category=MergeCategory.CROSS_FAMILY,
        relation_a_name="SUPERSEDES",
        relation_a_summary="A thay thế B (temporal/versioning)",
        relation_a_description="""Quan hệ thay thế phiên bản.
Family: TEMPORAL

Ví dụ: "SGK mới" SUPERSEDES "SGK cũ"
""",
        relation_b_name="CONTRADICTS",
        relation_b_summary="A mâu thuẫn với B (comparative)",
        relation_b_description="""Quan hệ mâu thuẫn logic.
Family: COMPARATIVE

Ví dụ: "Luận điểm A" CONTRADICTS "Luận điểm B"
""",
        should_merge=False,
        reasoning="SUPERSEDES: A thay thế B (B không còn valid), CONTRADICTS: A và B không thể cùng đúng. "
        "Khác về hậu quả và family.",
        is_boundary_case=False,
    ),
]


# =============================================================================
# COMBINED DATASET
# =============================================================================

ALL_SAMPLES: list[RelationSample] = (
    SEMANTIC_SYNONYM_SAMPLES
    + DIRECTIONAL_INVERSE_SAMPLES
    + GRANULARITY_VARIANT_SAMPLES
    + RELATED_DISTINCT_SAMPLES
    + DOMAIN_OVERLAP_SAMPLES
    + CROSS_FAMILY_SAMPLES
)


def get_samples_by_category(category: MergeCategory | str) -> list[RelationSample]:
    """Get samples filtered by category."""
    if isinstance(category, str):
        category = MergeCategory(category)
    return [s for s in ALL_SAMPLES if s.category == category]


def get_positive_samples() -> list[RelationSample]:
    """Get samples where should_merge=True."""
    return [s for s in ALL_SAMPLES if s.should_merge]


def get_negative_samples() -> list[RelationSample]:
    """Get samples where should_merge=False."""
    return [s for s in ALL_SAMPLES if not s.should_merge]


def get_boundary_cases() -> list[RelationSample]:
    """Get boundary case samples (near threshold)."""
    return [s for s in ALL_SAMPLES if s.is_boundary_case]


def print_statistics() -> None:
    """Print dataset statistics."""
    print("=" * 70)
    print("EXTRACTION_MERGE_THRESHOLD BENCHMARK DATASET")
    print("=" * 70)
    print(f"Total samples: {len(ALL_SAMPLES)}")
    print(f"  - Positive (should_merge=True): {len(get_positive_samples())}")
    print(f"  - Negative (should_merge=False): {len(get_negative_samples())}")
    print(f"  - Boundary cases: {len(get_boundary_cases())}")
    print()
    print("By category:")
    for cat in MergeCategory:
        samples = get_samples_by_category(cat)
        positive = len([s for s in samples if s.should_merge])
        negative = len([s for s in samples if not s.should_merge])
        print(f"  - {cat.value}: {len(samples)} (merge={positive}, no_merge={negative})")
    print("=" * 70)


if __name__ == "__main__":
    print_statistics()
