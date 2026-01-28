"""Benchmark samples for EXTRACTION_AUTO_MERGE_THRESHOLD and EXTRACTION_LLM_THRESHOLD.

100 sample pairs of content nodes with ground truth labels.
Domain: Lịch sử Việt Nam và Thế giới (Vietnamese History)

FOCUS: Testing threshold boundaries for node similarity decisions.
- Text length: 300-800 characters per node (realistic paragraph size)
- Content: Based on actual Vietnamese history curriculum

Each sample contains:
- node_a_text: First node content
- node_b_text: Second node content
- should_merge: Ground truth - True if semantically equivalent/duplicate
- expected_zone: AUTO_MERGE, LLM_DECIDE, or KEEP_SEPARATE
- category: Type of comparison
- reasoning: Explanation for the ground truth

Categories:
1. DUPLICATE (should_merge=True): Exact or near-exact duplicates -> AUTO_MERGE zone
2. PARAPHRASE (should_merge=True): Same content, different wording -> AUTO_MERGE zone
3. SUBSET (should_merge=True): One contains the other -> LLM_DECIDE zone
4. OVERLAPPING (boundary): Significant overlap but distinct -> LLM_DECIDE zone
5. RELATED_DISTINCT (should_merge=False): Same topic, different content -> KEEP_SEPARATE
6. TEMPORAL_DIFFERENT (should_merge=False): Same event, different period -> KEEP_SEPARATE
7. CAUSE_EFFECT (should_merge=False): Causal relationship -> KEEP_SEPARATE
8. UNRELATED (should_merge=False): Different topics entirely -> KEEP_SEPARATE
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class NodeCategory(str, Enum):
    """Categories of node pairs for analysis."""

    DUPLICATE = "duplicate"  # Exact/near-exact duplicates -> MERGE
    PARAPHRASE = "paraphrase"  # Same meaning, different words -> MERGE
    SUBSET = "subset"  # One contains the other -> MERGE (usually)
    OVERLAPPING = "overlapping"  # Significant overlap -> LLM decides
    RELATED_DISTINCT = "related_distinct"  # Same topic, different content -> NO MERGE
    TEMPORAL_DIFFERENT = "temporal_different"  # Same subject, different time -> NO MERGE
    CAUSE_EFFECT = "cause_effect"  # Causal relationship -> NO MERGE
    UNRELATED = "unrelated"  # Different topics -> NO MERGE


class ExpectedZone(str, Enum):
    """Expected similarity zone for threshold decisions."""

    AUTO_MERGE = "auto_merge"  # >= AUTO_MERGE_THRESHOLD
    LLM_DECIDE = "llm_decide"  # >= LLM_THRESHOLD and < AUTO_MERGE_THRESHOLD
    KEEP_SEPARATE = "keep_separate"  # < LLM_THRESHOLD


@dataclass
class NodePairSample:
    """A sample pair of nodes for benchmark.

    Format matches content_nodes in ArangoDB:
    - text: 300-800 characters of Vietnamese historical content
    """

    id: int
    category: NodeCategory
    expected_zone: ExpectedZone

    # Node A
    node_a_text: str

    # Node B
    node_b_text: str

    # Ground truth
    should_merge: bool
    reasoning: str

    # For analysis
    is_boundary_case: bool = False  # Near threshold - critical for evaluation


# =============================================================================
# CATEGORY 1: DUPLICATE (should_merge=True) - 15 samples
# Nội dung giống hệt hoặc gần hệt, chỉ khác vài từ/dấu câu
# Expected: HIGH similarity (>= 0.85), AUTO_MERGE zone
# =============================================================================

DUPLICATE_SAMPLES = [
    NodePairSample(
        id=1,
        category=NodeCategory.DUPLICATE,
        expected_zone=ExpectedZone.AUTO_MERGE,
        node_a_text="""Hội nghị I-an-ta diễn ra từ ngày 4 đến ngày 11-2-1945 tại thành phố I-an-ta (Liên Xô), với sự tham gia của ba cường quốc Liên Xô, Mỹ và Anh. Hội nghị đưa ra nhiều quyết định quan trọng: thống nhất mục tiêu tiêu diệt tận gốc chủ nghĩa phát xít Đức và chủ nghĩa quân phiệt Nhật, thành lập tổ chức Liên hiệp quốc để duy trì hoà bình và an ninh quốc tế.""",
        node_b_text="""Hội nghị I-an-ta diễn ra từ ngày 4 đến ngày 11-2-1945 tại thành phố I-an-ta (Liên Xô), với sự tham gia của ba cường quốc Liên Xô, Mỹ và Anh. Hội nghị đưa ra nhiều quyết định quan trọng: thống nhất mục tiêu tiêu diệt tận gốc chủ nghĩa phát xít Đức và chủ nghĩa quân phiệt Nhật; thành lập tổ chức Liên hiệp quốc để duy trì hoà bình và an ninh quốc tế.""",
        should_merge=True,
        reasoning="Nội dung giống hệt, chỉ khác dấu phẩy/chấm phẩy. Rõ ràng là duplicate.",
        is_boundary_case=False,
    ),
    NodePairSample(
        id=2,
        category=NodeCategory.DUPLICATE,
        expected_zone=ExpectedZone.AUTO_MERGE,
        node_a_text="""Trật tự thế giới hai cực I-an-ta hình thành đã chia thế giới thành hai cực đối lập: cực Liên Xô đứng đầu hệ thống xã hội chủ nghĩa và cực Mỹ đứng đầu hệ thống tư bản chủ nghĩa. Sự đối đầu này thể hiện qua các mặt tư tưởng, chính trị, kinh tế và quân sự, kéo dài suốt giai đoạn Chiến tranh lạnh từ 1945 đến 1991.""",
        node_b_text="""Trật tự thế giới hai cực I-an-ta hình thành đã chia thế giới thành hai cực đối lập: cực Liên Xô đứng đầu hệ thống xã hội chủ nghĩa và cực Mỹ đứng đầu hệ thống tư bản chủ nghĩa. Sự đối đầu này thể hiện qua các mặt tư tưởng, chính trị, kinh tế, quân sự, kéo dài suốt giai đoạn Chiến tranh lạnh từ năm 1945 đến năm 1991.""",
        should_merge=True,
        reasoning="Gần như giống hệt, chỉ khác 'từ 1945' vs 'từ năm 1945'. Duplicate rõ ràng.",
        is_boundary_case=False,
    ),
    NodePairSample(
        id=3,
        category=NodeCategory.DUPLICATE,
        expected_zone=ExpectedZone.AUTO_MERGE,
        node_a_text="""Năm 1988, Lực lượng gìn giữ hoà bình của Liên hợp quốc đã được trao tặng Giải thưởng Nô-ben Hoà bình. Năm 2001, Tổ chức Liên hợp quốc và Tổng Thư kí Cô-phi Át-ta An-nan cũng được trao tặng Giải thưởng Nô-ben Hoà bình. Đây là sự ghi nhận quan trọng cho những đóng góp của Liên hợp quốc trong việc duy trì hoà bình thế giới.""",
        node_b_text="""Năm 1988, Lực lượng gìn giữ hoà bình của Liên hợp quốc đã được trao tặng Giải thưởng Nô-ben Hoà bình. Năm 2001, Tổ chức Liên hợp quốc và Tổng Thư kí Cô-phi Át-ta An-nan được trao tặng Giải thưởng Nô-ben Hoà bình. Đây là sự ghi nhận quan trọng cho những đóng góp của LHQ trong việc duy trì hoà bình thế giới.""",
        should_merge=True,
        reasoning="Giống hệt, chỉ khác 'cũng được' vs 'được' và 'Liên hợp quốc' vs 'LHQ'. Duplicate.",
        is_boundary_case=False,
    ),
    NodePairSample(
        id=4,
        category=NodeCategory.DUPLICATE,
        expected_zone=ExpectedZone.AUTO_MERGE,
        node_a_text="""Sự sụp đổ của chủ nghĩa xã hội ở Đông Âu và Liên Xô những năm 1989-1991 đã chấm dứt sự tồn tại của Trật tự thế giới hai cực I-an-ta. Đây là một trong những sự kiện quan trọng nhất của lịch sử thế giới cuối thế kỷ XX, đánh dấu sự kết thúc của Chiến tranh lạnh và mở ra một kỷ nguyên mới trong quan hệ quốc tế.""",
        node_b_text="""Sự sụp đổ của chủ nghĩa xã hội ở Đông Âu và Liên Xô những năm 1989 – 1991 đã chấm dứt sự tồn tại của Trật tự thế giới hai cực I-an-ta. Đây là một trong những sự kiện quan trọng nhất của lịch sử thế giới cuối thế kỷ XX, đánh dấu sự kết thúc của Chiến tranh lạnh và mở ra một kỷ nguyên mới trong quan hệ quốc tế.""",
        should_merge=True,
        reasoning="Giống hệt hoàn toàn, chỉ khác format năm (1989-1991 vs 1989 – 1991). Duplicate.",
        is_boundary_case=False,
    ),
    NodePairSample(
        id=5,
        category=NodeCategory.DUPLICATE,
        expected_zone=ExpectedZone.AUTO_MERGE,
        node_a_text="""Ngày 7-6-2019, tại phiên họp Hội đồng Bảo an Liên hợp quốc ở Niu Y-oóc (Mỹ), với số phiếu bầu 192/193, Việt Nam chính thức trở thành Ủy viên không thường trực Hội đồng Bảo an Liên hợp quốc nhiệm kì 2020-2021. Đây là lần thứ hai Việt Nam đảm nhiệm vị trí này trong Liên hợp quốc.""",
        node_b_text="""Ngày 7-6-2019, tại phiên họp Hội đồng Bảo an Liên hợp quốc ở Niu Y-oóc (Mỹ), với số phiếu bầu 192/193 phiếu, Việt Nam chính thức trở thành Ủy viên không thường trực Hội đồng Bảo an Liên hợp quốc nhiệm kì 2020 – 2021. Đây là lần thứ hai Việt Nam đảm nhiệm vị trí này trong Liên hợp quốc.""",
        should_merge=True,
        reasoning="Gần như giống hệt, chỉ khác format năm và thêm chữ 'phiếu'. Duplicate.",
        is_boundary_case=False,
    ),
    NodePairSample(
        id=6,
        category=NodeCategory.DUPLICATE,
        expected_zone=ExpectedZone.AUTO_MERGE,
        node_a_text="""Liên hợp quốc có vai trò quan trọng trong cuộc đấu tranh nhằm thủ tiêu hệ thống thuộc địa của chủ nghĩa thực dân và thủ tiêu hoàn toàn các hình thức của chế độ phân biệt chủng tộc. Năm 1960, Đại hội đồng Liên hợp quốc đã thông qua bản Tuyên ngôn về thủ tiêu hoàn toàn chủ nghĩa thực dân, trao trả độc lập cho các quốc gia và dân tộc thuộc địa.""",
        node_b_text="""Liên hợp quốc có vai trò quan trọng trong cuộc đấu tranh nhằm thủ tiêu hệ thống thuộc địa của chủ nghĩa thực dân, thủ tiêu hoàn toàn các hình thức của chế độ phân biệt chủng tộc. Năm 1960, Đại hội đồng Liên hợp quốc đã thông qua bản Tuyên ngôn về thủ tiêu hoàn toàn chủ nghĩa thực dân, trao trả độc lập cho các quốc gia và dân tộc thuộc địa.""",
        should_merge=True,
        reasoning="Giống hệt, chỉ khác 'và thủ tiêu' vs ', thủ tiêu'. Duplicate.",
        is_boundary_case=False,
    ),
    NodePairSample(
        id=7,
        category=NodeCategory.DUPLICATE,
        expected_zone=ExpectedZone.AUTO_MERGE,
        node_a_text="""Chiến thắng Điện Biên Phủ năm 1954 là một trong những chiến thắng vĩ đại nhất trong lịch sử đấu tranh chống ngoại xâm của dân tộc Việt Nam. Chiến thắng này đã buộc thực dân Pháp phải ký Hiệp định Genève, chấm dứt cuộc chiến tranh xâm lược Đông Dương và công nhận độc lập, chủ quyền, thống nhất và toàn vẹn lãnh thổ của Việt Nam.""",
        node_b_text="""Chiến thắng Điện Biên Phủ năm 1954 là một trong những chiến thắng vĩ đại nhất trong lịch sử đấu tranh chống ngoại xâm của dân tộc Việt Nam. Chiến thắng này đã buộc thực dân Pháp phải ký Hiệp định Giơ-ne-vơ, chấm dứt cuộc chiến tranh xâm lược Đông Dương và công nhận độc lập, chủ quyền, thống nhất và toàn vẹn lãnh thổ của Việt Nam.""",
        should_merge=True,
        reasoning="Giống hệt, chỉ khác cách viết 'Genève' vs 'Giơ-ne-vơ' (phiên âm). Duplicate.",
        is_boundary_case=False,
    ),
    NodePairSample(
        id=8,
        category=NodeCategory.DUPLICATE,
        expected_zone=ExpectedZone.AUTO_MERGE,
        node_a_text="""Cách mạng tháng Tám năm 1945 là một cuộc cách mạng dân tộc dân chủ nhân dân. Thắng lợi của Cách mạng tháng Tám đã đập tan xiềng xích nô lệ của thực dân Pháp trong gần một thế kỷ, lật đổ chế độ quân chủ chuyên chế tồn tại hàng nghìn năm, lập nên nước Việt Nam Dân chủ Cộng hòa - Nhà nước dân chủ nhân dân đầu tiên ở Đông Nam Á.""",
        node_b_text="""Cách mạng tháng Tám năm 1945 là một cuộc cách mạng dân tộc dân chủ nhân dân. Thắng lợi của Cách mạng Tháng Tám đã đập tan xiềng xích nô lệ của thực dân Pháp trong gần một thế kỷ, lật đổ chế độ quân chủ chuyên chế tồn tại hàng nghìn năm, lập nên nước Việt Nam Dân chủ Cộng hoà - Nhà nước dân chủ nhân dân đầu tiên ở Đông Nam Á.""",
        should_merge=True,
        reasoning="Giống hệt, chỉ khác 'tháng Tám' vs 'Tháng Tám' và 'Cộng hòa' vs 'Cộng hoà'. Duplicate.",
        is_boundary_case=False,
    ),
    NodePairSample(
        id=9,
        category=NodeCategory.DUPLICATE,
        expected_zone=ExpectedZone.AUTO_MERGE,
        node_a_text="""Khởi nghĩa Lam Sơn (1418-1427) do Lê Lợi lãnh đạo là một trong những cuộc khởi nghĩa vĩ đại nhất trong lịch sử chống ngoại xâm của dân tộc ta. Sau 10 năm kháng chiến gian khổ, nghĩa quân Lam Sơn đã đánh bại quân Minh xâm lược, giành lại độc lập cho Tổ quốc, mở ra thời kỳ phát triển mới của đất nước dưới triều Lê sơ.""",
        node_b_text="""Khởi nghĩa Lam Sơn (1418 - 1427) do Lê Lợi lãnh đạo là một trong những cuộc khởi nghĩa vĩ đại nhất trong lịch sử chống ngoại xâm của dân tộc ta. Sau mười năm kháng chiến gian khổ, nghĩa quân Lam Sơn đã đánh bại quân Minh xâm lược, giành lại độc lập cho Tổ quốc, mở ra thời kỳ phát triển mới của đất nước dưới triều Lê sơ.""",
        should_merge=True,
        reasoning="Giống hệt, chỉ khác format năm và '10 năm' vs 'mười năm'. Duplicate.",
        is_boundary_case=False,
    ),
    NodePairSample(
        id=10,
        category=NodeCategory.DUPLICATE,
        expected_zone=ExpectedZone.AUTO_MERGE,
        node_a_text="""Triều đại nhà Trần (1225-1400) là một trong những triều đại huy hoàng nhất trong lịch sử phong kiến Việt Nam. Nhà Trần nổi tiếng với ba lần kháng chiến chống quân Nguyên-Mông xâm lược (1258, 1285, 1287-1288), thể hiện tinh thần yêu nước và ý chí quật cường của dân tộc Việt Nam.""",
        node_b_text="""Triều đại nhà Trần (1225 – 1400) là một trong những triều đại huy hoàng nhất trong lịch sử phong kiến Việt Nam. Nhà Trần nổi tiếng với 3 lần kháng chiến chống quân Nguyên Mông xâm lược (1258, 1285, 1287 - 1288), thể hiện tinh thần yêu nước và ý chí quật cường của dân tộc Việt Nam.""",
        should_merge=True,
        reasoning="Giống hệt, chỉ khác format số và gạch nối. Duplicate.",
        is_boundary_case=False,
    ),
    NodePairSample(
        id=11,
        category=NodeCategory.DUPLICATE,
        expected_zone=ExpectedZone.AUTO_MERGE,
        node_a_text="""Hiến chương Liên hợp quốc xác định mục tiêu của tổ chức là duy trì hoà bình và an ninh thế giới, phát triển các mối quan hệ hữu nghị giữa các dân tộc và tiến hành hợp tác quốc tế giữa các nước. Liên hợp quốc đóng vai trò là trung tâm điều hoà các nỗ lực quốc tế vì những mục tiêu chung.""",
        node_b_text="""Hiến chương Liên hợp quốc xác định mục tiêu của tổ chức là duy trì hoà bình và an ninh thế giới, phát triển các mối quan hệ hữu nghị giữa các dân tộc, tiến hành hợp tác quốc tế giữa các nước. Liên hợp quốc đóng vai trò là trung tâm điều hoà các nỗ lực quốc tế vì những mục tiêu chung.""",
        should_merge=True,
        reasoning="Giống hệt, chỉ khác 'và tiến hành' vs ', tiến hành'. Duplicate.",
        is_boundary_case=False,
    ),
    NodePairSample(
        id=12,
        category=NodeCategory.DUPLICATE,
        expected_zone=ExpectedZone.AUTO_MERGE,
        node_a_text="""Ngô Quyền (898-944) là vị anh hùng dân tộc đã lãnh đạo nhân dân ta đánh bại quân Nam Hán trong trận Bạch Đằng năm 938, chấm dứt hơn 1000 năm Bắc thuộc, mở ra kỷ nguyên độc lập tự chủ lâu dài cho dân tộc Việt Nam. Chiến thắng Bạch Đằng 938 có ý nghĩa lịch sử to lớn trong lịch sử dựng nước và giữ nước của dân tộc ta.""",
        node_b_text="""Ngô Quyền (898 - 944) là vị anh hùng dân tộc đã lãnh đạo nhân dân ta đánh bại quân Nam Hán trong trận Bạch Đằng năm 938, chấm dứt hơn 1000 năm Bắc thuộc, mở ra kỷ nguyên độc lập tự chủ lâu dài cho dân tộc Việt Nam. Chiến thắng Bạch Đằng năm 938 có ý nghĩa lịch sử to lớn trong lịch sử dựng nước và giữ nước của dân tộc ta.""",
        should_merge=True,
        reasoning="Giống hệt, chỉ khác format năm sinh/mất và 'Bạch Đằng 938' vs 'Bạch Đằng năm 938'. Duplicate.",
        is_boundary_case=False,
    ),
    NodePairSample(
        id=13,
        category=NodeCategory.DUPLICATE,
        expected_zone=ExpectedZone.AUTO_MERGE,
        node_a_text="""Liên hợp quốc đã xây dựng và soạn thảo được một hệ thống các công ước quốc tế về giải trừ quân bị, chống chạy đua vũ trang, hạn chế vũ khí hạt nhân. Tổ chức này đã đóng góp tích cực vào việc duy trì hoà bình và an ninh quốc tế, triển khai hoạt động giữ gìn hoà bình ở nhiều khu vực trên thế giới.""",
        node_b_text="""Liên hợp quốc đã xây dựng và soạn thảo được một hệ thống các công ước quốc tế về giải trừ quân bị, chống chạy đua vũ trang, hạn chế vũ khí hạt nhân,... đóng góp tích cực vào việc duy trì hoà bình và an ninh quốc tế, triển khai hoạt động giữ gìn hoà bình ở nhiều khu vực trên thế giới.""",
        should_merge=True,
        reasoning="Gần như giống hệt, chỉ khác '. Tổ chức này đã đóng góp' vs ',... đóng góp'. Duplicate.",
        is_boundary_case=False,
    ),
    NodePairSample(
        id=14,
        category=NodeCategory.DUPLICATE,
        expected_zone=ExpectedZone.AUTO_MERGE,
        node_a_text="""Hồ Chí Minh (1890-1969) là lãnh tụ vĩ đại của dân tộc Việt Nam, người sáng lập Đảng Cộng sản Việt Nam và nước Việt Nam Dân chủ Cộng hòa. Chủ tịch Hồ Chí Minh đã cống hiến trọn đời cho sự nghiệp giải phóng dân tộc, thống nhất Tổ quốc và xây dựng chủ nghĩa xã hội ở Việt Nam.""",
        node_b_text="""Hồ Chí Minh (1890 - 1969) là lãnh tụ vĩ đại của dân tộc Việt Nam, người sáng lập Đảng Cộng sản Việt Nam và nước Việt Nam Dân chủ Cộng hoà. Chủ tịch Hồ Chí Minh đã cống hiến trọn đời cho sự nghiệp giải phóng dân tộc, thống nhất Tổ quốc và xây dựng chủ nghĩa xã hội ở Việt Nam.""",
        should_merge=True,
        reasoning="Giống hệt, chỉ khác format năm và 'Cộng hòa' vs 'Cộng hoà'. Duplicate.",
        is_boundary_case=False,
    ),
    NodePairSample(
        id=15,
        category=NodeCategory.DUPLICATE,
        expected_zone=ExpectedZone.AUTO_MERGE,
        node_a_text="""Quá trình hình thành Liên hợp quốc kéo dài từ năm 1941 đến năm 1945, trải qua nhiều sự kiện quan trọng. Ngày 1-1-1942, tại Oa-sinh-tơn (Mỹ), đại diện 26 nước chống phát xít đã kí bản Tuyên ngôn về Liên hợp quốc, cam kết thành lập tổ chức quốc tế vì hoà bình và an ninh sau chiến tranh.""",
        node_b_text="""Quá trình hình thành Liên hợp quốc kéo dài từ năm 1941 đến năm 1945, trải qua nhiều sự kiện quan trọng. Ngày 1/1/1942, tại Oa-sinh-tơn (Mỹ), đại diện 26 nước chống phát xít đã ký bản Tuyên ngôn về Liên hợp quốc, cam kết thành lập tổ chức quốc tế vì hoà bình và an ninh sau chiến tranh.""",
        should_merge=True,
        reasoning="Giống hệt, chỉ khác format ngày (1-1-1942 vs 1/1/1942) và 'kí' vs 'ký'. Duplicate.",
        is_boundary_case=False,
    ),
]

# =============================================================================
# CATEGORY 2: PARAPHRASE (should_merge=True) - 20 samples
# Cùng nội dung nhưng viết lại với từ ngữ khác
# Expected: HIGH similarity (0.75-0.85), AUTO_MERGE zone
# =============================================================================

PARAPHRASE_SAMPLES = [
    NodePairSample(
        id=16,
        category=NodeCategory.PARAPHRASE,
        expected_zone=ExpectedZone.AUTO_MERGE,
        node_a_text="""Hội nghị I-an-ta được tổ chức vào tháng 2 năm 1945 tại Liên Xô, với sự tham dự của lãnh đạo ba cường quốc: Liên Xô, Mỹ và Anh. Tại hội nghị này, các nước đã đạt được thỏa thuận về việc tiêu diệt chủ nghĩa phát xít và thành lập một tổ chức quốc tế để duy trì hòa bình thế giới.""",
        node_b_text="""Tháng 2-1945, ba cường quốc Liên Xô, Mỹ, Anh đã họp tại I-an-ta (Liên Xô). Hội nghị đưa ra các quyết định quan trọng: thống nhất mục tiêu tiêu diệt hoàn toàn chủ nghĩa phát xít và lập tổ chức Liên hợp quốc nhằm giữ gìn hòa bình, an ninh quốc tế sau chiến tranh.""",
        should_merge=True,
        reasoning="Cùng nội dung về Hội nghị I-an-ta, chỉ paraphrase lại cách diễn đạt. Nên merge.",
        is_boundary_case=True,
    ),
    NodePairSample(
        id=17,
        category=NodeCategory.PARAPHRASE,
        expected_zone=ExpectedZone.AUTO_MERGE,
        node_a_text="""Sau khi Trật tự thế giới hai cực I-an-ta sụp đổ năm 1991, thế giới bước vào một giai đoạn mới. Mỹ trở thành siêu cường duy nhất nhưng ảnh hưởng bị hạn chế. Các cường quốc mới như Trung Quốc, Ấn Độ ngày càng có vai trò quan trọng hơn trong quan hệ quốc tế.""",
        node_b_text="""Sự sụp đổ của Trật tự hai cực I-an-ta năm 1991 mở ra kỷ nguyên mới cho thế giới. Mặc dù Mỹ vẫn là siêu cường nhưng phạm vi ảnh hưởng thu hẹp. Trung Quốc, Ấn Độ và một số nước lớn khác đang nổi lên với vị trí, vai trò lớn hơn trên trường quốc tế.""",
        should_merge=True,
        reasoning="Cùng nội dung về hậu sụp đổ Trật tự I-an-ta, paraphrase khác nhau. Nên merge.",
        is_boundary_case=True,
    ),
    NodePairSample(
        id=18,
        category=NodeCategory.PARAPHRASE,
        expected_zone=ExpectedZone.AUTO_MERGE,
        node_a_text="""Liên hợp quốc thúc đẩy hợp tác quốc tế về kinh tế, tài chính, thương mại và khoa học kỹ thuật thông qua các chương trình, quỹ và tổ chức chuyên môn. Tổ chức này có nhiều chương trình hỗ trợ các nước đang phát triển về vốn, tri thức và cán bộ để phát triển kinh tế.""",
        node_b_text="""LHQ tạo môi trường thuận lợi cho hợp tác quốc tế trong lĩnh vực kinh tế, tài chính, khoa học-kỹ thuật qua các chương trình và tổ chức chuyên môn trực thuộc. LHQ hỗ trợ các quốc gia đang phát triển về vốn, kiến thức, kỹ thuật và nhân lực để thúc đẩy phát triển kinh tế.""",
        should_merge=True,
        reasoning="Cùng nội dung về vai trò kinh tế của LHQ, paraphrase với từ ngữ khác. Nên merge.",
        is_boundary_case=True,
    ),
    NodePairSample(
        id=19,
        category=NodeCategory.PARAPHRASE,
        expected_zone=ExpectedZone.AUTO_MERGE,
        node_a_text="""Chiến thắng Điện Biên Phủ 1954 buộc thực dân Pháp ký Hiệp định Genève, chấm dứt chiến tranh xâm lược Đông Dương. Chiến thắng này có ý nghĩa lịch sử to lớn, đánh dấu sự thất bại hoàn toàn của chủ nghĩa thực dân cũ trên thế giới.""",
        node_b_text="""Thắng lợi của chiến dịch Điện Biên Phủ năm 1954 đã buộc Pháp phải ký kết Hiệp định Giơ-ne-vơ, kết thúc cuộc chiến tranh xâm lược Đông Dương. Đây là thắng lợi mang tầm vóc thời đại, báo hiệu sự sụp đổ của chủ nghĩa thực dân cũ trên phạm vi toàn cầu.""",
        should_merge=True,
        reasoning="Cùng nội dung về ý nghĩa chiến thắng ĐBP, paraphrase. Nên merge.",
        is_boundary_case=True,
    ),
    NodePairSample(
        id=20,
        category=NodeCategory.PARAPHRASE,
        expected_zone=ExpectedZone.AUTO_MERGE,
        node_a_text="""Cách mạng tháng Tám 1945 đã xóa bỏ ách đô hộ của thực dân Pháp kéo dài gần 80 năm, lật đổ chế độ phong kiến tồn tại hàng nghìn năm, thành lập nước Việt Nam Dân chủ Cộng hòa - nhà nước dân chủ nhân dân đầu tiên ở Đông Nam Á.""",
        node_b_text="""Thắng lợi của Cách mạng Tháng Tám năm 1945 đã đập tan xiềng xích nô lệ của thực dân Pháp trong gần một thế kỷ, xóa bỏ chế độ quân chủ chuyên chế tồn tại hàng ngàn năm, khai sinh nước Việt Nam Dân chủ Cộng hòa - Nhà nước dân chủ nhân dân tiên phong ở khu vực Đông Nam Á.""",
        should_merge=True,
        reasoning="Cùng nội dung về ý nghĩa CMT8, paraphrase với từ ngữ khác. Nên merge.",
        is_boundary_case=True,
    ),
    NodePairSample(
        id=21,
        category=NodeCategory.PARAPHRASE,
        expected_zone=ExpectedZone.AUTO_MERGE,
        node_a_text="""Từ đầu những năm 70 thế kỷ XX, xu thế hòa hoãn Đông-Tây xuất hiện với các cuộc gặp gỡ thương lượng giữa Liên Xô và Mỹ. Tháng 12-1989, tại Malta, lãnh đạo hai nước tuyên bố hướng đến chấm dứt Chiến tranh lạnh, mở ra kỷ nguyên mới trong quan hệ Xô-Mỹ.""",
        node_b_text="""Đầu thập niên 70 của thế kỷ XX, xu hướng hòa hoãn giữa hai khối Đông và Tây bắt đầu xuất hiện thông qua các cuộc đàm phán Xô-Mỹ. Cuộc gặp không chính thức tại đảo Man-ta (tháng 12/1989) đã đưa ra tuyên bố chấm dứt Chiến tranh lạnh, khởi đầu kỷ nguyên mới trong quan hệ hai siêu cường.""",
        should_merge=True,
        reasoning="Cùng nội dung về hòa hoãn Đông-Tây và gặp gỡ Malta, paraphrase. Nên merge.",
        is_boundary_case=True,
    ),
    NodePairSample(
        id=22,
        category=NodeCategory.PARAPHRASE,
        expected_zone=ExpectedZone.AUTO_MERGE,
        node_a_text="""Nhà Trần (1225-1400) nổi tiếng với ba lần kháng chiến chống quân Nguyên-Mông xâm lược. Quân và dân Đại Việt dưới sự lãnh đạo của các vua Trần đã đánh bại đội quân hùng mạnh nhất thế giới lúc bấy giờ, bảo vệ vững chắc độc lập dân tộc.""",
        node_b_text="""Triều Trần (1225-1400) đã ba lần đánh thắng quân xâm lược Nguyên-Mông (1258, 1285, 1287-1288). Dưới sự chỉ huy của các vị vua Trần, nhân dân Đại Việt đã chiến thắng đội quân mạnh nhất châu Á thời đó, giữ vững nền độc lập của đất nước.""",
        should_merge=True,
        reasoning="Cùng nội dung về kháng chiến chống Nguyên-Mông của nhà Trần, paraphrase. Nên merge.",
        is_boundary_case=True,
    ),
    NodePairSample(
        id=23,
        category=NodeCategory.PARAPHRASE,
        expected_zone=ExpectedZone.AUTO_MERGE,
        node_a_text="""Ngô Quyền đã lãnh đạo trận Bạch Đằng năm 938 đánh bại quân Nam Hán, chấm dứt hơn 1000 năm Bắc thuộc. Chiến thắng này mở ra kỷ nguyên độc lập tự chủ lâu dài cho dân tộc Việt Nam, có ý nghĩa lịch sử to lớn.""",
        node_b_text="""Trận Bạch Đằng năm 938 do Ngô Quyền chỉ huy đã đánh tan quân xâm lược Nam Hán, kết thúc thời kỳ Bắc thuộc hơn nghìn năm. Thắng lợi vĩ đại này đánh dấu bước ngoặt quan trọng, mở đầu thời kỳ độc lập tự chủ kéo dài của dân tộc.""",
        should_merge=True,
        reasoning="Cùng nội dung về trận Bạch Đằng 938 và Ngô Quyền, paraphrase. Nên merge.",
        is_boundary_case=True,
    ),
    NodePairSample(
        id=24,
        category=NodeCategory.PARAPHRASE,
        expected_zone=ExpectedZone.AUTO_MERGE,
        node_a_text="""Lê Lợi lãnh đạo khởi nghĩa Lam Sơn (1418-1427), sau 10 năm chiến đấu gian khổ đã đánh bại quân Minh xâm lược. Chiến thắng này giành lại độc lập cho đất nước và mở ra thời kỳ phát triển rực rỡ của triều Lê sơ.""",
        node_b_text="""Khởi nghĩa Lam Sơn do Lê Lợi lãnh đạo kéo dài 10 năm (1418-1427), cuối cùng đã đánh đuổi được quân Minh. Thắng lợi vĩ đại này khôi phục nền độc lập dân tộc, đặt nền móng cho giai đoạn thịnh trị của triều Lê sơ.""",
        should_merge=True,
        reasoning="Cùng nội dung về khởi nghĩa Lam Sơn và Lê Lợi, paraphrase. Nên merge.",
        is_boundary_case=True,
    ),
    NodePairSample(
        id=25,
        category=NodeCategory.PARAPHRASE,
        expected_zone=ExpectedZone.AUTO_MERGE,
        node_a_text="""Hồ Chí Minh là lãnh tụ vĩ đại của cách mạng Việt Nam, người sáng lập Đảng Cộng sản Việt Nam và nước Việt Nam Dân chủ Cộng hòa. Người đã cống hiến trọn đời cho sự nghiệp giải phóng dân tộc và thống nhất đất nước.""",
        node_b_text="""Chủ tịch Hồ Chí Minh - vị lãnh tụ kính yêu của nhân dân Việt Nam, đã thành lập Đảng Cộng sản và khai sinh nước Việt Nam Dân chủ Cộng hòa. Cả cuộc đời Người dành trọn cho sự nghiệp giải phóng và thống nhất Tổ quốc.""",
        should_merge=True,
        reasoning="Cùng nội dung về Hồ Chí Minh và công lao của Người, paraphrase. Nên merge.",
        is_boundary_case=True,
    ),
    NodePairSample(
        id=26,
        category=NodeCategory.PARAPHRASE,
        expected_zone=ExpectedZone.AUTO_MERGE,
        node_a_text="""LHQ có vai trò quan trọng trong việc giải quyết các cuộc xung đột quốc tế, làm trung gian hòa giải các cuộc khủng hoảng ở nhiều khu vực trên thế giới. Tổ chức đã góp phần ngăn chặn không để xảy ra một cuộc chiến tranh thế giới mới.""",
        node_b_text="""Liên hợp quốc đóng vai trò trung gian hòa giải trong các cuộc xung đột và khủng hoảng quốc tế tại nhiều khu vực. Nhờ nỗ lực của tổ chức này, thế giới đã tránh được nguy cơ bùng nổ một cuộc chiến tranh thế giới lần thứ ba.""",
        should_merge=True,
        reasoning="Cùng nội dung về vai trò hòa giải của LHQ, paraphrase. Nên merge.",
        is_boundary_case=True,
    ),
    NodePairSample(
        id=27,
        category=NodeCategory.PARAPHRASE,
        expected_zone=ExpectedZone.AUTO_MERGE,
        node_a_text="""Trần Hưng Đạo là vị danh tướng kiệt xuất của dân tộc Việt Nam. Ông đã lãnh đạo quân và dân Đại Việt đánh bại hai cuộc xâm lược của quân Nguyên-Mông (1285 và 1288), bảo vệ nền độc lập của đất nước.""",
        node_b_text="""Hưng Đạo Vương Trần Quốc Tuấn là danh tướng lỗi lạc trong lịch sử Việt Nam. Dưới sự chỉ huy của ông, quân dân Đại Việt đã hai lần chiến thắng đội quân xâm lược Mông-Nguyên hùng mạnh vào các năm 1285 và 1288, giữ vững độc lập dân tộc.""",
        should_merge=True,
        reasoning="Cùng nội dung về Trần Hưng Đạo và chiến công chống Nguyên-Mông, paraphrase. Nên merge.",
        is_boundary_case=True,
    ),
    NodePairSample(
        id=28,
        category=NodeCategory.PARAPHRASE,
        expected_zone=ExpectedZone.AUTO_MERGE,
        node_a_text="""Cuộc kháng chiến chống Mỹ cứu nước (1954-1975) là cuộc chiến tranh giải phóng dân tộc vĩ đại nhất trong lịch sử Việt Nam. Chiến thắng 30/4/1975 đã thống nhất đất nước, hoàn thành sự nghiệp giải phóng dân tộc.""",
        node_b_text="""Kháng chiến chống Mỹ kéo dài 21 năm (1954-1975) là cuộc chiến tranh giải phóng vĩ đại của dân tộc Việt Nam. Thắng lợi ngày 30 tháng 4 năm 1975 đánh dấu sự thống nhất Tổ quốc và hoàn thành trọn vẹn sự nghiệp giải phóng đất nước.""",
        should_merge=True,
        reasoning="Cùng nội dung về kháng chiến chống Mỹ và thống nhất đất nước, paraphrase. Nên merge.",
        is_boundary_case=True,
    ),
    NodePairSample(
        id=29,
        category=NodeCategory.PARAPHRASE,
        expected_zone=ExpectedZone.AUTO_MERGE,
        node_a_text="""Đầu năm 1945, Chiến tranh thế giới thứ hai sắp kết thúc. Các nước Đồng minh cần giải quyết nhiều vấn đề quan trọng: tiêu diệt hoàn toàn phát xít, tổ chức lại thế giới sau chiến tranh và phân chia quyền lợi giữa các nước thắng trận.""",
        node_b_text="""Đầu năm 1945, CTTG II bước vào giai đoạn cuối. Các cường quốc Đồng minh phải đối mặt với nhiều vấn đề cấp bách: đánh bại hoàn toàn chủ nghĩa phát xít, thiết lập trật tự thế giới mới và phân chia phạm vi ảnh hưởng sau chiến tranh.""",
        should_merge=True,
        reasoning="Cùng nội dung về bối cảnh cuối CTTG II và các vấn đề cần giải quyết, paraphrase. Nên merge.",
        is_boundary_case=True,
    ),
    NodePairSample(
        id=30,
        category=NodeCategory.PARAPHRASE,
        expected_zone=ExpectedZone.AUTO_MERGE,
        node_a_text="""Hai Bà Trưng khởi nghĩa năm 40 chống ách đô hộ của nhà Hán, giành lại độc lập cho đất nước trong ba năm. Cuộc khởi nghĩa thể hiện tinh thần yêu nước, ý chí quật cường của phụ nữ Việt Nam.""",
        node_b_text="""Năm 40, Hai Bà Trưng đã lãnh đạo nhân dân nổi dậy chống ách đô hộ của phong kiến phương Bắc, giành độc lập trong 3 năm. Khởi nghĩa Hai Bà Trưng là biểu tượng của tinh thần bất khuất và lòng yêu nước của người phụ nữ Việt Nam.""",
        should_merge=True,
        reasoning="Cùng nội dung về khởi nghĩa Hai Bà Trưng, paraphrase. Nên merge.",
        is_boundary_case=True,
    ),
    NodePairSample(
        id=31,
        category=NodeCategory.PARAPHRASE,
        expected_zone=ExpectedZone.AUTO_MERGE,
        node_a_text="""Nguyễn Trãi là nhà chính trị, quân sự, văn hóa lỗi lạc thời Lê sơ. Ông là tác giả của Bình Ngô đại cáo - áng thiên cổ hùng văn, được coi là bản Tuyên ngôn độc lập thứ hai của dân tộc Việt Nam.""",
        node_b_text="""Nguyễn Trãi - danh nhân văn hóa thế giới, là nhà chính trị, quân sự, văn học xuất sắc thời Lê sơ. Bình Ngô đại cáo do ông viết được xem như bản Tuyên ngôn độc lập lần thứ hai của nước ta, là áng văn bất hủ trong kho tàng văn học dân tộc.""",
        should_merge=True,
        reasoning="Cùng nội dung về Nguyễn Trãi và Bình Ngô đại cáo, paraphrase. Nên merge.",
        is_boundary_case=True,
    ),
    NodePairSample(
        id=32,
        category=NodeCategory.PARAPHRASE,
        expected_zone=ExpectedZone.AUTO_MERGE,
        node_a_text="""Năm 2015, Liên hợp quốc thông qua Chương trình nghị sự 2030 vì sự phát triển bền vững, đặt ra 17 mục tiêu toàn cầu cần đạt được đến năm 2030. Các mục tiêu này hướng đến xóa đói giảm nghèo, bảo vệ môi trường và phát triển kinh tế-xã hội bền vững.""",
        node_b_text="""LHQ đã thông qua Chương trình nghị sự 2030 với 17 Mục tiêu Phát triển Bền vững (SDGs) vào năm 2015. Các mục tiêu này nhằm xóa bỏ đói nghèo, bảo vệ hành tinh và đảm bảo mọi người được sống trong hòa bình, thịnh vượng vào năm 2030.""",
        should_merge=True,
        reasoning="Cùng nội dung về Chương trình nghị sự 2030 của LHQ, paraphrase. Nên merge.",
        is_boundary_case=True,
    ),
    NodePairSample(
        id=33,
        category=NodeCategory.PARAPHRASE,
        expected_zone=ExpectedZone.AUTO_MERGE,
        node_a_text="""Vua Quang Trung (Nguyễn Huệ) là vị anh hùng dân tộc kiệt xuất. Năm 1789, ông đã lãnh đạo quân Tây Sơn đánh tan 29 vạn quân Thanh xâm lược trong trận Ngọc Hồi - Đống Đa, bảo vệ nền độc lập của Tổ quốc.""",
        node_b_text="""Nguyễn Huệ (vua Quang Trung) là vị anh hùng áo vải xuất sắc của dân tộc. Mùa xuân năm 1789, ông chỉ huy quân Tây Sơn đại phá 29 vạn quân xâm lược nhà Thanh trong chiến thắng Ngọc Hồi-Đống Đa lừng lẫy, giữ vững độc lập dân tộc.""",
        should_merge=True,
        reasoning="Cùng nội dung về Quang Trung và chiến thắng Ngọc Hồi-Đống Đa, paraphrase. Nên merge.",
        is_boundary_case=True,
    ),
    NodePairSample(
        id=34,
        category=NodeCategory.PARAPHRASE,
        expected_zone=ExpectedZone.AUTO_MERGE,
        node_a_text="""Phan Bội Châu là nhà yêu nước tiêu biểu đầu thế kỷ XX. Ông khởi xướng phong trào Đông Du (1905-1908), đưa thanh niên Việt Nam sang Nhật Bản học tập để về cứu nước. Phong trào thể hiện xu hướng cứu nước mới của dân tộc.""",
        node_b_text="""Phan Bội Châu là lãnh tụ của phong trào yêu nước đầu thế kỷ XX. Phong trào Đông Du do ông khởi xướng (1905-1908) đã đưa nhiều thanh niên yêu nước sang Nhật du học với mục tiêu trở về giải phóng dân tộc. Đây là hướng đi mới trong phong trào cứu nước.""",
        should_merge=True,
        reasoning="Cùng nội dung về Phan Bội Châu và phong trào Đông Du, paraphrase. Nên merge.",
        is_boundary_case=True,
    ),
    NodePairSample(
        id=35,
        category=NodeCategory.PARAPHRASE,
        expected_zone=ExpectedZone.AUTO_MERGE,
        node_a_text="""Triều Tiên bị chia cắt thành hai quốc gia sau Hiệp định đình chiến năm 1953. Đây là biểu hiện điển hình của Trật tự thế giới hai cực I-an-ta: Bắc Triều Tiên theo chủ nghĩa xã hội, Nam Triều Tiên theo chủ nghĩa tư bản.""",
        node_b_text="""Sau Hiệp định đình chiến Bàn Môn Điểm 1953, bán đảo Triều Tiên chính thức bị chia đôi. CHDCND Triều Tiên ở phía Bắc theo phe xã hội chủ nghĩa, Đại Hàn Dân quốc ở phía Nam theo phe tư bản - một minh họa rõ nét cho Trật tự hai cực I-an-ta.""",
        should_merge=True,
        reasoning="Cùng nội dung về chia cắt Triều Tiên và Trật tự hai cực, paraphrase. Nên merge.",
        is_boundary_case=True,
    ),
]

# =============================================================================
# CATEGORY 3: SUBSET (should_merge=True, usually) - 10 samples
# Một node chứa đầy đủ thông tin của node kia + thêm chi tiết
# Expected: MEDIUM-HIGH similarity (0.65-0.80), LLM_DECIDE zone (boundary)
# =============================================================================

SUBSET_SAMPLES = [
    NodePairSample(
        id=36,
        category=NodeCategory.SUBSET,
        expected_zone=ExpectedZone.LLM_DECIDE,
        node_a_text="""Hội nghị I-an-ta diễn ra từ ngày 4 đến 11-2-1945 tại Liên Xô, với sự tham dự của Liên Xô, Mỹ và Anh. Hội nghị quyết định tiêu diệt phát xít và thành lập Liên hợp quốc.""",
        node_b_text="""Hội nghị I-an-ta diễn ra từ ngày 4 đến 11-2-1945 tại thành phố I-an-ta (Liên Xô). Ba cường quốc Liên Xô, Mỹ, Anh tham dự với các lãnh đạo: Xta-lin (Liên Xô), Ru-dơ-ven (Mỹ), Sóc-sin (Anh). Hội nghị đưa ra nhiều quyết định: thống nhất tiêu diệt chủ nghĩa phát xít, thành lập Liên hợp quốc, phân chia phạm vi ảnh hưởng ở châu Âu và châu Á.""",
        should_merge=True,
        reasoning="Node B mở rộng chi tiết từ Node A (thêm tên lãnh đạo, chi tiết quyết định). Có thể merge vì cùng nội dung chính.",
        is_boundary_case=True,
    ),
    NodePairSample(
        id=37,
        category=NodeCategory.SUBSET,
        expected_zone=ExpectedZone.LLM_DECIDE,
        node_a_text="""Chiến thắng Điện Biên Phủ 1954 buộc Pháp ký Hiệp định Genève, chấm dứt chiến tranh Đông Dương.""",
        node_b_text="""Chiến thắng Điện Biên Phủ (7/5/1954) là thắng lợi quân sự quyết định của quân và dân Việt Nam trước thực dân Pháp. Với chiến thắng này, Pháp buộc phải ký Hiệp định Genève (21/7/1954), chấm dứt cuộc chiến tranh xâm lược Đông Dương kéo dài 9 năm (1945-1954). Hiệp định công nhận độc lập, chủ quyền của Việt Nam, Lào, Campuchia.""",
        should_merge=True,
        reasoning="Node B mở rộng chi tiết từ Node A (thêm ngày tháng, thời gian chiến tranh, nội dung Hiệp định). Có thể merge.",
        is_boundary_case=True,
    ),
    NodePairSample(
        id=38,
        category=NodeCategory.SUBSET,
        expected_zone=ExpectedZone.LLM_DECIDE,
        node_a_text="""Nhà Trần ba lần đánh thắng quân Nguyên-Mông xâm lược, bảo vệ độc lập dân tộc.""",
        node_b_text="""Nhà Trần (1225-1400) đã ba lần kháng chiến thắng lợi chống quân Nguyên-Mông xâm lược vào các năm 1258, 1285 và 1287-1288. Dưới sự lãnh đạo của các vua Trần và Hưng Đạo Vương Trần Quốc Tuấn, quân dân Đại Việt đã đánh bại đội quân hùng mạnh nhất thế giới lúc bấy giờ, bảo vệ vững chắc nền độc lập dân tộc.""",
        should_merge=True,
        reasoning="Node B mở rộng chi tiết từ Node A (thêm năm tháng, tên lãnh đạo, mô tả quân Nguyên). Có thể merge.",
        is_boundary_case=True,
    ),
    NodePairSample(
        id=39,
        category=NodeCategory.SUBSET,
        expected_zone=ExpectedZone.LLM_DECIDE,
        node_a_text="""Liên hợp quốc được thành lập năm 1945 với mục tiêu duy trì hòa bình và an ninh thế giới.""",
        node_b_text="""Liên hợp quốc chính thức được thành lập ngày 24/10/1945, sau khi Hiến chương LHQ được phê chuẩn. Mục tiêu chính của tổ chức bao gồm: duy trì hòa bình và an ninh quốc tế, phát triển quan hệ hữu nghị giữa các dân tộc, hợp tác quốc tế để giải quyết các vấn đề kinh tế, xã hội, văn hóa, nhân đạo.""",
        should_merge=True,
        reasoning="Node B mở rộng chi tiết từ Node A (thêm ngày thành lập cụ thể, các mục tiêu khác). Có thể merge.",
        is_boundary_case=True,
    ),
    NodePairSample(
        id=40,
        category=NodeCategory.SUBSET,
        expected_zone=ExpectedZone.LLM_DECIDE,
        node_a_text="""Cách mạng tháng Tám 1945 lật đổ chế độ phong kiến và thực dân, lập nên nước Việt Nam Dân chủ Cộng hòa.""",
        node_b_text="""Cách mạng tháng Tám năm 1945 thành công đã đập tan ách thống trị của thực dân Pháp gần 80 năm, lật đổ chế độ phong kiến tồn tại hàng nghìn năm. Ngày 2/9/1945, Chủ tịch Hồ Chí Minh đọc Tuyên ngôn Độc lập tại Quảng trường Ba Đình, khai sinh nước Việt Nam Dân chủ Cộng hòa - nhà nước dân chủ nhân dân đầu tiên ở Đông Nam Á.""",
        should_merge=True,
        reasoning="Node B mở rộng chi tiết từ Node A (thêm ngày 2/9, tên Bác Hồ, địa điểm, ý nghĩa). Có thể merge.",
        is_boundary_case=True,
    ),
    NodePairSample(
        id=41,
        category=NodeCategory.SUBSET,
        expected_zone=ExpectedZone.LLM_DECIDE,
        node_a_text="""Trận Bạch Đằng năm 938 do Ngô Quyền lãnh đạo đã chấm dứt hơn 1000 năm Bắc thuộc.""",
        node_b_text="""Trận Bạch Đằng năm 938 là chiến thắng quyết định trong lịch sử dân tộc. Ngô Quyền đã dùng kế cắm cọc nhọn bọc sắt dưới lòng sông, lợi dụng thủy triều để đánh chìm đoàn thuyền chiến quân Nam Hán. Chiến thắng này chấm dứt hơn 1000 năm Bắc thuộc, mở ra kỷ nguyên độc lập tự chủ lâu dài của dân tộc Việt Nam.""",
        should_merge=True,
        reasoning="Node B mở rộng chi tiết từ Node A (thêm chiến thuật, ý nghĩa mở rộng). Có thể merge.",
        is_boundary_case=True,
    ),
    NodePairSample(
        id=42,
        category=NodeCategory.SUBSET,
        expected_zone=ExpectedZone.LLM_DECIDE,
        node_a_text="""Khởi nghĩa Lam Sơn (1418-1427) do Lê Lợi lãnh đạo đã đánh bại quân Minh xâm lược.""",
        node_b_text="""Khởi nghĩa Lam Sơn bùng nổ năm 1418 tại núi Lam Sơn (Thanh Hóa) dưới sự lãnh đạo của Lê Lợi và sự phò trợ của mưu sĩ Nguyễn Trãi. Sau 10 năm kháng chiến gian khổ với nhiều trận đánh quan trọng (Chi Lăng - Xương Giang 1427), nghĩa quân đã đánh bại hoàn toàn quân Minh, giành lại độc lập cho đất nước.""",
        should_merge=True,
        reasoning="Node B mở rộng chi tiết từ Node A (thêm địa điểm, Nguyễn Trãi, trận đánh cụ thể). Có thể merge.",
        is_boundary_case=True,
    ),
    NodePairSample(
        id=43,
        category=NodeCategory.SUBSET,
        expected_zone=ExpectedZone.LLM_DECIDE,
        node_a_text="""Năm 1960, LHQ thông qua Tuyên ngôn về trao trả độc lập cho các nước thuộc địa.""",
        node_b_text="""Năm 1960, Đại hội đồng Liên hợp quốc thông qua Nghị quyết 1514 - Tuyên ngôn về trao trả độc lập cho các quốc gia và dân tộc thuộc địa. Tuyên ngôn khẳng định quyền tự quyết của các dân tộc thuộc địa và yêu cầu các nước thực dân chấm dứt ách đô hộ. Đây là văn kiện quan trọng đánh dấu bước ngoặt trong cuộc đấu tranh giải phóng dân tộc trên thế giới.""",
        should_merge=True,
        reasoning="Node B mở rộng chi tiết từ Node A (thêm số nghị quyết, nội dung, ý nghĩa). Có thể merge.",
        is_boundary_case=True,
    ),
    NodePairSample(
        id=44,
        category=NodeCategory.SUBSET,
        expected_zone=ExpectedZone.LLM_DECIDE,
        node_a_text="""Quang Trung đánh tan 29 vạn quân Thanh năm 1789 trong chiến thắng Ngọc Hồi-Đống Đa.""",
        node_b_text="""Mùa xuân năm Kỷ Dậu (1789), vua Quang Trung thân chinh cầm quân tiến ra Bắc đánh quân Thanh. Chỉ trong 5 ngày (từ đêm 30 Tết đến mùng 5 Tết), quân Tây Sơn đã đánh tan 29 vạn quân xâm lược, giải phóng kinh thành Thăng Long. Chiến thắng Ngọc Hồi-Đống Đa là một trong những chiến công hiển hách nhất trong lịch sử chống ngoại xâm của dân tộc.""",
        should_merge=True,
        reasoning="Node B mở rộng chi tiết từ Node A (thêm thời gian cụ thể, diễn biến, ý nghĩa). Có thể merge.",
        is_boundary_case=True,
    ),
    NodePairSample(
        id=45,
        category=NodeCategory.SUBSET,
        expected_zone=ExpectedZone.LLM_DECIDE,
        node_a_text="""Trật tự thế giới hai cực I-an-ta hình thành sau CTTG II, chia thế giới thành hai cực Mỹ và Liên Xô.""",
        node_b_text="""Trật tự thế giới hai cực I-an-ta được hình thành từ các thỏa thuận tại Hội nghị I-an-ta (2/1945) và Pốt-xdam (7/1945) giữa ba cường quốc Liên Xô, Mỹ, Anh. Trật tự này chia thế giới thành hai cực đối lập: cực Liên Xô đứng đầu hệ thống XHCN và cực Mỹ đứng đầu hệ thống TBCN. Sự đối đầu này kéo dài suốt thời kỳ Chiến tranh lạnh (1945-1991).""",
        should_merge=True,
        reasoning="Node B mở rộng chi tiết từ Node A (thêm các hội nghị, hệ thống, thời gian). Có thể merge.",
        is_boundary_case=True,
    ),
]

# =============================================================================
# CATEGORY 4: OVERLAPPING (boundary cases) - 15 samples
# Có phần trùng lặp đáng kể nhưng cũng có thông tin khác biệt
# Expected: MEDIUM similarity (0.50-0.65), LLM_DECIDE zone
# should_merge depends on context - let LLM decide
# =============================================================================

OVERLAPPING_SAMPLES = [
    NodePairSample(
        id=46,
        category=NodeCategory.OVERLAPPING,
        expected_zone=ExpectedZone.LLM_DECIDE,
        node_a_text="""Hội nghị I-an-ta (2/1945) quyết định thành lập Liên hợp quốc để duy trì hòa bình thế giới sau chiến tranh. Đây là một trong những quyết định quan trọng nhất của hội nghị, đặt nền móng cho tổ chức quốc tế lớn nhất hiện nay.""",
        node_b_text="""Hội nghị I-an-ta (2/1945) thỏa thuận phân chia phạm vi ảnh hưởng giữa Liên Xô và phương Tây tại châu Âu và châu Á. Liên Xô kiểm soát Đông Âu, phương Tây kiểm soát Tây Âu. Đây là cơ sở hình thành Trật tự thế giới hai cực I-an-ta.""",
        should_merge=False,
        reasoning="Cùng về Hội nghị I-an-ta nhưng tập trung vào khía cạnh KHÁC NHAU (LHQ vs phân chia ảnh hưởng). Không nên merge, tạo relation RELATED.",
        is_boundary_case=True,
    ),
    NodePairSample(
        id=47,
        category=NodeCategory.OVERLAPPING,
        expected_zone=ExpectedZone.LLM_DECIDE,
        node_a_text="""Liên hợp quốc có vai trò quan trọng trong duy trì hòa bình thế giới. Tổ chức đã triển khai nhiều chiến dịch gìn giữ hòa bình, xây dựng hệ thống công ước giải trừ quân bị, góp phần ngăn chặn nguy cơ chiến tranh thế giới mới.""",
        node_b_text="""Liên hợp quốc đóng vai trò trung tâm trong hợp tác quốc tế về kinh tế và phát triển. Tổ chức hỗ trợ các nước đang phát triển về vốn, kỹ thuật, nhân lực thông qua các chương trình UNDP, UNICEF, WHO và nhiều cơ quan chuyên môn khác.""",
        should_merge=False,
        reasoning="Cùng về vai trò của LHQ nhưng tập trung vào lĩnh vực KHÁC NHAU (hòa bình vs kinh tế). Không nên merge, tạo relation RELATED.",
        is_boundary_case=True,
    ),
    NodePairSample(
        id=48,
        category=NodeCategory.OVERLAPPING,
        expected_zone=ExpectedZone.LLM_DECIDE,
        node_a_text="""Nhà Trần nổi tiếng với ba lần kháng chiến chống Nguyên-Mông. Quân dân Đại Việt dưới sự lãnh đạo của Trần Hưng Đạo đã đánh bại đội quân hùng mạnh nhất thế giới, thể hiện tinh thần yêu nước quật cường.""",
        node_b_text="""Nhà Trần (1225-1400) có nhiều đóng góp cho nền văn hóa dân tộc. Thời kỳ này văn học phát triển rực rỡ với Hịch tướng sĩ của Trần Hưng Đạo, thơ của Trần Nhân Tông. Kiến trúc đạt đỉnh cao với các công trình chùa, tháp nổi tiếng.""",
        should_merge=False,
        reasoning="Cùng về nhà Trần nhưng tập trung vào khía cạnh KHÁC NHAU (quân sự vs văn hóa). Không nên merge.",
        is_boundary_case=True,
    ),
    NodePairSample(
        id=49,
        category=NodeCategory.OVERLAPPING,
        expected_zone=ExpectedZone.LLM_DECIDE,
        node_a_text="""Chiến thắng Điện Biên Phủ 1954 có ý nghĩa lịch sử to lớn. Đây là thắng lợi quân sự quyết định buộc Pháp ký Hiệp định Genève, chấm dứt chiến tranh Đông Dương sau 9 năm.""",
        node_b_text="""Chiến thắng Điện Biên Phủ 1954 có tầm vóc thời đại. Thắng lợi này cổ vũ mạnh mẽ phong trào giải phóng dân tộc trên thế giới, đặc biệt là ở châu Á và châu Phi, góp phần làm sụp đổ hệ thống thuộc địa của chủ nghĩa thực dân cũ.""",
        should_merge=False,
        reasoning="Cùng về ý nghĩa ĐBP nhưng tập trung KHÁC (ý nghĩa với VN vs ý nghĩa quốc tế). Có thể merge hoặc tạo relation ELABORATES.",
        is_boundary_case=True,
    ),
    NodePairSample(
        id=50,
        category=NodeCategory.OVERLAPPING,
        expected_zone=ExpectedZone.LLM_DECIDE,
        node_a_text="""Trật tự thế giới hai cực I-an-ta tồn tại từ 1945 đến 1991. Giai đoạn đầu (1945-đầu 70s) đặc trưng bởi sự đối đầu căng thẳng giữa hai khối Đông-Tây, với nhiều cuộc khủng hoảng và xung đột.""",
        node_b_text="""Trật tự thế giới hai cực I-an-ta tồn tại từ 1945 đến 1991. Giai đoạn sau (đầu 70s-1991) chứng kiến xu thế hòa hoãn Đông-Tây với các cuộc đàm phán Xô-Mỹ, cuối cùng dẫn đến kết thúc Chiến tranh lạnh.""",
        should_merge=False,
        reasoning="Cùng về Trật tự hai cực nhưng mô tả HAI GIAI ĐOẠN KHÁC NHAU. Không nên merge, tạo relation TEMPORAL (FOLLOWED_BY).",
        is_boundary_case=True,
    ),
    NodePairSample(
        id=51,
        category=NodeCategory.OVERLAPPING,
        expected_zone=ExpectedZone.LLM_DECIDE,
        node_a_text="""Hồ Chí Minh là lãnh tụ cách mạng Việt Nam, người sáng lập Đảng Cộng sản và nước Việt Nam Dân chủ Cộng hòa. Người là biểu tượng của tinh thần yêu nước và ý chí độc lập dân tộc.""",
        node_b_text="""Hồ Chí Minh là danh nhân văn hóa thế giới được UNESCO vinh danh. Người để lại di sản tư tưởng phong phú về giải phóng dân tộc, về đạo đức cách mạng, về xây dựng con người mới, có giá trị lớn lao cho dân tộc Việt Nam và nhân loại.""",
        should_merge=False,
        reasoning="Cùng về Hồ Chí Minh nhưng góc nhìn KHÁC (lãnh tụ CM vs danh nhân văn hóa). Có thể tạo relation ELABORATES hoặc không merge.",
        is_boundary_case=True,
    ),
    NodePairSample(
        id=52,
        category=NodeCategory.OVERLAPPING,
        expected_zone=ExpectedZone.LLM_DECIDE,
        node_a_text="""Cách mạng tháng Tám 1945 thành công nhờ sự lãnh đạo đúng đắn của Đảng Cộng sản Việt Nam. Đảng đã nắm bắt thời cơ khi Nhật đầu hàng, phát động tổng khởi nghĩa giành chính quyền trong cả nước.""",
        node_b_text="""Cách mạng tháng Tám 1945 thành công có sự đóng góp to lớn của quần chúng nhân dân. Hàng triệu người dân đã hưởng ứng lời kêu gọi của Đảng, nổi dậy giành chính quyền ở các địa phương trong thời gian rất ngắn.""",
        should_merge=False,
        reasoning="Cùng về CMT8 nhưng nhấn mạnh NHÂN TỐ KHÁC (Đảng lãnh đạo vs quần chúng tham gia). Không nên merge.",
        is_boundary_case=True,
    ),
    NodePairSample(
        id=53,
        category=NodeCategory.OVERLAPPING,
        expected_zone=ExpectedZone.LLM_DECIDE,
        node_a_text="""Lê Lợi là vị anh hùng dân tộc đã lãnh đạo khởi nghĩa Lam Sơn đánh đuổi quân Minh. Ông là người có tài năng quân sự xuất sắc, biết quy tụ nhân tài và đoàn kết các lực lượng yêu nước.""",
        node_b_text="""Lê Lợi sau khi đánh thắng quân Minh đã lên ngôi vua, lập ra triều Lê sơ. Ông tiến hành nhiều cải cách quan trọng về chính trị, kinh tế, văn hóa, đặt nền móng cho thời kỳ thịnh trị của triều đại Lê.""",
        should_merge=False,
        reasoning="Cùng về Lê Lợi nhưng thời kỳ KHÁC (khởi nghĩa vs làm vua). Không nên merge, tạo relation TEMPORAL.",
        is_boundary_case=True,
    ),
    NodePairSample(
        id=54,
        category=NodeCategory.OVERLAPPING,
        expected_zone=ExpectedZone.LLM_DECIDE,
        node_a_text="""Nguyễn Trãi là nhà quân sự tài ba, đóng góp lớn cho chiến thắng của khởi nghĩa Lam Sơn. Ông đã soạn thảo nhiều thư từ ngoại giao, góp phần làm tan rã tinh thần chiến đấu của quân Minh.""",
        node_b_text="""Nguyễn Trãi là nhà văn hóa lớn của dân tộc, được UNESCO vinh danh danh nhân văn hóa thế giới. Ông để lại nhiều tác phẩm văn học có giá trị như Bình Ngô đại cáo, Quốc âm thi tập, Dư địa chí.""",
        should_merge=False,
        reasoning="Cùng về Nguyễn Trãi nhưng vai trò KHÁC (quân sự vs văn hóa). Không nên merge.",
        is_boundary_case=True,
    ),
    NodePairSample(
        id=55,
        category=NodeCategory.OVERLAPPING,
        expected_zone=ExpectedZone.LLM_DECIDE,
        node_a_text="""Kháng chiến chống Mỹ (1954-1975) diễn ra với nhiều chiến dịch lớn. Quân và dân miền Nam kiên cường chiến đấu chống lại lực lượng quân sự hùng hậu của Mỹ và chính quyền Sài Gòn.""",
        node_b_text="""Kháng chiến chống Mỹ (1954-1975) được hậu phương miền Bắc chi viện mạnh mẽ. Hàng triệu tấn vũ khí, lương thực được vận chuyển vào Nam qua đường Trường Sơn, cùng hàng vạn cán bộ, chiến sĩ.""",
        should_merge=False,
        reasoning="Cùng về kháng chiến chống Mỹ nhưng góc nhìn KHÁC (chiến đấu vs hậu phương). Không nên merge.",
        is_boundary_case=True,
    ),
    NodePairSample(
        id=56,
        category=NodeCategory.OVERLAPPING,
        expected_zone=ExpectedZone.LLM_DECIDE,
        node_a_text="""Sự sụp đổ của Trật tự hai cực I-an-ta bắt nguồn từ sự khủng hoảng của hệ thống XHCN ở Đông Âu và Liên Xô. Những cải cách của Goóc-ba-chốp đã dẫn đến sự tan rã của Liên Xô năm 1991.""",
        node_b_text="""Sự sụp đổ của Trật tự hai cực I-an-ta xuất phát từ cuộc chạy đua vũ trang làm kiệt quệ kinh tế Liên Xô. Mỹ duy trì ưu thế kinh tế trong khi Liên Xô không theo kịp, dẫn đến sự mất cân bằng quyền lực.""",
        should_merge=False,
        reasoning="Cùng về nguyên nhân sụp đổ Trật tự nhưng nhấn mạnh NGUYÊN NHÂN KHÁC (nội bộ vs chạy đua vũ trang). Không nên merge.",
        is_boundary_case=True,
    ),
    NodePairSample(
        id=57,
        category=NodeCategory.OVERLAPPING,
        expected_zone=ExpectedZone.LLM_DECIDE,
        node_a_text="""Việt Nam là thành viên tích cực của Liên hợp quốc từ năm 1977. Việt Nam đã tham gia nhiều hoạt động gìn giữ hòa bình, được bầu làm Ủy viên không thường trực HĐBA hai nhiệm kỳ (2008-2009 và 2020-2021).""",
        node_b_text="""Việt Nam hợp tác chặt chẽ với các cơ quan của Liên hợp quốc trong phát triển kinh tế-xã hội. UNDP, UNICEF, WHO và nhiều tổ chức LHQ đã triển khai các chương trình hỗ trợ Việt Nam xóa đói giảm nghèo, y tế, giáo dục.""",
        should_merge=False,
        reasoning="Cùng về Việt Nam và LHQ nhưng lĩnh vực KHÁC (chính trị vs phát triển). Không nên merge.",
        is_boundary_case=True,
    ),
    NodePairSample(
        id=58,
        category=NodeCategory.OVERLAPPING,
        expected_zone=ExpectedZone.LLM_DECIDE,
        node_a_text="""Trần Hưng Đạo là danh tướng kiệt xuất đã lãnh đạo quân dân Đại Việt đánh bại quân Nguyên-Mông. Ông nổi tiếng với tài dùng binh, đặc biệt là chiến thuật thủy chiến trên sông Bạch Đằng năm 1288.""",
        node_b_text="""Trần Hưng Đạo còn là nhà tư tưởng quân sự lớn của dân tộc. Hịch tướng sĩ của ông là áng văn bất hủ, thể hiện lòng yêu nước nồng nàn và khí phách anh hùng. Binh thư yếu lược là tác phẩm lý luận quân sự có giá trị.""",
        should_merge=False,
        reasoning="Cùng về Trần Hưng Đạo nhưng vai trò KHÁC (danh tướng vs nhà tư tưởng). Có thể tạo relation ELABORATES.",
        is_boundary_case=True,
    ),
    NodePairSample(
        id=59,
        category=NodeCategory.OVERLAPPING,
        expected_zone=ExpectedZone.LLM_DECIDE,
        node_a_text="""Chiến tranh lạnh (1945-1991) là cuộc đối đầu về tư tưởng giữa hai khối Đông-Tây. Mỹ và Liên Xô không trực tiếp giao chiến nhưng đối đầu thông qua các cuộc chiến tranh ủy nhiệm ở Triều Tiên, Việt Nam.""",
        node_b_text="""Chiến tranh lạnh (1945-1991) thể hiện qua cuộc chạy đua vũ trang hạt nhân gay gắt. Mỹ và Liên Xô tích lũy kho vũ khí hạt nhân khổng lồ, tạo ra thế cân bằng răn đe lẫn nhau (MAD - Mutually Assured Destruction).""",
        should_merge=False,
        reasoning="Cùng về Chiến tranh lạnh nhưng khía cạnh KHÁC (chiến tranh ủy nhiệm vs chạy đua vũ trang). Không nên merge.",
        is_boundary_case=True,
    ),
    NodePairSample(
        id=60,
        category=NodeCategory.OVERLAPPING,
        expected_zone=ExpectedZone.LLM_DECIDE,
        node_a_text="""Phong trào Đông Du (1905-1908) do Phan Bội Châu khởi xướng đưa thanh niên sang Nhật du học. Phong trào thể hiện xu hướng cứu nước mới, học tập kinh nghiệm của nước Nhật Minh Trị duy tân thành công.""",
        node_b_text="""Phong trào Đông Du thất bại năm 1908 khi Nhật Bản bắt tay với Pháp, trục xuất lưu học sinh Việt Nam. Tuy không thành công, phong trào đã đào tạo được một thế hệ thanh niên yêu nước có kiến thức mới.""",
        should_merge=False,
        reasoning="Cùng về Đông Du nhưng giai đoạn KHÁC (khởi đầu vs kết thúc). Không nên merge, có thể tạo relation TEMPORAL.",
        is_boundary_case=True,
    ),
]

# =============================================================================
# CATEGORY 5: RELATED_DISTINCT (should_merge=False) - 15 samples
# Cùng chủ đề lớn nhưng nội dung hoàn toàn khác nhau
# Expected: LOW-MEDIUM similarity (0.35-0.50), KEEP_SEPARATE zone
# =============================================================================

RELATED_DISTINCT_SAMPLES = [
    NodePairSample(
        id=61,
        category=NodeCategory.RELATED_DISTINCT,
        expected_zone=ExpectedZone.KEEP_SEPARATE,
        node_a_text="""Hội nghị I-an-ta quyết định thành lập Liên hợp quốc để duy trì hòa bình thế giới. Hiến chương LHQ xác định mục tiêu, nguyên tắc hoạt động của tổ chức quốc tế này. LHQ chính thức ra đời ngày 24/10/1945.""",
        node_b_text="""Hội nghị Pốt-xdam (7/1945) quyết định phân vùng chiếm đóng nước Đức sau chiến tranh. Đức bị chia thành 4 vùng do Mỹ, Anh, Pháp, Liên Xô kiểm soát. Sau đó nước Đức bị chia cắt thành Đông Đức và Tây Đức.""",
        should_merge=False,
        reasoning="Hai hội nghị KHÁC NHAU (I-an-ta vs Pốt-xdam) với nội dung KHÁC NHAU. Không merge.",
        is_boundary_case=False,
    ),
    NodePairSample(
        id=62,
        category=NodeCategory.RELATED_DISTINCT,
        expected_zone=ExpectedZone.KEEP_SEPARATE,
        node_a_text="""Chiến thắng Điện Biên Phủ 1954 là thắng lợi quyết định buộc Pháp ký Hiệp định Genève. Chiến dịch kéo dài 56 ngày đêm, quân ta tiêu diệt và bắt sống toàn bộ quân Pháp tại tập đoàn cứ điểm.""",
        node_b_text="""Chiến dịch Hồ Chí Minh (4/1975) là chiến dịch cuối cùng trong cuộc kháng chiến chống Mỹ. Quân ta tiến công vào Sài Gòn, chế độ Việt Nam Cộng hòa sụp đổ ngày 30/4/1975, đất nước thống nhất.""",
        should_merge=False,
        reasoning="Hai chiến dịch KHÁC NHAU (ĐBP 1954 vs HCM 1975) dù cùng về chiến tranh VN. Không merge.",
        is_boundary_case=False,
    ),
    NodePairSample(
        id=63,
        category=NodeCategory.RELATED_DISTINCT,
        expected_zone=ExpectedZone.KEEP_SEPARATE,
        node_a_text="""Nhà Lý (1009-1225) là triều đại phong kiến đầu tiên ổn định lâu dài sau thời kỳ Bắc thuộc. Lý Thái Tổ dời đô về Thăng Long năm 1010, xây dựng kinh đô mới cho quốc gia Đại Việt phát triển.""",
        node_b_text="""Nhà Trần (1225-1400) nổi tiếng với ba lần kháng chiến chống Nguyên-Mông. Các vua Trần và Hưng Đạo Vương đã lãnh đạo quân dân đánh bại đội quân xâm lược hùng mạnh nhất thế giới.""",
        should_merge=False,
        reasoning="Hai triều đại KHÁC NHAU (Lý vs Trần) với nội dung KHÁC NHAU. Không merge.",
        is_boundary_case=False,
    ),
    NodePairSample(
        id=64,
        category=NodeCategory.RELATED_DISTINCT,
        expected_zone=ExpectedZone.KEEP_SEPARATE,
        node_a_text="""Ngô Quyền lãnh đạo trận Bạch Đằng năm 938 đánh bại quân Nam Hán, chấm dứt hơn 1000 năm Bắc thuộc. Chiến thắng này mở ra kỷ nguyên độc lập tự chủ lâu dài cho dân tộc Việt Nam.""",
        node_b_text="""Trần Hưng Đạo lãnh đạo trận Bạch Đằng năm 1288 đánh bại quân Nguyên-Mông lần thứ ba. Ông dùng chiến thuật cọc nhọn đóng dưới lòng sông, tiêu diệt toàn bộ đoàn thuyền chiến của địch.""",
        should_merge=False,
        reasoning="Hai trận Bạch Đằng KHÁC NHAU (938 vs 1288) với bối cảnh và nhân vật KHÁC NHAU. Không merge.",
        is_boundary_case=False,
    ),
    NodePairSample(
        id=65,
        category=NodeCategory.RELATED_DISTINCT,
        expected_zone=ExpectedZone.KEEP_SEPARATE,
        node_a_text="""Cách mạng tháng Tám 1945 là cuộc cách mạng dân tộc dân chủ nhân dân. Thắng lợi đã lật đổ ách thống trị của thực dân phong kiến, lập nên nước Việt Nam Dân chủ Cộng hòa.""",
        node_b_text="""Cách mạng tháng Mười Nga 1917 là cuộc cách mạng xã hội chủ nghĩa đầu tiên trên thế giới. Giai cấp công nhân Nga dưới sự lãnh đạo của Đảng Bôn-sê-vích đã giành chính quyền, lập nên Nhà nước Xô viết.""",
        should_merge=False,
        reasoning="Hai cuộc cách mạng KHÁC NHAU (CMT8 VN vs CMT10 Nga) với tính chất và bối cảnh KHÁC. Không merge.",
        is_boundary_case=False,
    ),
    NodePairSample(
        id=66,
        category=NodeCategory.RELATED_DISTINCT,
        expected_zone=ExpectedZone.KEEP_SEPARATE,
        node_a_text="""Lê Lợi lãnh đạo khởi nghĩa Lam Sơn (1418-1427) đánh bại quân Minh xâm lược. Sau chiến thắng, ông lên ngôi vua, lập ra triều Lê sơ - triều đại hưng thịnh nhất trong lịch sử phong kiến Việt Nam.""",
        node_b_text="""Quang Trung (Nguyễn Huệ) lãnh đạo phong trào Tây Sơn đánh bại quân Thanh năm 1789. Ông là vị vua anh hùng đã tiến hành nhiều cải cách tiến bộ, tuy nhiên qua đời sớm năm 1792.""",
        should_merge=False,
        reasoning="Hai nhân vật lịch sử KHÁC NHAU (Lê Lợi vs Quang Trung) với thời kỳ và sự kiện KHÁC. Không merge.",
        is_boundary_case=False,
    ),
    NodePairSample(
        id=67,
        category=NodeCategory.RELATED_DISTINCT,
        expected_zone=ExpectedZone.KEEP_SEPARATE,
        node_a_text="""Phong trào Cần Vương (1885-1896) là phong trào yêu nước chống Pháp theo ngọn cờ phong kiến. Các văn thân, sĩ phu đã khởi nghĩa ở nhiều nơi nhưng cuối cùng bị thất bại do lạc hậu về tư tưởng và vũ khí.""",
        node_b_text="""Phong trào Đông Du (1905-1908) là phong trào yêu nước theo xu hướng dân chủ tư sản. Phan Bội Châu đưa thanh niên sang Nhật du học, học tập kinh nghiệm cải cách Minh Trị để về cứu nước.""",
        should_merge=False,
        reasoning="Hai phong trào yêu nước KHÁC NHAU (Cần Vương vs Đông Du) với tư tưởng và phương pháp KHÁC. Không merge.",
        is_boundary_case=False,
    ),
    NodePairSample(
        id=68,
        category=NodeCategory.RELATED_DISTINCT,
        expected_zone=ExpectedZone.KEEP_SEPARATE,
        node_a_text="""Liên hợp quốc được thành lập năm 1945 với 51 quốc gia thành viên. Hiện nay tổ chức này có 193 thành viên, là tổ chức quốc tế lớn nhất và có vai trò quan trọng nhất trong quan hệ quốc tế.""",
        node_b_text="""ASEAN được thành lập năm 1967 tại Bangkok bởi 5 nước Đông Nam Á. Việt Nam gia nhập ASEAN năm 1995. Hiện nay ASEAN gồm 10 thành viên, là tổ chức khu vực quan trọng ở Đông Nam Á.""",
        should_merge=False,
        reasoning="Hai tổ chức quốc tế KHÁC NHAU (LHQ vs ASEAN) với phạm vi và mục tiêu KHÁC. Không merge.",
        is_boundary_case=False,
    ),
    NodePairSample(
        id=69,
        category=NodeCategory.RELATED_DISTINCT,
        expected_zone=ExpectedZone.KEEP_SEPARATE,
        node_a_text="""Chiến tranh thế giới thứ nhất (1914-1918) bùng nổ ở châu Âu do mâu thuẫn giữa các nước đế quốc. Kết quả là sự sụp đổ của 4 đế quốc lớn: Đức, Áo-Hung, Ottoman và Nga hoàng.""",
        node_b_text="""Chiến tranh thế giới thứ hai (1939-1945) là cuộc chiến tranh lớn nhất lịch sử nhân loại. Nguyên nhân từ mâu thuẫn sau CTTG I và sự trỗi dậy của chủ nghĩa phát xít ở Đức, Ý, Nhật.""",
        should_merge=False,
        reasoning="Hai cuộc chiến tranh thế giới KHÁC NHAU (CTTG I vs CTTG II) với bối cảnh và diễn biến KHÁC. Không merge.",
        is_boundary_case=False,
    ),
    NodePairSample(
        id=70,
        category=NodeCategory.RELATED_DISTINCT,
        expected_zone=ExpectedZone.KEEP_SEPARATE,
        node_a_text="""Hai Bà Trưng khởi nghĩa năm 40 chống ách đô hộ của nhà Hán. Cuộc khởi nghĩa giành lại độc lập trong 3 năm, thể hiện tinh thần yêu nước của phụ nữ Việt Nam.""",
        node_b_text="""Bà Triệu khởi nghĩa năm 248 chống ách đô hộ của nhà Ngô. Với khẩu hiệu 'đánh đuổi quân Ngô, giành lại giang sơn', bà đã lãnh đạo nghĩa quân chiến đấu anh dũng.""",
        should_merge=False,
        reasoning="Hai cuộc khởi nghĩa KHÁC NHAU (Hai Bà Trưng năm 40 vs Bà Triệu năm 248). Không merge.",
        is_boundary_case=False,
    ),
    NodePairSample(
        id=71,
        category=NodeCategory.RELATED_DISTINCT,
        expected_zone=ExpectedZone.KEEP_SEPARATE,
        node_a_text="""Hiệp định Genève (1954) chia cắt Việt Nam tạm thời ở vĩ tuyến 17. Miền Bắc do Việt Nam Dân chủ Cộng hòa quản lý, miền Nam tạm thời do Pháp quản lý trước khi tổng tuyển cử thống nhất.""",
        node_b_text="""Hiệp định Paris (1973) chấm dứt chiến tranh, lập lại hòa bình ở Việt Nam. Mỹ rút quân khỏi miền Nam, hai bên ngừng bắn. Tuy nhiên chiến tranh vẫn tiếp diễn đến năm 1975.""",
        should_merge=False,
        reasoning="Hai hiệp định KHÁC NHAU (Genève 1954 vs Paris 1973) với nội dung và bối cảnh KHÁC. Không merge.",
        is_boundary_case=False,
    ),
    NodePairSample(
        id=72,
        category=NodeCategory.RELATED_DISTINCT,
        expected_zone=ExpectedZone.KEEP_SEPARATE,
        node_a_text="""Triều Nguyễn (1802-1945) là triều đại phong kiến cuối cùng trong lịch sử Việt Nam. Gia Long thống nhất đất nước sau khi đánh bại nhà Tây Sơn, lập kinh đô tại Huế.""",
        node_b_text="""Triều Lê sơ (1428-1527) là thời kỳ thịnh trị của chế độ phong kiến Việt Nam. Lê Thánh Tông được coi là vị vua anh minh nhất, tiến hành nhiều cải cách tiến bộ.""",
        should_merge=False,
        reasoning="Hai triều đại KHÁC NHAU (Nguyễn vs Lê sơ) với thời kỳ và đặc điểm KHÁC. Không merge.",
        is_boundary_case=False,
    ),
    NodePairSample(
        id=73,
        category=NodeCategory.RELATED_DISTINCT,
        expected_zone=ExpectedZone.KEEP_SEPARATE,
        node_a_text="""Phan Bội Châu là đại diện tiêu biểu của xu hướng bạo động trong phong trào yêu nước đầu thế kỷ XX. Ông chủ trương dựa vào Nhật để đánh Pháp, tổ chức các phong trào Đông Du, Việt Nam Quang phục hội.""",
        node_b_text="""Phan Châu Trinh là đại diện tiêu biểu của xu hướng cải cách trong phong trào yêu nước đầu thế kỷ XX. Ông chủ trương dựa vào Pháp để khai hóa dân trí, phát triển kinh tế rồi mới giành độc lập.""",
        should_merge=False,
        reasoning="Hai nhân vật KHÁC NHAU (Phan Bội Châu vs Phan Châu Trinh) với xu hướng tư tưởng ĐỐI LẬP. Không merge.",
        is_boundary_case=False,
    ),
    NodePairSample(
        id=74,
        category=NodeCategory.RELATED_DISTINCT,
        expected_zone=ExpectedZone.KEEP_SEPARATE,
        node_a_text="""Khởi nghĩa Yên Bái (2/1930) do Việt Nam Quốc dân đảng tổ chức. Cuộc khởi nghĩa thất bại nhanh chóng, 13 yếu nhân bị xử tử tại Yên Bái với câu nói bất hủ 'Không thành công cũng thành nhân'.""",
        node_b_text="""Xô viết Nghệ Tĩnh (1930-1931) là cao trào cách mạng do Đảng Cộng sản lãnh đạo. Công nhân, nông dân Nghệ An, Hà Tĩnh đã nổi dậy lập chính quyền Xô viết ở một số địa phương.""",
        should_merge=False,
        reasoning="Hai sự kiện KHÁC NHAU (KN Yên Bái vs Xô viết Nghệ Tĩnh) với lực lượng và tính chất KHÁC. Không merge.",
        is_boundary_case=False,
    ),
    NodePairSample(
        id=75,
        category=NodeCategory.RELATED_DISTINCT,
        expected_zone=ExpectedZone.KEEP_SEPARATE,
        node_a_text="""Trận Chi Lăng - Xương Giang (1427) là trận quyết định trong khởi nghĩa Lam Sơn. Nghĩa quân tiêu diệt 10 vạn viện binh Minh, giết chết tướng Liễu Thăng, buộc quân Minh phải rút về nước.""",
        node_b_text="""Trận Rạch Gầm - Xoài Mút (1785) là chiến thắng lớn của quân Tây Sơn trước quân Xiêm. Nguyễn Huệ chỉ huy đánh tan 5 vạn quân Xiêm xâm lược miền Nam, bảo vệ lãnh thổ đất nước.""",
        should_merge=False,
        reasoning="Hai trận đánh KHÁC NHAU (Chi Lăng vs Rạch Gầm) với thời kỳ và đối tượng KHÁC. Không merge.",
        is_boundary_case=False,
    ),
]

# =============================================================================
# CATEGORY 6: TEMPORAL_DIFFERENT (should_merge=False) - 10 samples
# Cùng chủ đề nhưng khác thời điểm/giai đoạn lịch sử
# Expected: MEDIUM similarity (0.40-0.55), KEEP_SEPARATE zone
# =============================================================================

TEMPORAL_DIFFERENT_SAMPLES = [
    NodePairSample(
        id=76,
        category=NodeCategory.TEMPORAL_DIFFERENT,
        expected_zone=ExpectedZone.KEEP_SEPARATE,
        node_a_text="""Giai đoạn đầu của Trật tự hai cực I-an-ta (1945-đầu 1970s) đặc trưng bởi sự đối đầu căng thẳng Đông-Tây. Chiến tranh lạnh bùng nổ với nhiều cuộc khủng hoảng: phong tỏa Berlin 1948, chiến tranh Triều Tiên 1950-1953, khủng hoảng Cuba 1962.""",
        node_b_text="""Giai đoạn cuối của Trật tự hai cực I-an-ta (đầu 1970s-1991) chứng kiến xu thế hòa hoãn. Các cuộc gặp gỡ Xô-Mỹ, hiệp ước hạn chế vũ khí chiến lược (SALT) được ký kết. Chiến tranh lạnh kết thúc năm 1991.""",
        should_merge=False,
        reasoning="Cùng về Trật tự hai cực nhưng HAI GIAI ĐOẠN KHÁC NHAU với đặc trưng đối lập. Không merge.",
        is_boundary_case=False,
    ),
    NodePairSample(
        id=77,
        category=NodeCategory.TEMPORAL_DIFFERENT,
        expected_zone=ExpectedZone.KEEP_SEPARATE,
        node_a_text="""Kháng chiến chống Pháp (1945-1954) là cuộc kháng chiến 9 năm bảo vệ nền độc lập non trẻ. Thắng lợi Điện Biên Phủ 1954 buộc Pháp ký Hiệp định Genève, miền Bắc hoàn toàn giải phóng.""",
        node_b_text="""Kháng chiến chống Mỹ (1954-1975) là cuộc kháng chiến 21 năm thống nhất đất nước. Chiến thắng 30/4/1975 kết thúc chiến tranh, Việt Nam hoàn toàn độc lập và thống nhất.""",
        should_merge=False,
        reasoning="Hai cuộc kháng chiến KHÁC NHAU về thời gian, đối tượng và mục tiêu. Không merge.",
        is_boundary_case=False,
    ),
    NodePairSample(
        id=78,
        category=NodeCategory.TEMPORAL_DIFFERENT,
        expected_zone=ExpectedZone.KEEP_SEPARATE,
        node_a_text="""Trước 1945, Việt Nam là thuộc địa của thực dân Pháp trong gần 80 năm. Nhân dân ta sống dưới ách đô hộ, bị bóc lột tàn bạo về kinh tế, bị kìm hãm về chính trị và văn hóa.""",
        node_b_text="""Sau 1975, Việt Nam bước vào thời kỳ xây dựng chủ nghĩa xã hội trong cả nước. Đất nước phải khắc phục hậu quả chiến tranh nặng nề, đồng thời đối mặt với bao vây cấm vận của phương Tây.""",
        should_merge=False,
        reasoning="Hai giai đoạn lịch sử HOÀN TOÀN KHÁC NHAU (thuộc Pháp vs sau thống nhất). Không merge.",
        is_boundary_case=False,
    ),
    NodePairSample(
        id=79,
        category=NodeCategory.TEMPORAL_DIFFERENT,
        expected_zone=ExpectedZone.KEEP_SEPARATE,
        node_a_text="""Triều Lý (1009-1225) là triều đại đầu tiên xây dựng nhà nước phong kiến tập quyền. Lý Thái Tổ dời đô về Thăng Long, mở ra thời kỳ phát triển ổn định đầu tiên sau Bắc thuộc.""",
        node_b_text="""Triều Nguyễn (1802-1945) là triều đại phong kiến cuối cùng của Việt Nam. Các vua Nguyễn không chống lại được sự xâm lược của thực dân Pháp, dẫn đến mất nước năm 1858-1884.""",
        should_merge=False,
        reasoning="Hai triều đại KHÁC NHAU về thời gian và đặc điểm lịch sử (thịnh vượng vs suy tàn). Không merge.",
        is_boundary_case=False,
    ),
    NodePairSample(
        id=80,
        category=NodeCategory.TEMPORAL_DIFFERENT,
        expected_zone=ExpectedZone.KEEP_SEPARATE,
        node_a_text="""Hồ Chí Minh thời kỳ 1911-1930 là Nguyễn Ái Quốc - người thanh niên yêu nước ra đi tìm đường cứu nước. Người đã đến Pháp, Liên Xô, Trung Quốc, tiếp thu chủ nghĩa Mác-Lênin, thành lập Đảng Cộng sản.""",
        node_b_text="""Hồ Chí Minh thời kỳ 1945-1969 là Chủ tịch nước Việt Nam Dân chủ Cộng hòa. Người lãnh đạo hai cuộc kháng chiến chống Pháp và chống Mỹ, cống hiến trọn đời cho sự nghiệp giải phóng dân tộc.""",
        should_merge=False,
        reasoning="Cùng về Hồ Chí Minh nhưng HAI GIAI ĐOẠN KHÁC NHAU (tìm đường cứu nước vs lãnh đạo CM). Không merge.",
        is_boundary_case=False,
    ),
    NodePairSample(
        id=81,
        category=NodeCategory.TEMPORAL_DIFFERENT,
        expected_zone=ExpectedZone.KEEP_SEPARATE,
        node_a_text="""Liên hợp quốc những năm đầu (1945-1960s) tập trung vào việc ngăn chặn chiến tranh thế giới mới và giải quyết các xung đột sau CTTG II. Hội đồng Bảo an đóng vai trò trung tâm với 5 cường quốc thường trực.""",
        node_b_text="""Liên hợp quốc từ năm 2000 đến nay tập trung vào các mục tiêu phát triển bền vững và ứng phó biến đổi khí hậu. Chương trình Mục tiêu Thiên niên kỷ (2000) và SDGs (2015) là các sáng kiến quan trọng.""",
        should_merge=False,
        reasoning="Cùng về LHQ nhưng HAI GIAI ĐOẠN KHÁC NHAU với ưu tiên và hoạt động KHÁC. Không merge.",
        is_boundary_case=False,
    ),
    NodePairSample(
        id=82,
        category=NodeCategory.TEMPORAL_DIFFERENT,
        expected_zone=ExpectedZone.KEEP_SEPARATE,
        node_a_text="""Việt Nam thời kỳ Pháp thuộc (1858-1945) là xã hội thuộc địa nửa phong kiến. Thực dân Pháp tiến hành khai thác thuộc địa, bóc lột tài nguyên và nhân lực, kìm hãm sự phát triển của dân tộc.""",
        node_b_text="""Việt Nam thời kỳ Đổi mới (từ 1986) chuyển đổi sang nền kinh tế thị trường định hướng XHCN. Đất nước phát triển nhanh chóng, hội nhập quốc tế sâu rộng, đời sống nhân dân được cải thiện.""",
        should_merge=False,
        reasoning="Hai thời kỳ HOÀN TOÀN KHÁC NHAU (thuộc địa vs đổi mới). Không merge.",
        is_boundary_case=False,
    ),
    NodePairSample(
        id=83,
        category=NodeCategory.TEMPORAL_DIFFERENT,
        expected_zone=ExpectedZone.KEEP_SEPARATE,
        node_a_text="""Phong trào yêu nước đầu thế kỷ XX theo khuynh hướng dân chủ tư sản. Các sĩ phu yêu nước như Phan Bội Châu, Phan Châu Trinh tìm kiếm con đường cứu nước mới, thoát khỏi lối cũ Cần Vương.""",
        node_b_text="""Phong trào yêu nước từ 1930 theo khuynh hướng vô sản. Đảng Cộng sản Việt Nam ra đời, lãnh đạo phong trào đấu tranh của công nhân và nông dân theo con đường cách mạng Mác-Lênin.""",
        should_merge=False,
        reasoning="Cùng về phong trào yêu nước nhưng HAI GIAI ĐOẠN với tư tưởng KHÁC NHAU. Không merge.",
        is_boundary_case=False,
    ),
    NodePairSample(
        id=84,
        category=NodeCategory.TEMPORAL_DIFFERENT,
        expected_zone=ExpectedZone.KEEP_SEPARATE,
        node_a_text="""Quan hệ Việt-Mỹ thời kỳ chiến tranh (1954-1975) là quan hệ đối đầu. Mỹ can thiệp quân sự vào Việt Nam, tiến hành cuộc chiến tranh xâm lược với quy mô lớn nhất kể từ sau CTTG II.""",
        node_b_text="""Quan hệ Việt-Mỹ sau bình thường hóa (1995 đến nay) phát triển tích cực. Hai nước thiết lập quan hệ ngoại giao, mở rộng hợp tác kinh tế, thương mại, giáo dục và an ninh.""",
        should_merge=False,
        reasoning="Cùng về quan hệ Việt-Mỹ nhưng HAI GIAI ĐOẠN với tính chất ĐỐI LẬP. Không merge.",
        is_boundary_case=False,
    ),
    NodePairSample(
        id=85,
        category=NodeCategory.TEMPORAL_DIFFERENT,
        expected_zone=ExpectedZone.KEEP_SEPARATE,
        node_a_text="""Kinh tế Việt Nam trước Đổi mới (1976-1986) vận hành theo cơ chế kế hoạch hóa tập trung. Nền kinh tế khủng hoảng nghiêm trọng với lạm phát cao, thiếu hụt hàng hóa, đời sống nhân dân khó khăn.""",
        node_b_text="""Kinh tế Việt Nam sau Đổi mới (1986 đến nay) chuyển đổi sang cơ chế thị trường. GDP tăng trưởng nhanh, xuất khẩu mạnh mẽ, thu hút đầu tư nước ngoài, Việt Nam trở thành nước có thu nhập trung bình.""",
        should_merge=False,
        reasoning="Cùng về kinh tế VN nhưng HAI GIAI ĐOẠN với đặc điểm và kết quả ĐỐI LẬP. Không merge.",
        is_boundary_case=False,
    ),
]

# =============================================================================
# CATEGORY 7: CAUSE_EFFECT (should_merge=False) - 10 samples
# Một node là nguyên nhân, node kia là kết quả
# Expected: MEDIUM similarity (0.45-0.55), KEEP_SEPARATE zone
# =============================================================================

CAUSE_EFFECT_SAMPLES = [
    NodePairSample(
        id=86,
        category=NodeCategory.CAUSE_EFFECT,
        expected_zone=ExpectedZone.KEEP_SEPARATE,
        node_a_text="""Chính sách khai thác thuộc địa của Pháp đã bóc lột tàn bạo tài nguyên và nhân lực Việt Nam. Ruộng đất bị chiếm đoạt, nông dân phá sản, công nhân bị bóc lột thậm tệ trong các nhà máy, đồn điền.""",
        node_b_text="""Mâu thuẫn sâu sắc giữa nhân dân Việt Nam với thực dân Pháp đã dẫn đến các phong trào đấu tranh liên tục. Từ phong trào Cần Vương đến Đông Du, Duy Tân, và cao trào cách mạng 1930-1931.""",
        should_merge=False,
        reasoning="Quan hệ NHÂN-QUẢ: Chính sách bóc lột (nguyên nhân) → Phong trào đấu tranh (kết quả). Không merge, tạo relation CAUSED_BY.",
        is_boundary_case=False,
    ),
    NodePairSample(
        id=87,
        category=NodeCategory.CAUSE_EFFECT,
        expected_zone=ExpectedZone.KEEP_SEPARATE,
        node_a_text="""Cuộc chạy đua vũ trang hạt nhân khốc liệt giữa Mỹ và Liên Xô làm kiệt quệ kinh tế Liên Xô. Chi phí quân sự chiếm tỷ lệ quá lớn trong ngân sách nhà nước, ảnh hưởng nghiêm trọng đến đời sống nhân dân.""",
        node_b_text="""Sự sụp đổ của Liên Xô năm 1991 chấm dứt Trật tự thế giới hai cực I-an-ta. Chiến tranh lạnh kết thúc, Mỹ trở thành siêu cường duy nhất, thế giới bước vào giai đoạn mới.""",
        should_merge=False,
        reasoning="Quan hệ NHÂN-QUẢ: Chạy đua vũ trang làm kiệt quệ kinh tế (nguyên nhân) → Liên Xô sụp đổ (kết quả). Không merge.",
        is_boundary_case=False,
    ),
    NodePairSample(
        id=88,
        category=NodeCategory.CAUSE_EFFECT,
        expected_zone=ExpectedZone.KEEP_SEPARATE,
        node_a_text="""Sự lãnh đạo đúng đắn của Đảng Cộng sản Việt Nam đã chuẩn bị lực lượng và tổ chức nhân dân sẵn sàng cho tổng khởi nghĩa. Thời cơ được nắm bắt khi Nhật đầu hàng Đồng minh tháng 8/1945.""",
        node_b_text="""Cách mạng tháng Tám 1945 thành công trong cả nước chỉ trong hơn hai tuần. Chính quyền nhân dân được thiết lập ở khắp nơi, nước Việt Nam Dân chủ Cộng hòa ra đời ngày 2/9/1945.""",
        should_merge=False,
        reasoning="Quan hệ NHÂN-QUẢ: Sự chuẩn bị của Đảng (nguyên nhân) → CMT8 thành công (kết quả). Không merge.",
        is_boundary_case=False,
    ),
    NodePairSample(
        id=89,
        category=NodeCategory.CAUSE_EFFECT,
        expected_zone=ExpectedZone.KEEP_SEPARATE,
        node_a_text="""Quân ta dùng chiến thuật đào hào vây lấn, pháo binh bắn phá dữ dội vào cứ điểm của Pháp tại Điện Biên Phủ. Sau 56 ngày đêm chiến đấu ác liệt, các cứ điểm của địch lần lượt bị tiêu diệt.""",
        node_b_text="""Thực dân Pháp buộc phải ký Hiệp định Genève ngày 21/7/1954, công nhận độc lập, chủ quyền của Việt Nam. Cuộc chiến tranh xâm lược Đông Dương kéo dài 9 năm của Pháp chấm dứt.""",
        should_merge=False,
        reasoning="Quan hệ NHÂN-QUẢ: Chiến thắng ĐBP (nguyên nhân) → Hiệp định Genève (kết quả). Không merge.",
        is_boundary_case=False,
    ),
    NodePairSample(
        id=90,
        category=NodeCategory.CAUSE_EFFECT,
        expected_zone=ExpectedZone.KEEP_SEPARATE,
        node_a_text="""Triều đình nhà Nguyễn thực hiện chính sách bế quan tỏa cảng, từ chối giao thương với phương Tây. Quân đội yếu kém, không được hiện đại hóa, vũ khí lạc hậu so với các nước châu Âu.""",
        node_b_text="""Thực dân Pháp nổ súng xâm lược Việt Nam tại Đà Nẵng năm 1858 và dễ dàng chiếm được các vùng đất ven biển. Triều Nguyễn buộc phải ký các hiệp ước bất bình đẳng, dần mất chủ quyền.""",
        should_merge=False,
        reasoning="Quan hệ NHÂN-QUẢ: Chính sách bế quan và quân đội yếu (nguyên nhân) → Mất nước (kết quả). Không merge.",
        is_boundary_case=False,
    ),
    NodePairSample(
        id=91,
        category=NodeCategory.CAUSE_EFFECT,
        expected_zone=ExpectedZone.KEEP_SEPARATE,
        node_a_text="""Mâu thuẫn giữa các nước đế quốc về thị trường và thuộc địa ngày càng gay gắt sau CTTG I. Chủ nghĩa phát xít trỗi dậy ở Đức, Ý, Nhật với chính sách bành trướng xâm lược.""",
        node_b_text="""Chiến tranh thế giới thứ hai bùng nổ năm 1939 khi Đức tấn công Ba Lan. Cuộc chiến lan rộng ra khắp thế giới với quy mô và sự tàn khốc chưa từng có trong lịch sử.""",
        should_merge=False,
        reasoning="Quan hệ NHÂN-QUẢ: Mâu thuẫn đế quốc và phát xít (nguyên nhân) → CTTG II bùng nổ (kết quả). Không merge.",
        is_boundary_case=False,
    ),
    NodePairSample(
        id=92,
        category=NodeCategory.CAUSE_EFFECT,
        expected_zone=ExpectedZone.KEEP_SEPARATE,
        node_a_text="""Hệ thống XHCN ở Đông Âu lâm vào khủng hoảng sâu sắc về kinh tế-chính trị. Nhân dân bất mãn với mô hình phát triển cứng nhắc, đòi hỏi cải cách dân chủ và tự do.""",
        node_b_text="""Làn sóng 'cách mạng nhung' lan khắp Đông Âu năm 1989. Các chế độ XHCN ở Ba Lan, Hungary, Tiệp Khắc, Đông Đức lần lượt sụp đổ, bức tường Berlin bị phá dỡ.""",
        should_merge=False,
        reasoning="Quan hệ NHÂN-QUẢ: Khủng hoảng XHCN Đông Âu (nguyên nhân) → Sụp đổ 1989 (kết quả). Không merge.",
        is_boundary_case=False,
    ),
    NodePairSample(
        id=93,
        category=NodeCategory.CAUSE_EFFECT,
        expected_zone=ExpectedZone.KEEP_SEPARATE,
        node_a_text="""Chiến tranh thế giới thứ hai kết thúc với sự thất bại hoàn toàn của phe phát xít. Liên Xô và Mỹ nổi lên là hai siêu cường với hệ tư tưởng và mô hình phát triển đối lập.""",
        node_b_text="""Trật tự thế giới hai cực I-an-ta hình thành sau CTTG II. Thế giới bị chia thành hai phe: phe XHCN do Liên Xô đứng đầu và phe TBCN do Mỹ đứng đầu, đối đầu nhau trong Chiến tranh lạnh.""",
        should_merge=False,
        reasoning="Quan hệ NHÂN-QUẢ: CTTG II kết thúc (nguyên nhân) → Trật tự hai cực hình thành (kết quả). Không merge.",
        is_boundary_case=False,
    ),
    NodePairSample(
        id=94,
        category=NodeCategory.CAUSE_EFFECT,
        expected_zone=ExpectedZone.KEEP_SEPARATE,
        node_a_text="""Cuộc Tổng tiến công và nổi dậy Xuân 1975 diễn ra mạnh mẽ với các chiến dịch Tây Nguyên, Huế-Đà Nẵng. Quân giải phóng liên tiếp giành thắng lợi, tiến công vào Sài Gòn.""",
        node_b_text="""Chế độ Việt Nam Cộng hòa sụp đổ ngày 30/4/1975. Đất nước thống nhất sau 21 năm chia cắt, mở ra kỷ nguyên mới trong lịch sử dân tộc Việt Nam.""",
        should_merge=False,
        reasoning="Quan hệ NHÂN-QUẢ: Tổng tiến công Xuân 1975 (nguyên nhân) → Giải phóng miền Nam (kết quả). Không merge.",
        is_boundary_case=False,
    ),
    NodePairSample(
        id=95,
        category=NodeCategory.CAUSE_EFFECT,
        expected_zone=ExpectedZone.KEEP_SEPARATE,
        node_a_text="""Nền kinh tế Việt Nam trước Đổi mới rơi vào khủng hoảng nghiêm trọng. Lạm phát phi mã, thiếu hụt hàng hóa, tem phiếu phân phối, đời sống nhân dân vô cùng khó khăn.""",
        node_b_text="""Đại hội VI của Đảng (12/1986) quyết định thực hiện đường lối Đổi mới. Đất nước chuyển đổi sang nền kinh tế thị trường định hướng XHCN, mở cửa hội nhập quốc tế.""",
        should_merge=False,
        reasoning="Quan hệ NHÂN-QUẢ: Khủng hoảng kinh tế (nguyên nhân) → Quyết định Đổi mới (kết quả). Không merge.",
        is_boundary_case=False,
    ),
]

# =============================================================================
# CATEGORY 8: UNRELATED (should_merge=False) - 5 samples
# Nội dung hoàn toàn khác nhau, không liên quan
# Expected: LOW similarity (< 0.35), KEEP_SEPARATE zone
# =============================================================================

UNRELATED_SAMPLES = [
    NodePairSample(
        id=96,
        category=NodeCategory.UNRELATED,
        expected_zone=ExpectedZone.KEEP_SEPARATE,
        node_a_text="""Chiến thắng Điện Biên Phủ 1954 là thắng lợi quân sự quyết định của nhân dân Việt Nam trước thực dân Pháp. Sau 56 ngày đêm chiến đấu, quân ta tiêu diệt toàn bộ tập đoàn cứ điểm của địch.""",
        node_b_text="""Năm 2015, Liên hợp quốc thông qua Chương trình nghị sự 2030 vì sự phát triển bền vững. Chương trình đặt ra 17 mục tiêu toàn cầu về xóa đói giảm nghèo, bảo vệ môi trường và phát triển kinh tế.""",
        should_merge=False,
        reasoning="HAI CHỦ ĐỀ HOÀN TOÀN KHÁC NHAU (ĐBP 1954 vs SDGs 2015). Không có liên quan. Không merge.",
        is_boundary_case=False,
    ),
    NodePairSample(
        id=97,
        category=NodeCategory.UNRELATED,
        expected_zone=ExpectedZone.KEEP_SEPARATE,
        node_a_text="""Nhà Trần (1225-1400) nổi tiếng với ba lần kháng chiến chống Nguyên-Mông xâm lược. Dưới sự lãnh đạo của các vua Trần và Hưng Đạo Vương, quân dân đánh bại đội quân hùng mạnh nhất thế giới.""",
        node_b_text="""Liên hợp quốc ưu tiên thúc đẩy hợp tác quốc tế về kinh tế, tài chính, khoa học-kỹ thuật. Tổ chức có nhiều chương trình hỗ trợ các nước đang phát triển về vốn, tri thức và nhân lực.""",
        should_merge=False,
        reasoning="HAI CHỦ ĐỀ HOÀN TOÀN KHÁC NHAU (nhà Trần vs LHQ kinh tế). Không có liên quan. Không merge.",
        is_boundary_case=False,
    ),
    NodePairSample(
        id=98,
        category=NodeCategory.UNRELATED,
        expected_zone=ExpectedZone.KEEP_SEPARATE,
        node_a_text="""Phan Bội Châu là lãnh tụ phong trào yêu nước đầu thế kỷ XX. Ông khởi xướng phong trào Đông Du, đưa thanh niên sang Nhật du học với mục tiêu về cứu nước.""",
        node_b_text="""Triều Tiên bị chia cắt thành hai quốc gia sau Hiệp định đình chiến năm 1953. CHDCND Triều Tiên theo chế độ XHCN, Đại Hàn Dân quốc theo chế độ tư bản, hai bên đối đầu đến ngày nay.""",
        should_merge=False,
        reasoning="HAI CHỦ ĐỀ HOÀN TOÀN KHÁC NHAU (Phan Bội Châu vs Triều Tiên chia cắt). Không liên quan. Không merge.",
        is_boundary_case=False,
    ),
    NodePairSample(
        id=99,
        category=NodeCategory.UNRELATED,
        expected_zone=ExpectedZone.KEEP_SEPARATE,
        node_a_text="""Ngô Quyền lãnh đạo trận Bạch Đằng năm 938 đánh bại quân Nam Hán. Chiến thắng này chấm dứt hơn 1000 năm Bắc thuộc, mở ra kỷ nguyên độc lập tự chủ lâu dài cho dân tộc Việt Nam.""",
        node_b_text="""Từ đầu những năm 70 thế kỷ XX, xu thế hòa hoãn Đông-Tây xuất hiện với các cuộc gặp gỡ Xô-Mỹ. Các hiệp ước hạn chế vũ khí chiến lược được ký kết, căng thẳng Chiến tranh lạnh giảm bớt.""",
        should_merge=False,
        reasoning="HAI CHỦ ĐỀ HOÀN TOÀN KHÁC NHAU (Ngô Quyền 938 vs hòa hoãn Đông-Tây 1970s). Không liên quan. Không merge.",
        is_boundary_case=False,
    ),
    NodePairSample(
        id=100,
        category=NodeCategory.UNRELATED,
        expected_zone=ExpectedZone.KEEP_SEPARATE,
        node_a_text="""Cách mạng tháng Tám 1945 là cuộc cách mạng dân tộc dân chủ nhân dân. Thắng lợi đã lật đổ ách thống trị của thực dân phong kiến, lập nên nước Việt Nam Dân chủ Cộng hòa.""",
        node_b_text="""Năm 1988, Lực lượng gìn giữ hòa bình của LHQ được trao tặng Giải thưởng Nobel Hòa bình. Năm 2001, tổ chức LHQ và Tổng Thư ký Kofi Annan cũng được vinh danh với giải thưởng này.""",
        should_merge=False,
        reasoning="HAI CHỦ ĐỀ HOÀN TOÀN KHÁC NHAU (CMT8 1945 vs Nobel Hòa bình LHQ). Không liên quan. Không merge.",
        is_boundary_case=False,
    ),
]


# =============================================================================
# COMBINED DATASET
# =============================================================================

ALL_SAMPLES: list[NodePairSample] = (
    DUPLICATE_SAMPLES
    + PARAPHRASE_SAMPLES
    + SUBSET_SAMPLES
    + OVERLAPPING_SAMPLES
    + RELATED_DISTINCT_SAMPLES
    + TEMPORAL_DIFFERENT_SAMPLES
    + CAUSE_EFFECT_SAMPLES
    + UNRELATED_SAMPLES
)


def get_samples_by_category(category: NodeCategory | str) -> list[NodePairSample]:
    """Get samples filtered by category."""
    if isinstance(category, str):
        category = NodeCategory(category)
    return [s for s in ALL_SAMPLES if s.category == category]


def get_samples_by_zone(zone: ExpectedZone | str) -> list[NodePairSample]:
    """Get samples filtered by expected zone."""
    if isinstance(zone, str):
        zone = ExpectedZone(zone)
    return [s for s in ALL_SAMPLES if s.expected_zone == zone]


def get_positive_samples() -> list[NodePairSample]:
    """Get samples where should_merge=True."""
    return [s for s in ALL_SAMPLES if s.should_merge]


def get_negative_samples() -> list[NodePairSample]:
    """Get samples where should_merge=False."""
    return [s for s in ALL_SAMPLES if not s.should_merge]


def get_boundary_cases() -> list[NodePairSample]:
    """Get boundary case samples (near threshold)."""
    return [s for s in ALL_SAMPLES if s.is_boundary_case]


def print_statistics() -> None:
    """Print dataset statistics."""
    print("=" * 70)
    print("NODE SIMILARITY THRESHOLD BENCHMARK DATASET")
    print("=" * 70)
    print(f"Total samples: {len(ALL_SAMPLES)}")
    print(f"  - Positive (should_merge=True): {len(get_positive_samples())}")
    print(f"  - Negative (should_merge=False): {len(get_negative_samples())}")
    print(f"  - Boundary cases: {len(get_boundary_cases())}")
    print()
    print("By category:")
    for cat in NodeCategory:
        samples = get_samples_by_category(cat)
        positive = len([s for s in samples if s.should_merge])
        negative = len([s for s in samples if not s.should_merge])
        print(f"  - {cat.value}: {len(samples)} (merge={positive}, no_merge={negative})")
    print()
    print("By expected zone:")
    for zone in ExpectedZone:
        samples = get_samples_by_zone(zone)
        print(f"  - {zone.value}: {len(samples)}")
    print("=" * 70)


if __name__ == "__main__":
    print_statistics()
