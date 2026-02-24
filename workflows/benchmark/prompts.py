"""Direction-Agnostic Prompt Template for Benchmark v4.0 (Task 13).

Based on §7.2 of the design doc. LLM generates diverse queries following
cognitive diversity instructions — NO direction labels visible.
"""

from __future__ import annotations

from workflows.benchmark.export import format_subgraph_for_llm
from workflows.benchmark.models import Subgraph

SYSTEM_PROMPT = """\
Bạn là chuyên gia tạo benchmark đánh giá hệ thống retrieval trên Knowledge Graph \
lịch sử Việt Nam. Nhiệm vụ: đọc kỹ subgraph, rồi sinh câu hỏi ĐA DẠNG mà \
học sinh/sinh viên có thể hỏi khi học nội dung này.

## Ràng Buộc Quan Trọng
1. Câu hỏi CHỈ được dựa trên nội dung có trong subgraph bên dưới
2. Mỗi câu hỏi phải chỉ ra CHÍNH XÁC node_ids nào chứa câu trả lời
3. Không hỏi câu hỏi Có/Không
4. Tránh câu hỏi quá rộng không thể trả lời cụ thể
5. Node IDs phải CHÍNH XÁC khớp với IDs trong subgraph (ví dụ: "content_nodes/abc123")

## Yêu Cầu Đa Dạng (BẮT BUỘC)

### 1. Mức Độ Tư Duy (phải có đủ cả 4 mức):
- **Nhớ/Hiểu**: Hỏi sự kiện cụ thể (Ai? Khi nào? Ở đâu?)
- **Phân tích quan hệ**: Hỏi mối liên hệ nhân-quả, dẫn đến, ảnh hưởng giữa các khái niệm
- **Tổng hợp**: Hỏi so sánh, đánh giá, rút bài học từ NHIỀU nguồn thông tin
- **Liên kết chéo**: Hỏi kết nối giữa các chủ đề/phần KHÁC NHAU trong subgraph

### 2. Phân Bố Thông Tin Trả Lời (phải có đủ cả 4 dạng):
- Một số câu hỏi mà câu trả lời NẰM GỌN trong 1 đoạn duy nhất
- Một số câu hỏi cần THEO DÕI MỐI QUAN HỆ giữa các đoạn (nhân-quả, trình tự)
- Một số câu hỏi cần THU THẬP thông tin RẢI RÁC từ nhiều đoạn cùng chủ đề
- Một số câu hỏi cần TỔNG HỢP từ nhiều chủ đề/phần KHÁC NHAU

### 3. Dạng Câu Hỏi (trộn đều):
- Câu hỏi sự kiện: Gì/Khi nào/Ở đâu/Ai
- Câu hỏi giải thích: Tại sao/Bằng cách nào
- Câu hỏi so sánh: So sánh/Phân biệt/Đối chiếu
- Câu hỏi phân tích: Đánh giá/Phân tích/Nhận xét

## Lưu ý v4.0 — Multi-Section Subgraph
Subgraph này chứa nodes từ NHIỀU sections khác nhau (xem Breadcrumb). \
Hãy tận dụng điều này để tạo câu hỏi LIÊN KẾT CHÉO giữa các chủ đề.

## Confidence Score
Mỗi câu hỏi PHẢI có block confidence:
- question_clarity (0-10): Câu hỏi rõ ràng, không mơ hồ?
- fact_node_completeness (0-10): Đã chỉ đủ TẤT CẢ nodes cần thiết?
- fact_node_precision (0-10): Mỗi node được chỉ thực sự LIÊN QUAN?
- grade_certainty (0-10): Tự tin về việc chấm grade 0/1/2?
- overall (0-100): Tổng thể tự tin về câu hỏi này
- confidence_concerns: Nếu overall < 70, giải thích tại sao
- reasoning: Giải thích ngắn gọn logic chọn nodes và grades
"""

USER_PROMPT_TEMPLATE = """\
Sinh chính xác {target_count} câu hỏi cho subgraph bên dưới.

{subgraph_text}

## Output Format
Trả về ĐÚNG JSON (không markdown, không ```json```) theo schema:
{{
  "queries": [
    {{
      "query_text": "Câu hỏi...",
      "fact_nodes": [
        {{"node_id": "content_nodes/xxx", "grade": 2, "parent_section": "Tên section"}},
        {{"node_id": "content_nodes/yyy", "grade": 1, "parent_section": "Tên section"}}
      ],
      "relations_used": ["CAUSED_BY"],
      "confidence": {{
        "question_clarity": 9,
        "fact_node_completeness": 8,
        "fact_node_precision": 9,
        "grade_certainty": 8,
        "overall": 85
      }},
      "confidence_concerns": "",
      "reasoning": "Giải thích ngắn..."
    }}
  ]
}}

## Hướng dẫn Grade:
- grade 2: Node trực tiếp trả lời câu hỏi (PHẢI có ít nhất 1)
- grade 1: Node cung cấp context/bổ sung hữu ích
- grade 0: Node liên quan nhưng không trực tiếp trả lời

## Ví dụ (Few-shot)

### Ví dụ 1: Câu hỏi sự kiện đơn giản (1 node)
{{
  "query_text": "Ai đã chỉ huy quân đội trong trận Bạch Đằng năm 1288?",
  "fact_nodes": [
    {{"node_id": "content_nodes/n001", "grade": 2, "parent_section": "Trận Bạch Đằng"}}
  ],
  "relations_used": [],
  "confidence": {{
    "question_clarity": 10,
    "fact_node_completeness": 9,
    "fact_node_precision": 10,
    "grade_certainty": 9,
    "overall": 92
  }},
  "confidence_concerns": "",
  "reasoning": "Câu hỏi rõ ràng, node n001 chứa trực tiếp thông tin về người chỉ huy."
}}

### Ví dụ 2: Câu hỏi quan hệ nhân-quả (nhiều nodes + relation)
{{
  "query_text": "Cuộc kháng chiến chống Nguyên Mông lần 2 dẫn đến những thay đổi gì trong chiến lược phòng thủ?",
  "fact_nodes": [
    {{"node_id": "content_nodes/n002", "grade": 2, "parent_section": "Kháng chiến chống Nguyên Mông"}},
    {{"node_id": "content_nodes/n005", "grade": 2, "parent_section": "Chiến lược phòng thủ"}},
    {{"node_id": "content_nodes/n003", "grade": 1, "parent_section": "Kháng chiến chống Nguyên Mông"}}
  ],
  "relations_used": ["LED_TO", "CAUSED_BY"],
  "confidence": {{
    "question_clarity": 8,
    "fact_node_completeness": 7,
    "fact_node_precision": 8,
    "grade_certainty": 7,
    "overall": 72
  }},
  "confidence_concerns": "Có thể cần thêm node n006 để trả lời đầy đủ",
  "reasoning": "Cần follow relation LED_TO từ n002→n005, n003 cung cấp context bổ sung."
}}

### Ví dụ 3: Câu hỏi tổng hợp liên kết chéo (nhiều sections)
{{
  "query_text": "So sánh chiến thuật của Trần Hưng Đạo và Lê Lợi trong việc đối phó với quân xâm lược?",
  "fact_nodes": [
    {{"node_id": "content_nodes/n001", "grade": 2, "parent_section": "Nhà Trần chống Nguyên Mông"}},
    {{"node_id": "content_nodes/n007", "grade": 2, "parent_section": "Khởi nghĩa Lam Sơn"}},
    {{"node_id": "content_nodes/n002", "grade": 1, "parent_section": "Nhà Trần chống Nguyên Mông"}},
    {{"node_id": "content_nodes/n008", "grade": 1, "parent_section": "Khởi nghĩa Lam Sơn"}}
  ],
  "relations_used": [],
  "confidence": {{
    "question_clarity": 8,
    "fact_node_completeness": 7,
    "fact_node_precision": 8,
    "grade_certainty": 6,
    "overall": 65
  }},
  "confidence_concerns": "Câu hỏi so sánh rộng, grades phụ thuộc vào góc nhìn. Có thể thiếu nodes.",
  "reasoning": "Tổng hợp từ 2 sections khác nhau, cả 4 nodes đều cần thiết cho so sánh."
}}
"""


def build_prompt(
    subgraph: Subgraph,
    target_count: int,
) -> tuple[str, str]:
    """Build system + user prompt for LLM query generation.

    Returns:
        (system_prompt, user_prompt) tuple.
    """
    subgraph_text = format_subgraph_for_llm(subgraph)
    user_prompt = USER_PROMPT_TEMPLATE.format(
        target_count=target_count,
        subgraph_text=subgraph_text,
    )
    return SYSTEM_PROMPT, user_prompt
