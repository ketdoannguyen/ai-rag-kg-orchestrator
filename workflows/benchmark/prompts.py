"""Direction-Constrained Prompt Template for Benchmark v4.0.

LLM generates queries for specified directions via explicit allocation.
Each generated query MUST carry a ``direction`` field matching one of
NT / TD / HZ / BU / CS.
"""

from __future__ import annotations

from typing import Literal

from workflows.benchmark.export import format_subgraph_for_llm
from workflows.benchmark.models import Subgraph

SYSTEM_PROMPT = """\
Bạn là chuyên gia tạo benchmark đánh giá hệ thống retrieval trên Knowledge Graph \
lịch sử Việt Nam. Nhiệm vụ: đọc kỹ subgraph, rồi sinh câu hỏi theo ĐÚNG phân bổ \
direction được chỉ định.

## Định Nghĩa Chiến Lược Retrieval (BẮT BUỘC)

**NT — Neutral/Point Lookup:**
Câu hỏi tra cứu 1 điểm dữ liệu. Câu trả lời nằm gọn trong 1 node, 1 section.
→ fact_nodes: ≤2 nodes, cùng 1 parent_section. relations_used: [] (rỗng).

**TD — Top-Down:**
Câu hỏi "tổng kết 1 chủ đề". Cần đọc nhiều nodes trong cùng 1 section.
→ fact_nodes: ≥3 nodes, cùng 1 parent_section. relations_used: [] (rỗng).

**HZ — Horizontal:**
Câu hỏi "theo dấu mối quan hệ". Cần follow ít nhất 1 relation giữa nodes.
→ fact_nodes: ≥2 nodes. relations_used: CÓ ít nhất 1 relation type (không rỗng).

**BU — Bottom-Up:**
Câu hỏi "tổng hợp từ nhiều góc độ". Cần thu thập từ ≥3 sections khác nhau.
→ fact_nodes: nodes từ ≥3 parent_sections khác nhau. relations_used: [] (rỗng).

**CS — Cross-Section:**
Câu hỏi so sánh đúng 2 chủ đề. Nodes từ đúng 2 sections.
→ fact_nodes: nodes từ đúng 2 parent_sections. relations_used: [] (rỗng).

## Ràng Buộc Quan Trọng
1. Câu hỏi CHỈ được dựa trên nội dung có trong subgraph bên dưới
2. Mỗi câu hỏi phải chỉ ra CHÍNH XÁC node_ids nào chứa câu trả lời
3. Không hỏi câu hỏi Có/Không
4. Tránh câu hỏi quá rộng không thể trả lời cụ thể
5. Node IDs phải CHÍNH XÁC khớp với IDs trong subgraph (ví dụ: "n1", "n2", ...)
6. Mỗi query PHẢI có field "direction" với đúng một giá trị: "NT", "TD", "HZ", "BU", "CS"

## Confidence Score
Mỗi câu hỏi PHẢI có block confidence:
- question_clarity (0-10): Câu hỏi rõ ràng, không mơ hồ?
- fact_node_completeness (0-10): Đã chỉ đủ TẤT CẢ nodes cần thiết?
- fact_node_precision (0-10): Mỗi node được chỉ thực sự LIÊN QUAN?
- grade_certainty (0-10): Tự tin về việc chấm grade 0/1/2?
- overall (0-100): Tổng thể tự tin về câu hỏi này
- confidence_concerns: Nếu overall < 70, giải thích tại sao
- reasoning: Giải thích ngắn gọn logic chọn nodes, grades, VÀ tại sao direction đúng
"""

USER_PROMPT_TEMPLATE = """\
Sinh chính xác {total} câu hỏi theo phân bổ bắt buộc sau:
- NT: {nt} câu (Point Lookup — ≤2 nodes, 1 section, relations_used=[])
- TD: {td} câu (Top-Down — ≥3 nodes, 1 section, relations_used=[])
- HZ: {hz} câu (Horizontal — PHẢI có relations_used không rỗng)
- BU: {bu} câu (Bottom-Up — nodes từ ≥3 sections, relations_used=[])
- CS: {cs} câu (Cross-Section — nodes từ đúng 2 sections, relations_used=[])

{subgraph_text}

## Output Format
Trả về ĐÚNG JSON (không markdown, không ```json```) theo schema:
{{
  "queries": [
    {{
      "query_text": "Câu hỏi...",
      "direction": "NT",
      "fact_nodes": [
        {{"node_id": "n1", "grade": 2, "parent_section": "Tên section"}},
        {{"node_id": "n2", "grade": 1, "parent_section": "Tên section"}}
      ],
      "relations_used": [],
      "confidence": {{
        "question_clarity": 9,
        "fact_node_completeness": 8,
        "fact_node_precision": 9,
        "grade_certainty": 8,
        "overall": 85
      }},
      "confidence_concerns": "",
      "reasoning": "Giải thích ngắn tại sao direction NT đúng: fact_nodes ≤2, cùng 1 section, relations_used=[]"
    }}
  ]
}}

## Hướng dẫn Grade:
- grade 2: Node trực tiếp trả lời câu hỏi (PHẢI có ít nhất 1)
- grade 1: Node cung cấp context/bổ sung hữu ích
- grade 0: Node liên quan nhưng không trực tiếp trả lời

## Ví dụ (Few-shot)

### Ví dụ 1: NT — Point Lookup (1 node, 1 section, không relation)
{{
  "query_text": "Ai đã chỉ huy quân đội trong trận Bạch Đằng năm 1288?",
  "direction": "NT",
  "fact_nodes": [
    {{"node_id": "n1", "grade": 2, "parent_section": "Trận Bạch Đằng"}}
  ],
  "relations_used": [],
  "confidence": {{
    "question_clarity": 10, "fact_node_completeness": 9,
    "fact_node_precision": 10, "grade_certainty": 9, "overall": 92
  }},
  "confidence_concerns": "",
  "reasoning": "NT vì fact_nodes=1 (≤2), cùng section 'Trận Bạch Đằng', relations_used=[]."
}}

### Ví dụ 2: HZ — Horizontal (follow relation, relations_used không rỗng)
{{
  "query_text": "Cuộc kháng chiến chống Nguyên Mông lần 2 dẫn đến những thay đổi gì trong chiến lược phòng thủ?",
  "direction": "HZ",
  "fact_nodes": [
    {{"node_id": "n2", "grade": 2, "parent_section": "Kháng chiến chống Nguyên Mông"}},
    {{"node_id": "n5", "grade": 2, "parent_section": "Chiến lược phòng thủ"}},
    {{"node_id": "n3", "grade": 1, "parent_section": "Kháng chiến chống Nguyên Mông"}}
  ],
  "relations_used": ["LED_TO"],
  "confidence": {{
    "question_clarity": 8, "fact_node_completeness": 7,
    "fact_node_precision": 8, "grade_certainty": 7, "overall": 72
  }},
  "confidence_concerns": "Có thể cần thêm node n6 để trả lời đầy đủ",
  "reasoning": "HZ vì cần follow relation LED_TO từ n2→n5. relations_used=['LED_TO'] (không rỗng)."
}}

### Ví dụ 3: BU — Bottom-Up (tổng hợp ≥3 sections, không relation)
{{
  "query_text": "Hãy tổng hợp các nguyên nhân chính dẫn đến thất bại của nhà Hồ trước quân Minh?",
  "direction": "BU",
  "fact_nodes": [
    {{"node_id": "n1", "grade": 2, "parent_section": "Kinh tế nhà Hồ"}},
    {{"node_id": "n7", "grade": 2, "parent_section": "Quân sự nhà Hồ"}},
    {{"node_id": "n12", "grade": 2, "parent_section": "Chính trị nhà Hồ"}},
    {{"node_id": "n8", "grade": 1, "parent_section": "Quân sự nhà Hồ"}}
  ],
  "relations_used": [],
  "confidence": {{
    "question_clarity": 8, "fact_node_completeness": 7,
    "fact_node_precision": 8, "grade_certainty": 6, "overall": 68
  }},
  "confidence_concerns": "Câu hỏi tổng hợp rộng, có thể thiếu nodes từ section khác.",
  "reasoning": "BU vì fact_nodes từ 3 sections khác nhau (Kinh tế/Quân sự/Chính trị). relations_used=[]."
}}
"""


def build_prompt(
    subgraph: Subgraph,
    target_count: int,
    direction_allocation: dict[str, int] | None = None,
    context_mode: Literal["full", "node_only", "relation", "breadth"] = "full",
) -> tuple[str, str, dict[str, str]]:
    """Build system + user prompt for LLM query generation.

    Args:
        subgraph: The subgraph to generate queries for.
        target_count: Total number of queries to generate.
        direction_allocation: Per-direction quota dict, e.g. ``{"NT": 1, "TD": 2, "HZ": 3}``.
            If None, distributes evenly across all 5 directions.
        context_mode: Controls context format passed to ``format_subgraph_for_llm``.

    Returns:
        ``(system_prompt, user_prompt, alias_map)`` tuple where ``alias_map`` maps
        full node ID → short alias used in the prompt.
    """
    if direction_allocation is None:
        # Fallback: distribute evenly
        per = target_count // 5
        remainder = target_count % 5
        direction_allocation = dict.fromkeys(("NT", "TD", "HZ", "BU", "CS"), per)
        for d in ("NT", "TD", "HZ", "BU", "CS")[:remainder]:
            direction_allocation[d] += 1

    subgraph_text, alias_map = format_subgraph_for_llm(subgraph, context_mode)
    user_prompt = USER_PROMPT_TEMPLATE.format(
        total=target_count,
        nt=direction_allocation.get("NT", 0),
        td=direction_allocation.get("TD", 0),
        hz=direction_allocation.get("HZ", 0),
        bu=direction_allocation.get("BU", 0),
        cs=direction_allocation.get("CS", 0),
        subgraph_text=subgraph_text,
    )
    return SYSTEM_PROMPT, user_prompt, alias_map
