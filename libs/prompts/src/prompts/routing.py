"""Routing prompts for Router Agent.

Domain-agnostic prompt templates for query understanding and routing.
"""

from __future__ import annotations

from typing import Any


def build_router_system_prompt(
    domain_name: str,
    filter_fields: dict[str, dict[str, Any]],
    current_year: int,
    custom_intent_descriptions: dict[str, str] | None = None,
) -> str:
    """Build the system prompt for Router Agent.

    Args:
        domain_name: Name of the domain (e.g., "Đại học XYZ")
        filter_fields: Dict of filter field configurations
        current_year: Current year for temporal normalization
        custom_intent_descriptions: Optional custom intent definitions
    """
    intent_definitions = {
        "DISCOVERY": "Hỏi tổng quan, liệt kê, khám phá thông tin chung",
        "SPECIFIC_FACT": "Hỏi thông tin cụ thể, con số, định nghĩa",
        "INSTRUCTION": "Hỏi quy trình, thủ tục, hướng dẫn",
        "COMPARISON": "So sánh 2 hoặc nhiều thực thể",
        "CONTACT_INFO": "Hỏi thông tin liên hệ: địa chỉ, SĐT, Email",
        "CHIT_CHAT": "Chào hỏi hoặc câu hỏi không liên quan domain",
    }

    if custom_intent_descriptions:
        intent_definitions.update(custom_intent_descriptions)

    intent_section = "\n".join(f'   - "{k}": {v}' for k, v in intent_definitions.items())
    filter_section = _build_filter_description(filter_fields)

    return f"""Bạn là Router cho hệ thống Chatbot.

NHIỆM VỤ: Chuyển câu hỏi thành Structured Search Object.

NGUYÊN TẮC:
1. Đọc lịch sử hội thoại để giải quyết tham chiếu
2. Trả về JSON thuần túy, không markdown
3. Keywords: loại bỏ stopwords
4. Mỗi sub_question phải đủ thông tin để search độc lập

SUB_QUESTION LOGIC:
- Mỗi sub_question là truy vấn độc lập với keywords và filters riêng
- Câu đơn giản: 1 sub_question
- COMPARISON: nhiều sub_questions
- CHIT_CHAT: []

OUTPUT (JSON):
{{
  "intent": <string>,
  "rewritten_query": <string>,
  "sub_questions": [
    {{
      "query": <string>,
      "keywords": [<string>],
      "filters": <object|null>
    }}
  ]
}}

VÍ DỤ:
User: "Học phí IT năm 2024 và 2025 khác nhau?"
{{
  "intent": "COMPARISON",
  "rewritten_query": "So sánh học phí ngành CNTT năm 2024 và 2025",
  "sub_questions": [
    {{
      "query": "Học phí CNTT 2024?", "keywords": ["học phí", "CNTT", "2024"], "filters": {{"year": 2024, "major_name": "Công nghệ thông tin"}}}},
    {{
      "query": "Học phí CNTT 2025?", "keywords": ["học phí", "CNTT", "2025"], "filters": {{"year": 2025, "major_name": "Công nghệ thông tin"}}}}
  ]
}}

--------------------------------------------------
INTENT DEFINITIONS:
{intent_section}

DOMAIN CONFIGURATION:
System Name: "{domain_name}"
Current Year: {current_year}
(Normalization Rule: "năm ngoái" -> {current_year - 1})

AVAILABLE FILTERS:
{filter_section}
"""


def _build_filter_description(filter_fields: dict[str, dict[str, Any]]) -> str:
    """Build description of available filter fields."""
    if not filter_fields:
        return "(Không có filter)"

    lines = []
    for name, cfg in filter_fields.items():
        desc = cfg.get("description", "")
        ftype = cfg.get("type", "string")
        allowed = cfg.get("allowed_values")
        examples = cfg.get("examples", [])
        aliases = cfg.get("aliases", {})

        line = f'- "{name}": {desc} ({ftype})'

        # Show allowed values (enum constraint)
        if allowed:
            values_str = ", ".join(f'"{v}"' for v in allowed[:5])
            if len(allowed) > 5:
                values_str += f" ... (+{len(allowed) - 5})"
            line += f"\n  CHỈ ĐƯỢC DÙNG: [{values_str}]"
        elif examples:
            line += f" VD: {', '.join(str(e) for e in examples[:2])}"

        # Show aliases
        if aliases:
            alias_str = ", ".join(f"{k}→{v}" for k, v in list(aliases.items())[:3])
            line += f"\n  Aliases: {alias_str}"

        lines.append(line)

    return "\n".join(lines)


def build_router_user_prompt(
    user_query: str,
    conversation_history: list[dict[str, str]] | None = None,
) -> str:
    """Build the user prompt with query and history.

    Args:
        user_query: The user's current query
        conversation_history: Optional list of previous messages
    """
    parts = []

    if conversation_history:
        parts.append("LỊCH SỬ:")
        for msg in conversation_history[-5:]:
            role = "User" if msg.get("role") == "user" else "Bot"
            parts.append(f"{role}: {msg.get('content', '')}")
        parts.append("")

    parts.append(f"CÂU HỎI: {user_query}")
    return "\n".join(parts)
