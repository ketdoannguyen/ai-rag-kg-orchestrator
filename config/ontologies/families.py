"""Relation family definitions.

Families group related relations for:
1. Family inference for new relations
2. Display/categorization purposes

Note: FAMILY_AFFINITY_MATRIX has been removed because:
- The affinity values were arbitrary without scientific justification
- Scoring is now handled by HybridRelationScorer using BM25 + Embedding + RRF
- See: Cormack et al. "Reciprocal rank fusion" (SIGIR 2009)
"""

from __future__ import annotations

# Family names as constants
CAUSAL = "causal"
TEMPORAL = "temporal"
COMPARATIVE = "comparative"
ELABORATIVE = "elaborative"
PARTICIPATION = "participation"
STRUCTURAL = "structural"

ALL_FAMILIES = [CAUSAL, TEMPORAL, COMPARATIVE, ELABORATIVE, PARTICIPATION, STRUCTURAL]


# Family metadata (kept - useful for display and keyword inference)
FAMILY_METADATA: dict[str, dict[str, str | list[str]]] = {
    CAUSAL: {
        "name": "Nhân quả",
        "description": "Quan hệ nguyên nhân - kết quả giữa các đoạn văn",
        "keywords": [
            "tại sao",
            "vì sao",
            "nguyên nhân",
            "hậu quả",
            "kết quả",
            "dẫn đến",
            "gây ra",
        ],
    },
    TEMPORAL: {
        "name": "Thời gian",
        "description": "Quan hệ về thời gian, trình tự",
        "keywords": ["trước", "sau", "trong thời kỳ", "cùng thời", "tiếp theo", "khi nào"],
    },
    COMPARATIVE: {
        "name": "So sánh",
        "description": "So sánh, tương đồng, đối lập",
        "keywords": ["so sánh", "giống", "khác", "tương tự", "đối lập", "khác biệt"],
    },
    ELABORATIVE: {
        "name": "Giải thích",
        "description": "Giải thích, mở rộng, tóm tắt nội dung",
        "keywords": ["là gì", "như thế nào", "ví dụ", "định nghĩa", "giải thích"],
    },
    PARTICIPATION: {
        "name": "Tham gia",
        "description": "Nhân vật/yếu tố tham gia sự kiện",
        "keywords": ["lãnh đạo", "tham gia", "đóng góp", "ai", "người nào", "chống lại"],
    },
    STRUCTURAL: {
        "name": "Cấu trúc",
        "description": "Cấu trúc, phân loại, phân cấp",
        "keywords": ["chương", "mục", "phần", "loại", "thuộc về", "bao gồm"],
    },
}


def get_family_metadata(family: str) -> dict[str, str | list[str]] | None:
    """Get metadata for a family.

    Args:
        family: Family name

    Returns:
        Family metadata dict or None if not found
    """
    if not family:
        return None
    return FAMILY_METADATA.get(family.lower())


def get_families_for_keywords(text: str) -> list[str]:
    """Get families that match keywords in text.

    Useful for inferring relation families based on text content.

    Args:
        text: Text to search for keywords

    Returns:
        List of matching family names
    """
    if not text:
        return []

    text_lower = text.lower()
    matched_families = []

    for family, metadata in FAMILY_METADATA.items():
        keywords = metadata.get("keywords", [])
        if isinstance(keywords, list):
            for keyword in keywords:
                if keyword in text_lower:
                    matched_families.append(family)
                    break

    return matched_families
