"""Relation family definitions and affinity matrix.

Families group related relations for:
1. Affinity scoring in query retrieval
2. Family inference for new relations
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

# Default affinity for unknown types
DEFAULT_AFFINITY = 0.5

# Family metadata
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

# Query Type → Family Affinity Matrix
# Values: 0.0 (no affinity) to 1.0 (perfect match)
FAMILY_AFFINITY_MATRIX: dict[str, dict[str, float]] = {
    "causal": {
        CAUSAL: 1.0,
        TEMPORAL: 0.6,
        COMPARATIVE: 0.2,
        ELABORATIVE: 0.3,
        PARTICIPATION: 0.4,
        STRUCTURAL: 0.1,
    },
    "comparison": {
        CAUSAL: 0.2,
        TEMPORAL: 0.3,
        COMPARATIVE: 1.0,
        ELABORATIVE: 0.4,
        PARTICIPATION: 0.2,
        STRUCTURAL: 0.2,
    },
    "figure": {
        CAUSAL: 0.4,
        TEMPORAL: 0.4,
        COMPARATIVE: 0.3,
        ELABORATIVE: 0.3,
        PARTICIPATION: 1.0,
        STRUCTURAL: 0.1,
    },
    "concept": {
        CAUSAL: 0.3,
        TEMPORAL: 0.2,
        COMPARATIVE: 0.5,
        ELABORATIVE: 1.0,
        PARTICIPATION: 0.2,
        STRUCTURAL: 0.6,
    },
    "factual": {
        CAUSAL: 0.4,
        TEMPORAL: 0.6,
        COMPARATIVE: 0.4,
        ELABORATIVE: 0.5,
        PARTICIPATION: 0.4,
        STRUCTURAL: 0.3,
    },
    "lesson_synthesis": {
        CAUSAL: 0.8,
        TEMPORAL: 0.5,
        COMPARATIVE: 0.5,
        ELABORATIVE: 0.5,
        PARTICIPATION: 0.4,
        STRUCTURAL: 0.2,
    },
    "structural": {
        CAUSAL: 0.1,
        TEMPORAL: 0.3,
        COMPARATIVE: 0.2,
        ELABORATIVE: 0.3,
        PARTICIPATION: 0.1,
        STRUCTURAL: 1.0,
    },
    "other": {
        CAUSAL: 0.0,
        TEMPORAL: 0.0,
        COMPARATIVE: 0.0,
        ELABORATIVE: 0.0,
        PARTICIPATION: 0.0,
        STRUCTURAL: 0.0,
    },
}


def get_affinity(query_type: str, family: str) -> float:
    """Get affinity score between query type and family.

    Args:
        query_type: Query type (causal, comparison, etc.)
        family: Family name (causal, temporal, etc.)

    Returns:
        Affinity score 0.0 - 1.0
    """
    # Null check
    if not query_type or not family:
        return DEFAULT_AFFINITY

    query_type_lower = query_type.lower()
    family_lower = family.lower()

    if query_type_lower not in FAMILY_AFFINITY_MATRIX:
        return DEFAULT_AFFINITY

    return FAMILY_AFFINITY_MATRIX[query_type_lower].get(family_lower, DEFAULT_AFFINITY)


def get_max_affinity(query_type: str, families: list[str]) -> float:
    """Get maximum affinity from a list of families.

    Args:
        query_type: Query type
        families: List of family names

    Returns:
        Maximum affinity score
    """
    if not families or not query_type:
        return DEFAULT_AFFINITY

    return max(get_affinity(query_type, f) for f in families)


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
