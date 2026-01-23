"""Education domain schema for university learning materials.

Updated from "Tuyển sinh Đại học" to "Tài liệu Học tập Đại học".
Includes smart filter support with is_filter_safe and dependencies.
"""

from __future__ import annotations

from .base import DomainSchema, MetadataField

EDUCATION_SCHEMA = DomainSchema(
    name="education",
    display_name="Tài liệu Học tập Đại học",
    metadata_fields=[
        # =====================================================================
        # FILTER-SAFE FIELDS (can be used for direct filtering)
        # Nodes from different values are independent
        # =====================================================================
        MetadataField(
            name="subject",
            type="list",
            description="Môn học/Lĩnh vực (FILTER-SAFE: nodes from different subjects are independent)",
            allowed_values=[
                "Toán học",
                "Vật lý",
                "Hóa học",
                "Sinh học",
                "Tin học",
                "Ngữ văn",
                "Lịch sử",
                "Địa lý",
                "Tiếng Anh",
                "Triết học",
                "Kinh tế",
                "Luật",
                "Y học",
                "Kỹ thuật",
                "Kiến trúc",
                "Nông nghiệp",
            ],
            aliases={
                "Math": "Toán học",
                "Toán": "Toán học",
                "Physics": "Vật lý",
                "Lý": "Vật lý",
                "Chemistry": "Hóa học",
                "Hóa": "Hóa học",
                "Biology": "Sinh học",
                "Sinh": "Sinh học",
                "IT": "Tin học",
                "CS": "Tin học",
                "CNTT": "Tin học",
                "Literature": "Ngữ văn",
                "Văn": "Ngữ văn",
                "History": "Lịch sử",
                "Sử": "Lịch sử",
                "Geography": "Địa lý",
                "Địa": "Địa lý",
                "English": "Tiếng Anh",
            },
            extraction_hints=["môn", "học phần", "subject", "ngành", "lĩnh vực"],
            is_filter_safe=True,
        ),
        MetadataField(
            name="academic_level",
            type="list",
            description="Cấp học (FILTER-SAFE: levels are independent)",
            allowed_values=[
                "Đại cương",
                "Cơ sở ngành",
                "Chuyên ngành",
                "Sau đại học",
            ],
            extraction_hints=["năm", "khóa", "cấp độ", "trình độ"],
            is_filter_safe=True,
        ),
        MetadataField(
            name="source_type",
            type="list",
            description="Loại nguồn tài liệu (FILTER-SAFE: sources are independent)",
            allowed_values=[
                "Giáo trình",
                "Slide bài giảng",
                "Đề thi",
                "Bài tập",
                "Tài liệu tham khảo",
                "Luận văn",
                "Báo cáo",
            ],
            extraction_hints=["nguồn", "tài liệu", "source", "loại tài liệu"],
            is_filter_safe=True,
        ),
        # =====================================================================
        # NON-FILTER-SAFE FIELDS (need dependency expansion when filtering)
        # Values have semantic relationships
        # =====================================================================
        MetadataField(
            name="content_type",
            type="list",
            description="Loại nội dung (NOT FILTER-SAFE: có phụ thuộc ngữ nghĩa)",
            allowed_values=[
                "Lý thuyết",
                "Định nghĩa",
                "Định lý",
                "Công thức",
                "Chứng minh",
                "Ví dụ",
                "Bài tập",
                "Lời giải",
                "Tóm tắt",
                "Ghi chú",
            ],
            extraction_hints=["loại", "type", "nội dung"],
            is_filter_safe=False,
            # Dependencies: when filtering for key, also include its dependencies
            # Example: filtering "Lý thuyết" should also include "Định nghĩa", "Định lý", "Công thức"
            dependencies={
                "Lý thuyết": ["Định nghĩa", "Định lý", "Công thức"],
                "Bài tập": ["Lời giải", "Công thức", "Ví dụ"],
                "Ví dụ": ["Định nghĩa", "Công thức"],
                "Chứng minh": ["Định lý", "Định nghĩa"],
                "Lời giải": ["Công thức", "Định lý"],
            },
        ),
        MetadataField(
            name="chapter",
            type="string",
            description="Chương/Phần của tài liệu (NOT FILTER-SAFE: use with HAS_SECTION edges)",
            extraction_hints=["chương", "phần", "chapter", "section"],
            is_filter_safe=False,
            # Chapters have hierarchical relationships, use graph traversal
            dependencies={},
        ),
        # =====================================================================
        # ADDITIONAL FIELDS (for enrichment, not primary filtering)
        # =====================================================================
        MetadataField(
            name="difficulty",
            type="list",
            description="Độ khó của nội dung",
            allowed_values=[
                "Cơ bản",
                "Trung bình",
                "Nâng cao",
            ],
            extraction_hints=["độ khó", "level", "cấp độ"],
            is_filter_safe=True,
        ),
        MetadataField(
            name="keywords",
            type="list",
            description="Từ khóa chính của nội dung",
            extraction_hints=["từ khóa", "keyword", "thuật ngữ"],
            is_filter_safe=True,  # Keywords are independent
        ),
    ],
)
