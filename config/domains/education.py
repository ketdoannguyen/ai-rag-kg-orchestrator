"""Education domain schema for university admission system."""

from .base import DomainSchema, MetadataField


EDUCATION_SCHEMA = DomainSchema(
    name="education",
    display_name="Tuyển sinh Đại học",
    metadata_fields=[
        MetadataField(
            name="year",
            type="integer",
            description="Năm tuyển sinh/học",
            allowed_values=[2023, 2024, 2025, 2026],
            extraction_hints=["năm", "niên khóa", "khóa"],
        ),
        MetadataField(
            name="major_name",
            type="string",
            description="Tên ngành đã chuẩn hóa",
            allowed_values=[
                "Công nghệ thông tin",
                "Khoa học máy tính",
                "Kỹ thuật phần mềm",
                "Trí tuệ nhân tạo",
                "Khoa học dữ liệu",
                "An toàn thông tin",
                "Quản trị kinh doanh",
                "Kế toán",
                "Marketing",
                "Tài chính ngân hàng",
                "Du lịch",
                "Ngôn ngữ Anh",
                "Thiết kế đồ họa",
            ],
            aliases={
                "IT": "Công nghệ thông tin",
                "CNTT": "Công nghệ thông tin",
                "CS": "Khoa học máy tính",
                "KHMT": "Khoa học máy tính",
                "SE": "Kỹ thuật phần mềm",
                "KTPM": "Kỹ thuật phần mềm",
                "AI": "Trí tuệ nhân tạo",
                "TTNT": "Trí tuệ nhân tạo",
                "DS": "Khoa học dữ liệu",
                "KHDL": "Khoa học dữ liệu",
                "ATTT": "An toàn thông tin",
                "BA": "Quản trị kinh doanh",
                "QTKD": "Quản trị kinh doanh",
                "MKT": "Marketing",
            },
            extraction_hints=["ngành", "khoa", "chuyên ngành", "lĩnh vực"],
        ),
        MetadataField(
            name="admission_method",
            type="string",
            description="Phương thức xét tuyển",
            allowed_values=[
                "Học bạ",
                "THPT",
                "Đánh giá năng lực",
                "Xét tuyển thẳng",
                "Liên thông",
            ],
            aliases={
                "Điểm thi THPT": "THPT",
                "Thi THPT": "THPT",
                "ĐGNL": "Đánh giá năng lực",
                "Xét học bạ": "Học bạ",
            },
            extraction_hints=["phương thức", "xét tuyển", "tuyển sinh"],
        ),
        MetadataField(
            name="training_system",
            type="string", 
            description="Hệ đào tạo",
            allowed_values=[
                "Chất lượng cao",
                "Đại trà",
                "Liên kết quốc tế",
                "Từ xa",
            ],
            aliases={
                "CLC": "Chất lượng cao",
                "Tiêu chuẩn": "Đại trà",
                "Quốc tế": "Liên kết quốc tế",
            },
            extraction_hints=["hệ", "chương trình", "đào tạo"],
        ),
    ],
)
