"""Domain schema base classes.

Provides MetadataField and DomainSchema for domain-agnostic metadata handling.
Used by both RouterAgent (query extraction) and KnowledgeTreeEnricher (content extraction).
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class MetadataField(BaseModel):
    """Single metadata field definition - shared between Router & Enricher.

    Smart Filter Support:
    - is_filter_safe: If True, this field can be used for direct filtering
      without losing related content. Example: 'subject' is filter-safe because
      nodes from different subjects are independent.
    - dependencies: For non-filter-safe fields, specifies which values should
      be included when filtering. Example: 'Lý thuyết' needs 'Định nghĩa' nodes.
    """

    name: str = Field(..., description="Field name, e.g. 'major_name', 'year'")
    type: str = Field(
        default="string",
        description="Data type: 'string', 'integer', 'float', 'boolean', 'list'",
    )
    description: str = Field(
        default="",
        description="Human-readable description for LLM prompts",
    )
    allowed_values: list[Any] | None = Field(
        default=None,
        description="Enum-like list of valid values. If set, extracted values must match.",
    )
    aliases: dict[str, Any] = Field(
        default_factory=dict,
        description="Mapping of abbreviations to normalized values, e.g. {'IT': 'Công nghệ thông tin'}",
    )
    extraction_hints: list[str] = Field(
        default_factory=list,
        description="Keywords that hint at this field in content, e.g. ['ngành', 'khoa']",
    )
    required: bool = Field(
        default=False,
        description="Whether this field is required in extraction",
    )
    # Smart Filter fields
    is_filter_safe: bool = Field(
        default=True,
        description="If True, this field can be filtered directly without losing related content. "
        "Example: 'subject' is filter-safe (nodes from different subjects are independent). "
        "Example: 'content_type' is NOT filter-safe ('Lý thuyết' needs 'Định nghĩa' nodes).",
    )
    dependencies: dict[str, list[str]] = Field(
        default_factory=dict,
        description="For non-filter-safe fields: mapping of value -> related values to include. "
        "Example: {'Lý thuyết': ['Định nghĩa', 'Công thức'], 'Bài tập': ['Lời giải']}",
    )

    def normalize_value(self, value: Any) -> Any:
        """Normalize value using aliases if available.

        For list type: normalize each element in the list.
        For single values: normalize directly.
        """
        if value is None:
            return None

        # Handle list type
        if isinstance(value, list):
            normalized_list = []
            for v in value:
                if isinstance(v, str):
                    normalized = self.aliases.get(v.upper()) or self.aliases.get(v) or v
                else:
                    normalized = v
                normalized_list.append(normalized)
            return normalized_list

        # Handle single value (legacy support)
        if isinstance(value, str):
            normalized = self.aliases.get(value.upper()) or self.aliases.get(value)
            if normalized:
                return [normalized]  # Convert to list for consistency
            return [value]  # Single value as list

        return [value]  # Wrap in list for consistency

    def validate_value(self, value: Any) -> bool:
        """Check if value(s) are in allowed_values (if defined).

        For list type: validates each element in the list.
        Returns True only if ALL elements are valid.
        """
        if value is None:
            return True
        if not self.allowed_values:
            return True

        # Handle list type
        if isinstance(value, list):
            return all(v in self.allowed_values for v in value)

        # Handle single value (legacy support)
        return value in self.allowed_values

    def expand_filter_value(self, value: str | list[str]) -> list[str]:
        """Expand filter value to include dependencies.

        For filter-safe fields, returns value as-is (wrapped in list).
        For non-filter-safe fields, expands with dependencies.

        Example:
            field.is_filter_safe = False
            field.dependencies = {"Lý thuyết": ["Định nghĩa", "Công thức"]}
            expand_filter_value("Lý thuyết")
            -> ["Lý thuyết", "Định nghĩa", "Công thức"]
        """
        # Normalize to list
        values = [value] if isinstance(value, str) else list(value)

        if self.is_filter_safe:
            return values

        # Expand with dependencies
        expanded = set(values)
        for v in values:
            deps = self.dependencies.get(v, [])
            expanded.update(deps)

        return list(expanded)


class DomainSchema(BaseModel):
    """Complete domain configuration for metadata extraction."""

    name: str = Field(..., description="Domain identifier, e.g. 'education'")
    display_name: str = Field(..., description="Human-readable name, e.g. 'Tuyển sinh Đại học'")
    metadata_fields: list[MetadataField] = Field(
        default_factory=list,
        description="List of metadata fields for this domain",
    )

    def get_field(self, name: str) -> MetadataField | None:
        """Get a metadata field by name."""
        for f in self.metadata_fields:
            if f.name == name:
                return f
        return None

    def to_extraction_prompt_section(self) -> str:
        """Build prompt section for content metadata extraction."""
        if not self.metadata_fields:
            return "(Không có metadata fields)"

        lines = ["Trích xuất các metadata sau từ nội dung:"]
        for f in self.metadata_fields:
            line = f'- "{f.name}": {f.description}'
            if f.type != "string":
                line += f" (type: {f.type})"
            if f.allowed_values:
                values = ", ".join(f'"{v}"' for v in f.allowed_values[:5])
                if len(f.allowed_values) > 5:
                    values += f" ... (+{len(f.allowed_values) - 5})"
                line += f"\n  CHỈ ĐƯỢC DÙNG: [{values}]"
            if f.aliases:
                alias_str = ", ".join(f"{k}→{v}" for k, v in list(f.aliases.items())[:3])
                line += f"\n  Aliases: {alias_str}"
            if f.extraction_hints:
                line += f"\n  Hints: {', '.join(f.extraction_hints[:3])}"
            lines.append(line)

        return "\n".join(lines)

    def normalize_metadata(self, raw: dict[str, Any]) -> dict[str, Any]:
        """Normalize extracted metadata using aliases."""
        result = {}
        for key, value in raw.items():
            field = self.get_field(key)
            if field:
                result[key] = field.normalize_value(value)
            else:
                result[key] = value
        return result

    def validate_metadata(self, data: dict[str, Any]) -> dict[str, list[str]]:
        """Validate metadata against allowed_values.

        Returns dict of field_name -> list of validation errors.
        """
        errors: dict[str, list[str]] = {}
        for field in self.metadata_fields:
            value = data.get(field.name)
            if value is not None and not field.validate_value(value):
                errors[field.name] = [
                    f"Value '{value}' not in allowed values: {field.allowed_values[:5]}"
                ]
        return errors

    def expand_filters(self, filters: dict[str, Any]) -> dict[str, list[str]]:
        """Expand filter values to include dependencies for non-filter-safe fields.

        This is used by the retrieval system to ensure related content
        is not accidentally filtered out.

        Example:
            filters = {"subject": "Toán học", "content_type": "Lý thuyết"}
            expanded = schema.expand_filters(filters)
            # expanded = {
            #     "subject": ["Toán học"],  # filter-safe, no expansion
            #     "content_type": ["Lý thuyết", "Định nghĩa", "Công thức"]  # expanded
            # }
        """
        result: dict[str, list[str]] = {}
        for field_name, value in filters.items():
            field = self.get_field(field_name)
            if field:
                result[field_name] = field.expand_filter_value(value)
            else:
                # Unknown field, keep as-is
                result[field_name] = [value] if isinstance(value, str) else list(value)
        return result

    def get_filter_safe_fields(self) -> list[MetadataField]:
        """Get all filter-safe fields (can be used for direct filtering)."""
        return [f for f in self.metadata_fields if f.is_filter_safe]

    def get_non_filter_safe_fields(self) -> list[MetadataField]:
        """Get all non-filter-safe fields (need dependency expansion)."""
        return [f for f in self.metadata_fields if not f.is_filter_safe]
