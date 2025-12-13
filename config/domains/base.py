"""Domain schema base classes.

Provides MetadataField and DomainSchema for domain-agnostic metadata handling.
Used by both RouterAgent (query extraction) and KnowledgeTreeEnricher (content extraction).
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class MetadataField(BaseModel):
    """Single metadata field definition - shared between Router & Enricher."""
    
    name: str = Field(..., description="Field name, e.g. 'major_name', 'year'")
    type: str = Field(
        default="string",
        description="Data type: 'string', 'integer', 'float', 'boolean', 'list'",
    )
    description: str = Field(
        default="",
        description="Human-readable description for LLM prompts",
    )
    allowed_values: Optional[List[Any]] = Field(
        default=None,
        description="Enum-like list of valid values. If set, extracted values must match.",
    )
    aliases: Dict[str, Any] = Field(
        default_factory=dict,
        description="Mapping of abbreviations to normalized values, e.g. {'IT': 'Công nghệ thông tin'}",
    )
    extraction_hints: List[str] = Field(
        default_factory=list,
        description="Keywords that hint at this field in content, e.g. ['ngành', 'khoa']",
    )
    required: bool = Field(
        default=False,
        description="Whether this field is required in extraction",
    )
    
    def normalize_value(self, value: Any) -> Any:
        """Normalize value using aliases if available."""
        if value is None:
            return None
        
        # Check aliases first
        if isinstance(value, str):
            normalized = self.aliases.get(value.upper()) or self.aliases.get(value)
            if normalized:
                return normalized
        
        return value
    
    def validate_value(self, value: Any) -> bool:
        """Check if value is in allowed_values (if defined)."""
        if value is None:
            return True
        if not self.allowed_values:
            return True
        return value in self.allowed_values


class DomainSchema(BaseModel):
    """Complete domain configuration for metadata extraction."""
    
    name: str = Field(..., description="Domain identifier, e.g. 'education'")
    display_name: str = Field(..., description="Human-readable name, e.g. 'Tuyển sinh Đại học'")
    metadata_fields: List[MetadataField] = Field(
        default_factory=list,
        description="List of metadata fields for this domain",
    )
    
    def get_field(self, name: str) -> Optional[MetadataField]:
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
                alias_str = ", ".join(f'{k}→{v}' for k, v in list(f.aliases.items())[:3])
                line += f"\n  Aliases: {alias_str}"
            if f.extraction_hints:
                line += f"\n  Hints: {', '.join(f.extraction_hints[:3])}"
            lines.append(line)
        
        return "\n".join(lines)
    
    def normalize_metadata(self, raw: Dict[str, Any]) -> Dict[str, Any]:
        """Normalize extracted metadata using aliases."""
        result = {}
        for key, value in raw.items():
            field = self.get_field(key)
            if field:
                result[key] = field.normalize_value(value)
            else:
                result[key] = value
        return result
    
    def validate_metadata(self, data: Dict[str, Any]) -> Dict[str, List[str]]:
        """Validate metadata against allowed_values.
        
        Returns dict of field_name -> list of validation errors.
        """
        errors: Dict[str, List[str]] = {}
        for field in self.metadata_fields:
            value = data.get(field.name)
            if value is not None and not field.validate_value(value):
                errors[field.name] = [
                    f"Value '{value}' not in allowed values: {field.allowed_values[:5]}"
                ]
        return errors
