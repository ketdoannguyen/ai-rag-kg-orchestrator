"""Input and output models for refine_json workflow."""

from pathlib import Path
from typing import Optional

from pydantic import BaseModel, Field, validator


class RefineJsonInput(BaseModel):
    """Input model for refine_json workflow."""

    input_path: str = Field(..., description="Path to input JSON file")
    output_path: str = Field(..., description="Path to output JSON file")
    llm_timeout_seconds: float = Field(default=60.0, gt=0.0, le=300.0, description="Timeout for LLM calls")
    max_concurrent_buffers: int = Field(default=5, ge=1, le=10, description="Max concurrent buffer processing")

    @validator("input_path")
    def _validate_input_path(cls, value: str) -> str:
        """Validate input file exists."""
        path = Path(value)
        if not path.exists():
            raise ValueError(f"Input file does not exist: {value}")
        if not path.is_file():
            raise ValueError(f"Input path is not a file: {value}")
        return value

    @validator("output_path")
    def _validate_output_path(cls, value: str) -> str:
        """Ensure output directory exists."""
        path = Path(value)
        path.parent.mkdir(parents=True, exist_ok=True)
        return value


class RefineJsonOutput(BaseModel):
    """Output model for refine_json workflow."""

    success: bool = Field(..., description="Whether refinement succeeded")
    output_path: str = Field(..., description="Path to output JSON file")
    minio_object: Optional[str] = Field(None, description="MinIO object path if uploaded")
    error: Optional[str] = Field(None, description="Error message if failed")
    buffers_processed: int = Field(default=0, description="Number of paragraph buffers processed")
    paragraphs_refined: int = Field(default=0, description="Total paragraphs refined")

