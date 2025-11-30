from typing import Any

from pydantic import BaseModel, Field


class GatewayModelConfig(BaseModel):
    """Provider configuration for a logical model."""

    model: str
    api_key: str | None = None
    api_base: str | None = None
    timeout: float | None = None
    extra: dict[str, Any] = Field(default_factory=dict)

    def to_litellm_params(self) -> dict[str, Any]:
        """Convert the config into kwargs accepted by LiteLLM."""
        params: dict[str, Any] = dict(self.extra)
        if self.api_key:
            params["api_key"] = self.api_key
        if self.api_base:
            params["base_url"] = self.api_base
        if self.timeout is not None:
            params["timeout"] = self.timeout
        return params

