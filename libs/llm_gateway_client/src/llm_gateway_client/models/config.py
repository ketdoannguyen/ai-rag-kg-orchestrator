from typing import Any

from pydantic import BaseModel, Field


class ProviderConfig(BaseModel):
    """OpenRouter provider routing configuration.

    Controls which inference providers OpenRouter uses when routing requests.
    See: https://openrouter.ai/docs/provider-routing
    """

    order: list[str]
    """Preferred providers in order of priority (e.g., ["together", "fireworks"])."""

    allow_fallbacks: bool = False
    """Whether to allow fallback to other providers if preferred ones fail."""


class GatewayModelConfig(BaseModel):
    """Provider configuration for a logical model."""

    model: str
    api_key: str | None = None
    api_base: str | None = None
    timeout: float | None = None
    extra: dict[str, Any] = Field(default_factory=dict)
    provider: ProviderConfig | None = None
    """OpenRouter provider routing configuration."""

    def to_litellm_params(self) -> dict[str, Any]:
        """Convert the config into kwargs accepted by LiteLLM."""
        params: dict[str, Any] = dict(self.extra)
        if self.api_key:
            params["api_key"] = self.api_key
        if self.api_base:
            params["api_base"] = self.api_base
        if self.timeout is not None:
            params["timeout"] = self.timeout
        if self.provider:
            extra_body = params.setdefault("extra_body", {})
            extra_body["provider"] = self.provider.model_dump()
        return params
