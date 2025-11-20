from __future__ import annotations

from typing import Any

from loguru import logger

from .exceptions import ConfigurationError, ModelNotFoundError
from .models.config import GatewayModelConfig


class ModelRegistry:
    """Prefix-aware registry that resolves logical model names to provider configs."""

    def __init__(
        self,
        model_type: str,
        models: dict[str, GatewayModelConfig | dict[str, Any] | str] | None = None,
    ):
        self.model_type = model_type
        self._models = self._normalize(models or {})
        self._default_key = next(iter(self._models), None)

        if not self._models:
            logger.warning("Model registry for {type} initialized without entries.", type=model_type)

    @property
    def default_key(self) -> str | None:
        return self._default_key

    @property
    def entries(self) -> dict[str, GatewayModelConfig]:
        return self._models

    def get(self, requested_name: str | None) -> GatewayModelConfig:
        if not self._models:
            raise ConfigurationError(f"No {self.model_type} models have been configured.")

        if requested_name in self._models:
            return self._models[requested_name]

        if requested_name:
            for prefix, cfg in self._models.items():
                if requested_name.startswith(prefix):
                    return cfg
            logger.warning(
                "Unable to route {type} model '{requested}'. Falling back to default.",
                type=self.model_type,
                requested=requested_name,
            )

        if not self._default_key:
            raise ModelNotFoundError(
                f"No provider for '{requested_name or 'default'}' and no default {self.model_type} configured."
            )
        return self._models[self._default_key]

    def _normalize(
        self, models: dict[str, GatewayModelConfig | dict[str, Any] | str]
    ) -> dict[str, GatewayModelConfig]:
        normalized: dict[str, GatewayModelConfig] = {}
        for key, value in models.items():
            if isinstance(value, GatewayModelConfig):
                normalized[key] = value
            elif isinstance(value, str):
                normalized[key] = GatewayModelConfig(model=value)
            elif isinstance(value, dict):
                normalized[key] = GatewayModelConfig(**value)
            else:
                raise ConfigurationError(
                    f"Unsupported model configuration for key '{key}': {type(value)}"
                )
        return normalized

