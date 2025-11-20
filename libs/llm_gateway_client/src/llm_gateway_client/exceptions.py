class GatewayError(Exception):
    """Base exception for the gateway client."""

    pass


class GatewayAPIError(GatewayError):
    """Raised for API errors from the downstream provider."""

    pass


class ModelNotFoundError(GatewayError):
    """Raised when a requested model cannot be found or is not supported by any configured provider."""

    pass


class ConfigurationError(GatewayError):
    """Raised for configuration-related errors."""

    pass
