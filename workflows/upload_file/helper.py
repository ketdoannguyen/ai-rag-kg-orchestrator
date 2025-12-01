from __future__ import annotations

from typing import Any

from loguru import logger
from tenacity import RetryCallState


def log_upload_retry(retry_state: RetryCallState) -> None:
    """Log tenacity retry attempts for MinIO uploads."""
    exc: Exception | None = retry_state.outcome.exception() if retry_state.outcome else None
    sleep: Any = getattr(retry_state.next_action, "sleep", None)
    logger.warning(
        "Upload attempt failed; retrying",
        attempt=retry_state.attempt_number,
        sleep_seconds=sleep,
        error=str(exc) if exc else "unknown",
    )


__all__ = ["log_upload_retry"]


