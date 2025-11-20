from __future__ import annotations

from contextlib import AbstractContextManager
from typing import Any, Optional

from sqlalchemy import create_engine
from sqlalchemy.engine import Engine


class PostgresClient(AbstractContextManager["PostgresClient"]):
    """Thin Postgres client wrapper, owned by services/workflows."""

    def __init__(self, url: str) -> None:
        self._url = url
        self._engine: Engine = create_engine(url)

    @property
    def engine(self) -> Engine:
        return self._engine

    def close(self) -> None:
        self._engine.dispose()

    def __exit__(
        self,
        exc_type: Optional[type[BaseException]],
        exc: Optional[BaseException],
        tb: Optional[Any],
    ) -> None:
        self.close()


