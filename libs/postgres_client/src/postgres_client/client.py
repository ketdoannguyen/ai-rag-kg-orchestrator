from __future__ import annotations

from contextlib import asynccontextmanager
from typing import Any, Optional

from sqlalchemy.ext.asyncio import create_async_engine, AsyncEngine, async_sessionmaker, AsyncSession


class PostgresClient:
    """Thin Postgres client wrapper, owned by services/workflows."""

    def __init__(self, url: str) -> None:
        # Convert postgresql:// to postgresql+asyncpg:// for async support
        if url.startswith("postgresql://"):
            url = url.replace("postgresql://", "postgresql+asyncpg://", 1)
        elif url.startswith("postgresql+psycopg2://"):
            url = url.replace("postgresql+psycopg2://", "postgresql+asyncpg://", 1)
        
        self._url = url
        self._engine: AsyncEngine = create_async_engine(url, echo=False)
        self._sessionmaker = async_sessionmaker(self._engine, class_=AsyncSession, expire_on_commit=False)

    @property
    def engine(self) -> AsyncEngine:
        return self._engine

    @property
    def sessionmaker(self) -> async_sessionmaker[AsyncSession]:
        """Get async sessionmaker for creating sessions."""
        return self._sessionmaker

    async def close(self) -> None:
        await self._engine.dispose()

    async def __aenter__(self) -> "PostgresClient":
        return self

    async def __aexit__(
        self,
        exc_type: Optional[type[BaseException]],
        exc: Optional[BaseException],
        tb: Optional[Any],
    ) -> None:
        await self.close()


