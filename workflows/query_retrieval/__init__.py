"""Query retrieval workflow for Vietnamese History domain.

This module provides:
- Query classification (LLM-based)
- Dynamic relation selection (embedding-first)
- Multi-strategy retrieval

Entry point: start_query_retrieval()
"""

from __future__ import annotations

from .models import (
    QueryRetrievalInput,
    QueryRetrievalOutput,
    QueryType,
    RetrievalStrategy,
)
from .run import start_query_retrieval

__all__ = [
    "start_query_retrieval",
    "QueryRetrievalInput",
    "QueryRetrievalOutput",
    "QueryType",
    "RetrievalStrategy",
]
