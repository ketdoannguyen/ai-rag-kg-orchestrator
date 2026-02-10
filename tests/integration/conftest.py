"""Shared fixtures for integration tests."""

from __future__ import annotations


def pytest_addoption(parser):
    """Add --run-llm option for tests requiring LLM API calls."""
    parser.addoption(
        "--run-llm",
        action="store_true",
        default=False,
        help="Run tests that require LLM API calls (embedding, routing)",
    )
