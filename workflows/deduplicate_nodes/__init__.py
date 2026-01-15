"""Deduplicate nodes workflow for merge/deduplication pipeline.

Provides:
- start_deduplicate_nodes: Main entry point
- DeduplicateInput: Input dataclass
- DeduplicateOutput: Output dataclass
- DedupMode: Mode enum (inline/batch/hybrid)
- DeduplicateNodesWorkflow: Workflow implementation class
- LLMNodeVerifier: LLM verification class
- LLMVerificationResult: Result from LLM verification
"""

from .llm_verifier import LLMNodeVerifier
from .models import (
    DeduplicateInput,
    DeduplicateOutput,
    DedupMode,
    LLMVerificationResult,
    NodeProcessResult,
)
from .run import start_deduplicate_nodes
from .workflow import DeduplicateNodesWorkflow

__all__ = [
    "start_deduplicate_nodes",
    "DeduplicateInput",
    "DeduplicateOutput",
    "DedupMode",
    "DeduplicateNodesWorkflow",
    "LLMNodeVerifier",
    "LLMVerificationResult",
    "NodeProcessResult",
]
