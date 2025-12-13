"""Input and output models for Router Agent.

This module defines clear contracts for the Router Agent:
- RouterInput: What the agent receives
- RouterOutput: What the agent produces
- SubQuestion: Individual searchable sub-question with its own keywords/filters
- DomainConfig: Domain-specific configuration
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class Intent(str, Enum):
    """Query intent classification.
    
    Each intent maps to a different retrieval strategy:
    - DISCOVERY: Broad search, multiple results
    - SPECIFIC_FACT: Precise search, single answer
    - INSTRUCTION: Step-by-step content retrieval
    - COMPARISON: Multi-entity retrieval for comparison
    - CONTACT_INFO: Contact/location specific search
    - CHIT_CHAT: No retrieval needed, direct response
    """
    
    DISCOVERY = "DISCOVERY"
    SPECIFIC_FACT = "SPECIFIC_FACT"
    INSTRUCTION = "INSTRUCTION"
    COMPARISON = "COMPARISON"
    CONTACT_INFO = "CONTACT_INFO"
    CHIT_CHAT = "CHIT_CHAT"


class ChatMessage(BaseModel):
    """Single message in conversation history."""
    
    role: str = Field(
        ...,
        description="Message role: 'user' or 'assistant'",
    )
    content: str = Field(
        ...,
        description="Message content",
    )


class FilterFieldConfig(BaseModel):
    """Configuration for a single filter field."""
    
    type: str = Field(
        default="string",
        description="Data type: 'string', 'integer', 'float', 'boolean', 'list'",
    )
    description: str = Field(
        default="",
        description="Human-readable description of this filter field",
    )
    allowed_values: Optional[List[Any]] = Field(
        default=None,
        description="Enum-like list of valid values. LLM must output one of these.",
    )
    examples: List[Any] = Field(
        default_factory=list,
        description="Example values (used if allowed_values not set)",
    )
    aliases: Dict[str, Any] = Field(
        default_factory=dict,
        description="Mapping of abbreviations to normalized values, e.g. {'IT': 'Công nghệ thông tin'}",
    )


class DomainConfig(BaseModel):
    """Domain-specific configuration for Router Agent."""
    
    domain_name: str = Field(
        ...,
        description="Human-readable domain name, e.g., 'Tuyển sinh Đại học'",
    )
    filter_fields: Dict[str, FilterFieldConfig] = Field(
        default_factory=dict,
        description="Available filter fields for this domain",
    )
    custom_intent_descriptions: Optional[Dict[str, str]] = Field(
        default=None,
        description="Optional custom descriptions for each intent",
    )
    
    def get_filter_fields_dict(self) -> Dict[str, Dict[str, Any]]:
        """Convert filter_fields to dict format for prompt building."""
        return {
            name: config.model_dump()
            for name, config in self.filter_fields.items()
        }


class RouterInput(BaseModel):
    """Input model for Router Agent."""
    
    model_config = {"validate_default": True}
    
    user_query: str = Field(
        ...,
        min_length=1,
        description="The current user query to analyze",
    )
    conversation_history: List[ChatMessage] = Field(
        default_factory=list,
        description="Previous messages for context and coreference resolution",
    )
    domain_config: DomainConfig = Field(
        ...,
        description="Domain-specific configuration",
    )
    current_year: int = Field(
        default_factory=lambda: datetime.now().year,
        description="Current year for temporal inference",
    )
    
    def get_history_as_dicts(self) -> List[Dict[str, str]]:
        """Convert conversation history to dict format."""
        return [msg.model_dump() for msg in self.conversation_history]


class SubQuestion(BaseModel):
    """A single searchable sub-question with its own retrieval parameters.
    
    This is the atomic unit for downstream retrieval.
    """
    
    query: str = Field(
        ...,
        description="The question text, rewritten for clarity",
    )
    keywords: List[str] = Field(
        default_factory=list,
        description="Keywords for full-text search (BM25)",
    )
    filters: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Metadata filters for vector/hybrid search",
    )


class RouterOutput(BaseModel):
    """Output model for Router Agent.
    
    Structured search object:
    - intent: Query classification
    - rewritten_query: Original query rewritten for context
    - sub_questions: List of searchable units, each with keywords and filters
    
    For simple queries: 1 sub_question
    For COMPARISON: Multiple sub_questions
    For CHIT_CHAT: Empty sub_questions
    """
    
    intent: Intent = Field(
        ...,
        description="Classified query intent",
    )
    rewritten_query: str = Field(
        ...,
        description="Rewritten query with resolved references (for display/logging)",
    )
    sub_questions: List[SubQuestion] = Field(
        default_factory=list,
        description="Searchable units, each with query/keywords/filters",
    )
    
    # Metadata
    raw_llm_response: Optional[str] = Field(
        default=None,
        description="Raw LLM response for debugging",
        exclude=True,
    )
    
    @classmethod
    def create_chit_chat_response(cls, original_query: str) -> "RouterOutput":
        """Create a default CHIT_CHAT response."""
        return cls(
            intent=Intent.CHIT_CHAT,
            rewritten_query=original_query,
            sub_questions=[],
        )


class RouterConfig(BaseModel):
    """Configuration for Router Agent behavior."""
    
    llm_model: Optional[str] = Field(
        default=None,
        description="LLM model to use. None uses gateway's default.",
    )
    temperature: float = Field(
        default=0.0,
        ge=0.0,
        le=2.0,
        description="LLM temperature for deterministic routing",
    )
    max_retries: int = Field(
        default=3,
        ge=1,
        le=10,
        description="Maximum retry attempts for LLM calls",
    )
