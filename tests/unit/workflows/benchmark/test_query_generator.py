"""Unit tests for workflows/benchmark/query_generator.py.

Covers: _extract_json, parse_llm_response.

Note: generate_queries_for_subgraph and generate_batch are async and require
LLM clients — they are excluded from unit tests per project conventions.
"""

from __future__ import annotations

import json

import pytest

from workflows.benchmark.models import Subgraph, SubgraphNode
from workflows.benchmark.query_generator import _extract_json, parse_llm_response

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_subgraph_node(node_id: str, text: str = "some text") -> SubgraphNode:
    return SubgraphNode(node_id=node_id, text=text)


def make_subgraph(
    subgraph_id: str = "sg-001",
    nodes: list[SubgraphNode] | None = None,
) -> Subgraph:
    if nodes is None:
        nodes = [
            make_subgraph_node("content_nodes/n1", "First paragraph text."),
            make_subgraph_node("content_nodes/n2", "Second paragraph text."),
        ]
    node_ids = [n.node_id for n in nodes]
    return Subgraph(
        subgraph_id=subgraph_id,
        anchor_key="anchor1",
        anchor_title="Test Subgraph",
        breadcrumb="A > B",
        nodes=nodes,
        core_node_ids=node_ids[:1],
        enriched_node_ids=node_ids[1:],
        relations=[],
        content_count=len(nodes),
        relation_count=0,
        tier="small",
        query_allocation=5,
    )


def make_valid_query_payload(
    query_text: str = "What happened?",
    fact_nodes: list[dict] | None = None,
    relations_used: list[str] | None = None,
    confidence: dict | None = None,
) -> dict:
    if fact_nodes is None:
        fact_nodes = [
            {"node_id": "content_nodes/n1", "grade": 2, "parent_section": "s1"},
            {"node_id": "content_nodes/n2", "grade": 1, "parent_section": "s1"},
        ]
    return {
        "query_text": query_text,
        "fact_nodes": fact_nodes,
        "relations_used": relations_used or [],
        "confidence": confidence
        or {
            "question_clarity": 8,
            "fact_node_completeness": 7,
            "fact_node_precision": 9,
            "grade_certainty": 8,
            "overall": 75,
        },
        "confidence_concerns": "none",
        "reasoning": "Because of facts.",
    }


def make_llm_response_text(queries: list[dict]) -> str:
    return json.dumps({"queries": queries})


# ---------------------------------------------------------------------------
# _extract_json
# ---------------------------------------------------------------------------


class TestExtractJson:
    def test_direct_json_string(self):
        payload = {"key": "value", "number": 42}
        text = json.dumps(payload)
        result = _extract_json(text)
        assert result == payload

    def test_json_in_code_fence(self):
        payload = {"queries": []}
        text = f"```json\n{json.dumps(payload)}\n```"
        result = _extract_json(text)
        assert result == payload

    def test_json_in_generic_code_fence(self):
        payload = {"hello": "world"}
        text = f"```\n{json.dumps(payload)}\n```"
        result = _extract_json(text)
        assert result == payload

    def test_json_embedded_in_text(self):
        payload = {"key": "value"}
        text = f"Here is the output: {json.dumps(payload)} and some trailing text."
        result = _extract_json(text)
        assert result == payload

    def test_json_with_leading_and_trailing_whitespace(self):
        payload = {"a": 1}
        text = f"   {json.dumps(payload)}   "
        result = _extract_json(text)
        assert result == payload

    def test_invalid_json_raises_value_error(self):
        with pytest.raises(ValueError, match="Could not extract valid JSON"):
            _extract_json("this is not json at all")

    def test_invalid_json_in_code_fence_raises(self):
        text = "```json\n{broken: json}\n```"
        with pytest.raises(ValueError):
            _extract_json(text)

    def test_nested_json_extracted_correctly(self):
        payload = {"nested": {"a": [1, 2, 3]}, "top": True}
        result = _extract_json(json.dumps(payload))
        assert result["nested"]["a"] == [1, 2, 3]

    def test_empty_object_extracted(self):
        result = _extract_json("{}")
        assert result == {}

    def test_json_array_raises_or_extracts(self):
        # _extract_json always returns a dict; a bare array text may either
        # be parsed as non-dict or raise — test that it doesn't silently fail
        # For a bare array, json.loads returns a list, not a dict.
        # The function just returns whatever json.loads returns for direct parse.
        text = "[1, 2, 3]"
        # This is valid JSON but returns a list, not dict. Function returns it.
        result = _extract_json(text)
        assert result == [1, 2, 3]

    def test_json_with_brace_prefix_text(self):
        payload = {"answer": 42}
        text = f"LLM says: {json.dumps(payload)} end."
        result = _extract_json(text)
        assert result == payload


# ---------------------------------------------------------------------------
# parse_llm_response
# ---------------------------------------------------------------------------


class TestParseLlmResponse:
    def test_valid_single_query_parsed(self):
        subgraph = make_subgraph()
        raw_text = make_llm_response_text([make_valid_query_payload()])
        valid, rejected = parse_llm_response(raw_text, subgraph)
        assert len(valid) == 1
        assert len(rejected) == 0
        assert valid[0].query_text == "What happened?"

    def test_valid_multiple_queries_parsed(self):
        subgraph = make_subgraph()
        queries = [make_valid_query_payload(query_text=f"Question {i}?") for i in range(3)]
        raw_text = make_llm_response_text(queries)
        valid, rejected = parse_llm_response(raw_text, subgraph)
        assert len(valid) == 3
        assert len(rejected) == 0

    def test_invalid_node_ids_rejected(self):
        subgraph = make_subgraph()
        query = make_valid_query_payload(
            fact_nodes=[
                {"node_id": "content_nodes/nonexistent", "grade": 2, "parent_section": "s1"},
            ]
        )
        raw_text = make_llm_response_text([query])
        valid, rejected = parse_llm_response(raw_text, subgraph)
        assert len(valid) == 0
        assert len(rejected) == 1
        assert "Invalid node_ids" in rejected[0]["reason"]

    def test_missing_grade2_node_rejected(self):
        subgraph = make_subgraph()
        # No grade-2 node — all grade 1
        query = make_valid_query_payload(
            fact_nodes=[
                {"node_id": "content_nodes/n1", "grade": 1, "parent_section": "s1"},
                {"node_id": "content_nodes/n2", "grade": 0, "parent_section": "s1"},
            ]
        )
        raw_text = make_llm_response_text([query])
        valid, rejected = parse_llm_response(raw_text, subgraph)
        assert len(valid) == 0
        assert len(rejected) == 1
        assert "grade-2" in rejected[0]["reason"]

    def test_mixed_valid_and_invalid(self):
        subgraph = make_subgraph()
        good_query = make_valid_query_payload(query_text="Good question?")
        bad_query = make_valid_query_payload(
            query_text="Bad question?",
            fact_nodes=[{"node_id": "content_nodes/ghost", "grade": 2, "parent_section": "s1"}],
        )
        raw_text = make_llm_response_text([good_query, bad_query])
        valid, rejected = parse_llm_response(raw_text, subgraph)
        assert len(valid) == 1
        assert len(rejected) == 1
        assert valid[0].query_text == "Good question?"

    def test_malformed_entry_rejected_gracefully(self):
        subgraph = make_subgraph()
        # Missing required 'query_text' key
        malformed = {
            "fact_nodes": [{"node_id": "content_nodes/n1", "grade": 2, "parent_section": "s1"}],
            "relations_used": [],
            "confidence": {},
        }
        raw_text = make_llm_response_text([malformed])
        valid, rejected = parse_llm_response(raw_text, subgraph)
        assert len(valid) == 0
        assert len(rejected) == 1

    def test_empty_queries_list_returns_empty(self):
        subgraph = make_subgraph()
        raw_text = json.dumps({"queries": []})
        valid, rejected = parse_llm_response(raw_text, subgraph)
        assert valid == []
        assert rejected == []

    def test_query_id_is_auto_generated_uuid(self):
        subgraph = make_subgraph()
        raw_text = make_llm_response_text([make_valid_query_payload()])
        valid, _ = parse_llm_response(raw_text, subgraph)
        assert len(valid[0].query_id) == 36  # UUID format

    def test_each_query_gets_unique_id(self):
        subgraph = make_subgraph()
        queries = [make_valid_query_payload(query_text=f"Question {i}?") for i in range(5)]
        raw_text = make_llm_response_text(queries)
        valid, _ = parse_llm_response(raw_text, subgraph)
        ids = {q.query_id for q in valid}
        assert len(ids) == 5

    def test_subgraph_id_assigned_to_queries(self):
        subgraph = make_subgraph(subgraph_id="sg-specific")
        raw_text = make_llm_response_text([make_valid_query_payload()])
        valid, _ = parse_llm_response(raw_text, subgraph)
        assert valid[0].subgraph_id == "sg-specific"

    def test_confidence_defaults_applied_when_missing(self):
        subgraph = make_subgraph()
        # Provide no confidence dict
        query_payload = make_valid_query_payload()
        query_payload.pop("confidence", None)
        raw_text = make_llm_response_text([query_payload])
        valid, rejected = parse_llm_response(raw_text, subgraph)
        assert len(valid) == 1
        # Default confidence overall = 50
        assert valid[0].confidence.overall == 50

    def test_json_in_markdown_fence_parsed(self):
        subgraph = make_subgraph()
        payload = {"queries": [make_valid_query_payload()]}
        raw_text = f"```json\n{json.dumps(payload)}\n```"
        valid, rejected = parse_llm_response(raw_text, subgraph)
        assert len(valid) == 1

    def test_multiple_invalid_queries_all_rejected(self):
        subgraph = make_subgraph()
        # Both queries reference nonexistent nodes
        bad_queries = [
            make_valid_query_payload(
                fact_nodes=[
                    {"node_id": "content_nodes/ghost1", "grade": 2, "parent_section": "s1"}
                ]
            ),
            make_valid_query_payload(
                fact_nodes=[
                    {"node_id": "content_nodes/ghost2", "grade": 2, "parent_section": "s1"}
                ]
            ),
        ]
        raw_text = make_llm_response_text(bad_queries)
        valid, rejected = parse_llm_response(raw_text, subgraph)
        assert len(valid) == 0
        assert len(rejected) == 2

    def test_relations_used_preserved(self):
        subgraph = make_subgraph()
        query = make_valid_query_payload(relations_used=["rel_type_1", "rel_type_2"])
        raw_text = make_llm_response_text([query])
        valid, _ = parse_llm_response(raw_text, subgraph)
        assert valid[0].relations_used == ["rel_type_1", "rel_type_2"]

    def test_rejected_entry_contains_audit_info(self):
        subgraph = make_subgraph()
        bad_query = make_valid_query_payload(
            query_text="Bad node query?",
            fact_nodes=[{"node_id": "content_nodes/ghost", "grade": 2, "parent_section": "s1"}],
        )
        raw_text = make_llm_response_text([bad_query])
        _, rejected = parse_llm_response(raw_text, subgraph)
        entry = rejected[0]
        assert "index" in entry
        assert "reason" in entry
        assert "query_text" in entry
