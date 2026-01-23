"""Extraction prompts for merge/relation decisions.

These prompts are used by the LLM extractor to determine if two nodes
should be merged or if a relation should be extracted between them.
"""

VERIFICATION_PROMPT = """You are an expert at knowledge graph deduplication for Vietnamese educational content.
Analyze these two content nodes and determine if they should be merged, related, or kept separate.

## Node A
- Type: {node_a_type}
- Text: {node_a_text}

## Node B
- Type: {node_b_type}
- Text: {node_b_text}

## Embedding Similarity: {similarity_score:.2f}
(This is embedding similarity only - nodes may or may not have a semantic relation)

{valid_relations}

## Task
1. Determine if nodes should be MERGED (identical/duplicate content)
2. If NOT merging, check if there is a meaningful RELATION between them
3. If no meaningful relation exists, return relation_type: null

## Response Format (JSON only)
{{
    "should_merge": true/false,
    "relation_type": "RELATION_TYPE" or null,
    "confidence": 0.0-1.0,
    "reasoning": "brief explanation in Vietnamese",
    "is_new_relation": false,
    "new_relation_description": "Description if proposing new relation, null otherwise"
}}

## CRITICAL RULES:
1. should_merge=true: Nodes are duplicates, merge them
2. should_merge=false AND relation_type="...": Nodes have a semantic relation
3. should_merge=false AND relation_type=null: Nodes are NOT related, keep separate
4. **NEVER force a relation** - null is valid when nodes aren't truly related!
5. **NEVER use SIMILAR_TO** - this is a forbidden fallback relation
6. If proposing a new relation, set is_new_relation=true AND provide new_relation_description
7. new_relation_description MUST be in Vietnamese and explain what this relation means
8. Respond with JSON only, no markdown
"""

VERIFICATION_BATCH_PROMPT = """You are an expert at knowledge graph deduplication for Vietnamese educational content.
Analyze these node pairs and determine if each should be merged, related, or kept separate.

## Node Pairs
{pairs_text}

{valid_relations}

## Task
For EACH pair, determine:
1. Should nodes be MERGED (identical/duplicate content)?
2. If NOT merging, is there a meaningful RELATION between them?
3. If no meaningful relation exists, return relation_type: null

## Response Format (JSON array only)
[
    {{
        "pair_index": 1,
        "should_merge": true/false,
        "relation_type": "RELATION_TYPE" or null,
        "confidence": 0.0-1.0,
        "reasoning": "brief explanation",
        "is_new_relation": false,
        "new_relation_description": "Description if proposing new relation, null otherwise"
    }},
    ...
]

## CRITICAL RULES:
1. Return exactly {expected_count} results in order
2. should_merge=true: Nodes are duplicates, merge them
3. should_merge=false AND relation_type="...": Nodes have a semantic relation
4. should_merge=false AND relation_type=null: Nodes are NOT related, keep separate
5. **NEVER force a relation** - null is valid when nodes aren't truly related!
6. **NEVER use SIMILAR_TO** - this is a forbidden fallback relation
7. If proposing a new relation, set is_new_relation=true AND provide new_relation_description
8. new_relation_description MUST be in Vietnamese and explain what this relation means
9. Respond with JSON array only, no markdown
"""
