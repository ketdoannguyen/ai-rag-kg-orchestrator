"""Extraction prompts for merge/relation decisions.

These prompts are used by the LLM extractor to determine if two nodes
should be merged or if a relation should be extracted between them.
"""

VERIFICATION_PROMPT = """Analyze two Vietnamese content nodes: merge, relate, or keep separate.

## Rules
- MERGE: Only if 90%+ identical FACTS (not just same topic). Same topic ≠ merge.
- RELATION: Different info about related topics. Use valid relation or propose new.
- SEPARATE: No meaningful connection. Return relation_type: null.
- NEVER use SIMILAR_TO. When in doubt, don't merge.
- NEVER use structural relations: HAS_SECTION, BELONGS_TO_SECTION, HAS_CONTENT, NEXT_SIBLING, PREVIOUS_SIBLING
- NEVER use fallback relations: SIMILAR_TO, STRUCTURALLY_EQUIVALENT
- Focus on SEMANTIC relations between the content of the nodes

## Output (JSON only)
{{"should_merge": bool, "relation_type": "TYPE" or null, "confidence": 0.0-1.0, "reasoning": "brief", "is_new_relation": false, "new_relation_description": null}}

## Valid Relations
{valid_relations}

## Node A
- Type: {node_a_type}
- Text: {node_a_text}

## Node B
- Type: {node_b_type}
- Text: {node_b_text}

## Embedding Similarity: {similarity_score:.2f}
"""

VERIFICATION_BATCH_PROMPT = """Analyze Vietnamese content node pairs: merge, relate, or keep separate.

## Rules
- MERGE: Only if 90%+ identical FACTS (not just same topic). Same topic ≠ merge.
- RELATION: Different info about related topics. Use valid relation or propose new.
- SEPARATE: No meaningful connection. Return relation_type: null.
- NEVER use SIMILAR_TO. When in doubt, don't merge.
- NEVER use structural relations: HAS_SECTION, BELONGS_TO_SECTION, HAS_CONTENT, NEXT_SIBLING, PREVIOUS_SIBLING
- NEVER use fallback relations: SIMILAR_TO, STRUCTURALLY_EQUIVALENT
- Focus on SEMANTIC relations between the content of the nodes

## Output (JSON array only)
[{{"pair_index": 1, "should_merge": bool, "relation_type": "TYPE" or null, "confidence": 0.0-1.0, "reasoning": "brief", "is_new_relation": false, "new_relation_description": null}}, ...]

## Valid Relations
{valid_relations}

## Node Pairs
{pairs_text}
"""
