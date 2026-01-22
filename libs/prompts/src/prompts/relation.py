"""Relation extraction and proposal prompts.

Prompts for:
- Proposing new relation types when existing ones don't fit
- Validating proposed relations with strong models
"""

# Prompt for LLM to propose new relations
RELATION_PROPOSAL_PROMPT = """You are an expert ontology designer. The existing relation types do not fit the relationship between these nodes. Propose a new relation type.

## Node A ({node_a_type})
{node_a_text}

## Node B ({node_b_type})
{node_b_text}

## Embedding Similarity: {similarity_score:.2f}

## Existing Relations (none fit well)
{existing_relations}

## Task
Propose a NEW relation type that accurately captures the relationship.

## Response Format (JSON only)
{{
    "relation_name": "RELATION_NAME_IN_VIETNAMESE",
    "relation_name_english": "ENGLISH_NAME",
    "description": "What this relation means in Vietnamese",
    "is_symmetric": true/false,
    "is_transitive": true/false,
    "confidence": 0.0-1.0,
    "reasoning": "Why existing relations don't fit and why this new one is needed"
}}

IMPORTANT:
- Use SCREAMING_SNAKE_CASE for relation names
- Prefer Vietnamese names for educational content (e.g., "ĐỊNH_NGHĨA", "GIẢI_THÍCH")
- Be specific - avoid generic relations like "RELATED_TO"
- Consider if the relation is symmetric (A->B implies B->A)
- Consider if the relation is transitive (A->B, B->C implies A->C)
"""

# Prompt for strong model to validate proposed relation
RELATION_VALIDATION_PROMPT = """You are validating a proposed new relation type for an ontology.

## Proposed Relation
- Name: {relation_name} ({relation_name_english})
- Description: {description}
- Source Type: {source_type}
- Target Type: {target_type}
- Symmetric: {is_symmetric}
- Transitive: {is_transitive}

## Example Context
Source: {example_source}
Target: {example_target}

## Existing Ontology Relations
{existing_relations}

## Validation Criteria
1. Is this relation genuinely new? (not a duplicate/synonym of existing)
2. Is the name clear and descriptive?
3. Are symmetric/transitive properties correct?
4. Is it specific enough to be useful?
5. Does it follow naming conventions?

## Response Format (JSON only)
{{
    "is_valid": true/false,
    "validation_score": 0.0-1.0,
    "issues": ["list of issues if any"],
    "suggested_name": "better name if applicable or null",
    "reasoning": "explanation in Vietnamese"
}}
"""

# Prompt for strong model to validate a specific relation instance between two nodes
RELATION_INSTANCE_VALIDATION_PROMPT = """You are a knowledge graph expert validating a proposed relationship.

## Node A ({node_a_type})
{node_a_text}

## Node B ({node_b_type})
{node_b_text}

## Proposed Relationship
- Type: {proposed_relation}
- Description: {description}
- Original Reasoning: {reasoning}

## Task
Validate if this relationship accurately describes the connection between Node A and Node B.

## Validation Criteria
1. Does the relationship type accurately capture the semantic connection?
2. Is the relationship direction correct (A → B)?
3. Is this relationship meaningful and not overly generic?
4. Is the description accurate for this specific pair?

## Response Format (JSON only)
{{
    "is_valid": true/false,
    "confidence": 0.0-1.0,
    "reasoning": "Explanation in Vietnamese"
}}

IMPORTANT:
- Be strict but fair
- A relationship is valid if it accurately describes the connection
- Reject if the relationship is too generic or misleading
- Respond with JSON only, no markdown
"""

# Prompt for strong model to re-extract relation after rejection
RELATION_REEXTRACTION_PROMPT = """You are a knowledge graph expert. The previous relationship proposal was rejected.
Extract the correct relationship between these nodes.

## Node A ({node_a_type})
{node_a_text}

## Node B ({node_b_type})
{node_b_text}

## Previous Rejected Proposal
{previous_attempt}

{ontology_section}

## Task
1. Analyze the semantic relationship between Node A and Node B
2. Choose the BEST fitting relationship from the list above
3. Or propose a NEW relationship if none fit well

## Response Format (JSON only)
{{
    "relation_type": "RELATION_TYPE",
    "description": "Description of what this relation means (in Vietnamese)",
    "is_new_relation": true/false,
    "confidence": 0.0-1.0,
    "reasoning": "Brief explanation in Vietnamese"
}}

## CRITICAL RULES:
1. Do NOT use the previously rejected relation: {previous_attempt}
2. Do NOT use SIMILAR_TO - this is forbidden
3. If proposing new relation, provide clear description
4. Respond with JSON only, no markdown
"""
