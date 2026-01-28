"""Relation extraction and proposal prompts.

Prompts for:
- Strong LLM verification and extraction of new relations
"""

from config.ontologies.families import ALL_FAMILIES, FAMILY_METADATA


def _build_families_section() -> str:
    """Build families section dynamically from families.py."""
    lines = []
    for family in ALL_FAMILIES:
        metadata = FAMILY_METADATA.get(family, {})
        name = metadata.get("name", family)
        description = metadata.get("description", "")
        lines.append(f'- "{family}": {name} - {description}')
    return "\n".join(lines)


# Prompt for Strong LLM to verify AND extract full info in one step
STRONG_VERIFY_AND_EXTRACT_PROMPT = (
    """You are an expert in Knowledge Graphs and semantic relationships.

## TASK
Verify and extract complete information for a new relation type.

## INPUT
- Node A: {node_a_text}
- Node B: {node_b_text}
- Proposed Relation: {proposed_relation}
- Summary (from weak LLM): {summary}

## REQUIREMENTS

1. **ANALYZE** whether the relation "{proposed_relation}" accurately describes the relationship between Node A and Node B.

2. **IF ACCEPTED** (accepted=true):
   - Extract complete information for relation "{proposed_relation}"

3. **IF REJECTED** (accepted=false):
   - Propose a more appropriate relation type
   - Extract complete information for the new relation

## OUTPUT FORMAT (JSON)

```json
{{
  "accepted": true,
  "reasoning": "Reason for accepting or rejecting the relation...",

  "id": "RELATION_NAME",
  "name": "RELATION_NAME",
  "summary": "Short description in 10-20 words",
  "description": "Detailed description of this relation, including meaning, usage, examples. Write in Vietnamese, 50-100 words.",

  "source_node_types": ["paragraph"],
  "target_node_types": ["paragraph"],

  "is_symmetric": false,
  "is_transitive": false,

  "families": ["family1", "family2"],
  "keywords": ["keyword1", "keyword2", "keyword3"]
}}
```

## DETAILED GUIDELINES

### id and name
- Format: SCREAMING_SNAKE_CASE (e.g., FATHER_OF, LED_BY, CAUSED_BY)
- Must be in English, no accents

### summary
- 10-20 words in Vietnamese
- Brief description of the relation meaning

### description
- 50-100 words in Vietnamese
- Detailed description including:
  - Meaning of the relation
  - Direction: what A → B means
  - Specific examples
  - Distinction from similar relations

### source_node_types / target_node_types
- Valid values: "paragraph", "structure", "entity"
- Describes which node types can be source/target

### is_symmetric
- true: If A→B then B→A is also true (e.g., SIMILAR_TO, MARRIED_TO)
- false: Direction matters (e.g., FATHER_OF, LED_BY)

### is_transitive
- true: If A→B and B→C then A→C (e.g., PART_OF, ANCESTOR_OF)
- false: No transitivity

### families
Choose from the following families:
"""
    + _build_families_section()
    + """

### keywords
- 5-10 Vietnamese keywords
- Used to match user queries
- Include synonyms

## RESPONSE
Return JSON object in the format above. Do NOT add any other text.
"""
)
