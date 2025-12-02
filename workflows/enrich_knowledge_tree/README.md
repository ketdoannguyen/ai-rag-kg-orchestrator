## enrich_knowledge_tree Workflow

This workflow enriches a `KnowledgeNode` tree **in one bottom-up pass**:

- **Refine detail nodes** (PARAGRAPH/TABLE) into better, self-contained RAG chunks.
- **Enrich structure nodes** (STRUCTURE) with section summaries built from refined children.

It is designed to run **after** the ingest workflows (`upload_file`, `ingest_file`, `ingest_web`, `embed_nodes`) have created a full `KnowledgeNode` tree.

---

### High-Level Flow

```text
Raw Knowledge Tree (STRUCTURE + PARAGRAPH/TABLE/IMAGE)
        │
        ▼
Bottom‑up traversal (post‑order, async per level)
  ├─ For each STRUCTURE node:
  │    1) Refine direct PARAGRAPH/TABLE children with LLM
  │    2) Build a mini-summary from refined details
  │    3) Combine mini-summary + child STRUCTURE summaries
  │    4) Generate section summary (description)
        ▼
Enriched Knowledge Tree
  - Refined detail chunks (better for RAG)
  - All STRUCTURE nodes have `attributes.description`
  - Statistics + change summary returned
```

Key ideas:

- **Single pass**: walk the tree once, bottom‑up.
- **Async per level**: sibling sections (e.g., section 1, 2, 3) are processed **in parallel**.
- **Context preserved**: detail refinement and structure enrichment share the same local context.

---

### Public API

The workflow lives in `workflows.enrich_knowledge_tree` and exposes:

- `EnrichKnowledgeTreeInput`
- `EnrichKnowledgeTreeOutput`
- `KnowledgeTreeEnricher`

```python
from llm_gateway_client import AIGatewayClient
from workflows.enrich_knowledge_tree import (
    EnrichKnowledgeTreeInput,
    EnrichKnowledgeTreeOutput,
    KnowledgeTreeEnricher,
)

async def run_enrich_knowledge_tree(
    root_node,
    llm_client: AIGatewayClient,
) -> EnrichKnowledgeTreeOutput:
    enricher = KnowledgeTreeEnricher(llm_client=llm_client)

    workflow_input = EnrichKnowledgeTreeInput(
        knowledge_tree=root_node,
        # Optional configs – sensible defaults are provided
        # refine_config=RefineConfig(...),
        # enrich_config=EnrichConfig(...),
    )

    result = await enricher.enrich(workflow_input)
    return result
```

---

### Input / Output Models

#### Input

```python
class EnrichKnowledgeTreeInput(BaseModel):
    knowledge_tree: KnowledgeNode       # Root STRUCTURE node (tree with children)
    refine_config: RefineConfig         # Detail-level refinement config (optional)
    enrich_config: EnrichConfig         # Structure-level enrichment config (optional)
```

Default `refine_config`:

```json
{
  "min_paragraph_length": 100,
  "max_paragraph_length": 800,
  "batch_size": 10,
  "llm_model": "gpt-4o-mini",
  "llm_temperature": 0.3,
  "preserve_structure": true
}
```

Default `enrich_config`:

```json
{
  "max_summary_length": 300,
  "summary_style": "informative",
  "llm_model": "gpt-4o-mini",
  "llm_temperature": 0.5
}
```

#### Output

```python
class EnrichKnowledgeTreeOutput(BaseModel):
    enriched_tree: KnowledgeNode    # Root node of enriched tree
    statistics: EnrichStatistics    # Aggregated stats
    changes_summary: ChangesSummary # Per-section change summary
```

Statistics:

- `paragraphs_before` / `paragraphs_after`
- `structures_enriched`
- `detail_llm_calls`
- `structure_llm_calls`

Change summary:

- `refined_sections`: `{ node_id: { title, original_detail_count, refined_detail_count } }`
- `root_summary_preview`: first ~300 chars of root summary (if available)

---

### Traversal & Concurrency Model

The core algorithm uses **post‑order DFS** with **async concurrency** for sibling `StructureNode`s:

1. For a given `StructureNode`:
   - Recursively process all `StructureNode` children **concurrently** (`asyncio.gather`).
   - After all child summaries are ready, process the current node.
2. At the current node:
   - Split children into:
     - `detail_nodes`   → `ParagraphNode` + `TableNode`
     - `structure_nodes`→ `StructureNode` (already summarized in previous step)
     - `image_nodes`    → `ImageNode` (kept as‑is)
   - If `len(detail_nodes) <= 1`:
     - Skip LLM merge, keep detail nodes.
     - Optionally create a short heuristic mini-summary.
   - Else:
     - Collect text from paragraphs/tables.
     - Call detail‑level LLM to merge into refined `ParagraphNode`s.
     - Build a mini-summary from refined paragraphs.
   - Rebuild `node.children` as:
     - refined paragraphs + image nodes + structure children.
   - Call structure‑level LLM:
     - Input = mini-summary + child structure summaries.
     - Output = `attributes.description` for the current `StructureNode`.

This satisfies the requirement:

- Sibling sections (e.g., “Mục 1”, “Mục 2”) are refined **in parallel**.
- A parent section (“I”) only gets summarized **after** all its child sections are done.

---

### Prompt Design

Prompts live in `services/prompts/enrichment.py`:

- `build_detail_merge_prompt(items, breadcrumb)`
  - Input is a list of items (paragraph/table) with indices.
  - Asks the LLM to merge them into self‑contained segments.
  - Output JSON: `{ "segments": [ { "source_indices": [...], "content": "..." } ] }`.

- `build_structure_enrich_prompt(title, children_content, config, short=False)`
  - Input includes mini-summary + child structure summaries.
  - Asks the LLM to generate a JSON object: `{ "summary": "..." }`.
  - `short=True` is used for mini‑summaries.

The workflow reuses the same **AIGatewayClient + GatewayInvokeRequest** pattern as the existing `refine_json` workflow, including `extract_json_from_response` to robustly handle code‑block formatted outputs.

---

### Performance & Cost Characteristics

- **Asynchronous sibling processing**:
  - All direct child sections of a given node are processed with `asyncio.gather()`.
  - Good fit for trees where a chapter has many sections or subsections.

- **LLM call budget**:
  - Roughly:
    - 1 detail LLM call per `StructureNode` that has ≥ 2 detail children.
    - 1 structure LLM call per `StructureNode` that has any textual content to summarize.
  - For a large textbook‑like tree:
    - Cost is dominated by number of sections, not number of paragraphs (paragraphs are merged per section).

- **Memory**:
  - The entire `KnowledgeNode` tree is kept in memory.
  - For extremely large documents, consider:
    - Processing subtrees per chapter.
    - Or introducing streaming / chunked traversal in a follow‑up iteration.

---

### Strategy & Future Optimizations

Some pragmatic strategies if you need to push performance further:

- **Limit depth or breadth**:
  - Only run structure summaries for selected levels (e.g., chapters + main sections).
  - Skip very deep or very small sections to save LLM calls.

- **Control prompt size**:
  - Already caps table rows and concatenated text per section.
  - If you hit context limits:
    - Sample first N paragraphs + all child summaries.
    - Or introduce a two‑stage summarization (chunked → merge summaries).

- **Model choice**:
  - Use cheaper models (e.g., `gpt-4o-mini`) for:
    - Detail merging (where exact wording matters less).
  - Reserve stronger models for:
    - High‑level structure summaries (root or major sections).

- **Caching**:
  - If the tree is re‑enriched with the same content/config:
    - Add a small cache layer keyed by node id + hash of children texts.

---

### Testing

Integration tests for other workflows live in `tests/workflows/` (e.g., `test_ingest_file.py`, `test_ingest_web.py`).

For `enrich_knowledge_tree`, a typical test should:

- Build or load a small `StructureNode` tree.
- Run `KnowledgeTreeEnricher.enrich(...)` with a mocked `AIGatewayClient`.
- Assert:
  - All STRUCTURE nodes got `attributes.description` set.
  - The number of detail nodes decreased after refinement (where expected).
  - `statistics` and `changes_summary` contain consistent values.

You can run the test suite from an activated virtualenv:

```bash
source .venv/bin/activate
pytest tests/workflows/test_enrich_knowledge_tree.py -q
```



