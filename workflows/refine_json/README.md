# refine_json Workflow

This workflow refines processed textbook JSON content by intelligently merging fragmented paragraphs into coherent segments using an LLM. It processes each Section by collecting all its direct Paragraphs and refining them together, ensuring better context understanding and more accurate merging. The goal is to improve content quality for downstream RAG indexing and knowledge graph construction.

## How It Works

```
┌─────────────────┐      ┌──────────────────┐      ┌─────────────────┐
│ Processed JSON  │ ───> │ Collect Section │ ───>  │ Refine All      │
│ (Section tree)  │      │ Paragraphs       │      │ Paragraphs      │
└─────────────────┘      └──────────────────┘      │ with LLM        │
       │                                              └──────────────┘
       │                                                      │
       │                                                      ▼
       │                                            ┌─────────────────┐
       │                                            │ Merge Segments  │
       │                                            │ + Union BBoxes  │
       │                                            └─────────────────┘
       │                                                      │
       │                                                      ▼
       └──────────────────────────────────────────────────────────┐
                                                                  │
       ┌──────────────────────────────────────────────────────────┘
       │
       ▼
┌──────────────────────────────────────────────────────────────┐
│ Save refined JSON locally + Upload to MinIO                  │
│ Output: textbook/<date>/<job_id>/refined.json                │
└──────────────────────────────────────────────────────────────┘
```

1. **Input** – Load processed JSON file with `Section` tree structure (typically from `process_raw_json` workflow).
2. **Section Processing** – For each Section, collect all direct Paragraphs (excluding Paragraphs in child Sections).
3. **Concurrent Refinement** – Process multiple Sections concurrently using `asyncio.gather()` for better performance.
4. **LLM Merging** – Send all Paragraphs of a Section to LLM together, allowing it to see full context and merge intelligently.
5. **Segment Creation** – Parse LLM response to create refined Paragraph objects with merged content and union bounding boxes.
6. **Persistence** – Save refined JSON locally and upload to MinIO storage.

## Key Features

- **Section-based Refinement**: Each Section's Paragraphs are refined together, ensuring better context understanding.
- **Concurrent Processing**: Multiple Sections are processed in parallel for faster execution.
- **BBox Preservation**: Calculates union bounding boxes for merged segments to maintain spatial grounding.
- **Async I/O**: Uses `asyncio.to_thread()` for non-blocking file operations.
- **Error Handling**: Graceful degradation - returns original Paragraphs if LLM call fails or times out.

## Running It Manually

### Python API

```python
from workflows.refine_json import TextbookRefiner
from config.settings import settings
from config.llm_gateway import build_llm_gateway_client
import asyncio

async def run_refinement():
    # Initialize LLM Client
    llm_client = build_llm_gateway_client(settings)
    
    # Create Refiner
    refiner = TextbookRefiner(
        llm_client=llm_client,
        model=settings.LLM_GATEWAY_CHAT_MODEL
    )
    
    # Run Refinement
    result = await refiner.refine(
        input_path="processed_textbook.json",
        output_path="refined_textbook.json",
        llm_timeout=60.0  # Optional: timeout in seconds
    )
    
    # Check results
    if result.success:
        print(f"✅ Refined {result.paragraphs_refined} paragraphs")
        print(f"📁 Saved to: {result.output_path}")
        if result.minio_object:
            print(f"☁️  Uploaded to MinIO: {result.minio_object}")
    else:
        print(f"❌ Error: {result.error}")

if __name__ == "__main__":
    asyncio.run(run_refinement())
```

## Configuration

The workflow relies on global `settings` and `llm_gateway_client`. Ensure the following environment variables are set (usually in `.env`):

- `LLM_GATEWAY_CHAT_MODEL`: The LLM model to use (e.g., `openrouter/openai/gpt-4o`).
- `LLM_GATEWAY_CHAT_API_KEY`: API key for the LLM provider.
- `MINIO_BUCKET`: MinIO bucket name (default: from settings).
- `MINIO_ENDPOINT`, `MINIO_ACCESS_KEY`, `MINIO_SECRET_KEY`: MinIO connection details.

## Key Files

- `refiner.py` – Main orchestrator: collects Section Paragraphs, refines them with LLM, processes Sections concurrently.
- `helpers.py` – Pure helper functions: JSON parsing, bbox calculation, paragraph creation.
- `models.py` – Input/Output models: `RefineJsonInput`, `RefineJsonOutput` with validation.
- `services/prompts/refinement.py` – LLM prompts for paragraph merging.

## How Refinement Works

1. **Section Traversal**: Recursively traverse the Section tree.
2. **Paragraph Collection**: For each Section, collect all direct Paragraphs (not from child Sections).
3. **Concurrent Processing**: Process multiple Sections in parallel using `asyncio.gather()`.
4. **LLM Call**: Send all Paragraphs of a Section to LLM with context (breadcrumb path).
5. **Response Parsing**: Extract JSON segments from LLM response, handle code blocks.
6. **Paragraph Creation**: Create new Paragraph objects with:
   - Merged text content
   - Union bounding boxes from source Paragraphs
   - Page number from first source Paragraph
7. **Tree Rebuilding**: Replace original Paragraphs with refined ones, maintain other children (Images, Tables, Sections).

## Performance Considerations

- **Concurrent Section Processing**: Multiple Sections are refined in parallel for faster execution.
- **Single LLM Call per Section**: All Paragraphs of a Section are sent together, reducing API calls and improving context.
- **Async File I/O**: Non-blocking file operations prevent event loop blocking.
- **Timeout Protection**: LLM calls have configurable timeouts to prevent hanging.

Keep changes small and readable—each helper covers a single responsibility so we can reason about failures quickly.
