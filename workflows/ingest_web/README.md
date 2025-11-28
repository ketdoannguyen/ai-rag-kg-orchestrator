# ingest_web Workflow

This workflow crawls a set of web pages, extracts the readable content, uploads the raw/converted artifacts to MinIO, and (optionally) writes local copies for inspection. The goal is to keep the orchestration simple and predictable so we can extend it later (for example adding OCR) without rewriting core pieces.

## How It Works

```
┌──────────┐      ┌────────────┐      ┌────────────────┐      ┌─────────────┐
│ URLs /   │ ───> │ Crawl4AI   │ ───> │ Content        │ ───> │ Markdown →  │
│ payload  │      │ client     │      │ extraction     │      │ structured  │
└──────────┘      └────────────┘      └────────────────┘      │ JSON        │
       │                                                    ┌─┴─────────────┐
       │                                                    │ Image handler │
       │                                                    └─┬─────────────┘
       │                                                      │
       ▼                                                      ▼
┌──────────────────────────────────────────────────────────────┐
│ Persist artifacts (HTML / Markdown / JSON + images) to MinIO │
│ Optionally mirror the same files into `data/ingest_web/...`  │
└──────────────────────────────────────────────────────────────┘
```

1. **Input** – either pass a ready-made `WebIngestionInput` or provide raw URLs; the workflow wraps them for you.
2. **Crawler** – a shared `Crawl4AIClient` fetches each page asynchronously with bounded concurrency.
3. **Content extractor** – trims navigation, ads, and boilerplate, returning markdown-like text.
4. **Image handler** – finds markdown image references and replaces them with MinIO locations.
5. **Markdown → JSON** – converts headings/paragraphs into a hierarchical `Section` tree ready for downstream processing.
6. **Persistence** – uploads HTML/Markdown/JSON (and any images) to MinIO; optionally saves the same assets locally for debugging.

## Running It Manually

Use the helper script to kick off a run from the project root:

```bash
python scripts/run_ingest_web.py https://example.com https://another.example \
  --output tmp/ingest.json
```

Flags:

- `--enable-ocr` – passed through to the workflow (currently unused hook for future OCR support).
- `--output` – optional path where the script writes the JSON result that `start_ingest()` returns.

## Key Files

- `run.py` – orchestrates retries, concurrency, storage, and local persistence.
- `content_extractor.py` – HTML cleanup → Markdown.
- `image_handler.py` – downloads images and uploads them to MinIO.
- `markdown_to_json.py` – converts Markdown into the shared knowledge graph section model.
- `crawl4ai_client.py` – thin wrapper around `crawl4ai.AsyncWebCrawler`.

Keep changes small and readable—each helper covers a single responsibility so we can reason about failures quickly.
