# ingest_file Workflow

This workflow processes PDF files using the Marker SDK, extracts structured content and images, uploads artifacts to MinIO, and optionally writes local copies for inspection. The goal is to keep the orchestration simple and predictable so we can extend it later without rewriting core pieces.

## How It Works

```
┌──────────┐      ┌────────────┐      ┌────────────────┐      ┌─────────────┐
│ PDF file │ ───> │ Marker SDK │ ───> │ Extract content │ ───> │ Structured  │
│ / input  │      │ converter  │      │ (JSON/MD/images) │      │ JSON output │
└──────────┘      └────────────┘      └────────────────┘      └─────────────┘
       │                                                    ┌─┴─────────────┐
       │                                                    │ Image handler │
       │                                                    └─┬─────────────┘
       │                                                      │
       ▼                                                      ▼
┌──────────────────────────────────────────────────────────────┐
│ Persist artifacts (PDF / JSON / Markdown + images) to MinIO │
│ Optionally mirror the same files into workspace directory   │
└──────────────────────────────────────────────────────────────┘
```

1. **Input** – either pass a ready-made `FileIngestionInput` or provide a file path; the workflow wraps it for you with validation.
2. **Marker conversion** – the `MarkerClient` converts PDF to structured JSON and markdown using local inference (runs in thread pool to avoid blocking event loop).
3. **Content extraction** – extracts JSON structure, markdown text, and embedded images from the PDF.
4. **Image handler** – saves extracted images to workspace and uploads them to MinIO.
5. **Persistence** – uploads source PDF, JSON, Markdown, and images to MinIO; saves the same assets locally for debugging.

## Running It Manually

Use the helper script to kick off a run from the project root:

```bash
python scripts/run_ingest_file.py path/to/document.pdf \
  --output tmp/ingest.json
```

Flags:

- `--enable-ocr` – passed through to the workflow (enables OCR processing in Marker).
- `--output` – optional path where the script writes the JSON result that `start_ingest()` returns.

## Key Files

- `run.py` – orchestrates conversion, storage, and local persistence with retry logic.
- `marker_client.py` – thin wrapper around the Marker SDK for PDF conversion.
- `models.py` – Pydantic models for input validation and output structure.

Keep changes small and readable—each helper covers a single responsibility so we can reason about failures quickly.
