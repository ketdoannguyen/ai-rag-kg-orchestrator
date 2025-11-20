# Project RAG KG Tuyensinh

Hyper-modular UV workspace for RAG + knowledge-graph orchestration.

## Structure

- `config/`: global settings (`pydantic BaseSettings`).
- `libs/`: infra clients (Postgres, Neo4j, LLM, Embedding, Storage, Parser).
- `workflows/`: domain pipelines (ingest file/web, query chat, admin CMS, etc.).
- `services/`: FastAPI API + Celery worker exposing workflows.

```
uv workspace
├─ libs/
├─ workflows/
└─ services/
```

## Getting Started

```bash
uv sync            # install workspace
uv run make lint   # run lint (ruff + mypy placeholder)
uv run make test   # run pytest
uv run --project services/api uvicorn api.main:app --reload
```

Copy `data/.env.example` to `data/.env` and adjust secrets.


