from fastapi import FastAPI

from .routers import chat, cms, ingest

app = FastAPI(title="RAG KG Orchestrator API")

app.include_router(chat.router)
app.include_router(ingest.router)
app.include_router(cms.router)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}
