from fastapi import APIRouter, Depends

from ..dependencies import get_ingest_workflow

router = APIRouter(prefix="/ingest", tags=["ingest"])


@router.post("/file")
def ingest_file(file_path: str, workflow=Depends(get_ingest_workflow)) -> dict:
    return workflow(file_path)
