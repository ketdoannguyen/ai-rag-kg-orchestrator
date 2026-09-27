from fastapi import APIRouter, Depends

from ..dependencies import get_ingest_workflow

router = APIRouter(prefix="/ingest", tags=["ingest"])


@router.post("/file")
def ingest_file(data_source_id: str, workflow=Depends(get_ingest_workflow)) -> dict:
    return workflow(data_source_id)
