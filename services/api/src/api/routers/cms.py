from fastapi import APIRouter, Depends

from ..dependencies import get_cms_workflow

router = APIRouter(prefix="/cms", tags=["cms"])


@router.post("/chunk/{chunk_id}")
def update_chunk(chunk_id: str, text: str, workflow=Depends(get_cms_workflow)) -> dict:
    return workflow(chunk_id, text)
