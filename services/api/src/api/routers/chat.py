from fastapi import APIRouter, Depends

from ..dependencies import get_chat_workflow

router = APIRouter(prefix="/chat", tags=["chat"])


@router.post("/query")
def chat_query(
    query: str,
    workflow=Depends(get_chat_workflow),
) -> dict:
    return workflow(query)


