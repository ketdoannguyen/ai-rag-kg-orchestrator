from celery import shared_task

from worker.dependencies import get_ingest_workflow


@shared_task
def ingest_file_task(data_source_id: str) -> dict:
    workflow = get_ingest_workflow()
    return workflow(data_source_id)
