from celery import shared_task

from worker.dependencies import get_ingest_workflow


@shared_task
def ingest_file_task(file_path: str) -> dict:
    workflow = get_ingest_workflow()
    return workflow(file_path)


