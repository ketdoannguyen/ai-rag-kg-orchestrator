from celery import Celery

app = Celery(
    "worker",
    broker="redis://localhost:6379/0",
    backend="redis://localhost:6379/0",
)

app.autodiscover_tasks(["worker.tasks"])
