FROM python:3.11-slim

WORKDIR /app

RUN pip install uv
COPY . .

RUN uv sync --project services/worker

CMD ["uv", "run", "--project", "services/worker", "celery", "-A", "worker.celery_app", "worker", "--loglevel=info"]


