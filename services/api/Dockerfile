FROM python:3.11-slim

WORKDIR /app

RUN pip install uv
COPY . .

RUN uv sync --project services/api

CMD ["uv", "run", "--project", "services/api", "uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]


