COMPOSE ?= docker compose --env-file .env -f docker/docker-compose.yml
COMPOSE_SERVICES ?= docker compose -f docker/docker-compose.services.yml

lint:
	ruff check .

test:
	pytest

format:
	ruff format .

clean:
	echo "Cleaning Python artifacts..."
	sudo find . -type f -name "*.pyc" -delete
	sudo find . -type f -name "*:Zone.Identifier" -delete
	sudo find . -type d -name "__pycache__" -empty -delete
	sudo find . -type d -name "*.egg-info" -exec rm -rf {} +
	echo "✅ Cleanup complete"

check: format lint test
	@echo "✅ All checks passed!"

mypy:
	mypy libs/ workflows/ --ignore-missing-imports

up:
	$(COMPOSE) up -d

down:
	$(COMPOSE) down

restart:
	$(COMPOSE) restart

logs:
	$(COMPOSE) logs -f

ps:
	$(COMPOSE) ps

build:
	$(COMPOSE) build

# Local data services (ArangoDB + MinIO). Redis is never part of these targets.
storage-up:
	$(COMPOSE) up -d arangodb minio

storage-down:
	$(COMPOSE) stop arangodb minio

# --all-packages is required: the workspace libs (arango_client, minio_client) are
# not dependencies of the root project, so a plain `uv run` would leave them out
# of the environment and these scripts would fail to import them.
storage-init:
	uv run --all-packages python scripts/init_local_storage.py

storage-verify:
	uv run --all-packages python scripts/verify_local_storage.py

# Destructive. Without CONFIRM this only prints what would be deleted.
# Run with: make storage-reset CONFIRM=RESET-ARANGO-MINIO
storage-reset:
	@bash scripts/reset_local_storage.sh --confirm "$(CONFIRM)"

up-services:
	$(COMPOSE_SERVICES) up -d

down-services:
	$(COMPOSE_SERVICES) down

logs-services:
	$(COMPOSE_SERVICES) logs -f

build-services:
	$(COMPOSE_SERVICES) build


claude:
	@/home/solid/code/router/start-proxy.sh
	@echo ""
	@echo "source /home/solid/code/router/env-claude.sh && claude"

# ============================================
# Database Migration Commands (Alembic)
# ============================================
# Load biến môi trường từ .env (nếu có)
# Sử dụng shell để load .env thay vì include (tránh lỗi nếu .env không tồn tại)
ifneq (,$(wildcard .env))
    include .env
    export
endif
