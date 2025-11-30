COMPOSE ?= docker compose -f docker/docker-compose.yml
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

up-services:
	$(COMPOSE_SERVICES) up -d

down-services:
	$(COMPOSE_SERVICES) down

logs-services:
	$(COMPOSE_SERVICES) logs -f

build-services:
	$(COMPOSE_SERVICES) build

# ============================================
# Database Migration Commands (Alembic)
# ============================================
# Load biến môi trường từ .env (nếu có)
# Sử dụng shell để load .env thay vì include (tránh lỗi nếu .env không tồn tại)
ifneq (,$(wildcard .env))
    include .env
    export
endif

# Chuyển đổi DATABASE_URL từ psycopg sang asyncpg cho Alembic
# Nếu DB_URL đã được set thì dùng nó, nếu không thì convert từ DATABASE_URL
# Format: postgresql+asyncpg://user:password@localhost:5432/db
DB_URL ?= $(shell if [ -n "$$DATABASE_URL" ]; then \
	echo "$$DATABASE_URL" | sed 's|postgresql+psycopg://|postgresql+asyncpg://|' | sed 's|postgresql://|postgresql+asyncpg://|'; \
	else \
	echo "postgresql+asyncpg://user:password@localhost:5432/rag"; \
	fi)

# Lệnh tạo file migration mới (Tự động so sánh Code vs DB)
db-gen:
	@if [ -z "$(msg)" ]; then \
		echo "❌ Error: msg parameter is required. Usage: make db-gen msg='your message'"; \
		exit 1; \
	fi
	cd libs/postgres_client && \
	PYTHONPATH=src DB_URL=$(DB_URL) uv run alembic revision --autogenerate -m "$(msg)"

# Lệnh chạy migration (Update DB lên version mới nhất)
db-up:
	cd libs/postgres_client && \
	PYTHONPATH=src DB_URL=$(DB_URL) uv run alembic upgrade head

# Lệnh rollback (Quay lui 1 bước)
db-down:
	cd libs/postgres_client && \
	PYTHONPATH=src DB_URL=$(DB_URL) uv run alembic downgrade -1

# Lệnh xem lịch sử migration
db-history:
	cd libs/postgres_client && \
	PYTHONPATH=src DB_URL=$(DB_URL) uv run alembic history

# Lệnh xem trạng thái hiện tại
db-current:
	cd libs/postgres_client && \
	PYTHONPATH=src DB_URL=$(DB_URL) uv run alembic current