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
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*:Zone.Identifier" -delete
	find . -type d -name "__pycache__" -empty -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} +
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