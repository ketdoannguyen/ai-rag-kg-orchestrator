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
	docker compose up -d

up-services:
	docker compose -f docker-compose.services.yml up -d


