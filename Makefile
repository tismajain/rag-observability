.PHONY: install dev ingest test test-unit lint migrate shell down clean help

help:
	@echo "Targets:"
	@echo "  install     Install package + dev deps in editable mode"
	@echo "  dev         Bring up docker services and run the API with reload"
	@echo "  ingest      Run the ingestion CLI (placeholder until Phase 2)"
	@echo "  test        Run the full test suite"
	@echo "  test-unit   Run unit tests only"
	@echo "  lint        Ruff + mypy on src/"
	@echo "  migrate     Apply Alembic migrations (lands Phase 6)"
	@echo "  shell       IPython shell with settings preloaded"
	@echo "  down        Stop docker services"
	@echo "  clean       Stop services and remove named volumes"

install:
	pip install -e ".[dev]"

dev:
	docker compose up -d
	uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000

ingest:
	python -m cli.ingest --dir ./data/documents

test:
	pytest tests/ -v

test-unit:
	pytest tests/unit/ -v

lint:
	ruff check src/
	mypy src/

migrate:
	alembic upgrade head

shell:
	ipython -i -c "from src.config.settings import settings; print(settings.model_dump())"

down:
	docker compose down

clean:
	docker compose down -v
