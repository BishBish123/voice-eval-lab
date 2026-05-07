.DEFAULT_GOAL := help
SHELL := /bin/bash
UV ?= uv

.PHONY: help
help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) | sort \
		| awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2}'

.PHONY: install
install: ## Install dev deps
	$(UV) sync --extra dev

.PHONY: fmt
fmt: ## Format with ruff
	$(UV) run ruff format src tests

.PHONY: lint
lint: ## Lint with ruff
	$(UV) run ruff check src tests

.PHONY: typecheck
typecheck: ## mypy --strict
	$(UV) run mypy src

.PHONY: check
check: lint typecheck ## Lint + typecheck

.PHONY: test
test: ## Unit tests
	$(UV) run pytest -m "not integration"

.PHONY: eval
eval: ## Run the eval harness on the bundled golden set
	$(UV) run voice-eval run --out evals/REPORT.md --json evals/scores.json

.PHONY: calibrate
calibrate: evals/calibration.csv  ## Run kappa calibration (requires evals/calibration.csv with human labels)
	$(UV) run voice-eval calibrate --labels evals/calibration.csv --out evals/CALIBRATION.md

.PHONY: eval-canonical
eval-canonical: evals/calibration.csv  ## Regenerate the tracked canonical report + WER-injection variant + calibration
	$(UV) run voice-eval run --out evals/REPORT.md --json evals/scores.json
	$(UV) run voice-eval run --wer-substitution-rate 0.1 \
		--out evals/REPORT.with-wer-injection.md
	$(UV) run voice-eval calibrate --labels evals/calibration.csv --out evals/CALIBRATION.md

.PHONY: notes-up
notes-up: ## Start local Postgres + pgvector for the PgVectorNotesStore backend
	docker compose -f docker-compose.notes.yml up -d

.PHONY: notes-down
notes-down: ## Stop the local Postgres + pgvector container
	docker compose -f docker-compose.notes.yml down

.PHONY: backend-up
backend-up: ## Start local Postgres for the PostgresSessionStore backend
	docker compose -f docker-compose.backend.yml up -d

.PHONY: backend-down
backend-down: ## Stop the local Postgres container for the backend
	docker compose -f docker-compose.backend.yml down

.PHONY: docker-build
docker-build: ## Build the production Docker image (tag: voice-eval-lab:latest)
	docker build --tag voice-eval-lab:latest .

.PHONY: docker-run
docker-run: ## Run the production Docker image locally (in-memory store, port 8000)
	docker run --rm \
		--name voice-eval-lab-local \
		--publish 8000:8000 \
		voice-eval-lab:latest

.PHONY: deploy
deploy: ## Deploy to Fly.io via scripts/deploy.sh (requires fly CLI + auth)
	bash scripts/deploy.sh

.PHONY: test-integration-mock
test-integration-mock: ## Run mock-server integration tests (real-mode adapter HTTP paths)
	$(UV) run pytest -m integration -v

.PHONY: clean
clean: ## Wipe caches
	rm -rf .pytest_cache .mypy_cache .ruff_cache .coverage htmlcov build dist
	find . -name __pycache__ -type d -exec rm -rf {} +
