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
check: lint typecheck

.PHONY: test
test: ## Unit tests
	$(UV) run pytest -m "not integration"

.PHONY: eval
eval: ## Run the eval harness on the bundled golden set
	$(UV) run voice-eval run --out evals/REPORT.md --json evals/scores.json

.PHONY: eval-canonical
eval-canonical: ## Regenerate the tracked canonical report + WER-injection variant
	$(UV) run voice-eval run --out evals/REPORT.md --json evals/scores.json
	$(UV) run voice-eval run --wer-substitution-rate 0.1 \
		--out evals/REPORT.with-wer-injection.md

.PHONY: clean
clean: ## Wipe caches
	rm -rf .pytest_cache .mypy_cache .ruff_cache .coverage htmlcov build dist
	find . -name __pycache__ -type d -exec rm -rf {} +
