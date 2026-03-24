.PHONY: help setup install-hooks test test-fast test-unit test-cov lint format fix-all pre-commit clean info tui validate docker-mlflow-up docker-mlflow-down

PYTHON := python3

# ============================================
# Help
# ============================================

help:
	@echo "RyotenkAI - Automated CI/CD for LLM Training"
	@echo ""
	@echo "Setup:"
	@echo "  make setup          - Install dependencies"
	@echo "  make install-hooks  - Install pre-commit hooks"
	@echo ""
	@echo "Code Quality:"
	@echo "  make fix-all        - Auto-fix all issues"
	@echo "  make format         - Format code"
	@echo "  make lint           - Run linters"
	@echo "  make pre-commit     - Run pre-commit on all files"
	@echo ""
	@echo "Testing:"
	@echo "  make test           - Run all tests"
	@echo "  make test-fast      - Skip slow tests"
	@echo "  make test-unit      - Unit tests only"
	@echo "  make test-cov       - With coverage"
	@echo ""
	@echo "Pipeline:"
	@echo "  make tui            - Launch interactive TUI"
	@echo "  make validate CONFIG=path  - Validate config"
	@echo ""
	@echo "Infrastructure:"
	@echo "  make docker-mlflow-up   - Start MLflow stack"
	@echo "  make docker-mlflow-down - Stop MLflow stack"
	@echo ""
	@echo "Utilities:"
	@echo "  make clean          - Clean temp files"
	@echo "  make info           - Project info"

# ============================================
# Setup
# ============================================

setup:
	pip install --upgrade pip
	pip install -e ".[dev]"
	@echo "Setup complete"

install-hooks:
	pip install pre-commit
	pre-commit install
	pre-commit install --hook-type commit-msg
	@echo "Hooks installed"

# ============================================
# Testing
# ============================================

test:
	pytest

test-fast:
	pytest -m "not slow"

test-unit:
	pytest src/tests/unit -v

test-cov:
	pytest --cov=src --cov-report=html:src/tests/coverage/htmlcov --cov-report=term-missing
	@echo "Report: src/tests/coverage/htmlcov/index.html"

# ============================================
# Code Quality
# ============================================

lint:
	@ruff check src/ || true
	@ruff format --check src/ || true
	@mypy src/ --config-file pyproject.toml || true

format:
	ruff check --fix src/
	ruff format src/
	black src/

fix-all:
	@echo "Auto-fixing..."
	ruff check --fix --unsafe-fixes src/ || true
	ruff format src/
	black src/
	@echo "Done. Run 'make lint' to verify"

pre-commit:
	pre-commit run --all-files

# ============================================
# Pipeline
# ============================================

tui:
	ryotenkai tui

validate:
	ryotenkai config-validate --config $(CONFIG)

# ============================================
# Infrastructure
# ============================================

docker-mlflow-up:
	bash docker/mlflow/start.sh up

docker-mlflow-down:
	bash docker/mlflow/start.sh stop

# ============================================
# Utilities
# ============================================

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".mypy_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".ruff_cache" -exec rm -rf {} + 2>/dev/null || true
	rm -rf htmlcov/ .coverage build/ dist/ .test_venv/ mlflow.db src/tests/coverage/ 2>/dev/null || true
	@echo "Cleaned"

info:
	@echo "Project: ryotenkai v1.0.0"
	@echo "Python: $$($(PYTHON) --version 2>/dev/null)"
	@echo ""
	@echo "Config example: src/config/pipeline_config.yaml"
	@echo "Config ref:     src/config/CONFIG_REFERENCE.md"
