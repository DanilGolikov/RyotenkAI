.PHONY: help setup install-hooks test test-fast test-unit test-cov lint format fix-all pre-commit clean info validate docker-mlflow-up docker-mlflow-down web-install web-build web-start web-stop web-restart web-status web-logs web-backend-start web-backend-stop web-backend-restart web-frontend-start web-frontend-stop web-frontend-restart web-openapi-dump gen-api verify-api-sync _check-venv

# Pin all Python tooling to the project-local venv so `make` works regardless
# of which venv is active in the shell. Override with e.g. `make VENV=.venv2`.
VENV       := .venv
VENV_BIN   := $(VENV)/bin
PYTHON     := $(VENV_BIN)/python
PIP        := $(PYTHON) -m pip
PYTEST     := $(PYTHON) -m pytest
RUFF       := $(PYTHON) -m ruff
MYPY       := $(PYTHON) -m mypy
BLACK      := $(PYTHON) -m black
PRECOMMIT  := $(VENV_BIN)/pre-commit
RYOTENKAI  := $(VENV_BIN)/ryotenkai

# Fail fast with a clear message if the venv is missing.
_check-venv:
	@test -x "$(PYTHON)" || { \
		echo "error: $(PYTHON) not found. Create the venv first:"; \
		echo "  python3 -m venv $(VENV) && $(PIP) install -e \".[dev]\""; \
		exit 1; \
	}

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
	@echo "  make validate CONFIG=path  - Validate config"
	@echo "  make smoke DIR=path        - Batch smoke (parallel) over a config dir"
	@echo ""
	@echo "Infrastructure:"
	@echo "  make docker-mlflow-up   - Start MLflow stack"
	@echo "  make docker-mlflow-down - Stop MLflow stack"
	@echo ""
	@echo "Web UI:"
	@echo "  make web-install        - Install frontend npm deps"
	@echo "  make web-build          - Build frontend (web/dist)"
	@echo "  make web-start          - Start backend + frontend (detached)"
	@echo "  make web-stop           - Stop backend + frontend"
	@echo "  make web-restart        - Restart backend + frontend"
	@echo "  make web-status         - Show backend + frontend status"
	@echo "  make web-logs           - Tail backend + frontend logs"
	@echo "  make web-backend-start  - Start only backend"
	@echo "  make web-frontend-start - Start only frontend"
	@echo ""
	@echo "Utilities:"
	@echo "  make clean          - Clean temp files"
	@echo "  make info           - Project info"

# ============================================
# Setup
# ============================================

setup:
	@test -x "$(PYTHON)" || python3 -m venv $(VENV)
	$(PIP) install --upgrade pip
	$(PIP) install -e ".[dev]"
	@echo "Setup complete"

install-hooks: _check-venv
	$(PIP) install pre-commit
	$(PRECOMMIT) install
	$(PRECOMMIT) install --hook-type commit-msg
	@echo "Hooks installed"

# ============================================
# Testing
# ============================================

test: _check-venv
	$(PYTEST)

test-fast: _check-venv
	$(PYTEST) -m "not slow"

test-unit: _check-venv
	$(PYTEST) src/tests/unit -v

test-cov: _check-venv
	$(PYTEST) --cov=src --cov-report=html:src/tests/coverage/htmlcov --cov-report=term-missing
	@echo "Report: src/tests/coverage/htmlcov/index.html"

# ============================================
# Code Quality
# ============================================

lint: _check-venv
	@$(RUFF) check src/ || true
	@$(RUFF) format --check src/ || true
	@$(MYPY) src/ --config-file pyproject.toml || true

format: _check-venv
	$(RUFF) check --fix src/
	$(RUFF) format src/
	$(BLACK) src/

fix-all: _check-venv
	@echo "Auto-fixing..."
	$(RUFF) check --fix --unsafe-fixes src/ || true
	$(RUFF) format src/
	$(BLACK) src/
	@echo "Done. Run 'make lint' to verify"

pre-commit: _check-venv
	$(PRECOMMIT) run --all-files

# ============================================
# Pipeline
# ============================================

validate: _check-venv
	$(RYOTENKAI) config validate --config $(CONFIG)

smoke: _check-venv
	$(RYOTENKAI) smoke $(DIR)

# ============================================
# Infrastructure
# ============================================

docker-mlflow-up:
	bash docker/mlflow/start.sh up

docker-mlflow-down:
	bash docker/mlflow/start.sh stop

# ============================================
# Web UI (FastAPI backend + React/Vite frontend)
# ============================================
# All commands are delegated to web/scripts/*.sh so the same logic works when
# invoked from outside make (e.g. shell aliases, IDE run configs).

web-install:
	cd web && npm install

web-build:
	cd web && npm run build

web-start:
	bash web/scripts/start.sh all

web-stop:
	bash web/scripts/stop.sh all

web-restart:
	bash web/scripts/restart.sh all

web-status:
	bash web/scripts/status.sh

web-logs:
	@echo "--- backend (web/.run/backend.log) ---"
	@tail -n 50 web/.run/backend.log 2>/dev/null || echo "(not running)"
	@echo ""
	@echo "--- frontend (web/.run/frontend.log) ---"
	@tail -n 50 web/.run/frontend.log 2>/dev/null || echo "(not running)"

web-backend-start:
	bash web/scripts/start.sh backend

web-backend-stop:
	bash web/scripts/stop.sh backend

web-backend-restart:
	bash web/scripts/restart.sh backend

web-frontend-start:
	bash web/scripts/start.sh frontend

web-frontend-stop:
	bash web/scripts/stop.sh frontend

web-frontend-restart:
	bash web/scripts/restart.sh frontend

# ============================================
# API codegen (OpenAPI → TypeScript)
# ============================================

# Dump the live FastAPI OpenAPI spec to a checked-in snapshot so codegen
# is offline-safe and CI can fail on drift without a running backend.
web-openapi-dump:
	$(PYTHON) -m src.api.openapi_dump > web/src/api/openapi.json

# Regenerate web/src/api/schema.d.ts from the checked-in spec.
gen-api: web-openapi-dump
	cd web && npm run gen:api

# CI drift guard: regenerate and fail if the working tree differs. Commit
# the updated openapi.json + schema.d.ts alongside your backend changes.
verify-api-sync: gen-api
	git diff --exit-code -- web/src/api/openapi.json web/src/api/schema.d.ts

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
