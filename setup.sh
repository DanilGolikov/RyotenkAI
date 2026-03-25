#!/usr/bin/env bash
set -euo pipefail

MIN_PYTHON="3.12"
VENV_DIR=".venv"

echo "=== RyotenkAI Setup ==="
echo ""

# --- Check Python version ---
if ! command -v python3 &>/dev/null; then
    echo "ERROR: python3 not found. Install Python >= $MIN_PYTHON"
    exit 1
fi

PY_VERSION=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
PY_MAJOR=$(echo "$PY_VERSION" | cut -d. -f1)
PY_MINOR=$(echo "$PY_VERSION" | cut -d. -f2)
MIN_MAJOR=$(echo "$MIN_PYTHON" | cut -d. -f1)
MIN_MINOR=$(echo "$MIN_PYTHON" | cut -d. -f2)

if [ "$PY_MAJOR" -lt "$MIN_MAJOR" ] || { [ "$PY_MAJOR" -eq "$MIN_MAJOR" ] && [ "$PY_MINOR" -lt "$MIN_MINOR" ]; }; then
    echo "ERROR: Python >= $MIN_PYTHON required, found $PY_VERSION"
    exit 1
fi
echo "[OK] Python $PY_VERSION"

# --- Create virtual environment ---
if [ ! -d "$VENV_DIR" ]; then
    echo "Creating virtual environment in $VENV_DIR ..."
    python3 -m venv "$VENV_DIR"
    echo "[OK] Virtual environment created"
else
    echo "[OK] Virtual environment already exists"
fi

# --- Activate venv ---
# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"

# --- Install dependencies ---
echo "Installing dependencies ..."
pip install --upgrade pip --quiet
pip install -e ".[dev]" --quiet
echo "[OK] Dependencies installed"

# --- Install pre-commit hooks ---
if command -v pre-commit &>/dev/null; then
    pre-commit install --install-hooks
    echo "[OK] Pre-commit hooks installed"
else
    echo "[WARN] pre-commit not found, skipping hook installation"
fi

# --- Copy .env.example if secrets.env does not exist ---
if [ ! -f "secrets.env" ] && [ -f ".env.example" ]; then
    cp .env.example secrets.env
    echo "[OK] Created secrets.env from .env.example — fill in your API keys"
elif [ -f "secrets.env" ]; then
    echo "[OK] secrets.env already exists"
fi

# --- MLflow stack (PostgreSQL + MinIO + MLflow UI) ---
MLFLOW_START="docker/mlflow/start.sh"
if command -v docker &>/dev/null && docker info &>/dev/null; then
    if docker compose version &>/dev/null || docker-compose version &>/dev/null; then
        if [ -f "$MLFLOW_START" ]; then
            echo ""
            echo "Starting MLflow stack ..."
            if bash "$MLFLOW_START" up; then
                echo "[OK] MLflow stack is up (see URLs below)"
            else
                echo "[WARN] MLflow stack did not start — fix Docker or docker/mlflow/.env.mlflow, then run:  make docker-mlflow-up"
            fi
        fi
    else
        echo "[WARN] Docker Compose plugin not found — MLflow not started. Install it, then run:  make docker-mlflow-up"
    fi
else
    echo "[WARN] Docker is not running or not installed — MLflow stack skipped. When ready:  make docker-mlflow-up"
fi

echo ""
echo "=== Setup complete ==="
echo ""
echo "Next steps:"
echo "  1. Activate the venv:  source $VENV_DIR/bin/activate"
echo "  2. Fill in secrets:    edit secrets.env with your API keys"
echo "  3. MLflow:             http://localhost:5002  (if stack started; set passwords in docker/mlflow/.env.mlflow)"
echo "  4. Try the CLI:        ryotenkai --help"
echo "  5. Run tests:          make test"
echo "  6. Launch TUI:         ryotenkai tui"
