#!/usr/bin/env bash
# ==============================================================================
# Start MLflow Stack (PostgreSQL + MinIO + MLflow Server)
# ==============================================================================
#
# Usage:
#   ./start.sh          # Start all services
#   ./start.sh stop     # Stop all services
#   ./start.sh restart  # Restart all services
#   ./start.sh logs     # Follow logs
#   ./start.sh status   # Show service status
#
# ==============================================================================

set -euo pipefail

cd "$(dirname "$0")" || exit 1

COMPOSE_FILE="docker-compose.mlflow.yml"
ENV_FILE=".env.mlflow"

if [[ -f "$ENV_FILE" ]]; then
    set -a
    # shellcheck disable=SC1090
    source "$ENV_FILE"
    set +a
fi

if [[ ! -f "$COMPOSE_FILE" ]]; then
    echo "Error: $COMPOSE_FILE not found"
    exit 1
fi

ACTION="${1:-up}"

case "$ACTION" in
    -h|--help|help)
        head -n 15 "$0" | grep "^#" | sed 's/^# \?//'
        exit 0
        ;;
    up|start)
        # Phase M6: Caddy basic-auth is mandatory for the stack to come up.
        if [[ -z "${CADDY_BASIC_AUTH_USER:-}" ]] || [[ -z "${CADDY_BASIC_AUTH_HASH:-}" ]]; then
            echo "Error: CADDY_BASIC_AUTH_USER and CADDY_BASIC_AUTH_HASH must be set in ${ENV_FILE}."
            echo ""
            echo "Generate a bcrypt hash for your password with:"
            echo "  docker run --rm caddy:2-alpine caddy hash-password --plaintext 'YOUR_PASSWORD'"
            echo ""
            echo "Then append to ${ENV_FILE}:"
            echo "  CADDY_BASIC_AUTH_USER=mlflow_user"
            echo "  CADDY_BASIC_AUTH_HASH=<paste-the-hash-output-above>"
            exit 1
        fi

        echo "Starting MLflow stack..."
        docker compose --env-file "$ENV_FILE" -f "$COMPOSE_FILE" up -d --build
        echo ""
        echo "Services:"
        echo "  MLflow UI:      http://localhost:${MLFLOW_PORT:-5002}   (basic-auth via Caddy)"
        echo "  MinIO Console:  http://localhost:${MINIO_CONSOLE_PORT:-9001}"
        echo "  PostgreSQL:     localhost:${POSTGRES_PORT:-5432}"
        ;;
    down|stop)
        echo "Stopping MLflow stack..."
        docker compose --env-file "$ENV_FILE" -f "$COMPOSE_FILE" down
        ;;
    restart)
        echo "Restarting MLflow stack..."
        docker compose --env-file "$ENV_FILE" -f "$COMPOSE_FILE" down
        docker compose --env-file "$ENV_FILE" -f "$COMPOSE_FILE" up -d --build
        ;;
    logs)
        docker compose --env-file "$ENV_FILE" -f "$COMPOSE_FILE" logs -f
        ;;
    status|ps)
        docker compose --env-file "$ENV_FILE" -f "$COMPOSE_FILE" ps
        ;;
    *)
        echo "Usage: $0 {up|stop|restart|logs|status}"
        exit 1
        ;;
esac
