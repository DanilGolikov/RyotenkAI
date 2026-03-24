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
        echo "Starting MLflow stack..."
        docker compose --env-file "$ENV_FILE" -f "$COMPOSE_FILE" up -d
        echo ""
        echo "Services:"
        echo "  MLflow UI:      http://localhost:${MLFLOW_PORT:-5002}"
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
        docker compose --env-file "$ENV_FILE" -f "$COMPOSE_FILE" up -d
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
