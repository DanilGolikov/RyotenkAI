#!/usr/bin/env sh
set -eu

# When the server is exposed publicly (allowed-hosts set), the Flask session
# secret MUST be set explicitly — an empty/default secret allows session
# forgery and CSRF token bypass. Refuse to start in that combination.
if [ -n "${MLFLOW_SERVER_ALLOWED_HOSTS:-}" ] && [ -z "${MLFLOW_FLASK_SERVER_SECRET_KEY:-}" ]; then
    echo "FATAL: MLFLOW_SERVER_ALLOWED_HOSTS is set but MLFLOW_FLASK_SERVER_SECRET_KEY is empty." >&2
    echo "Public exposure without a session secret allows session forgery." >&2
    echo "Set MLFLOW_FLASK_SERVER_SECRET_KEY in .env.mlflow before going public." >&2
    exit 1
fi

if [ -z "${BACKEND_STORE_URI:-}" ]; then
    echo "BACKEND_STORE_URI is required" >&2
    exit 1
fi

if [ -z "${ARTIFACTS_DESTINATION:-}" ]; then
    echo "ARTIFACTS_DESTINATION is required" >&2
    exit 1
fi

# Internal listen port. Phase M6 moved MLflow off the host-visible 5002 to
# 5102 (internal-only); external traffic now goes through mlflow_caddy which
# enforces basic-auth.
INTERNAL_PORT="${MLFLOW_INTERNAL_PORT:-5102}"

# Apply any pending schema migrations before starting the server.
# Idempotent: no-op when the schema is already current. Required when the
# server image is bumped across versions that introduced new schema
# revisions (e.g. 3.8.x -> 3.11.x).
echo "Running mlflow db upgrade..."
mlflow db upgrade "${BACKEND_STORE_URI}"

set -- \
    mlflow server \
    --host 0.0.0.0 \
    --port "${INTERNAL_PORT}" \
    --backend-store-uri "${BACKEND_STORE_URI}" \
    --serve-artifacts \
    --artifacts-destination "${ARTIFACTS_DESTINATION}"

if [ -n "${MLFLOW_APP_NAME:-}" ]; then
    set -- "$@" --app-name "${MLFLOW_APP_NAME}"
fi

# --allowed-hosts is only set when MLFLOW_SERVER_ALLOWED_HOSTS is explicitly
# provided (e.g. by expose-tailscale.sh for public access). Without it MLflow
# accepts all hosts, which is the correct default for local-only usage.
if [ -n "${MLFLOW_SERVER_ALLOWED_HOSTS:-}" ]; then
    HOST_PORT="${MLFLOW_HOST_PORT:-5002}"
    LOCAL_HOSTS="localhost,localhost:${INTERNAL_PORT},localhost:${HOST_PORT},127.0.0.1,127.0.0.1:${INTERNAL_PORT},127.0.0.1:${HOST_PORT}"
    set -- "$@" --allowed-hosts "${LOCAL_HOSTS},${MLFLOW_SERVER_ALLOWED_HOSTS}"
fi

if [ -n "${MLFLOW_SERVER_CORS_ALLOWED_ORIGINS:-}" ]; then
    set -- "$@" --cors-allowed-origins "${MLFLOW_SERVER_CORS_ALLOWED_ORIGINS}"
fi

exec "$@"
