#!/usr/bin/env sh
set -eu

if [ -z "${BACKEND_STORE_URI:-}" ]; then
    echo "BACKEND_STORE_URI is required" >&2
    exit 1
fi

if [ -z "${ARTIFACTS_DESTINATION:-}" ]; then
    echo "ARTIFACTS_DESTINATION is required" >&2
    exit 1
fi

set -- \
    mlflow server \
    --host 0.0.0.0 \
    --port 5000 \
    --backend-store-uri "${BACKEND_STORE_URI}" \
    --serve-artifacts \
    --artifacts-destination "${ARTIFACTS_DESTINATION}"

if [ -n "${MLFLOW_APP_NAME:-}" ]; then
    set -- "$@" --app-name "${MLFLOW_APP_NAME}"
fi

# --allowed-hosts is only set when MLFLOW_SERVER_ALLOWED_HOSTS is explicitly provided
# (e.g. by expose-tailscale.sh for public access). Without it MLflow accepts all hosts,
# which is the correct default for local-only usage.
if [ -n "${MLFLOW_SERVER_ALLOWED_HOSTS:-}" ]; then
    HOST_PORT="${MLFLOW_HOST_PORT:-5000}"
    LOCAL_HOSTS="localhost,localhost:5000,localhost:${HOST_PORT},127.0.0.1,127.0.0.1:5000,127.0.0.1:${HOST_PORT}"
    set -- "$@" --allowed-hosts "${LOCAL_HOSTS},${MLFLOW_SERVER_ALLOWED_HOSTS}"
fi

if [ -n "${MLFLOW_SERVER_CORS_ALLOWED_ORIGINS:-}" ]; then
    set -- "$@" --cors-allowed-origins "${MLFLOW_SERVER_CORS_ALLOWED_ORIGINS}"
fi

exec "$@"
