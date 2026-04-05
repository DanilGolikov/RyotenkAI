#!/usr/bin/env bash
# ==============================================================================
# Publish local MLflow over Tailscale Funnel
# ==============================================================================
#
# Usage:
#   ./expose-tailscale.sh up       # Start/recreate MLflow and expose it publicly
#   ./expose-tailscale.sh down     # Disable Funnel and stop the managed daemon
#   ./expose-tailscale.sh status   # Show Funnel and local MLflow status
#   ./expose-tailscale.sh url      # Print the public MLflow URL
#   ./expose-tailscale.sh help
#
# Notes:
# - Requires Tailscale CLI to be installed and authenticated (`tailscale up`)
# - Exposes only MLflow (`localhost:${MLFLOW_PORT:-5002}`), not MinIO
# - Uses HTTPS port 443 by default
# - Asks for confirmation before destructive/elevated actions
# - Prefers the system Tailscale daemon; falls back to rootless
# - If rootless fails for Funnel, offers automatic escalation to system daemon
#
# ==============================================================================

set -euo pipefail

cd "$(dirname "$0")" || exit 1

COMPOSE_FILE="docker-compose.mlflow.yml"
ENV_FILE=".env.mlflow"
LOCAL_HOST="127.0.0.1"
TAILSCALE_STATE_DIR="${TAILSCALE_STATE_DIR:-${XDG_STATE_HOME:-$HOME/.local/state}/ryotenkai-mlflow-tailscale}"
TAILSCALE_SOCKET="${TAILSCALE_SOCKET:-${TAILSCALE_STATE_DIR}/tailscaled.sock}"
TAILSCALE_STATE_FILE="${TAILSCALE_STATE_FILE:-${TAILSCALE_STATE_DIR}/tailscaled.state}"
TAILSCALE_LOG_FILE="${TAILSCALE_LOG_FILE:-${TAILSCALE_STATE_DIR}/tailscaled.log}"
TAILSCALE_PID_FILE="${TAILSCALE_PID_FILE:-${TAILSCALE_STATE_DIR}/tailscaled.pid}"
TAILSCALE_MODE_FILE="${TAILSCALE_STATE_DIR}/daemon_mode"

if [[ ! -f "$ENV_FILE" ]]; then
    echo "Error: $ENV_FILE not found. It should be in $(pwd)."
    exit 1
fi

set -a
# shellcheck disable=SC1090
source "$ENV_FILE"
set +a

MLFLOW_PORT="${MLFLOW_PORT:-5002}"
TAILSCALE_FUNNEL_HTTPS_PORT="${TAILSCALE_FUNNEL_HTTPS_PORT:-443}"
LOCAL_TARGET="http://${LOCAL_HOST}:${MLFLOW_PORT}"
PUBLIC_HEALTH_PATH="${PUBLIC_HEALTH_PATH:-/health}"

usage() {
    sed -n '2,15p' "$0" | sed 's/^# \{0,1\}//'
}

confirm_or_exit() {
    local prompt="$1"
    local reply=""

    if [[ "${AUTO_APPROVE:-0}" == "1" ]]; then
        return 0
    fi

    read -r -p "$prompt [y/N] " reply
    case "$reply" in
        y|Y|yes|YES)
            return 0
            ;;
        *)
            echo "Cancelled."
            exit 1
            ;;
    esac
}

confirm_or_skip() {
    local prompt="$1"
    local reply=""

    if [[ "${AUTO_APPROVE:-0}" == "1" ]]; then
        return 0
    fi

    read -r -p "$prompt [y/N] " reply
    case "$reply" in
        y|Y|yes|YES) return 0 ;;
        *) return 1 ;;
    esac
}

require_command() {
    local cmd="$1"
    local install_url="$2"

    if ! command -v "$cmd" >/dev/null 2>&1; then
        echo "Error: '$cmd' is not installed."
        echo "Install it first: $install_url"
        exit 1
    fi
}

# ---------------------------------------------------------------------------
# Tailscale daemon management
# ---------------------------------------------------------------------------

tailscale_cli() {
    if [[ "${TAILSCALE_USE_SYSTEM_DAEMON:-0}" == "1" ]]; then
        TAILSCALE_BE_CLI=1 tailscale "$@"
    else
        TAILSCALE_BE_CLI=1 tailscale --socket="$TAILSCALE_SOCKET" "$@"
    fi
}

tailscale_status_json() {
    tailscale_cli status --json
}

detect_system_tailscale() {
    TAILSCALE_BE_CLI=1 tailscale status --json >/dev/null 2>&1
}

local_tailscaled_running() {
    TAILSCALE_BE_CLI=1 tailscale --socket="$TAILSCALE_SOCKET" status --json >/dev/null 2>&1
}

cleanup_rootless_daemons() {
    local killed=0

    if [[ -f "$TAILSCALE_PID_FILE" ]]; then
        local stored_pid
        stored_pid="$(cat "$TAILSCALE_PID_FILE" 2>/dev/null || true)"
        if [[ -n "$stored_pid" ]] && kill -0 "$stored_pid" 2>/dev/null; then
            kill "$stored_pid" 2>/dev/null || true
            killed=$((killed + 1))
        fi
        rm -f "$TAILSCALE_PID_FILE"
    fi

    local state_dir_basename
    state_dir_basename="$(basename "$TAILSCALE_STATE_DIR")"
    local extra_pids
    extra_pids="$(pgrep -f "tailscaled.*${state_dir_basename}" 2>/dev/null || true)"
    for pid in $extra_pids; do
        if kill -0 "$pid" 2>/dev/null; then
            kill "$pid" 2>/dev/null || true
            killed=$((killed + 1))
        fi
    done

    rm -f "$TAILSCALE_SOCKET"

    if [[ "$killed" -gt 0 ]]; then
        echo "Stopped $killed rootless tailscaled process(es)."
        sleep 2
    fi
}

start_local_tailscaled() {
    require_command "tailscaled" "https://tailscale.com/download"

    cleanup_rootless_daemons

    mkdir -p "$TAILSCALE_STATE_DIR"

    echo "Starting rootless Tailscale daemon (userspace networking)..."
    nohup tailscaled \
        --tun=userspace-networking \
        --socket="$TAILSCALE_SOCKET" \
        --state="$TAILSCALE_STATE_FILE" \
        --statedir="$TAILSCALE_STATE_DIR" \
        >"$TAILSCALE_LOG_FILE" 2>&1 &
    echo $! >"$TAILSCALE_PID_FILE"
    echo "rootless" >"$TAILSCALE_MODE_FILE"

    for _ in {1..20}; do
        if local_tailscaled_running; then
            return 0
        fi
        sleep 1
    done

    echo "Error: rootless Tailscale daemon did not become ready."
    echo "Check log: $TAILSCALE_LOG_FILE"
    exit 1
}

start_sudo_tailscaled() {
    require_command "tailscaled" "https://tailscale.com/download"

    cleanup_rootless_daemons

    mkdir -p "$TAILSCALE_STATE_DIR"

    echo "Starting tailscaled with kernel networking (sudo required)..."
    sudo tailscaled \
        --socket="$TAILSCALE_SOCKET" \
        --state="$TAILSCALE_STATE_FILE" \
        --statedir="$TAILSCALE_STATE_DIR" \
        >"$TAILSCALE_LOG_FILE" 2>&1 &
    echo $! >"$TAILSCALE_PID_FILE"
    echo "sudo" >"$TAILSCALE_MODE_FILE"

    for _ in {1..20}; do
        if local_tailscaled_running; then
            TAILSCALE_USE_SYSTEM_DAEMON=0
            return 0
        fi
        sleep 1
    done

    echo "Error: system tailscaled did not start."
    echo "Check log: $TAILSCALE_LOG_FILE"
    exit 1
}

select_tailscale_backend() {
    if detect_system_tailscale; then
        TAILSCALE_USE_SYSTEM_DAEMON=1
        echo "Using system Tailscale daemon."
        return 0
    fi

    TAILSCALE_USE_SYSTEM_DAEMON=0

    if local_tailscaled_running; then
        local dup_count
        local state_dir_basename
        state_dir_basename="$(basename "$TAILSCALE_STATE_DIR")"
        dup_count="$(pgrep -fc "tailscaled.*${state_dir_basename}" 2>/dev/null || echo "0")"
        if [[ "$dup_count" -gt 1 ]]; then
            echo "Warning: found $dup_count rootless tailscaled processes (expected 1)."
            echo "Cleaning up duplicates..."
            cleanup_rootless_daemons
            start_local_tailscaled
        else
            echo "Using existing rootless Tailscale daemon."
        fi
        return 0
    fi

    start_local_tailscaled
}

# ---------------------------------------------------------------------------
# MLflow helpers
# ---------------------------------------------------------------------------

get_backend_state() {
    tailscale_status_json | python3 -c 'import json, sys; print(json.load(sys.stdin).get("BackendState", ""))'
}

ensure_tailscale_login() {
    local state
    state="$(get_backend_state)"

    if [[ "$state" == "Running" ]]; then
        return 0
    fi

    echo "Tailscale state: ${state:-unknown}"
    echo "A login URL may open in the terminal. Complete authentication in your browser, then return here."
    tailscale_cli up
}

check_local_mlflow() {
    python3 - "$LOCAL_TARGET/health" <<'PY'
import sys
import urllib.request

url = sys.argv[1]
try:
    with urllib.request.urlopen(url, timeout=2) as response:
        ok = 200 <= response.status < 300
    sys.exit(0 if ok else 1)
except Exception:
    sys.exit(1)
PY
}

wait_for_local_mlflow() {
    local attempts=30

    for ((i = 1; i <= attempts; i++)); do
        if check_local_mlflow; then
            return 0
        fi
        sleep 2
    done

    echo "Error: MLflow is not healthy at ${LOCAL_TARGET} after waiting."
    echo "Try './start.sh logs' for details."
    exit 1
}

get_tailnet_dns_name() {
    tailscale_status_json | python3 -c '
import json
import sys

data = json.load(sys.stdin)

def visit(node):
    if isinstance(node, dict):
        for value in node.values():
            found = visit(value)
            if found:
                return found
    elif isinstance(node, list):
        for value in node:
            found = visit(value)
            if found:
                return found
    elif isinstance(node, str):
        candidate = node.rstrip(".")
        if candidate.endswith(".ts.net"):
            return candidate
    return None

self_node = data.get("Self")
if isinstance(self_node, dict):
    for key in ("DNSName", "HostName", "Hostname", "Name"):
        value = self_node.get(key)
        if isinstance(value, str):
            candidate = value.rstrip(".")
            if candidate.endswith(".ts.net"):
                print(candidate)
                sys.exit(0)

candidate = visit(data)
if candidate:
    print(candidate)
    sys.exit(0)

sys.exit(1)
'
}

get_public_url() {
    local dns_name="$1"
    if [[ "$TAILSCALE_FUNNEL_HTTPS_PORT" == "443" ]]; then
        printf 'https://%s\n' "$dns_name"
    else
        printf 'https://%s:%s\n' "$dns_name" "$TAILSCALE_FUNNEL_HTTPS_PORT"
    fi
}

ensure_stack_running_for_host() {
    local allowed_hosts="$1"

    if check_local_mlflow; then
        echo "MLflow is already healthy locally; recreating with updated allowed-hosts..."
    else
        echo "Ensuring MLflow stack is running..."
    fi

    MLFLOW_SERVER_ALLOWED_HOSTS="$allowed_hosts" \
        docker compose --env-file "$ENV_FILE" -f "$COMPOSE_FILE" up -d
    wait_for_local_mlflow
}

check_public_mlflow() {
    local public_url="$1"
    curl --silent --show-error --fail --max-time 10 "${public_url%/}${PUBLIC_HEALTH_PATH}" >/dev/null 2>&1
}

# ---------------------------------------------------------------------------
# Funnel verification & progressive recovery
# ---------------------------------------------------------------------------

try_funnel_and_verify() {
    local public_url="$1"
    local max_attempts="${2:-15}"

    echo "Waiting for public HTTPS endpoint..."
    for ((i = 1; i <= max_attempts; i++)); do
        if check_public_mlflow "$public_url"; then
            return 0
        fi
        sleep 2
    done
    return 1
}

setup_funnel() {
    local public_url="$1"
    echo "Publishing MLflow via Tailscale Funnel..."
    tailscale_cli funnel --yes --bg --https="${TAILSCALE_FUNNEL_HTTPS_PORT}" "${LOCAL_TARGET}"
}

print_success() {
    local public_url="$1"
    echo ""
    echo "============================================"
    echo "  MLflow is publicly available!"
    echo "============================================"
    echo ""
    echo "  Public URL:    ${public_url}"
    echo "  Tracking URI:  ${public_url}"
    echo ""
    echo "  Use on a remote machine:"
    echo "    export MLFLOW_TRACKING_URI=${public_url}"
    echo ""
    echo "${public_url}"
}

do_up() {
    require_command "docker" "https://docs.docker.com/get-started/get-docker/"
    require_command "tailscale" "https://tailscale.com/download"
    confirm_or_exit "This will start Tailscale, may rebuild the MLflow stack, and publish MLflow on the public internet. Continue?"

    select_tailscale_backend
    ensure_tailscale_login

    if ! dns_name="$(get_tailnet_dns_name)"; then
        echo "Error: could not determine the Tailscale DNS name after login."
        exit 1
    fi

    local public_url
    public_url="$(get_public_url "$dns_name")"
    local allowed_hosts="${dns_name},${dns_name}:${TAILSCALE_FUNNEL_HTTPS_PORT}"

    ensure_stack_running_for_host "$allowed_hosts"

    # --- Attempt 1: current backend ---
    setup_funnel "$public_url"

    if try_funnel_and_verify "$public_url" 15; then
        print_success "$public_url"
        return 0
    fi

    # --- Diagnosis ---
    echo ""
    echo "Public endpoint not responding: ${public_url}"
    echo ""

    if [[ "${TAILSCALE_USE_SYSTEM_DAEMON:-0}" == "1" ]]; then
        echo "System Tailscale daemon is running, but Funnel traffic is not arriving."
        echo "Possible causes:"
        echo "  - Funnel is not enabled on your tailnet (enable at https://login.tailscale.com/admin/dns)"
        echo "  - Firewall blocking port ${TAILSCALE_FUNNEL_HTTPS_PORT}"
        echo "  - DNS propagation delay"
        echo ""
        echo "Check: tailscale funnel status"
        echo "Check: curl -v ${public_url}/health"
        exit 1
    fi

    # --- Attempt 2: clean restart of rootless daemon ---
    echo "Current backend: rootless (userspace networking)"
    echo "Funnel may not work reliably in this mode."
    echo ""

    if confirm_or_skip "Perform clean restart of Tailscale daemon and retry?"; then
        cleanup_rootless_daemons
        start_local_tailscaled
        ensure_tailscale_login
        setup_funnel "$public_url"

        if try_funnel_and_verify "$public_url" 15; then
            print_success "$public_url"
            return 0
        fi

        echo ""
        echo "Rootless daemon still cannot serve Funnel traffic."
    fi

    # --- Attempt 3: escalate to sudo (kernel networking) ---
    echo ""
    echo "Tailscale Funnel requires the daemon to accept incoming TLS connections."
    echo "With kernel networking (requires sudo), Funnel works more reliably."
    echo ""

    if confirm_or_skip "Switch to system-level Tailscale daemon (requires sudo password)?"; then
        start_sudo_tailscaled
        ensure_tailscale_login
        setup_funnel "$public_url"

        if try_funnel_and_verify "$public_url" 20; then
            print_success "$public_url"
            return 0
        fi

        echo ""
        echo "Error: Funnel still not working even with system daemon."
    fi

    # --- All attempts failed ---
    echo ""
    echo "Could not make Funnel endpoint reachable."
    echo ""
    echo "Manual debugging:"
    echo "  tailscale funnel status"
    echo "  curl -v ${public_url}/health"
    echo "  Check Funnel is enabled: https://login.tailscale.com/admin/dns"
    exit 1
}

do_down() {
    require_command "tailscale" "https://tailscale.com/download"
    confirm_or_exit "Disable public Tailscale Funnel access for MLflow?"

    if detect_system_tailscale; then
        TAILSCALE_USE_SYSTEM_DAEMON=1
    elif local_tailscaled_running; then
        TAILSCALE_USE_SYSTEM_DAEMON=0
    else
        echo "No Tailscale daemon running. Nothing to disable."
        return 0
    fi

    echo "Disabling Tailscale Funnel for MLflow..."
    tailscale_cli funnel --https="${TAILSCALE_FUNNEL_HTTPS_PORT}" "${LOCAL_TARGET}" off 2>/dev/null || true

    local daemon_mode=""
    if [[ -f "$TAILSCALE_MODE_FILE" ]]; then
        daemon_mode="$(cat "$TAILSCALE_MODE_FILE" 2>/dev/null || true)"
    fi

    if [[ "${TAILSCALE_USE_SYSTEM_DAEMON:-0}" != "1" ]] || [[ "$daemon_mode" == "rootless" ]] || [[ "$daemon_mode" == "sudo" ]]; then
        local state_dir_basename
        state_dir_basename="$(basename "$TAILSCALE_STATE_DIR")"
        local daemon_count
        daemon_count="$(pgrep -fc "tailscaled.*${state_dir_basename}" 2>/dev/null || echo "0")"

        if [[ "$daemon_count" -gt 0 ]]; then
            if confirm_or_skip "Also stop the managed Tailscale daemon ($daemon_count process(es))?"; then
                if [[ "$daemon_mode" == "sudo" ]]; then
                    echo "Stopping sudo-started daemon (may need password)..."
                    local pids
                    pids="$(pgrep -f "tailscaled.*${state_dir_basename}" 2>/dev/null || true)"
                    for pid in $pids; do
                        sudo kill "$pid" 2>/dev/null || kill "$pid" 2>/dev/null || true
                    done
                else
                    cleanup_rootless_daemons
                fi
                rm -f "$TAILSCALE_MODE_FILE"
            fi
        fi
    fi

    echo "Restarting MLflow without external host restrictions..."
    docker compose --env-file "$ENV_FILE" -f "$COMPOSE_FILE" up -d
    echo "Done. MLflow is available at ${LOCAL_TARGET} (local only)."
}

show_status() {
    if detect_system_tailscale; then
        TAILSCALE_USE_SYSTEM_DAEMON=1
    elif local_tailscaled_running; then
        TAILSCALE_USE_SYSTEM_DAEMON=0
    fi

    echo "=== Local MLflow ==="
    if check_local_mlflow; then
        echo "  Status: healthy at ${LOCAL_TARGET}"
    else
        echo "  Status: NOT reachable at ${LOCAL_TARGET}"
    fi

    echo ""
    echo "=== Tailscale Daemon ==="
    if [[ "${TAILSCALE_USE_SYSTEM_DAEMON:-0}" == "1" ]]; then
        echo "  Backend: system daemon"
    elif local_tailscaled_running; then
        local state_dir_basename
        state_dir_basename="$(basename "$TAILSCALE_STATE_DIR")"
        local daemon_count
        daemon_count="$(pgrep -fc "tailscaled.*${state_dir_basename}" 2>/dev/null || echo "0")"
        local daemon_mode=""
        [[ -f "$TAILSCALE_MODE_FILE" ]] && daemon_mode="$(cat "$TAILSCALE_MODE_FILE" 2>/dev/null || true)"
        echo "  Backend: managed daemon (mode=${daemon_mode:-rootless}, processes=${daemon_count})"
        echo "  State dir: ${TAILSCALE_STATE_DIR}"
    else
        echo "  Backend: no daemon running"
    fi

    echo ""
    echo "=== Tailscale Status ==="
    tailscale_status_json 2>/dev/null | python3 -c '
import json, sys
data = json.load(sys.stdin)
print(f"  State: {data.get(\"BackendState\", \"unknown\")}")
self_node = data.get("Self", {})
dns = self_node.get("DNSName", "unknown").rstrip(".")
print(f"  Node:  {dns}")
' 2>/dev/null || echo "  Not connected"

    echo ""
    echo "=== Tailscale Funnel ==="
    tailscale_cli funnel status 2>/dev/null || echo "  No active Funnel routes"

    echo ""
    echo "=== Public Endpoint ==="
    if dns_name="$(get_tailnet_dns_name 2>/dev/null)"; then
        local public_url
        public_url="$(get_public_url "$dns_name")"
        if check_public_mlflow "$public_url"; then
            echo "  Status: REACHABLE at ${public_url}"
        else
            echo "  Status: NOT reachable at ${public_url}"
        fi
    else
        echo "  Status: Tailscale not connected, cannot determine URL"
    fi
}

# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

ACTION="${1:-up}"

case "$ACTION" in
    -h|--help|help)
        usage
        ;;
    up|start)
        do_up
        ;;
    down|stop)
        do_down
        ;;
    status)
        require_command "tailscale" "https://tailscale.com/download"
        show_status
        ;;
    url)
        require_command "tailscale" "https://tailscale.com/download"
        if detect_system_tailscale; then
            TAILSCALE_USE_SYSTEM_DAEMON=1
        elif local_tailscaled_running; then
            TAILSCALE_USE_SYSTEM_DAEMON=0
        fi
        if ! dns_name="$(get_tailnet_dns_name)"; then
            echo "Error: Tailscale is not connected."
            exit 1
        fi
        get_public_url "$dns_name"
        ;;
    *)
        echo "Usage: $0 {up|down|status|url|help}"
        exit 1
        ;;
esac
