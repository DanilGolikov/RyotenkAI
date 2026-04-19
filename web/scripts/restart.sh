#!/usr/bin/env bash
# Restart backend and/or frontend by calling stop.sh then start.sh.
#
# Usage:
#   ./scripts/restart.sh             # both
#   ./scripts/restart.sh backend
#   ./scripts/restart.sh frontend

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

TARGET="${1:-all}"
"${SCRIPT_DIR}/stop.sh"  "${TARGET}"
"${SCRIPT_DIR}/start.sh" "${TARGET}"
