#!/usr/bin/env bash
# scripts/mutation/run_on_diff.sh
#
# Run mutation testing on production files changed in the current diff.
# Thin wrapper around `scripts/mutation/orchestrate.py --diff BASE_REF`.
#
# Usage:
#   bash scripts/mutation/run_on_diff.sh [BASE_REF]
#   bash scripts/mutation/run_on_diff.sh --help
#
# Defaults: BASE_REF=origin/main (CI) or main (local).
# Exit code: 0 if all gates pass OR mode=advisory; 1 if mode=blocking
# and any hotspot is below its threshold.
set -euo pipefail

if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
    sed -n '2,15p' "$0"
    exit 0
fi

cd "$(dirname "$0")/../.."

BASE_REF="${1:-}"
if [ -z "$BASE_REF" ]; then
    # Auto-detect: prefer origin/main if it exists (CI), else main.
    if git rev-parse --verify --quiet origin/main >/dev/null; then
        BASE_REF="origin/main"
    else
        BASE_REF="main"
    fi
fi

echo "Mutation testing (incremental) — base_ref=$BASE_REF"

CHANGED=$(git diff --name-only --diff-filter=ACM "$BASE_REF"...HEAD -- 'packages/*/src/*.py' || true)

if [ -z "$CHANGED" ]; then
    echo "No production files changed vs $BASE_REF; skipping mutation testing."
    exit 0
fi

echo "Changed production files:"
echo "$CHANGED" | sed 's/^/  /'

# Hand off to the Python orchestrator. It handles per-file mutation,
# filtering, budget enforcement, kill-rate accounting, and exit code.
exec .venv/bin/python scripts/mutation/orchestrate.py \
    --diff "$BASE_REF" \
    --report scripts/mutation/reports/pr_$(git rev-parse --short HEAD).json
