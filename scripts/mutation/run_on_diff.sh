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
# Default BASE_REF resolution order:
#   1. Explicit CLI arg ($1)
#   2. ``$MUTATION_BASE_REF`` env var
#   3. ``origin/RESEACRH`` (current integration branch) if it exists
#   4. ``RESEACRH`` (local)
#   5. ``origin/main`` / ``main`` fallback (for downstream that's moved
#      past the RESEACRH-as-integration model)
#
# When the integration branch is renamed (e.g. ``dev``), update this
# resolver in one place AND ``.github/workflows/mutation-pr.yml``.
#
# Exit code: 0 if all gates pass OR mode=advisory; 1 if mode=blocking
# and any hotspot is below its threshold.
set -euo pipefail

if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
    sed -n '2,20p' "$0"
    exit 0
fi

cd "$(dirname "$0")/../.."

BASE_REF="${1:-${MUTATION_BASE_REF:-}}"
if [ -z "$BASE_REF" ]; then
    if git rev-parse --verify --quiet origin/RESEACRH >/dev/null; then
        BASE_REF="origin/RESEACRH"
    elif git rev-parse --verify --quiet RESEACRH >/dev/null; then
        BASE_REF="RESEACRH"
    elif git rev-parse --verify --quiet origin/main >/dev/null; then
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
