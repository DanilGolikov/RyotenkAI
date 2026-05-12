#!/usr/bin/env bash
# scripts/mutation/validate_agent_output.sh
#
# Agent self-validation gate (D7 in mutation testing policy).
# Subagents call this BEFORE declaring "done" on a feature/bugfix to
# confirm their test suite actually kills mutations in the production
# code they just changed.
#
# Usage:
#   bash scripts/mutation/validate_agent_output.sh [BASE_REF]
#
# Default BASE_REF resolution (matches scripts/mutation/run_on_diff.sh):
#   1. Explicit CLI arg ($1)
#   2. ``$MUTATION_BASE_REF`` env var
#   3. ``origin/RESEACRH`` if it exists (current integration branch)
#   4. ``RESEACRH`` (local)
#   5. ``origin/main`` / ``main`` fallback
#
# Returns:
#   0 — every changed production file meets its threshold (or no
#       production files were changed at all).
#   1 — at least one hotspot or warned file is below threshold.
#   2 — script error (cosmic-ray missing, config missing, etc).
#
# This is NOT enforced by CI on its own — the per-PR workflow is the
# authoritative gate. But running this locally before pushing catches
# issues without burning a CI iteration.
set -euo pipefail

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

if [ ! -x .venv/bin/cosmic-ray ]; then
    cat >&2 <<EOF
error: cosmic-ray not installed in .venv.
       Run: .venv/bin/pip install cosmic-ray
       or:  uv sync --extra dev
EOF
    exit 2
fi

# Use --strict so advisory-mode hotspot violations still fail this
# script (we want the agent to see them BEFORE they hit CI).
exec .venv/bin/python scripts/mutation/orchestrate.py \
    --diff "$BASE_REF" \
    --strict \
    --report "scripts/mutation/reports/agent_$(date +%Y%m%d_%H%M%S).json"
