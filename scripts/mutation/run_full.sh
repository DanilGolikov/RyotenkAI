#!/usr/bin/env bash
# scripts/mutation/run_full.sh
#
# Run mutation testing on the full hotspot list (no diff filtering).
# Used by the nightly CI job. The per-file budget here is larger than
# the per-PR budget because nightly can afford ~2h.
#
# Usage:
#   bash scripts/mutation/run_full.sh
set -euo pipefail

cd "$(dirname "$0")/../.."

# Generous budget for the nightly run — each hotspot gets up to 35 min
# of mutation-testing wall-clock. Aggregate cap is left to the workflow
# (`timeout-minutes`).
exec .venv/bin/python scripts/mutation/orchestrate.py \
    --all-hotspots \
    --budget-minutes 35 \
    --report scripts/mutation/reports/nightly_$(date +%Y%m%d).json
