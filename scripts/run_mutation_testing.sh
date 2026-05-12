#!/usr/bin/env bash
# scripts/run_mutation_testing.sh
# Run Cosmic Ray on a curated list of hotspot modules and aggregate
# results into sessions/SUMMARY.md.
#
# Usage:
#   scripts/run_mutation_testing.sh                # run all modules
#   scripts/run_mutation_testing.sh event_bus      # run a single named target
#   FAST=1 scripts/run_mutation_testing.sh         # skip already-run sessions
#
# Output:
#   sessions/<module>.sqlite            — per-module session DB
#   sessions/<module>_report.html       — cr-html report
#   sessions/<module>_dump.jsonl        — raw cosmic-ray dump
#   sessions/SUMMARY.md                 — aggregated kill rates
#
# Each target tuple is "name|module-path|test-path". The harness
# rewrites cosmic-ray.toml per target before each run.
set -euo pipefail

cd "$(dirname "$0")/.."

# (name, module-path, pytest target). Add/remove rows to extend coverage.
TARGETS=(
    "event_bus|packages/pod/src/ryotenkai_pod/runner/event_bus.py|tests/unit/pod/runner/event_bus/"
    "deployment_manager|packages/control/src/ryotenkai_control/pipeline/stages/managers/deployment_manager.py|tests/unit/control/pipeline/stages/managers/test_deployment_manager.py"
    "mlflow_relay|packages/pod/src/ryotenkai_pod/runner/mlflow_relay.py|tests/unit/pod/runner/test_mlflow_relay.py"
    "pod_terminator|packages/pod/src/ryotenkai_pod/runner/pod_terminator.py|tests/unit/pod/runner/test_pod_terminator.py"
    "supervisor|packages/pod/src/ryotenkai_pod/runner/supervisor.py|tests/unit/pod/runner/test_supervisor.py"
)

mkdir -p sessions

SELECTED="${1:-}"
FAST="${FAST:-0}"

run_one() {
    local name="$1" mod="$2" test_path="$3"
    local session="sessions/${name}.sqlite"

    if [[ "$FAST" == "1" && -f "$session" ]]; then
        echo "[SKIP] $name (session already exists; FAST=1)"
        return
    fi
    rm -f "$session"

    echo "[RUN ] $name  ->  $mod"
    # Per-run TOML lives in a temp file so the repo-root template
    # (which documents the operator-filter strategy) stays unchanged.
    local toml
    toml=$(mktemp -t cosmic-ray.XXXX.toml)
    trap "rm -f '$toml'" RETURN
    cat > "$toml" <<EOF
[cosmic-ray]
module-path = "${mod}"
timeout = 30.0
excluded-modules = []
test-command = ".venv/bin/python -m pytest -c tests/pytest.ini ${test_path} -x --no-header -q --disable-warnings"

[cosmic-ray.distributor]
name = "local"

[cosmic-ray.filters.operators-filter]
exclude-operators = [
    "core/ReplaceBinaryOperator_BitOr_.*",
    "core/ReplaceBinaryOperator_.*_BitOr",
]
EOF

    .venv/bin/cosmic-ray init "$toml" "$session"
    .venv/bin/cr-filter-operators "$session" "$toml" || echo "[warn] cr-filter-operators failed"
    .venv/bin/cr-filter-pragma "$session" 2>/dev/null || true
    local total
    total=$(sqlite3 "$session" "SELECT COUNT(*) FROM work_items" 2>/dev/null || echo "?")
    echo "       $total mutations queued (post-filter)"

    local t0 t1
    t0=$(date +%s)
    .venv/bin/cosmic-ray exec "$toml" "$session"
    t1=$(date +%s)
    echo "       exec took $((t1 - t0))s"

    .venv/bin/cr-html "$session" > "sessions/${name}_report.html"
    # `cosmic-ray dump` crashes on SKIPPED jobs in 8.4.6; emit a
    # plain SQL summary instead.
    sqlite3 "$session" \
        "SELECT worker_outcome, COALESCE(test_outcome, 'N/A'), COUNT(*) FROM work_results GROUP BY worker_outcome, test_outcome" \
        > "sessions/${name}_summary.tsv" 2>/dev/null || true
    .venv/bin/cr-report "$session" | tail -3
    echo ""
}

main() {
    for entry in "${TARGETS[@]}"; do
        IFS='|' read -r name mod test_path <<< "$entry"
        if [[ -n "$SELECTED" && "$SELECTED" != "$name" ]]; then continue; fi
        run_one "$name" "$mod" "$test_path"
    done

    echo "=== SUMMARY ==="
    {
        echo "# Mutation testing summary"
        echo
        echo "| module | total | survived | kill rate |"
        echo "|---|---|---|---|"
        for entry in "${TARGETS[@]}"; do
            IFS='|' read -r name _ _ <<< "$entry"
            local session="sessions/${name}.sqlite"
            [[ -f "$session" ]] || continue
            local rep
            rep=$(.venv/bin/cr-report "$session" | tail -3)
            local total survived kill_rate
            total=$(echo "$rep" | grep -E '^total jobs' | awk '{print $3}')
            survived=$(echo "$rep" | grep -E '^surviving' | awk '{print $3}')
            if [[ -n "$total" && -n "$survived" ]]; then
                kill_rate=$(python3 -c "print(f'{(1 - $survived/$total) * 100:.1f}%')")
            else
                kill_rate="?"
            fi
            echo "| ${name} | ${total} | ${survived} | ${kill_rate} |"
        done
    } | tee sessions/SUMMARY.md
}

main "$@"
