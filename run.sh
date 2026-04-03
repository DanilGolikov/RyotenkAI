#!/bin/bash
# RyotenkAI — entry point for local runs
#
# Usage:
#   ./run.sh config.yaml                                       — fresh run
#   ./run.sh config.yaml --validate-only                       — dataset validation only
#   ./run.sh config.yaml --validate-config                     — static config checks
#   ./run.sh runs/<id> --resume                                — auto-resume
#   ./run.sh runs/<id> --restart-from-stage "Inference Deployer"
#   ./run.sh runs/<id> --restart-from-stage 5
#   ./run.sh runs/<id> --list-restart-points                   — list restart points
#   ./run.sh runs/<id> --inspect [-v] [--logs]                 — inspect run
#   ./run.sh runs/<id> --status [--interval 5]                 — live monitoring (Rich)
#   ./run.sh runs/<id> --diff [--attempt 1 --attempt 3]        — config diff
#   ./run.sh runs/<id> --show-logs [--attempt N] [--follow]    — view logs
#   ./run.sh runs/<id> --report                                — generate report
#   ./run.sh --list-runs [runs/]                               — list all runs
#   ./run.sh config.yaml runs/<id> --resume                    — resume with explicit config
#
# Batch smoke testing:
#   ./run.sh /path/to/configs --smoke                          — run all configs in parallel
#   ./run.sh /path/to/configs --smoke --workers 2              — limit parallelism
#   ./run.sh /path/to/configs --smoke --workers -1             — 1 worker per config (unlimited)
#   ./run.sh /path/to/configs --smoke --timeout 1200           — idle timeout 20 min per config
#   ./run.sh /path/to/configs --smoke --stagger 10             — 10s between launches (default: 5)
#   ./run.sh /path/to/configs --smoke --dry-run                — list configs only
#
# ryotenkai TUI (interactive terminal UI):
#   ./run.sh --tui                                             — browse all runs (uses ./runs)
#   ./run.sh runs/<id> --tui                                   — live monitor for a run
#   ./run.sh runs/<id> --tui --interval 10                     — refresh interval
set -e
cd "$(dirname "$0")"

# ── Help ───────────────────────────────────────────────────────────────────────

_show_help() {
    cat <<'EOF'
Usage: ./run.sh <config.yaml | runs/<id>> [options]

Run pipeline:
  ./run.sh config.yaml                                     — fresh run
  ./run.sh config.yaml --validate-only                     — dataset validation only
  ./run.sh runs/<id> --resume                              — auto-resume from first failed stage
  ./run.sh runs/<id> --restart-from-stage 'Stage Name'    — restart from a stage
  ./run.sh runs/<id> --restart-from-stage 5               — restart by stage number
  ./run.sh config.yaml runs/<id> --resume                  — resume with explicit config

Inspect and diagnostics:
  ./run.sh runs/<id> --inspect [-v] [--logs]               — inspect run
  ./run.sh runs/<id> --list-restart-points                 — list available restart points
  ./run.sh runs/<id> --status [--interval N]               — live monitoring (polling)
  ./run.sh runs/<id> --diff [--attempt N --attempt M]      — config diff between attempts
  ./run.sh runs/<id> --show-logs [--attempt N] [--follow]  — view/stream logs
  ./run.sh runs/<id> --report                              — generate markdown report
  ./run.sh --list-runs [runs/]                             — list all runs

ryotenkai TUI (interactive browser):
  ./run.sh --tui                                           — browse all runs (./runs)
  ./run.sh runs/<id> --tui                                 — live monitor for a run
  ./run.sh runs/<id> --tui --interval 10                   — refresh interval 10s

Batch smoke testing:
  ./run.sh /path/to/configs --smoke                        — run all *.yaml configs in parallel
  ./run.sh /path/to/configs --smoke --workers 2            — limit parallelism (default: 4)
  ./run.sh /path/to/configs --smoke --workers -1           — 1 worker per config (unlimited)
  ./run.sh /path/to/configs --smoke --timeout 1200         — idle timeout per config (default: 600s)
  ./run.sh /path/to/configs --smoke --stagger 10           — delay between launches (default: 5s)
  ./run.sh /path/to/configs --smoke --dry-run              — list configs without running

Pre-flight:
  ./run.sh config.yaml --validate-config                   — static config checks
EOF
}

for arg in "$@"; do
    if [[ "$arg" == "--help" || "$arg" == "-h" ]]; then
        _show_help
        exit 0
    fi
done

# ── Argument parsing ─────────────────────────────────────────────────────────

CONFIG=""
RUN_DIR=""
VALIDATE_ONLY=false
LIST_RESTART_POINTS=false
INSPECT=false
LIST_RUNS=false
SHOW_LOGS=false
RUN_DIFF=false
RUN_STATUS=false
VALIDATE_CONFIG=false
REPORT=false
TUI=false
SMOKE=false
SMOKE_WORKERS=""
SMOKE_TIMEOUT=""
SMOKE_STAGGER=""
SMOKE_DRY_RUN=false
SMOKE_REPORT_DIR=""
PASSTHROUGH=()

i=0
args=("$@")
while [[ $i -lt ${#args[@]} ]]; do
    arg="${args[$i]}"
    case "$arg" in
        --validate-only|--validate-dataset-only)
            VALIDATE_ONLY=true
            ;;
        --list-restart-points)
            LIST_RESTART_POINTS=true
            ;;
        --inspect)
            INSPECT=true
            ;;
        --list-runs)
            LIST_RUNS=true
            ;;
        --show-logs)
            SHOW_LOGS=true
            ;;
        --diff)
            RUN_DIFF=true
            ;;
        --status)
            RUN_STATUS=true
            ;;
        --validate-config)
            VALIDATE_CONFIG=true
            ;;
        --report)
            REPORT=true
            ;;
        --tui)
            TUI=true
            ;;
        --smoke)
            SMOKE=true
            ;;
        --workers)
            i=$((i + 1))
            SMOKE_WORKERS="${args[$i]}"
            ;;
        --workers=*)
            SMOKE_WORKERS="${arg#--workers=}"
            ;;
        --smoke-report-dir)
            i=$((i + 1))
            SMOKE_REPORT_DIR="${args[$i]}"
            ;;
        --smoke-report-dir=*)
            SMOKE_REPORT_DIR="${arg#--smoke-report-dir=}"
            ;;
        --timeout)
            i=$((i + 1))
            SMOKE_TIMEOUT="${args[$i]}"
            ;;
        --timeout=*)
            SMOKE_TIMEOUT="${arg#--timeout=}"
            ;;
        --stagger)
            i=$((i + 1))
            SMOKE_STAGGER="${args[$i]}"
            ;;
        --stagger=*)
            SMOKE_STAGGER="${arg#--stagger=}"
            ;;
        --dry-run)
            SMOKE_DRY_RUN=true
            ;;
        --run-dir)
            i=$((i + 1))
            RUN_DIR="${args[$i]}"
            ;;
        --run-dir=*)
            RUN_DIR="${arg#--run-dir=}"
            ;;
        --restart-from-stage)
            i=$((i + 1))
            PASSTHROUGH+=("$arg" "${args[$i]}")
            ;;
        --restart-from-stage=*)
            PASSTHROUGH+=("$arg")
            ;;
        --attempt)
            i=$((i + 1))
            PASSTHROUGH+=("$arg" "${args[$i]}")
            ;;
        --attempt=*)
            PASSTHROUGH+=("$arg")
            ;;
        --interval)
            i=$((i + 1))
            PASSTHROUGH+=("$arg" "${args[$i]}")
            ;;
        --interval=*)
            PASSTHROUGH+=("$arg")
            ;;
        -v|--verbose|--logs|-f|--follow)
            PASSTHROUGH+=("$arg")
            ;;
        -*)
            PASSTHROUGH+=("$arg")
            ;;
        *)
            # Autodetect: yaml file → --config, directory → --run-dir
            if [[ "$arg" == *.yaml || "$arg" == *.yml ]]; then
                CONFIG="$arg"
            elif [[ -d "$arg" ]]; then
                RUN_DIR="$arg"
            elif [[ -f "$arg" ]]; then
                CONFIG="$arg"
            else
                echo "Error: path not found: $arg"
                exit 1
            fi
            ;;
    esac
    i=$((i + 1))
done

# ── Check: any query command set ────────────────────────────────────────────

IS_QUERY=$([[ "$LIST_RESTART_POINTS" == true || "$INSPECT" == true || "$LIST_RUNS" == true || \
              "$SHOW_LOGS" == true || "$RUN_DIFF" == true || "$RUN_STATUS" == true || \
              "$VALIDATE_ONLY" == true || "$VALIDATE_CONFIG" == true || "$REPORT" == true || \
              "$TUI" == true || "$SMOKE" == true ]] && echo true || echo false)

if [[ -z "$CONFIG" && -z "$RUN_DIR" && "$IS_QUERY" == false ]]; then
    _show_help
    exit 1
fi

if [[ -n "$CONFIG" && ! -f "$CONFIG" ]]; then
    echo "Config not found: $CONFIG"
    exit 1
fi

# ── Activate environment ───────────────────────────────────────────────────

VENV_ACTIVATE=""
for ENV_DIR in ".venv" "venv"; do
    if [[ -f "$ENV_DIR/bin/activate" ]]; then
        VENV_ACTIVATE="$ENV_DIR/bin/activate"
        break
    fi
done

if [[ -z "$VENV_ACTIVATE" ]]; then
    echo "Virtual environment not found (.venv or venv). Run: bash setup.sh"
    exit 1
fi

# shellcheck disable=SC1090
source "$VENV_ACTIVATE"

# ── Run ─────────────────────────────────────────────────────────────────────

if [[ "$VALIDATE_ONLY" == true ]]; then
    if [[ -z "$CONFIG" ]]; then
        echo "Error: --validate-only requires a config path"
        exit 1
    fi
    echo "Config: $CONFIG"
    echo "Mode:   validate-dataset"
    echo ""
    LOG_LEVEL=DEBUG python -m src.main validate-dataset --config "$CONFIG"

elif [[ "$VALIDATE_CONFIG" == true ]]; then
    if [[ -z "$CONFIG" ]]; then
        echo "Error: --validate-config requires a config path"
        exit 1
    fi
    python -m src.main config-validate --config "$CONFIG" "${PASSTHROUGH[@]}"

elif [[ "$LIST_RESTART_POINTS" == true ]]; then
    if [[ -z "$RUN_DIR" ]]; then
        echo "Error: --list-restart-points requires a run directory"
        exit 1
    fi
    ARGS=("$RUN_DIR")
    [[ -n "$CONFIG" ]] && ARGS+=(--config "$CONFIG")
    python -m src.main list-restart-points "${ARGS[@]}"

elif [[ "$INSPECT" == true ]]; then
    if [[ -z "$RUN_DIR" ]]; then
        echo "Error: --inspect requires a run directory"
        exit 1
    fi
    python -m src.main inspect-run "$RUN_DIR" "${PASSTHROUGH[@]}"

elif [[ "$LIST_RUNS" == true ]]; then
    ARGS=()
    [[ -n "$RUN_DIR" ]] && ARGS+=("$RUN_DIR")
    python -m src.main runs-list "${ARGS[@]}" "${PASSTHROUGH[@]}"

elif [[ "$SHOW_LOGS" == true ]]; then
    if [[ -z "$RUN_DIR" ]]; then
        echo "Error: --show-logs requires a run directory"
        exit 1
    fi
    python -m src.main logs "$RUN_DIR" "${PASSTHROUGH[@]}"

elif [[ "$RUN_DIFF" == true ]]; then
    if [[ -z "$RUN_DIR" ]]; then
        echo "Error: --diff requires a run directory"
        exit 1
    fi
    python -m src.main run-diff "$RUN_DIR" "${PASSTHROUGH[@]}"

elif [[ "$RUN_STATUS" == true ]]; then
    if [[ -z "$RUN_DIR" ]]; then
        echo "Error: --status requires a run directory"
        exit 1
    fi
    python -m src.main run-status "$RUN_DIR" "${PASSTHROUGH[@]}"

elif [[ "$REPORT" == true ]]; then
    if [[ -z "$RUN_DIR" ]]; then
        echo "Error: --report requires a run directory"
        exit 1
    fi
    python -m src.main report "$RUN_DIR" "${PASSTHROUGH[@]}"

elif [[ "$TUI" == true ]]; then
    ARGS=()
    [[ -n "$RUN_DIR" ]] && ARGS+=("$RUN_DIR")
    python -m src.main tui "${ARGS[@]}" "${PASSTHROUGH[@]}"

elif [[ "$SMOKE" == true ]]; then
    # Smoke requires a directory with configs (passed as CONFIG or RUN_DIR positional)
    SMOKE_DIR="${RUN_DIR:-$CONFIG}"
    if [[ -z "$SMOKE_DIR" || ! -d "$SMOKE_DIR" ]]; then
        echo "Error: --smoke requires a directory with *.yaml configs"
        echo "Usage: ./run.sh /path/to/configs --smoke [--workers N] [--timeout S] [--dry-run]"
        exit 1
    fi
    ARGS=("$SMOKE_DIR")
    [[ -n "$SMOKE_WORKERS" ]]    && ARGS+=(--workers "$SMOKE_WORKERS")
    [[ -n "$SMOKE_TIMEOUT" ]]    && ARGS+=(--idle-timeout "$SMOKE_TIMEOUT")
    [[ -n "$SMOKE_STAGGER" ]]    && ARGS+=(--stagger "$SMOKE_STAGGER")
    [[ -n "$SMOKE_REPORT_DIR" ]] && ARGS+=(--report-dir "$SMOKE_REPORT_DIR")
    [[ "$SMOKE_DRY_RUN" == true ]] && ARGS+=(--dry-run)
    python scripts/batch_smoke.py "${ARGS[@]}"

else
    [[ -n "$CONFIG" ]]  && echo "Config:  $CONFIG"
    [[ -n "$RUN_DIR" ]] && echo "Run dir: $RUN_DIR"
    [[ ${#PASSTHROUGH[@]} -gt 0 ]] && echo "Options: ${PASSTHROUGH[*]}"
    echo ""

    CMD_ARGS=()
    [[ -n "$CONFIG" ]]  && CMD_ARGS+=(--config "$CONFIG")
    [[ -n "$RUN_DIR" ]] && CMD_ARGS+=(--run-dir "$RUN_DIR")

    LOG_LEVEL=DEBUG python -m src.main train "${CMD_ARGS[@]}" "${PASSTHROUGH[@]}"
fi

EXIT=$?
echo ""
if [[ $EXIT -eq 0 ]]; then
    echo "Done"
else
    echo "Failed (exit $EXIT)"
fi
exit $EXIT
