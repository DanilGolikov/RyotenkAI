"""
Checkpoint directory utilities and pipeline run discovery.

Pure functions — no class state, no DataBuffer dependency.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from src.training.managers.constants import (
    CHECKPOINT_FINAL_DIR,
    CHECKPOINT_SIZE_ESTIMATE_MB,  # noqa: F401  (re-exported for convenience)
    KEY_PHASES,
    KEY_STARTED_AT,
    KEY_STATUS,
)
from src.utils.logger import logger


# ---------------------------------------------------------------------------
# Serialization helper
# ---------------------------------------------------------------------------


def _sanitize_metrics(metrics: dict[str, Any]) -> dict[str, Any]:
    """
    Convert metrics to JSON-serializable types.

    Handles numpy/torch scalars and arrays.

    Args:
        metrics: Dict with potentially non-serializable values

    Returns:
        Dict with JSON-serializable values only
    """
    result: dict[str, Any] = {}
    for key, value in metrics.items():
        if hasattr(value, "item"):  # numpy/torch scalar
            result[key] = value.item()
        elif hasattr(value, "tolist"):  # numpy array
            result[key] = value.tolist()
        elif isinstance(value, int | float | str | bool | type(None)):
            result[key] = value
        elif isinstance(value, dict):
            result[key] = _sanitize_metrics(value)  # Recurse for nested dicts
        elif isinstance(value, list):
            result[key] = [v.item() if hasattr(v, "item") else v for v in value]
        else:
            result[key] = str(value)  # Fallback to string
    return result


# ---------------------------------------------------------------------------
# Checkpoint directory helpers
# ---------------------------------------------------------------------------


def _extract_checkpoint_step(checkpoint_path: Path) -> int | float:
    """
    Extract step number from checkpoint-N directory name.

    Args:
        checkpoint_path: Path to checkpoint directory

    Returns:
        Step number, inf for checkpoint-final, -1 for unknown format
    """
    name = checkpoint_path.name
    if name == CHECKPOINT_FINAL_DIR:
        return float("inf")  # Always last
    try:
        # checkpoint-100 -> 100
        return int(name.split("-")[-1])
    except ValueError:
        return -1  # Unknown format — sort first


def _get_sorted_checkpoints(output_dir: Path) -> list[Path]:
    """
    Get checkpoint directories sorted by step number.

    Filters directories only (excludes files) and sorts numerically.

    Args:
        output_dir: Directory containing checkpoints

    Returns:
        List of checkpoint Paths sorted by step number
    """
    checkpoints = [p for p in output_dir.glob("checkpoint-*") if p.is_dir()]
    return sorted(checkpoints, key=_extract_checkpoint_step)


# ---------------------------------------------------------------------------
# Run discovery
# ---------------------------------------------------------------------------

# Kept as a module-level constant to avoid importing DataBuffer (circular).
_STATE_FILENAME = "pipeline_state.json"


def list_available_runs(base_output_dir: str | Path) -> list[dict[str, Any]]:
    """
    List all available training runs in output directory.

    Args:
        base_output_dir: Base directory for checkpoints

    Returns:
        List of dicts with run_id, status, created_at
    """
    base_path = Path(base_output_dir)
    runs: list[dict[str, Any]] = []

    if not base_path.exists():
        return runs

    for run_dir in base_path.iterdir():
        if not run_dir.is_dir():
            continue

        # Support both layouts:
        # - v2: <run_dir>/output/pipeline_state.json (provider run workspace layout)
        # - legacy: <run_dir>/pipeline_state.json
        state_file = run_dir / "output" / _STATE_FILENAME
        if not state_file.exists():
            state_file = run_dir / _STATE_FILENAME
        if state_file.exists():
            try:
                with state_file.open() as f:
                    state = json.load(f)
                runs.append(
                    {
                        "run_id": run_dir.name,
                        KEY_STATUS: state.get(KEY_STATUS, "unknown"),
                        "progress": (
                            f"{len([p for p in state.get(KEY_PHASES, []) if p.get(KEY_STATUS) == 'completed'])}"
                            f"/{state.get('total_phases', 0)}"
                        ),
                        KEY_STARTED_AT: state.get(KEY_STARTED_AT),
                    }
                )
            except Exception as e:
                logger.warning(f"Could not read state for {run_dir.name}: {e}")

    return sorted(runs, key=lambda x: x.get(KEY_STARTED_AT, ""), reverse=True)


__all__ = [
    "_sanitize_metrics",
    "_extract_checkpoint_step",
    "_get_sorted_checkpoints",
    "list_available_runs",
]
