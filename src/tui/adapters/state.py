"""Backward-compatibility shim. Real module lives in src/pipeline/state/queries."""

from __future__ import annotations

from src.pipeline.state.queries import (
    discover_run_dirs,
    find_running_attempt_no,
    get_attempt_by_no,
    get_running_attempt_no,
    latest_attempt_no,
    load_pipeline_state,
    predict_next_attempt_no,
    resolve_config_path_from_state,
    run_state_exists,
    state_store,
)

# Re-exported symbol used by monkeypatch paths in legacy tests.
from src.pipeline.state import PipelineStateStore  # noqa: F401

__all__ = [
    "PipelineStateStore",
    "discover_run_dirs",
    "find_running_attempt_no",
    "get_attempt_by_no",
    "get_running_attempt_no",
    "latest_attempt_no",
    "load_pipeline_state",
    "predict_next_attempt_no",
    "resolve_config_path_from_state",
    "run_state_exists",
    "state_store",
]
