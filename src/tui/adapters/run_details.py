from __future__ import annotations

from pathlib import Path
from typing import Any

from src.pipeline.run_queries import RunInspectionData, RunInspector, diff_attempts as query_diff_attempts, tail_lines


def load_run_inspection(run_dir: Path, *, include_logs: bool = False) -> RunInspectionData:
    return RunInspector(run_dir).load(include_logs=include_logs)


def diff_attempts(state, attempt_a: int, attempt_b: int) -> dict[str, Any]:
    return query_diff_attempts(state, attempt_a, attempt_b)


def resolve_run_config_path(raw_path: str | None) -> Path | None:
    if not raw_path:
        return None
    return Path(raw_path).expanduser().resolve()
