from __future__ import annotations

from pathlib import Path

from src.api.schemas.log import LogChunk, LogFileInfo
from src.pipeline.state import PipelineStateStore
from src.utils.logs_layout import LOGS_DIR_NAME, PIPELINE_LOG_NAME, REMOTE_TRAINING_LOG_NAME, LogLayout

# Log files that live at the run root (not inside any attempt).
_RUN_ROOT_LOG_FILES: tuple[str, ...] = ("tui_launch.log",)

# Legacy per-attempt log files that lived directly under attempt_dir (pre-LogLayout).
# Used only as a fallback when a run's state has no log_paths registry.
_LEGACY_ATTEMPT_LOG_FILES: tuple[str, ...] = (
    PIPELINE_LOG_NAME,
    REMOTE_TRAINING_LOG_NAME,
    "inference.log",
    "eval.log",
)


def _is_run_root_file(file: str) -> bool:
    return file in _RUN_ROOT_LOG_FILES


def _discover_from_state(run_dir: Path, attempt_no: int) -> dict[str, Path]:
    """
    Build a map of logical log file name -> resolved absolute Path
    from the pipeline state file. Empty map if state is unavailable.
    """
    try:
        store = PipelineStateStore(run_dir)
        if not store.exists():
            return {}
        state = store.load()
    except Exception:
        return {}

    attempt = next((a for a in state.attempts if a.attempt_no == attempt_no), None)
    if attempt is None:
        return {}

    attempt_dir = store.next_attempt_dir(attempt_no)
    layout = LogLayout(attempt_dir)
    out: dict[str, Path] = {}

    # Always expose the aggregated pipeline log when the new layout is in use.
    pipeline_log = layout.pipeline_log
    if pipeline_log.exists():
        out[PIPELINE_LOG_NAME] = pipeline_log.resolve()

    for stage_name, stage_state in attempt.stage_runs.items():
        paths = stage_state.log_paths or {}
        for key, rel_path in paths.items():
            if not isinstance(rel_path, str) or not rel_path:
                continue
            abs_path = (attempt_dir / rel_path).resolve()
            if key == "remote_training":
                # Expose the remote training log under its historical file name.
                out[REMOTE_TRAINING_LOG_NAME] = abs_path
            else:
                out[f"{stage_name}.log"] = abs_path
    return out


def _discover_from_legacy_layout(run_dir: Path, attempt_no: int) -> dict[str, Path]:
    """Fallback: runs created before LogLayout had logs directly under attempt_dir."""
    attempt_dir = PipelineStateStore(run_dir).next_attempt_dir(attempt_no)
    out: dict[str, Path] = {}
    # New-layout files (if present after upgrade mid-life).
    logs_dir = attempt_dir / LOGS_DIR_NAME
    if logs_dir.is_dir():
        for path in logs_dir.glob("*.log"):
            out[path.name] = path.resolve()
    # Legacy files directly under attempt_dir.
    for name in _LEGACY_ATTEMPT_LOG_FILES:
        legacy = attempt_dir / name
        if legacy.exists() and name not in out:
            out[name] = legacy.resolve()
    return out


def _discover_log_files(run_dir: Path, attempt_no: int) -> dict[str, Path]:
    discovered = _discover_from_state(run_dir, attempt_no)
    if discovered:
        return discovered
    return _discover_from_legacy_layout(run_dir, attempt_no)


def resolve_log_path(run_dir: Path, attempt_no: int, file: str) -> Path:
    if _is_run_root_file(file):
        return (run_dir / file).resolve()

    files = _discover_log_files(run_dir, attempt_no)
    resolved = files.get(file)
    if resolved is None:
        raise ValueError(f"unsupported log file: {file}")
    return resolved


def list_log_files(run_dir: Path, attempt_no: int) -> list[LogFileInfo]:
    out: list[LogFileInfo] = []

    # Run-root files
    for name in _RUN_ROOT_LOG_FILES:
        path = (run_dir / name).resolve()
        exists = path.exists()
        size = path.stat().st_size if exists else 0
        out.append(LogFileInfo(name=name, path=str(path), size_bytes=size, exists=exists))

    # Attempt-scoped files
    for name, path in sorted(_discover_log_files(run_dir, attempt_no).items()):
        exists = path.exists()
        size = path.stat().st_size if exists else 0
        out.append(LogFileInfo(name=name, path=str(path), size_bytes=size, exists=exists))

    return out


def read_chunk(
    run_dir: Path,
    attempt_no: int,
    file: str,
    *,
    offset: int = 0,
    max_bytes: int = 1_048_576,
) -> LogChunk:
    path = resolve_log_path(run_dir, attempt_no, file)
    if not path.exists():
        return LogChunk(file=file, offset=offset, next_offset=offset, eof=True, content="")

    size = path.stat().st_size
    if offset > size:
        # File was truncated / rotated — reset to start.
        offset = 0
    with path.open("rb") as handle:
        handle.seek(offset)
        raw = handle.read(max_bytes)
    next_offset = offset + len(raw)
    eof = next_offset >= size
    content = raw.decode("utf-8", errors="replace")
    return LogChunk(file=file, offset=offset, next_offset=next_offset, eof=eof, content=content)


__all__ = ["list_log_files", "read_chunk", "resolve_log_path"]
