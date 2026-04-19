from __future__ import annotations

from pathlib import Path

from src.api.schemas.log import LogChunk, LogFileInfo
from src.pipeline.state import PipelineStateStore

ALLOWED_LOG_FILES: tuple[str, ...] = (
    "pipeline.log",
    "training.log",
    "inference.log",
    "eval.log",
    "tui_launch.log",
)


def resolve_log_path(run_dir: Path, attempt_no: int, file: str) -> Path:
    if file not in ALLOWED_LOG_FILES:
        raise ValueError(f"unsupported log file: {file}")
    if file == "tui_launch.log":
        # Launcher log lives at run root, not in attempt dir.
        return (run_dir / file).resolve()
    attempt_dir = PipelineStateStore(run_dir).next_attempt_dir(attempt_no)
    return (attempt_dir / file).resolve()


def list_log_files(run_dir: Path, attempt_no: int) -> list[LogFileInfo]:
    out: list[LogFileInfo] = []
    for name in ALLOWED_LOG_FILES:
        path = resolve_log_path(run_dir, attempt_no, name)
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


__all__ = ["ALLOWED_LOG_FILES", "list_log_files", "read_chunk", "resolve_log_path"]
