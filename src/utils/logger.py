"""
Logging configuration with structured logging support.
"""

from __future__ import annotations

import contextlib
import logging
import os
import sys
from contextlib import contextmanager
from contextvars import ContextVar
from pathlib import Path
from typing import TYPE_CHECKING

import colorlog
from rich.console import Console

from src.utils.logs_layout import LogLayout

if TYPE_CHECKING:
    from collections.abc import Iterator

_current_stage: ContextVar[str | None] = ContextVar("_current_stage", default=None)

# Module-level fallback for threads that don't receive the ContextVar copy
# (ThreadPoolExecutor workers, signal handlers, third-party-started threads).
# Valid because the orchestrator runs stages strictly sequentially and
# ThreadPoolExecutor's ``with``-block waits for all workers on exit
# (``shutdown(wait=True)``). If concurrent stages ever land in the same
# process, drop this fallback and rely solely on ``copy_context()``.
_active_stage: str | None = None


def _set_aligned_location(record: logging.LogRecord, location_width: int) -> None:
    """Set `record.location` as fixed-width `module:line` for log formatting."""
    module = record.module
    line_str = str(record.lineno)
    # Reserve space for ":line"
    max_module_len = location_width - len(line_str) - 1

    if len(module) > max_module_len:
        # Truncate from left, keep end (more unique): "…ployment_manager"
        module = "…" + module[-(max_module_len - 1) :]

    location = f"{module}:{line_str}"
    record.location = location.ljust(location_width)


class AlignedFormatter(logging.Formatter):
    """Custom formatter with fixed-width module:line column."""

    def __init__(self, fmt: str, datefmt: str | None = None, location_width: int = 28):
        super().__init__(fmt, datefmt)
        self.location_width = location_width

    def format(self, record: logging.LogRecord) -> str:
        _set_aligned_location(record, self.location_width)
        return super().format(record)


class AlignedColorFormatter(colorlog.ColoredFormatter):
    """Colored formatter with fixed-width module:line column."""

    def __init__(self, fmt: str, datefmt: str | None = None, location_width: int = 28, **kwargs):
        super().__init__(fmt, datefmt, **kwargs)
        self.location_width = location_width

    def format(self, record: logging.LogRecord) -> str:
        _set_aligned_location(record, self.location_width)
        return super().format(record)


class _StageContextFilter(logging.Filter):
    """Filter that admits records only when _current_stage matches ``stage_name``."""

    def __init__(self, stage_name: str) -> None:
        super().__init__()
        self._stage_name = stage_name

    def filter(self, record: logging.LogRecord) -> bool:  # noqa: ARG002 — logging.Filter API
        ctx = _current_stage.get()
        if ctx is None:
            # ThreadPoolExecutor workers and signal handlers don't inherit
            # the ContextVar — fall back to the module-level marker.
            ctx = _active_stage
        return ctx == self._stage_name


class _ExcludeRyotenkaiFilter(logging.Filter):
    """Reject records whose logger starts with ``ryotenkai``.

    Used on the root-logger-attached per-stage handler to avoid duplicate
    writes: ``ryotenkai.*`` records already land via the ryotenkai-attached
    handler. This filter is defensive — today ``ryotenkai.propagate = False``
    prevents propagation anyway, but if that ever changes we stay correct.
    """

    def filter(self, record: logging.LogRecord) -> bool:
        return not record.name.startswith("ryotenkai")


def _build_file_formatter() -> logging.Formatter:
    return AlignedFormatter(
        "%(asctime)s  %(location)s  %(levelname)-7s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        location_width=24,
    )


def setup_logger(
    name: str,
    level: int = logging.INFO,
    log_file: str | Path | None = None,
    use_color: bool = True,
) -> logging.Logger:
    """
    Setup logger with console and optionally file output.

    Args:
        name: Logger name
        level: Logging level
        log_file: Optional log file path
        use_color: Use colored output for console

    Returns:
        logging.Logger: Configured logger
    """
    base_logger = logging.getLogger(name)
    base_logger.setLevel(level)
    # Propagate to root so the aggregated pipeline.log FileHandler (attached
    # to root by ``_attach_pipeline_file_handler``) captures ``ryotenkai.*``
    # records too. Root has no StreamHandler, so the console output does not
    # duplicate — only the file handlers receive the propagated record.
    base_logger.propagate = True

    # Remove existing handlers
    base_logger.handlers.clear()

    # Console handler: use sys.__stderr__ to avoid captured streams being closed
    # and to share the same console stream as tqdm progress bars.
    console_handler = logging.StreamHandler(sys.__stderr__)
    console_handler.setLevel(level)

    # Format: DATETIME  LOCATION  LEVEL - MESSAGE
    # location = module:line with fixed 24-char width, long names truncated with …
    formatter: logging.Formatter  # Type annotation
    if use_color:
        formatter = AlignedColorFormatter(
            "%(asctime)s  %(location)s  %(log_color)s%(levelname)-7s%(reset)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
            location_width=24,
            log_colors={
                "DEBUG": "cyan",
                "INFO": "green",
                "WARNING": "yellow",
                "ERROR": "red",
                "CRITICAL": "red,bg_white",
            },
        )
    else:
        formatter = AlignedFormatter(
            "%(asctime)s  %(location)s  %(levelname)-7s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
            location_width=24,
        )

    console_handler.setFormatter(formatter)
    base_logger.addHandler(console_handler)

    # File handler (no colors)
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)

        file_handler = logging.FileHandler(log_path)
        file_handler.setLevel(level)
        file_handler.setFormatter(_build_file_formatter())
        base_logger.addHandler(file_handler)

    return base_logger


def get_logger(name: str) -> logging.Logger:
    """
    Get a child logger with the same formatting as main logger.

    All loggers use the same handlers (console + file) via propagation
    to the root 'ryotenkai' logger.
    """
    # Return child of main logger to inherit handlers and formatting
    return logging.getLogger(f"ryotenkai.{name}")


# Default logger instance (console-only at import time).
#
# IMPORTANT:
# - Run naming is owned by PipelineOrchestrator (single source of truth).
# - File logging must be initialized explicitly via init_run_logging(run_name).
_base_log_dir = Path("runs")

# Enable file logging by default, can be disabled with HELIX_NO_FILE_LOGS=1
_enable_file_logs = os.getenv("HELIX_NO_FILE_LOGS") != "1"

# Environment-based log level (dev: DEBUG, prod: INFO/WARNING)
_log_level_str = os.getenv("LOG_LEVEL", "INFO").upper()
_log_level = getattr(logging, _log_level_str, logging.INFO)

# Start with console-only logging; file handler is attached later (init_run_logging).
logger = setup_logger("ryotenkai", level=_log_level, log_file=None, use_color=False)

console = Console()  # Rich console for UI

_run_name: str | None = None
_run_log_dir: Path | None = None
_run_log_layout: LogLayout | None = None
_pipeline_file_handler: logging.FileHandler | None = None

# Third-party libraries that spam INFO/DEBUG without useful signal.
# Quieted at run-logging init so pipeline.log / stage.log stay readable.
_NOISY_LIBRARIES: tuple[str, ...] = ("httpx", "urllib3", "filelock", "botocore")

# Third-party libraries whose records we want to see in pipeline.log / stage.log.
# Some of them (notably mlflow) set ``propagate=False`` on import, which hides
# their records from our root FileHandler. We re-enable propagation so the
# aggregate + per-stage logs are complete.
_PROPAGATED_THIRD_PARTY: tuple[str, ...] = (
    "mlflow",
    "transformers",
    "datasets",
    "paramiko",
    "huggingface_hub",
)


def _quiet_noisy_libraries() -> None:
    """Raise the level of known-noisy third-party loggers to WARNING."""
    for name in _NOISY_LIBRARIES:
        logging.getLogger(name).setLevel(logging.WARNING)


def _force_propagation_for_third_party() -> None:
    """Make sure third-party loggers propagate to root.

    Libraries like ``mlflow`` disable propagation on import, which would
    prevent our root-attached pipeline.log / stage.log handlers from seeing
    their records.
    """
    for name in _PROPAGATED_THIRD_PARTY:
        logging.getLogger(name).propagate = True


def _attach_pipeline_file_handler(log_file: Path) -> logging.FileHandler:
    """(Re-)attach the aggregated pipeline.log FileHandler to the ROOT logger.

    Attaching to root is what lets the aggregate capture records from
    third-party libraries (mlflow, transformers, paramiko, etc.) which live in
    their own logger hierarchies and never propagate through ``ryotenkai``.
    """
    global _pipeline_file_handler
    root = logging.getLogger()

    if _pipeline_file_handler is not None:
        root.removeHandler(_pipeline_file_handler)
        with contextlib.suppress(Exception):  # pragma: no cover — best-effort cleanup
            _pipeline_file_handler.close()

    log_file.parent.mkdir(parents=True, exist_ok=True)
    handler = logging.FileHandler(log_file)
    handler.setLevel(_log_level)
    handler.setFormatter(_build_file_formatter())
    root.addHandler(handler)

    # Root defaults to WARNING — raise it so INFO records actually reach the
    # handler (Python checks the logger's effective level before the handler's).
    if root.level == logging.NOTSET or root.level > _log_level:
        root.setLevel(_log_level)

    _pipeline_file_handler = handler
    return handler


def set_log_level(level_name: str) -> logging.Logger:
    """Reconfigure the base logger at runtime."""
    global _log_level, _log_level_str, logger

    normalized = str(level_name).upper()
    level = getattr(logging, normalized, None)
    if not isinstance(level, int):
        raise ValueError(f"Unsupported log level: {level_name}")

    _log_level_str = normalized
    _log_level = level
    os.environ["LOG_LEVEL"] = normalized

    # Rebuild the console-only ``ryotenkai`` logger at the new level.
    logger = setup_logger("ryotenkai", level=_log_level, log_file=None, use_color=False)

    # Re-attach the pipeline.log handler to root at the new level.
    if _enable_file_logs and _run_log_layout is not None:
        _attach_pipeline_file_handler(_run_log_layout.pipeline_log)

    return logger


def init_run_logging(run_name: str, log_dir: str | Path | None = None) -> Path:
    """
    Initialize run-scoped logging and artifacts directory.

    This must be called exactly once per process, early in PipelineOrchestrator,
    after run_name is generated.

    ``log_dir`` is the attempt directory. The aggregated pipeline log lives
    inside ``<attempt_dir>/logs/pipeline.log`` and is attached to the ROOT
    logger so it captures third-party libraries too. Console output stays on
    the ``ryotenkai`` logger (``propagate=False``) to keep the terminal quiet.
    Returns the attempt directory (preserved behavior for callers).
    """
    global _run_name, _run_log_dir, _run_log_layout, logger

    if not isinstance(run_name, str) or not run_name:
        raise ValueError("run_name must be a non-empty string")

    if _run_name is not None and run_name != _run_name:
        logger.warning(f"Re-initializing run logging: {_run_name!r} -> {run_name!r}")

    _run_name = run_name
    attempt_dir = Path(log_dir) if log_dir is not None else (_base_log_dir / run_name)
    attempt_dir.mkdir(parents=True, exist_ok=True)

    layout = LogLayout(attempt_dir)
    layout.ensure_logs_dir()

    _run_log_dir = attempt_dir
    _run_log_layout = layout

    # Console handler stays on ``ryotenkai`` — third-party libs won't spam it.
    logger = setup_logger("ryotenkai", level=_log_level, log_file=None, use_color=False)

    if _enable_file_logs:
        _attach_pipeline_file_handler(layout.pipeline_log)
        _quiet_noisy_libraries()
        _force_propagation_for_third_party()
        logger.info(f"📝 File logging: {layout.pipeline_log} (level: {_log_level_str})")
    else:
        logger.info("📝 File logging disabled (HELIX_NO_FILE_LOGS=1)")

    return attempt_dir


def get_run_name() -> str:
    """Get canonical run name (requires init_run_logging)."""
    if _run_name is None:
        raise RuntimeError("Run logging not initialized. Call init_run_logging(run_name) in PipelineOrchestrator.")
    return _run_name


def get_run_log_dir() -> Path:
    """Get the local runs/artifacts directory for this run (requires init_run_logging)."""
    if _run_log_dir is None:
        raise RuntimeError("Run logging not initialized. Call init_run_logging(run_name) in PipelineOrchestrator.")
    return _run_log_dir


def get_run_log_layout() -> LogLayout:
    """Get the LogLayout for the current run (requires init_run_logging)."""
    if _run_log_layout is None:
        raise RuntimeError("Run logging not initialized. Call init_run_logging(run_name) in PipelineOrchestrator.")
    return _run_log_layout


@contextmanager
def stage_logging_context(stage_name: str, layout: LogLayout) -> Iterator[Path]:
    """
    Route per-stage log output to ``<logs>/<slug>.log`` while inside the context.

    Two FileHandlers are attached to the same stage file:
      * one on the ``ryotenkai`` logger — captures our own code,
      * one on the root logger — captures third-party loggers (mlflow,
        transformers, paramiko, httpx, ...).

    Both handlers share a ``_StageContextFilter`` that admits records only
    when the ``_current_stage`` ContextVar equals ``stage_name``. The root
    handler additionally rejects records from ``ryotenkai.*`` (prevented
    by ``propagate=False`` today, but filtered defensively in case the
    propagation contract ever changes).

    The aggregated ``pipeline.log`` handler on root keeps receiving every
    record untouched.
    """
    if not stage_name:
        raise ValueError("stage_name must be non-empty")

    stage_log_path = layout.stage_log(stage_name)
    stage_log_path.parent.mkdir(parents=True, exist_ok=True)

    # Per-scope bookkeeping: keep references so ``finally`` can undo
    # exactly what this context established, even on exception.
    installed: list[tuple[logging.Logger, logging.Handler]] = []

    # Set both the ContextVar (asyncio-safe) and the module-level marker
    # (thread-safe fallback). Save the previous active marker so nested
    # contexts restore the outer stage on exit (mirrors the ContextVar token).
    global _active_stage
    prev_active = _active_stage
    _active_stage = stage_name
    token = _current_stage.set(stage_name)

    try:
        if _enable_file_logs:
            stage_filter = _StageContextFilter(stage_name)

            # Handler A: ryotenkai logger (our own code).
            ryotenkai_handler = logging.FileHandler(stage_log_path)
            ryotenkai_handler.setLevel(_log_level)
            ryotenkai_handler.setFormatter(_build_file_formatter())
            ryotenkai_handler.addFilter(stage_filter)
            ryotenkai_logger = logging.getLogger("ryotenkai")
            ryotenkai_logger.addHandler(ryotenkai_handler)
            installed.append((ryotenkai_logger, ryotenkai_handler))

            # Handler B: root logger (third-party libs).
            root_handler = logging.FileHandler(stage_log_path)
            root_handler.setLevel(_log_level)
            root_handler.setFormatter(_build_file_formatter())
            root_handler.addFilter(stage_filter)
            root_handler.addFilter(_ExcludeRyotenkaiFilter())
            root_logger = logging.getLogger()
            root_logger.addHandler(root_handler)
            installed.append((root_logger, root_handler))

        yield stage_log_path
    finally:
        _current_stage.reset(token)
        _active_stage = prev_active
        for logger_obj, handler in installed:
            logger_obj.removeHandler(handler)
            with contextlib.suppress(Exception):  # pragma: no cover — best-effort cleanup
                handler.close()
