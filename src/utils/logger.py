"""
Logging configuration with structured logging support.
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

import colorlog
from rich.console import Console


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
    base_logger.propagate = False

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
        # File logs - same format as console but without colors
        file_formatter = AlignedFormatter(
            "%(asctime)s  %(location)s  %(levelname)-7s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
            location_width=24,
        )
        file_handler.setFormatter(file_formatter)
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

    log_file: Path | None = None
    if _enable_file_logs and _run_log_dir is not None:
        log_file = _run_log_dir / "pipeline.log"

    logger = setup_logger("ryotenkai", level=_log_level, log_file=log_file, use_color=False)
    return logger


def init_run_logging(run_name: str, log_dir: str | Path | None = None) -> Path:
    """
    Initialize run-scoped logging and artifacts directory.

    This must be called exactly once per process, early in PipelineOrchestrator,
    after run_name is generated.
    """
    global _run_name, _run_log_dir, logger

    if not isinstance(run_name, str) or not run_name:
        raise ValueError("run_name must be a non-empty string")

    # Support multiple pipeline runs per process (e.g., unit tests, REPL usage).
    # Still a single source of truth: caller must pass run_name explicitly.
    if _run_name is not None and run_name != _run_name:
        logger.warning(f"Re-initializing run logging: {_run_name!r} -> {run_name!r}")

    _run_name = run_name
    _run_log_dir = Path(log_dir) if log_dir is not None else (_base_log_dir / run_name)
    _run_log_dir.mkdir(parents=True, exist_ok=True)

    log_file = _run_log_dir / "pipeline.log"

    if _enable_file_logs:
        # Reconfigure base logger to add file handler.
        logger = setup_logger("ryotenkai", level=_log_level, log_file=log_file, use_color=False)
        logger.info(f"📝 File logging: {log_file} (level: {_log_level_str})")
    else:
        logger.info("📝 File logging disabled (HELIX_NO_FILE_LOGS=1)")

    return _run_log_dir


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
