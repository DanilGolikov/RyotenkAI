from __future__ import annotations

import contextlib
import logging
from collections.abc import Iterator
from pathlib import Path

_BASE_LOGGER_NAME = "ryotenkai.tui"


def init_tui_logging(log_path: Path, *, level: int = logging.INFO) -> logging.Logger:
    """Configure a file-only logger for TUI runtime diagnostics."""
    logger = logging.getLogger(_BASE_LOGGER_NAME)
    logger.setLevel(level)
    logger.propagate = False
    logger.handlers.clear()

    resolved_path = log_path.expanduser().resolve()
    resolved_path.parent.mkdir(parents=True, exist_ok=True)

    file_handler = logging.FileHandler(resolved_path, encoding="utf-8")
    file_handler.setLevel(level)
    file_handler.setFormatter(
        logging.Formatter(
            "%(asctime)s %(levelname)s %(name)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
    )
    logger.addHandler(file_handler)
    return logger


def get_tui_logger(name: str | None = None) -> logging.Logger:
    """Return the shared TUI logger or one of its children."""
    if not name:
        return logging.getLogger(_BASE_LOGGER_NAME)
    return logging.getLogger(f"{_BASE_LOGGER_NAME}.{name}")


@contextlib.contextmanager
def suppress_project_console_logging() -> Iterator[None]:
    """Temporarily detach project console handlers while TUI owns the terminal."""
    base_logger = logging.getLogger("ryotenkai")
    removed_handlers = [
        handler
        for handler in list(base_logger.handlers)
        if isinstance(handler, logging.StreamHandler) and not isinstance(handler, logging.FileHandler)
    ]
    for handler in removed_handlers:
        base_logger.removeHandler(handler)
    try:
        yield
    finally:
        for handler in removed_handlers:
            if handler not in base_logger.handlers:
                base_logger.addHandler(handler)
