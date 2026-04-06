from __future__ import annotations

import logging
import sys
import time
import traceback
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Protocol

from src.tui.logging import get_tui_logger, init_tui_logging, suppress_project_console_logging


class _RunnableApp(Protocol):
    def run(self) -> object: ...


_INITIAL_RESTART_DELAY_SECONDS = 1.0
_MAX_RESTART_DELAY_SECONDS = 30.0


def _utc_now_iso() -> str:
    return datetime.now(UTC).isoformat()


def default_errors_log_path(runs_dir: Path) -> Path:
    return runs_dir.expanduser().resolve() / "errors.log"


def default_tui_log_path(runs_dir: Path) -> Path:
    return runs_dir.expanduser().resolve() / "tui.log"


@dataclass(frozen=True, slots=True)
class TuiRuntimeConfig:
    errors_log_path: Path
    tui_log_path: Path
    restart_delay_seconds: float = _INITIAL_RESTART_DELAY_SECONDS
    max_restart_delay_seconds: float = _MAX_RESTART_DELAY_SECONDS


def _append_crash_to_errors_log(
    log_path: Path,
    *,
    exc: Exception,
    restart_count: int,
) -> None:
    formatted_traceback = traceback.format_exc()
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as stream:
        stream.write(f"[{_utc_now_iso()}] tui crash #{restart_count}\n")
        stream.write(f"exception: {type(exc).__name__}: {exc}\n")
        stream.write(formatted_traceback)
        if not formatted_traceback.endswith("\n"):
            stream.write("\n")
        stream.write("\n")


def run_tui_with_restart(
    app_factory: Callable[[], _RunnableApp],
    *,
    config: TuiRuntimeConfig,
) -> None:
    restart_count = 0
    restart_delay = max(0.0, float(config.restart_delay_seconds))
    max_restart_delay = max(restart_delay, float(config.max_restart_delay_seconds))
    tui_logger = _init_runtime_logger(config.tui_log_path)
    tui_logger.info(
        "Starting TUI runtime (errors_log_path=%s, restart_delay=%.1fs, max_restart_delay=%.1fs)",
        config.errors_log_path,
        restart_delay,
        max_restart_delay,
    )

    while True:
        try:
            tui_logger.info("Launching TUI app instance #%s", restart_count + 1)
            with suppress_project_console_logging():
                app_factory().run()
            tui_logger.info("TUI app exited normally")
            return
        except KeyboardInterrupt:
            tui_logger.info("TUI interrupted by KeyboardInterrupt")
            raise
        except Exception as exc:
            restart_count += 1
            log_hint = f"See {config.errors_log_path}"
            try:
                _append_crash_to_errors_log(config.errors_log_path, exc=exc, restart_count=restart_count)
            except OSError as log_error:
                log_hint = f"Failed to write {config.errors_log_path}: {log_error}"
                tui_logger.exception("Failed to append crash details to errors log: %s", log_error)
            tui_logger.exception("TUI crashed and will be restarted (crash #%s)", restart_count)
            print(
                (
                    f"TUI crashed with {type(exc).__name__}: {exc}. "
                    f"Restarting in {restart_delay:.1f}s. "
                    f"{log_hint}"
                ),
                file=sys.stderr,
            )
            if restart_delay > 0:
                tui_logger.info("Sleeping %.1fs before TUI restart", restart_delay)
                time.sleep(restart_delay)
            restart_delay = min(max_restart_delay, restart_delay * 2 if restart_delay else 0.0)


def _init_runtime_logger(log_path: Path) -> logging.Logger:
    try:
        return init_tui_logging(log_path)
    except OSError:
        fallback_logger = get_tui_logger("runtime")
        fallback_logger.addHandler(logging.NullHandler())
        return fallback_logger
