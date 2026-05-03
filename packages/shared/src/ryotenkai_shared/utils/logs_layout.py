"""
LogLayout — single source of truth for pipeline FS log layout.

All pipeline textual logs live under ``<attempt_dir>/logs/``:

    attempts/attempt_N/
    ├── logs/
    │   ├── pipeline.log              # aggregated stream (all stages)
    │   ├── <stage_name>.log          # per-stage stream
    │   ├── trainer.stdio.log         # remote trainer subprocess stdout/stderr (pulled by LogManager)
    │   └── runner.log                # remote uvicorn / runner stdout (pulled by LogManager)
    ├── evaluation/                   # stage artifacts (not logs)
    └── inference/                    # stage artifacts (not logs)

Two remote logs, two channels:

* ``trainer.stdio.log`` — trainer subprocess stdout/stderr ground-truth,
  written pod-side by the Supervisor's pump (NOT by the trainer itself,
  so it survives import-time crashes). Includes Python tracebacks, native
  faulthandler dumps (SEGV/ABRT), and HuggingFace progress chatter.
* ``runner.log`` — uvicorn / FastAPI runner stdout, redirected by the
  Mac-orchestrated ``runner_launcher.py`` from the very first byte of
  Python boot. Captures ImportError / SyntaxError that fire BEFORE the
  trainer ever spawns — this is the file that surfaces "what killed the
  runner before /healthz answered".

Pod↔Mac symmetry: the pod-side filenames (see :mod:`src.utils.pod_layout`)
match the Mac-side filenames here exactly. LogManager scp does a 1:1 mapping.

Consumers (orchestrator, logger, log_manager, log_service) MUST go through
this class — no module is allowed to construct log paths from raw string
literals.
"""

from __future__ import annotations

import re
from pathlib import Path

from ryotenkai_shared.utils.log_filenames import (
    PIPELINE_LOG,
    RUNNER_LOG,
    STAGE_LOG_SUFFIX,
    TRAINER_STDIO_LOG,
)

LOGS_DIR_NAME = "logs"

# Re-export filename constants under their LogLayout-historical names so
# existing consumers of this module keep working. Single source of truth
# is :mod:`src.utils.log_filenames` — pod-side and Mac-side share the
# same literals because LogManager scp does a 1:1 by-filename mapping.
PIPELINE_LOG_NAME = PIPELINE_LOG
REMOTE_TRAINER_STDIO_LOG_NAME = TRAINER_STDIO_LOG
REMOTE_RUNNER_LOG_NAME = RUNNER_LOG
# STAGE_LOG_SUFFIX is itself the canonical constant (re-imported above
# so users of this module don't notice the move).

STAGE_LOG_PATHS_KEY = "stage"
REMOTE_TRAINER_STDIO_LOG_PATHS_KEY = "remote_trainer_stdio"
REMOTE_RUNNER_LOG_PATHS_KEY = "remote_runner"

_SLUG_FALLBACK = "stage"
_SLUG_PATTERN = re.compile(r"[^a-z0-9]+")


def _slugify(name: str) -> str:
    """Normalize a stage name for use as a filename.

    Lowercases, replaces any non-[a-z0-9] run with a single underscore, and
    strips leading/trailing underscores. Empty input or non-alphanumeric-only
    input collapses to a safe fallback so we never produce an empty filename.
    """
    normalized = _SLUG_PATTERN.sub("_", name.lower()).strip("_")
    return normalized or _SLUG_FALLBACK


class LogLayout:
    """Filesystem layout for attempt-scoped logs."""

    def __init__(self, attempt_dir: Path) -> None:
        self._attempt_dir = Path(attempt_dir)

    @property
    def attempt_dir(self) -> Path:
        return self._attempt_dir

    @property
    def logs_dir(self) -> Path:
        return self._attempt_dir / LOGS_DIR_NAME

    @property
    def pipeline_log(self) -> Path:
        return self.logs_dir / PIPELINE_LOG_NAME

    def stage_log(self, stage_name: str) -> Path:
        if not stage_name:
            raise ValueError("stage_name must be non-empty")
        return self.logs_dir / f"{_slugify(stage_name)}{STAGE_LOG_SUFFIX}"

    @property
    def remote_trainer_stdio_log(self) -> Path:
        """Pod-side trainer stdout/stderr ground-truth, mirrored on Mac.

        Source: ``<workspace>/logs/trainer.stdio.log`` written by the
        Supervisor's pump in the runner. Captures everything the trainer
        subprocess emits — including import-time tracebacks and native
        faulthandler dumps.
        """
        return self.logs_dir / REMOTE_TRAINER_STDIO_LOG_NAME

    @property
    def remote_runner_log(self) -> Path:
        """Pod-side uvicorn/runner stdout, mirrored on Mac.

        Source: ``/workspace/runner.log`` written by entrypoint.sh's
        redirect at the beginning of the Python boot — captures both
        the runner's normal lifecycle log AND any pre-import crash
        (ImportError / SyntaxError) that the trainer.log channel
        misses because it never reaches the trainer subprocess.
        """
        return self.logs_dir / REMOTE_RUNNER_LOG_NAME

    def ensure_logs_dir(self) -> Path:
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        return self.logs_dir

    def relative(self, path: Path) -> str:
        """Path string relative to attempt_dir. Absolute path used as fallback."""
        try:
            return str(Path(path).relative_to(self._attempt_dir))
        except ValueError:
            return str(path)

    def stage_log_registry(self, stage_name: str, *, include_remote_trainer_stdio: bool = False) -> dict[str, str]:
        """
        Registry of log paths owned by a stage, for persistence in StageRunState.log_paths.

        Keys are logical identifiers; values are paths relative to attempt_dir.
        """
        registry: dict[str, str] = {
            STAGE_LOG_PATHS_KEY: self.relative(self.stage_log(stage_name)),
        }
        if include_remote_trainer_stdio:
            registry[REMOTE_TRAINER_STDIO_LOG_PATHS_KEY] = self.relative(self.remote_trainer_stdio_log)
        return registry


__all__ = [
    "LOGS_DIR_NAME",
    "PIPELINE_LOG_NAME",
    "REMOTE_RUNNER_LOG_NAME",
    "REMOTE_RUNNER_LOG_PATHS_KEY",
    "REMOTE_TRAINER_STDIO_LOG_NAME",
    "REMOTE_TRAINER_STDIO_LOG_PATHS_KEY",
    "STAGE_LOG_PATHS_KEY",
    "STAGE_LOG_SUFFIX",
    "LogLayout",
    "_slugify",
]
