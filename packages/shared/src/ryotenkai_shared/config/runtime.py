"""Runtime settings resolved once from environment at process start.

All env vars used by the control-plane are read here and nowhere else.
Modules receive ``RuntimeSettings`` via constructor injection — they never
call ``os.environ`` directly for these keys.

Usage::

    settings = load_runtime_settings()
    orchestrator = PipelineOrchestrator(config, settings=settings)
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path


def workspace_root() -> Path:
    """Walk up from this file to the uv workspace root.

    Identified by an ancestor that has both ``pyproject.toml`` and a
    ``packages/`` directory. Mirrors the same shape used by
    :func:`ryotenkai_shared.config.secrets.loader.load_secrets`. Falls
    back to ``Path.cwd()`` if the marker isn't found (e.g. tests
    running outside the workspace tree).
    """
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "pyproject.toml").is_file() and (parent / "packages").is_dir():
            return parent
    return Path.cwd()


def _resolve_runs_base_dir(raw: str) -> Path:
    """Anchor a relative ``runs_base_dir`` at the workspace root.

    The pipeline worker subprocess runs with ``cwd=packages/control/src/``
    (legacy spawn cwd from the pre-packagization layout) — interpreting
    ``"runs"`` against that CWD silently buries each pipeline run under
    ``packages/control/src/runs/``. Relative paths now resolve against
    the workspace root so ``runs/`` lives at the repo root regardless of
    the worker's CWD; absolute paths are passed through unchanged.
    """
    candidate = Path(raw)
    if candidate.is_absolute():
        return candidate
    return workspace_root() / candidate


@dataclass(frozen=True, slots=True)
class RuntimeSettings:
    """Immutable snapshot of runtime environment at process start."""

    runs_base_dir: Path = field(default_factory=lambda: _resolve_runs_base_dir("runs"))
    log_level: str = "INFO"


def load_runtime_settings() -> RuntimeSettings:
    """Read all control-plane env vars once and return a frozen settings object."""
    return RuntimeSettings(
        runs_base_dir=_resolve_runs_base_dir(os.environ.get("RYOTENKAI_RUNS_DIR", "runs")),
        log_level=os.environ.get("LOG_LEVEL", "INFO").upper(),
    )
