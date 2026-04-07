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


@dataclass(frozen=True, slots=True)
class RuntimeSettings:
    """Immutable snapshot of runtime environment at process start."""

    runs_base_dir: Path = field(default_factory=lambda: Path("runs"))
    log_level: str = "INFO"
    file_logs_enabled: bool = True


def load_runtime_settings() -> RuntimeSettings:
    """Read all control-plane env vars once and return a frozen settings object."""
    return RuntimeSettings(
        runs_base_dir=Path(os.environ.get("RYOTENKAI_RUNS_DIR", "runs")),
        log_level=os.environ.get("LOG_LEVEL", "INFO").upper(),
        file_logs_enabled=os.environ.get("HELIX_NO_FILE_LOGS") != "1",
    )
