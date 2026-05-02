"""PodLayout — single source of truth for pod-side filesystem layout.

All pod-side artefacts for a single run live under ``<root>/`` where
``root = <provider-base>/runs/<run_id>``:

    <root>/
    ├── src/                          # CodeSyncer rsync target
    ├── config/
    │   └── pipeline_config.yaml
    ├── data/                         # uploaded datasets
    ├── community/                    # PluginUnpacker output
    ├── output/                       # checkpoints, model artifacts
    ├── logs/
    │   ├── runner.log                # uvicorn / FastAPI runner stdout
    │   └── trainer.stdio.log         # trainer subprocess stdout/stderr
    ├── events/                       # EventJournal (control + telemetry)
    │   └── events.NNN.jsonl
    └── state/                        # JobLifecycleFSM persistence
        ├── job.json
        └── job.jsonl

Provider-agnostic by design: providers own only the ``root`` value;
the directory structure under ``root`` is identical for every
provider. RunPod's ``root`` is ``/workspace/runs/<run_id>``,
single_node's is ``<config.workspace_path>/runs/<run_id>``, etc.

Parallel structure to :class:`src.utils.logs_layout.LogLayout` (Mac
side). The ``runner.log`` / ``trainer.stdio.log`` filenames match on
both sides — LogManager scp does a 1:1 mapping.

Consumers (orchestrator, supervisor, runner main, log_manager,
gpu_deployer, code_syncer, file_uploader) MUST go through this class
— no module is allowed to construct pod-side paths from raw string
literals.
"""

from __future__ import annotations

import shlex
from dataclasses import dataclass
from pathlib import PurePosixPath

LOGS_DIR_NAME = "logs"
EVENTS_DIR_NAME = "events"
STATE_DIR_NAME = "state"
SRC_DIR_NAME = "src"
CONFIG_DIR_NAME = "config"
DATA_DIR_NAME = "data"
COMMUNITY_DIR_NAME = "community"
OUTPUT_DIR_NAME = "output"

CONFIG_FILE_NAME = "pipeline_config.yaml"
RUNNER_LOG_NAME = "runner.log"
TRAINER_STDIO_LOG_NAME = "trainer.stdio.log"
FSM_STATE_JSON_NAME = "job.json"
FSM_STATE_JSONL_NAME = "job.jsonl"


@dataclass(frozen=True)
class PodLayout:
    """Pod-side filesystem layout for a single run — single source of truth.

    All paths are :class:`pathlib.PurePosixPath` because the pod is always
    Linux. Mac-side conversion (when paths cross the SSH boundary) is the
    caller's responsibility — convert via ``str(layout.runner_log)``.

    Provider-agnostic: identical structure under any ``root``. Providers
    construct PodLayout via :meth:`from_root` with their own root.
    """

    root: PurePosixPath

    @classmethod
    def from_root(cls, root: PurePosixPath | str) -> "PodLayout":
        """Construct a layout for the given run-root path.

        ``root`` must be an absolute path. Relative roots are rejected
        because every consumer of this class assumes absolute paths
        (SSH commands, mkdir -p, etc.).
        """
        path = PurePosixPath(root)
        if not path.is_absolute():
            raise ValueError(
                f"PodLayout.root must be absolute, got: {root!r}",
            )
        return cls(root=path)

    @property
    def src_dir(self) -> PurePosixPath:
        return self.root / SRC_DIR_NAME

    @property
    def config_dir(self) -> PurePosixPath:
        return self.root / CONFIG_DIR_NAME

    @property
    def data_dir(self) -> PurePosixPath:
        return self.root / DATA_DIR_NAME

    @property
    def community_dir(self) -> PurePosixPath:
        return self.root / COMMUNITY_DIR_NAME

    @property
    def output_dir(self) -> PurePosixPath:
        return self.root / OUTPUT_DIR_NAME

    @property
    def logs_dir(self) -> PurePosixPath:
        return self.root / LOGS_DIR_NAME

    @property
    def events_dir(self) -> PurePosixPath:
        return self.root / EVENTS_DIR_NAME

    @property
    def state_dir(self) -> PurePosixPath:
        return self.root / STATE_DIR_NAME

    @property
    def config_file(self) -> PurePosixPath:
        return self.config_dir / CONFIG_FILE_NAME

    @property
    def runner_log(self) -> PurePosixPath:
        return self.logs_dir / RUNNER_LOG_NAME

    @property
    def trainer_stdio_log(self) -> PurePosixPath:
        return self.logs_dir / TRAINER_STDIO_LOG_NAME

    @property
    def fsm_state_json(self) -> PurePosixPath:
        return self.state_dir / FSM_STATE_JSON_NAME

    @property
    def fsm_state_jsonl(self) -> PurePosixPath:
        return self.state_dir / FSM_STATE_JSONL_NAME

    def all_dirs(self) -> tuple[PurePosixPath, ...]:
        """Every directory the layout owns. Used by :meth:`ensure_dirs_command`
        and by tests asserting completeness.
        """
        return (
            self.src_dir,
            self.config_dir,
            self.data_dir,
            self.community_dir,
            self.output_dir,
            self.logs_dir,
            self.events_dir,
            self.state_dir,
        )

    def ensure_dirs_command(self) -> str:
        """Single idempotent shell command that creates every directory
        in the layout.

        Each path is shell-escaped, so spaces and metacharacters in
        ``root`` are safe. Use via ``ssh ... '<command>'`` or
        ``subprocess.run(["sh", "-c", command])``.
        """
        quoted = " ".join(shlex.quote(str(d)) for d in self.all_dirs())
        return f"mkdir -p {quoted}"


__all__ = [
    "COMMUNITY_DIR_NAME",
    "CONFIG_DIR_NAME",
    "CONFIG_FILE_NAME",
    "DATA_DIR_NAME",
    "EVENTS_DIR_NAME",
    "FSM_STATE_JSONL_NAME",
    "FSM_STATE_JSON_NAME",
    "LOGS_DIR_NAME",
    "OUTPUT_DIR_NAME",
    "PodLayout",
    "RUNNER_LOG_NAME",
    "SRC_DIR_NAME",
    "STATE_DIR_NAME",
    "TRAINER_STDIO_LOG_NAME",
]
