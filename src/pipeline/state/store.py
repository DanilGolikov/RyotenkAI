from __future__ import annotations

import hashlib
import json
import os
import tempfile
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from src.pipeline.state.models import PipelineAttemptState, PipelineState, StageLineageRef, StageRunState, utc_now_iso

if TYPE_CHECKING:
    from src.pipeline.domain.run_context import RunContext

SCHEMA_VERSION = 1


class PipelineStateError(RuntimeError):
    """Base error for pipeline state issues."""


class PipelineStateLoadError(PipelineStateError):
    """State load failure."""


class PipelineStateLockError(PipelineStateError):
    """Run lock failure."""


@dataclass(frozen=True, slots=True)
class PipelineRunLock:
    path: Path
    fd: int

    def release(self) -> None:
        with suppress(OSError):
            os.close(self.fd)
        with suppress(FileNotFoundError):
            self.path.unlink()


def acquire_run_lock(lock_path: Path) -> PipelineRunLock:
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError as exc:
        raise PipelineStateLockError(f"Logical run is already locked: {lock_path}") from exc

    payload = f"pid={os.getpid()}\nstarted_at={utc_now_iso()}\n"
    os.write(fd, payload.encode("utf-8"))
    return PipelineRunLock(path=lock_path, fd=fd)


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", delete=False, dir=path.parent, encoding="utf-8") as tmp_file:
        json.dump(payload, tmp_file, ensure_ascii=False, indent=2, sort_keys=True)
        tmp_file.flush()
        os.fsync(tmp_file.fileno())
        tmp_name = tmp_file.name
    Path(tmp_name).replace(path)


class PipelineStateStore:
    def __init__(self, run_directory: Path):
        self.run_directory = run_directory
        self.state_path = self.run_directory / "pipeline_state.json"

    @property
    def attempts_dir(self) -> Path:
        return self.run_directory / "attempts"

    @property
    def lock_path(self) -> Path:
        return self.run_directory / "run.lock"

    def exists(self) -> bool:
        return self.state_path.exists()

    def load(self) -> PipelineState:
        if not self.state_path.exists():
            raise PipelineStateLoadError(f"Missing pipeline state: {self.state_path}")
        try:
            raw = json.loads(self.state_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise PipelineStateLoadError(f"Corrupted pipeline state: {self.state_path}") from exc
        if not isinstance(raw, dict):
            raise PipelineStateLoadError(f"Invalid pipeline state shape: {self.state_path}")
        state = PipelineState.from_dict(raw)
        if state.schema_version != SCHEMA_VERSION:
            raise PipelineStateLoadError(
                f"Unsupported pipeline_state schema_version={state.schema_version}; expected {SCHEMA_VERSION}"
            )
        if not state.logical_run_id:
            raise PipelineStateLoadError(f"Corrupt pipeline state: missing logical_run_id in {self.state_path}")
        if not state.run_directory:
            raise PipelineStateLoadError(f"Corrupt pipeline state: missing run_directory in {self.state_path}")
        return state

    def save(self, state: PipelineState) -> None:
        atomic_write_json(self.state_path, state.to_dict())

    def init_state(
        self,
        *,
        logical_run_id: str,
        config_path: str,
        training_critical_config_hash: str,
        late_stage_config_hash: str,
        root_mlflow_run_id: str | None = None,
    ) -> PipelineState:
        state = PipelineState(
            schema_version=SCHEMA_VERSION,
            logical_run_id=logical_run_id,
            run_directory=str(self.run_directory),
            config_path=config_path,
            active_attempt_id=None,
            pipeline_status=StageRunState.STATUS_PENDING,
            training_critical_config_hash=training_critical_config_hash,
            late_stage_config_hash=late_stage_config_hash,
            root_mlflow_run_id=root_mlflow_run_id,
            attempts=[],
            current_output_lineage={},
        )
        self.save(state)
        return state

    def next_attempt_dir(self, attempt_no: int) -> Path:
        return self.attempts_dir / f"attempt_{attempt_no}"


def build_attempt_id(logical_run_id: str, attempt_no: int) -> str:
    return f"{logical_run_id}:attempt:{attempt_no}"


def build_attempt_state(
    *,
    state: PipelineState,
    run_ctx: RunContext,
    requested_action: str,
    effective_action: str,
    restart_from_stage: str | None,
    enabled_stage_names: list[str],
    training_critical_config_hash: str,
    late_stage_config_hash: str,
) -> PipelineAttemptState:
    attempt_no = len(state.attempts) + 1
    attempt_id = build_attempt_id(state.logical_run_id, attempt_no)
    return PipelineAttemptState(
        attempt_id=attempt_id,
        attempt_no=attempt_no,
        runtime_name=run_ctx.name,
        requested_action=requested_action,
        effective_action=effective_action,
        restart_from_stage=restart_from_stage,
        status=StageRunState.STATUS_RUNNING,
        started_at=utc_now_iso(),
        training_critical_config_hash=training_critical_config_hash,
        late_stage_config_hash=late_stage_config_hash,
        root_mlflow_run_id=state.root_mlflow_run_id,
        enabled_stage_names=list(enabled_stage_names),
        stage_runs={},
    )


def hash_payload(payload: dict[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, ensure_ascii=False, default=str).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def update_lineage(
    lineage: dict[str, StageLineageRef],
    *,
    stage_name: str,
    attempt_id: str,
    outputs: dict[str, Any] | None = None,
    remove: bool = False,
    source_ref: StageLineageRef | None = None,
) -> dict[str, StageLineageRef]:
    new_lineage = dict(lineage)
    if remove:
        new_lineage.pop(stage_name, None)
        return new_lineage
    if source_ref is not None:
        new_lineage[stage_name] = source_ref
        return new_lineage
    new_lineage[stage_name] = StageLineageRef(
        attempt_id=attempt_id,
        stage_name=stage_name,
        outputs=dict(outputs or {}),
    )
    return new_lineage
