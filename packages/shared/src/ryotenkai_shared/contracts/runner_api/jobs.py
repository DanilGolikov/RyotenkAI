"""Job lifecycle DTOs — multipart submit + status snapshot + stop."""

from __future__ import annotations

from pydantic import Field

from ._strict import _StrictModel


class JobSpec(_StrictModel):
    """Body of the multipart ``POST /jobs`` request — the JSON part.

    Phase 1 keeps the schema minimal so the wire contract stays
    forward-compatible. Phase 6 (cutover) will add the strategy
    chain, dataset paths, and topology fields once the pipeline
    integration writes them.
    """

    job_id: str = Field(
        min_length=1,
        max_length=128,
        description=(
            "Caller-supplied identifier (typically the attempt's "
            "logical_run_id). Echoed back in every event and used as "
            "the path key for ``GET /jobs/{job_id}``."
        ),
    )
    command: list[str] = Field(
        min_length=1,
        description=(
            "argv-style command the supervisor exec()s as the trainer "
            "subprocess. The Mac client typically sends "
            "``['python', '-m', 'ryotenkai_pod.trainer.run_training', ...]``. "
            "The first element must resolve in PATH or be an "
            "absolute path; ``min_length=1`` rejects empty argv."
        ),
    )
    env: dict[str, str] = Field(
        default_factory=dict,
        description=(
            "Extra environment variables to merge over the runner's "
            "own ``os.environ`` for the trainer subprocess. Use "
            "this for HF_TOKEN, MLFLOW_*, RYOTENKAI_RUNNER_URL, etc."
        ),
    )
    workdir: str | None = Field(
        default=None,
        description=(
            "Absolute pod-side directory to spawn the trainer in. "
            "Forwarded to ``asyncio.create_subprocess_exec(cwd=...)`` "
            "so the trainer's relative-path resolution lines up with "
            "the run's workspace (config/, data/, src/ are all "
            "rsync'd under this directory). When ``None`` the trainer "
            "inherits uvicorn's cwd — usually ``/root`` post-SSH "
            "launch, which would misresolve ``--config "
            "config/pipeline_config.yaml`` to ``/root/config/...`` "
            "and crash with FileNotFoundError. Mac clients should "
            "always set this to ``ssh_info.workspace_path``."
        ),
    )


class JobSubmittedResponse(_StrictModel):
    """Reply to a successful ``POST /jobs``."""

    job_id: str = Field(description="Echo of :attr:`JobSpec.job_id`.")
    sequence: int = Field(
        description="FSM sequence at acceptance (always 0 — initial PREPARING).",
    )
    offset: int = Field(
        description=(
            "Event-bus offset of the synthetic ``job_submitted`` event "
            "the bus emits in lockstep with the FSM transition. The "
            "client uses this as the ``since=`` cursor for an "
            "immediate WebSocket subscribe to avoid a race where it "
            "misses early events."
        ),
    )


class JobSnapshotResponse(_StrictModel):
    """``GET /jobs/{job_id}`` — current FSM snapshot."""

    job_id: str
    state: str = Field(
        description=(
            "Current FSM state — one of ``preparing``, ``running``, "
            "``stopping``, ``completed``, ``failed``, ``cancelled``."
        ),
    )
    sequence: int
    started_at: str
    updated_at: str
    message: str
    last_event_offset: int = Field(
        description=(
            "Cursor of the most recent event on the bus. Lets the "
            "client subscribe with ``since=last_event_offset + 1`` to "
            "consume only events newer than this snapshot."
        ),
    )


class JobStopAcceptedResponse(_StrictModel):
    """``POST /jobs/{job_id}/stop`` — graceful-stop request accepted."""

    job_id: str
    state: str = Field(description="State after the synchronous transition (always ``stopping``).")
    sequence: int


__all__ = [
    "JobSnapshotResponse",
    "JobSpec",
    "JobStopAcceptedResponse",
    "JobSubmittedResponse",
]
