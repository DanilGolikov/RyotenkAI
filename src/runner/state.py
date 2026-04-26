"""Job lifecycle FSM — placeholder for Phase 1.

State diagram (canonical):

    POST /jobs (multipart)
            │
            ▼
       [preparing]   ← unpacking plugins, validating
            │
       prep ok / prep failed
            │
            ▼
        [running] ──► [failed]
            │
       ┌────┴──────┬─────────────┐
       │           │             │
       ▼           ▼             ▼
   [completed] [stopping]   [failed]
                  │
                  ▼
            [completed | cancelled]

Persistence: ``/workspace/.ryotenkai/state.jsonl`` (append-only) plus
``state.json`` (latest snapshot). Both written via
:func:`src.utils.atomic_fs.atomic_write_text`.

Phase 0 ships the enum + dataclass. Real transitions / persistence /
restore arrive in Phase 1.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

__all__ = [
    "JobState",
    "JobSnapshot",
]


class JobState(StrEnum):
    """Canonical FSM states. Order matches the diagram in the module docstring."""

    PREPARING = "preparing"
    RUNNING = "running"
    STOPPING = "stopping"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

    @property
    def is_terminal(self) -> bool:
        """``True`` for states no further transition leaves."""
        return self in {JobState.COMPLETED, JobState.FAILED, JobState.CANCELLED}


@dataclass(frozen=True, slots=True)
class JobSnapshot:
    """Immutable point-in-time view of a single job's state.

    Phase 1 expands this with started_at / finished_at / exit_code etc.
    """

    job_id: str
    state: JobState
    message: str = ""
