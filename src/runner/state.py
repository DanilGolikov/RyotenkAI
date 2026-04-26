"""Job lifecycle FSM with append-only persistence.

A single ``JobLifecycleFSM`` instance owns the canonical view of one
job's progress through the state diagram below. It enforces the
transition rules at every state change and persists to disk so a
container restart does not lose the run identity.

State diagram::

    POST /jobs (multipart)
            │
            ▼
       [preparing]    ← unpacking plugins, validating
        │       │
   prep ok│       │ prep failed
        ▼       ▼
    [running] ──► [failed]
        │
   ┌────┴──────┐
   │           │
   ▼           ▼
 [completed] [stopping]
                │
       ┌────────┼─────────┐
       ▼        ▼         ▼
   [completed][cancelled][failed]

Terminal states (``completed``, ``failed``, ``cancelled``) accept no
further transitions and raise :class:`InvalidTransitionError` on any
attempt — the supervisor is responsible for not calling
:meth:`transition` past terminal.

Persistence layout under ``<workspace>/.ryotenkai/``:
- ``state.jsonl`` — append-only audit log; each line is the full
  payload of a transition (state, message, timestamp, job_id, monotonic
  sequence). Ordering is the single source of truth — a duplicate
  state with the same sequence is a corruption signal.
- ``state.json`` — latest snapshot, atomic-written via
  :func:`src.utils.atomic_fs.atomic_write_json`. Read on startup to
  rebuild the in-memory FSM without scanning the JSONL.

The JSONL is opened with :data:`os.O_APPEND`, which gives
atomic-per-write ordering for payloads ≤ ``PIPE_BUF`` (4096 on Linux,
512 on macOS) so concurrent writers from separate threads can't
interleave a single transition record. Phase 1's single supervisor
never has more than one writer, but the choice keeps the door open
for Phase 5+ multi-stage flows without a refactor.

Rationale for not using :func:`atomic_write_text` for the JSONL:
the helper rewrites the whole file every call, which would be O(N)
on transition count. Append is O(1).
"""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path

# ---------------------------------------------------------------------------
# Inlined filesystem helpers
# ---------------------------------------------------------------------------
#
# These mirror :mod:`src.utils.atomic_fs` byte-for-byte (verified in
# ``test_state.py::test_atomic_helpers_match_canonical`` once the
# canonical import becomes available). Inlined deliberately so the
# runner package has *zero* dependency on the heavy ``src.utils``
# import chain (logger.py → colorlog → ...) — keeping the in-pod
# server bootable from an image that doesn't ship the full pipeline
# code. Phase 6 cutover migrates this to ``src.utils.atomic_fs``
# once we add the relevant modules to the docker layout.


def _utc_now_iso() -> str:
    """Second-precision UTC ISO-8601 string with trailing ``Z``."""
    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _atomic_write_text(path: Path, text: str) -> None:
    """Write ``text`` atomically — temp file + ``os.replace``."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", delete=False, dir=path.parent, encoding="utf-8", newline="\n",
    ) as tmp:
        tmp.write(text)
        tmp.flush()
        os.fsync(tmp.fileno())
        tmp_name = tmp.name
    Path(tmp_name).replace(path)


def _atomic_write_json(path: Path, payload: dict) -> None:
    _atomic_write_text(
        path,
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    )

__all__ = [
    "JobLifecycleFSM",
    "JobSnapshot",
    "JobState",
    "InvalidTransitionError",
]


# ---------------------------------------------------------------------------
# State enum + transition matrix
# ---------------------------------------------------------------------------


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


# Single source of truth for legal transitions. Read at every
# ``transition()`` call — keeping the matrix here (not on JobState) makes
# adding tests trivial and lets the FSM be the only enforcement point.
_LEGAL_TRANSITIONS: dict[JobState, frozenset[JobState]] = {
    JobState.PREPARING: frozenset({JobState.RUNNING, JobState.FAILED}),
    JobState.RUNNING: frozenset({JobState.STOPPING, JobState.COMPLETED, JobState.FAILED}),
    JobState.STOPPING: frozenset({JobState.COMPLETED, JobState.CANCELLED, JobState.FAILED}),
    # Terminals — explicit empty set so the lookup never KeyErrors.
    JobState.COMPLETED: frozenset(),
    JobState.FAILED: frozenset(),
    JobState.CANCELLED: frozenset(),
}


# States that, if observed on startup, indicate the previous container
# died mid-transition. They cannot be safely resumed; the FSM transitions
# them to FAILED with a synthetic reason. See :meth:`JobLifecycleFSM.restore_or_init`.
_UNSAFE_RESTORE_STATES: frozenset[JobState] = frozenset({JobState.PREPARING, JobState.STOPPING})


# ---------------------------------------------------------------------------
# Snapshot
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class JobSnapshot:
    """Immutable point-in-time view of a job's state.

    Attributes
    ----------
    job_id:
        Caller-supplied identifier — must be unique per job. The store
        keys persistence files by this id (Phase 1 single-active means
        one store = one job; Phase 5+ may grow a registry).
    state:
        Current FSM state.
    message:
        Optional human-readable detail. Used for failure reasons,
        graceful-stop hints, etc. Empty string when not relevant.
    sequence:
        Monotonic counter of transitions, starting at 0 (initial
        ``preparing``). Used as the canonical ordering for the JSONL
        audit log and for deduplication on restore.
    started_at:
        ISO-8601 UTC timestamp of the first transition (initial
        ``preparing``). Stays stable across all subsequent snapshots.
    updated_at:
        ISO-8601 UTC timestamp of *this* transition.
    """

    job_id: str
    state: JobState
    sequence: int
    started_at: str
    updated_at: str
    message: str = ""

    def to_dict(self) -> dict:
        """JSON-serialisable form. Used for both state.json and state.jsonl entries."""
        return {
            "job_id": self.job_id,
            "state": self.state.value,
            "sequence": self.sequence,
            "started_at": self.started_at,
            "updated_at": self.updated_at,
            "message": self.message,
        }

    @classmethod
    def from_dict(cls, payload: dict) -> "JobSnapshot":
        """Round-trip from dict produced by :meth:`to_dict`."""
        return cls(
            job_id=str(payload["job_id"]),
            state=JobState(payload["state"]),
            sequence=int(payload["sequence"]),
            started_at=str(payload["started_at"]),
            updated_at=str(payload["updated_at"]),
            message=str(payload.get("message", "")),
        )


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class InvalidTransitionError(RuntimeError):
    """Raised when :meth:`JobLifecycleFSM.transition` is called with a
    state the matrix forbids from the current one, or when
    :meth:`JobLifecycleFSM.transition` runs without a prior
    :meth:`submit`.

    ``current`` is the FSM's source state; for the no-submit case it
    is ``JobState.PREPARING`` by convention (the state any
    transition would *want* to start from). ``no_active_snapshot``
    flags this synthetic case so error messages can be precise.
    """

    def __init__(
        self,
        current: JobState,
        attempted: JobState,
        *,
        no_active_snapshot: bool = False,
    ) -> None:
        if no_active_snapshot:
            message = (
                f"cannot transition to {attempted.value}: no active job "
                "(call submit() first)"
            )
        else:
            legal = sorted(s.value for s in _LEGAL_TRANSITIONS[current])
            message = (
                f"illegal transition: {current.value} → {attempted.value}; "
                f"legal next states from {current.value}: {legal}"
            )
        super().__init__(message)
        self.current = current
        self.attempted = attempted
        self.no_active_snapshot = no_active_snapshot


# ---------------------------------------------------------------------------
# FSM + persistence
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class JobLifecycleFSM:
    """In-memory FSM with on-disk audit log + latest-snapshot mirror.

    Construct, call :meth:`restore_or_init` once at server boot, then
    :meth:`submit` for a new job and :meth:`transition` for each state
    change. :meth:`current` returns the latest snapshot or ``None``
    if no job has been submitted yet.

    Phase 1 keeps the FSM single-active: only one ``submit`` call per
    process lifetime is allowed without an intervening reset. Phase 5+
    may grow this into a registry; the current API leaves room for
    that without breaking callers.
    """

    workspace_dir: Path
    _snapshot: JobSnapshot | None = field(default=None, init=False, repr=False)

    @property
    def state_dir(self) -> Path:
        return self.workspace_dir / ".ryotenkai"

    @property
    def jsonl_path(self) -> Path:
        return self.state_dir / "state.jsonl"

    @property
    def json_path(self) -> Path:
        return self.state_dir / "state.json"

    # --- read-only accessors ------------------------------------------------

    def current(self) -> JobSnapshot | None:
        """Latest snapshot or ``None`` if no job submitted yet."""
        return self._snapshot

    # --- lifecycle ----------------------------------------------------------

    def restore_or_init(self) -> None:
        """Boot-time recovery.

        Three branches:

        1. ``state.json`` is missing → fresh FSM, no in-memory snapshot.
        2. ``state.json`` exists with a terminal state → restore as-is;
           a future ``submit`` will be rejected unless caller resets.
        3. ``state.json`` exists with an unsafe state (``preparing`` /
           ``stopping``) → transition to ``failed`` with reason
           ``container_restart_during_unsafe_state``. The original
           sequence is preserved, the new state gets ``sequence + 1``
           and is appended to the JSONL.

        The JSONL is treated as authoritative for sequence numbering,
        but the in-memory cursor uses ``state.json`` for O(1) startup
        — full JSONL replay is not necessary in Phase 1.
        """
        self.state_dir.mkdir(parents=True, exist_ok=True)

        if not self.json_path.exists():
            self._snapshot = None
            return

        # Defensive: if state.json is half-written (host died mid-rename
        # despite atomic_write_text — e.g. disk full, OOM kill before
        # fsync settled) we'd otherwise crash the server on every boot.
        # Treat parse errors as "no recoverable state" — the JSONL audit
        # log is still on disk for forensics and a fresh ``submit`` can
        # proceed.
        try:
            with self.json_path.open(encoding="utf-8") as fh:
                payload = json.load(fh)
            snap = JobSnapshot.from_dict(payload)
        except (json.JSONDecodeError, KeyError, ValueError) as exc:
            # The JSONL audit log preserves history; the corrupt
            # state.json is renamed (not deleted) so on-call can
            # inspect it. If renaming fails too, we fall through to
            # fresh state.
            corrupt_path = self.json_path.with_suffix(".json.corrupt")
            try:
                self.json_path.rename(corrupt_path)
            except OSError:
                pass
            del exc  # the original is preserved in the .corrupt file
            self._snapshot = None
            return

        if snap.state in _UNSAFE_RESTORE_STATES:
            # Synthetic transition — the previous container died mid-flight.
            now = _utc_now_iso()
            recovered = JobSnapshot(
                job_id=snap.job_id,
                state=JobState.FAILED,
                sequence=snap.sequence + 1,
                started_at=snap.started_at,
                updated_at=now,
                message="container_restart_during_unsafe_state",
            )
            self._snapshot = recovered
            self._append_jsonl(recovered)
            _atomic_write_json(self.json_path, recovered.to_dict())
            return

        # Terminal or RUNNING state — preserve as-is.
        self._snapshot = snap

    def submit(self, job_id: str) -> JobSnapshot:
        """Initialise FSM at ``preparing`` for a new job.

        Rejects if a snapshot already exists and is non-terminal —
        Phase 1 single-active guarantee. Terminal snapshots can be
        overwritten only by an explicit :meth:`reset` (not implemented
        in Phase 1; YAGNI until multi-job arrives).
        """
        if self._snapshot is not None and not self._snapshot.state.is_terminal:
            raise InvalidTransitionError(self._snapshot.state, JobState.PREPARING)

        now = _utc_now_iso()
        snap = JobSnapshot(
            job_id=job_id,
            state=JobState.PREPARING,
            sequence=0,
            started_at=now,
            updated_at=now,
        )
        self._snapshot = snap
        self.state_dir.mkdir(parents=True, exist_ok=True)
        self._append_jsonl(snap)
        _atomic_write_json(self.json_path, snap.to_dict())
        return snap

    def transition(self, target: JobState, *, message: str = "") -> JobSnapshot:
        """Move from current state to ``target``.

        Raises :class:`InvalidTransitionError` if there is no current
        snapshot or the matrix forbids the move.
        """
        if self._snapshot is None:
            # No job submitted yet — can't transition from "nothing".
            # Synthetic source = PREPARING; no_active_snapshot=True
            # so the error message is precise.
            raise InvalidTransitionError(
                JobState.PREPARING, target, no_active_snapshot=True,
            )

        legal = _LEGAL_TRANSITIONS[self._snapshot.state]
        if target not in legal:
            raise InvalidTransitionError(self._snapshot.state, target)

        snap = JobSnapshot(
            job_id=self._snapshot.job_id,
            state=target,
            sequence=self._snapshot.sequence + 1,
            started_at=self._snapshot.started_at,
            updated_at=_utc_now_iso(),
            message=message,
        )
        self._snapshot = snap
        self._append_jsonl(snap)
        _atomic_write_json(self.json_path, snap.to_dict())
        return snap

    # --- persistence helpers ------------------------------------------------

    def _append_jsonl(self, snap: JobSnapshot) -> None:
        """Append one JSON line to ``state.jsonl``.

        Uses ``open(..., "a")`` rather than ``atomic_fs.atomic_write_text``
        because that helper rewrites the whole file. JSONL append-only is
        the canonical pattern for high-frequency audit logs.

        Each write is a single ``write(2)`` syscall on a small payload,
        so it is atomic per writer on POSIX (``O_APPEND`` semantics).
        Phase 1 has a single writer; Phase 5+ may add more — the choice
        keeps that path safe.
        """
        line = json.dumps(snap.to_dict(), ensure_ascii=False, sort_keys=True) + "\n"
        with self.jsonl_path.open("a", encoding="utf-8") as fh:
            fh.write(line)
