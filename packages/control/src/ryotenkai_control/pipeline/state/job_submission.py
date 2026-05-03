"""Persistent record of an in-pod job submission.

When :class:`TrainingLauncher` (Phase 6.3a) submits a job to the
in-pod runner, it stashes the live :class:`JobClient` and
:class:`SSHTunnelManager` on the pipeline context for the monitor
to consume in-process. That handle dies with the launcher process —
which is fine for the happy path, but useless for ``ryotenkai job
status / events / stop`` invoked from a fresh shell ten minutes
later, or for resume after a Mac sleep.

This module bridges that gap: the launcher serialises the SSH
endpoint + ``job_id`` to ``attempts/<n>/job_submission.json`` right
after :meth:`JobClient.submit_job` succeeds; CLI commands rebuild
the connection from the file. No live-state mutation — the file
is written once and treated as immutable for the run's lifetime.

The Mac control plane is the source of truth for "where is the
pod"; the runner is the source of truth for "what is the job
doing". This split keeps the JSON schema stable: it never carries
FSM state, only enough metadata to dial the runner back in.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, ClassVar

from ryotenkai_shared.utils.atomic_fs import atomic_write_json

__all__ = [
    "JOB_SUBMISSION_FILENAME",
    "JobSubmission",
    "JobSubmissionLoadError",
    "load_job_submission",
    "save_job_submission",
]


JOB_SUBMISSION_FILENAME = "job_submission.json"


# ---------------------------------------------------------------------------
# Error
# ---------------------------------------------------------------------------


class JobSubmissionLoadError(RuntimeError):
    """Submission file missing, malformed, or schema-incompatible."""


# ---------------------------------------------------------------------------
# Record
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class JobSubmission:
    """Immutable snapshot of where a job lives in the cloud.

    Field set is deliberately narrow — anything that can change at
    runtime (FSM state, latest event offset, GPU usage) is fetched
    live from the runner over the SSH tunnel, never read from this
    file. ``schema_version`` exists so a future change to the
    record shape can fail loud rather than silently load with
    missing keys.
    """

    schema_version: int
    job_id: str
    provider_name: str
    pod_id: str | None
    ssh_host: str
    ssh_port: int
    ssh_username: str
    ssh_key_path: str | None
    created_at_iso: str

    # Bumped when the JSON shape changes in a backward-incompatible
    # way (field renamed / removed). Adding optional fields is
    # version-neutral — the loader fills them in with ``None``.
    # ``ClassVar`` so dataclass treats this as a class constant, not
    # an instance field — otherwise the slots+frozen dance promotes
    # it to a slot and ``asdict`` chokes on the slot descriptor.
    CURRENT_VERSION: ClassVar[int] = 1

    @classmethod
    def now(
        cls,
        *,
        job_id: str,
        provider_name: str,
        pod_id: str | None,
        ssh_host: str,
        ssh_port: int,
        ssh_username: str,
        ssh_key_path: str | None,
    ) -> JobSubmission:
        """Construct with ``created_at_iso`` set to the current UTC
        instant. Tests inject a fixed timestamp via the regular
        :class:`JobSubmission` constructor."""
        return cls(
            schema_version=cls.CURRENT_VERSION,
            job_id=job_id,
            provider_name=provider_name,
            pod_id=pod_id,
            ssh_host=ssh_host,
            ssh_port=int(ssh_port),
            ssh_username=ssh_username,
            ssh_key_path=ssh_key_path,
            created_at_iso=datetime.now(tz=timezone.utc).isoformat(),
        )

    def to_dict(self) -> dict[str, Any]:
        """Plain-dict view for :func:`atomic_write_json`."""
        return asdict(self)


# ---------------------------------------------------------------------------
# IO
# ---------------------------------------------------------------------------


def save_job_submission(attempt_dir: Path, submission: JobSubmission) -> Path:
    """Write ``submission`` to ``<attempt_dir>/job_submission.json``.

    Creates ``attempt_dir`` if it doesn't exist (the orchestrator
    usually pre-creates it, but we don't want a launcher race to
    eat the submission). Returns the resolved path for logging.
    """
    attempt_dir = attempt_dir.expanduser().resolve()
    attempt_dir.mkdir(parents=True, exist_ok=True)
    target = attempt_dir / JOB_SUBMISSION_FILENAME
    atomic_write_json(target, submission.to_dict())
    return target


def load_job_submission(attempt_dir: Path) -> JobSubmission:
    """Read the submission file and validate the schema version.

    Raises:
        JobSubmissionLoadError: file missing, JSON malformed, or the
            ``schema_version`` field is newer / unknown.
    """
    target = attempt_dir.expanduser().resolve() / JOB_SUBMISSION_FILENAME
    if not target.is_file():
        raise JobSubmissionLoadError(
            f"job submission record not found: {target}",
        )

    import json as _json

    try:
        payload = _json.loads(target.read_text(encoding="utf-8"))
    except (OSError, _json.JSONDecodeError) as exc:
        raise JobSubmissionLoadError(
            f"failed to read {target}: {exc}",
        ) from exc

    version = payload.get("schema_version")
    if version != JobSubmission.CURRENT_VERSION:
        raise JobSubmissionLoadError(
            f"unsupported job submission schema_version={version!r} "
            f"(this build expects {JobSubmission.CURRENT_VERSION})",
        )

    try:
        return JobSubmission(
            schema_version=int(payload["schema_version"]),
            job_id=str(payload["job_id"]),
            provider_name=str(payload["provider_name"]),
            pod_id=payload.get("pod_id"),
            ssh_host=str(payload["ssh_host"]),
            ssh_port=int(payload["ssh_port"]),
            ssh_username=str(payload["ssh_username"]),
            ssh_key_path=payload.get("ssh_key_path"),
            created_at_iso=str(payload["created_at_iso"]),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise JobSubmissionLoadError(
            f"malformed job submission at {target}: {exc}",
        ) from exc
