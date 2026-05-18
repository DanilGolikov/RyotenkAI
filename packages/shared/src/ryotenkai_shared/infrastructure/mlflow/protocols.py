"""Narrow Protocols for MLflow integration (target architecture).

Replaces the wide ``IMLflowManager`` (20+ methods, ``Any``-leaky) with
six focused Protocols, each ≤7 methods and fully typed:

* :class:`ITrackingClient`     — open / close / tag / adopt runs.
* :class:`IMetricSink`          — buffered, async metrics.
* :class:`IArtifactSink`        — file upload, checksum-verified.
* :class:`IRunQuery`            — read-path lookup (DI'd; not constructed
  ad-hoc).
* :class:`IModelRegistry`       — alias-based model promotion (replaces
  deprecated stages).
* :class:`IJournalUploader`     — SSOT journal artifact upload, idempotent.
* :class:`IPromptRegistry`      — narrow surface for SystemPromptLoader.

These Protocols live in ``shared`` and may be imported by ``control``,
``pod``, and ``providers``. Concrete implementations live in
``shared/.../mlflow/`` (transport, sinks, registry) or downstream
packages (e.g. ``control.pipeline.mlflow.read.client.MlflowReadClient``).

See ``docs/plans/vectorized-fluttering-mist.md`` for the design.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from enum import StrEnum
from pathlib import Path
from typing import Protocol, runtime_checkable

from ryotenkai_shared.infrastructure.mlflow.run_handle import RunHandle

__all__ = [
    "IArtifactSink",
    "IJournalUploader",
    "IMetricSink",
    "IModelRegistry",
    "IPromptRegistry",
    "IRunHandle",
    "IRunQuery",
    "ITrackingClient",
    "ModelVersion",
    "PromptArtifact",
    "RunStatus",
]


class RunStatus(StrEnum):
    """Terminal and non-terminal MLflow run statuses.

    Maps 1:1 to MLflow's ``RunStatus`` proto enum but pinned as
    ``StrEnum`` so taxonomy values are stable across SDK upgrades.
    """

    RUNNING = "RUNNING"
    FINISHED = "FINISHED"
    FAILED = "FAILED"
    KILLED = "KILLED"


# Re-export RunHandle as IRunHandle alias for naming consistency in
# Protocol-facing surfaces. Callers may use either; type-checker
# treats them as identical.
IRunHandle = RunHandle


class ModelVersion(Protocol):
    """Read-only view of an MLflow model version row.

    Concrete implementations come from ``mlflow.entities.ModelVersion``;
    we restrict to the fields we actually consume to make fakes trivial.
    """

    name: str
    version: str
    run_id: str | None


class PromptArtifact(Protocol):
    """Read-only view of an MLflow Prompt Registry artifact.

    Mirrors the shape returned by ``mlflow.genai.load_prompt`` without
    coupling to the concrete return type.
    """

    name: str
    version: str
    template: str


@runtime_checkable
class ITrackingClient(Protocol):
    """Open / close / tag / adopt runs.

    The ONLY surface allowed to call ``mlflow.start_run`` /
    ``mlflow.end_run`` / ``set_terminated``. All other producers
    (HF MLflowCallback, etc.) attach to runs via env vars.
    """

    def ping(self, timeout_s: float) -> None:
        """Verify reachability + auth. Raises on failure."""
        ...

    def start_run(
        self,
        experiment: str,
        name: str,
        tags: Mapping[str, str],
        params: Mapping[str, str],
    ) -> RunHandle: ...

    def start_nested_run(
        self,
        parent_run_id: str,
        name: str,
        tags: Mapping[str, str],
    ) -> RunHandle: ...

    def adopt_run(self, run_id: str) -> RunHandle:
        """Re-open an existing run by id (resume path)."""
        ...

    def set_terminated(self, run_id: str, status: RunStatus) -> None: ...

    def set_tags(self, run_id: str, tags: Mapping[str, str]) -> None: ...


@runtime_checkable
class IMetricSink(Protocol):
    """Buffered, async metric writer.

    Implementations should wrap ``MlflowClient.log_batch(synchronous=False)``
    with the existing ``MetricsBuffer`` for offline durability.
    """

    def log(self, run_id: str, metrics: Mapping[str, float], step: int) -> None: ...

    def flush(self, run_id: str, blocking: bool) -> None: ...


@runtime_checkable
class IArtifactSink(Protocol):
    """File upload with optional sha256 checksum verification."""

    def upload_file(
        self,
        run_id: str,
        local_path: Path,
        artifact_path: str,
        checksum_sha256: str | None,
    ) -> None: ...


@runtime_checkable
class IRunQuery(Protocol):
    """Read-path lookup. DI'd into every read-site (reports, deletion,
    summary_reporter). Ad-hoc ``MlflowClient()`` constructions are
    lint-forbidden outside the concrete implementation."""

    def get_run(self, run_id: str) -> RunHandle: ...

    def list_children(self, parent_run_id: str) -> Sequence[RunHandle]: ...

    def search(
        self,
        experiment: str,
        filter_: str,
        max_results: int,
    ) -> Sequence[RunHandle]: ...


@runtime_checkable
class IModelRegistry(Protocol):
    """Alias-based model promotion.

    Replaces deprecated stages (``Staging`` / ``Production``). Per
    MLflow 3.x community guidance, aliases (``@champion``,
    ``@challenger``) are the supported mechanism.
    """

    def register(self, model_uri: str, name: str) -> ModelVersion: ...

    def set_alias(self, name: str, alias: str, version: str) -> None: ...

    def resolve_alias(self, name: str, alias: str) -> ModelVersion: ...


@runtime_checkable
class IJournalUploader(Protocol):
    """SSOT journal artifact upload (idempotent via sha256 tag).

    Calls ``set_tags(run_id, {"ryotenkai.journal.sha256": <hex>})`` on
    success; checks the tag before re-uploading to guarantee
    at-most-once semantics under retry."""

    def upload(self, run_id: str, journal_path: Path, sha256: str) -> None: ...


@runtime_checkable
class IPromptRegistry(Protocol):
    """Narrow surface for SystemPromptLoader.

    Concrete impl lives in ``shared.infrastructure.mlflow.transport``
    (delegates to ``mlflow.genai.load_prompt`` with a per-call timeout).
    """

    def load(self, name_or_uri: str, timeout_s: float) -> PromptArtifact | None: ...
