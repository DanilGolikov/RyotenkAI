from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any, ClassVar

from ryotenkai_shared.contracts.pipeline_conditions import Condition

if TYPE_CHECKING:
    from ryotenkai_shared.errors import RyotenkAIError


def utc_now_iso() -> str:
    return datetime.now(UTC).isoformat()


def _copy_conditions(value: Any) -> list[Condition]:
    """Deserialise ``conditions[]`` defensively.

    Accepts a list of :class:`Condition` instances (already typed) or
    of dicts (just loaded from JSON). Invalid entries are silently
    dropped — the conditions side-channel is observability, never a
    correctness gate; one malformed entry should not block load of an
    otherwise-valid attempt.
    """
    if not isinstance(value, list):
        return []
    out: list[Condition] = []
    for item in value:
        if isinstance(item, Condition):
            out.append(item)
        elif isinstance(item, dict):
            try:
                out.append(Condition.model_validate(item))
            except Exception:  # noqa: BLE001 — observability fall-through
                continue
    return out


def _copy_dict(value: dict[str, Any] | None) -> dict[str, Any]:
    if not isinstance(value, dict):
        return {}
    return dict(value)


def _copy_str_dict(value: Any) -> dict[str, str]:
    if not isinstance(value, dict):
        return {}
    return {str(k): str(v) for k, v in value.items() if isinstance(v, (str, int, float))}


@dataclass(slots=True)
class StageLineageRef:
    attempt_id: str
    stage_name: str
    outputs: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "attempt_id": self.attempt_id,
            "stage_name": self.stage_name,
            "outputs": dict(self.outputs),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> StageLineageRef:
        return cls(
            attempt_id=str(data.get("attempt_id", "")),
            stage_name=str(data.get("stage_name", "")),
            outputs=_copy_dict(data.get("outputs")),
        )


@dataclass(slots=True)
class StageRunState:
    STATUS_PENDING: ClassVar[str] = "pending"
    STATUS_RUNNING: ClassVar[str] = "running"
    STATUS_COMPLETED: ClassVar[str] = "completed"
    STATUS_FAILED: ClassVar[str] = "failed"
    STATUS_INTERRUPTED: ClassVar[str] = "interrupted"
    STATUS_STALE: ClassVar[str] = "stale"
    STATUS_SKIPPED: ClassVar[str] = "skipped"

    MODE_EXECUTED: ClassVar[str] = "executed"
    MODE_REUSED: ClassVar[str] = "reused"
    MODE_SKIPPED: ClassVar[str] = "skipped"

    stage_name: str
    status: str = STATUS_PENDING
    execution_mode: str | None = None
    outputs: dict[str, Any] = field(default_factory=dict)
    error: str | None = None
    failure_kind: str | None = None
    reuse_from: dict[str, Any] | None = None
    skip_reason: str | None = None
    started_at: str | None = None
    completed_at: str | None = None
    log_paths: dict[str, str] = field(default_factory=dict)
    #: Phase G (Layer 10) — k8s/Operator-style conditions[]. Each entry
    #: is an observation (type + status + reason + message + ts);
    #: multiple can be true simultaneously. Source of truth for
    #: lifecycle state remains ``status`` (the FSM); conditions are a
    #: side-channel for warnings / progress hints. Empty list for
    #: legacy state.json files — additive field, no schema bump
    #: required.
    conditions: list[Condition] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "stage_name": self.stage_name,
            "status": self.status,
            "execution_mode": self.execution_mode,
            "outputs": dict(self.outputs),
            "error": self.error,
            "failure_kind": self.failure_kind,
            "reuse_from": dict(self.reuse_from) if isinstance(self.reuse_from, dict) else self.reuse_from,
            "skip_reason": self.skip_reason,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "log_paths": dict(self.log_paths),
            # Phase G — empty list ⇒ omit from JSON so legacy diff
            # tools don't see spurious empty arrays on stages that
            # haven't emitted any conditions yet.
            **(
                {"conditions": [c.model_dump(mode="json") for c in self.conditions]}
                if self.conditions
                else {}
            ),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any], *, stage_name: str | None = None) -> StageRunState:
        return cls(
            stage_name=str(stage_name or data.get("stage_name", "")),
            status=str(data.get("status", cls.STATUS_PENDING)),
            execution_mode=data.get("execution_mode"),
            outputs=_copy_dict(data.get("outputs")),
            error=data.get("error"),
            failure_kind=data.get("failure_kind"),
            reuse_from=data.get("reuse_from"),
            skip_reason=data.get("skip_reason"),
            started_at=data.get("started_at"),
            completed_at=data.get("completed_at"),
            log_paths=_copy_str_dict(data.get("log_paths")),
            conditions=_copy_conditions(data.get("conditions")),
        )


@dataclass(slots=True)
class PodMetadata:
    """Phase 11.C — pod identity persisted on the attempt.

    Lets ``ryotenkai run resume`` see whether there's a stopped pod
    waiting for retrieval (pod sleeping with /workspace intact) vs. a
    legacy run (no metadata recorded) vs. a terminated pod (gone for
    good). The probe (``PodAvailabilityProbe``) reads this on resume
    to decide whether to call ``podResume`` before re-running the
    pipeline.

    Fields:
        pod_id: Provider-side identifier (RunPod pod_id).
        provider: ``"runpod"`` / ``"single_node"`` / etc.
        created_at: ISO-8601 — for forensics.
        last_known_status: Coarse status string updated by the runner
            on transition events. ``"running"`` while training,
            ``"stopped"`` after Phase 11.B's ``podStop`` action,
            ``"terminated"`` after ``podTerminate``. ``None`` when
            we haven't queried yet (fresh attempt).

    Sealed-defaults: every field except pod_id is optional. Legacy
    runs (no metadata in JSON) deserialize to ``None``; the probe
    treats them as "RUNNING / unknown" and skips the resume step.
    """
    pod_id: str
    provider: str = "runpod"
    created_at: str = ""
    last_known_status: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "pod_id": self.pod_id,
            "provider": self.provider,
            "created_at": self.created_at,
            "last_known_status": self.last_known_status,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PodMetadata | None:
        # Empty dict / missing pod_id ⇒ no metadata. Legacy attempts
        # produce {} or omit the key entirely; both yield ``None``
        # so the probe knows to skip.
        pod_id = data.get("pod_id")
        if not isinstance(pod_id, str) or not pod_id:
            return None
        return cls(
            pod_id=pod_id,
            provider=str(data.get("provider", "runpod")),
            created_at=str(data.get("created_at", "")),
            last_known_status=data.get("last_known_status"),
        )


@dataclass(slots=True)
class AttemptFailure:
    """Phase H2 — typed failure record persisted in ``pipeline_state.json``.

    Replaces (alongside, for backward-compat) the legacy ``error: str | None``
    plain-string field with typed semantics matching the
    :class:`RyotenkAIError` hierarchy. Lives on the
    :class:`PipelineAttemptState` so resume/web-UI/automation can read
    typed failure info without re-parsing stderr.

    Schema migration: ``PipelineState.from_dict`` (and the lower-level
    :func:`AttemptFailure.from_legacy_error_string`) auto-back-fill a
    minimal ``AttemptFailure`` when a legacy state file carries only
    the plain ``error`` string and no structured ``failure`` block.
    """

    code: str = ""                       # ErrorCode value
    title: str = ""
    detail: str | None = None
    stage_name: str | None = None        # None if pre-stage failure
    stage_idx: int | None = None
    stage_total: int | None = None
    trace_id: str | None = None
    request_id: str | None = None
    context: dict[str, Any] | None = None
    failed_at: str = ""                  # ISO timestamp

    def to_dict(self) -> dict[str, Any]:
        return {
            "code": self.code,
            "title": self.title,
            "detail": self.detail,
            "stage_name": self.stage_name,
            "stage_idx": self.stage_idx,
            "stage_total": self.stage_total,
            "trace_id": self.trace_id,
            "request_id": self.request_id,
            "context": dict(self.context) if self.context else None,
            "failed_at": self.failed_at,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> AttemptFailure:
        context_raw = data.get("context")
        context: dict[str, Any] | None
        if isinstance(context_raw, dict):
            context = dict(context_raw)
        else:
            context = None
        return cls(
            code=str(data.get("code", "")),
            title=str(data.get("title", "")),
            detail=data.get("detail"),
            stage_name=data.get("stage_name"),
            stage_idx=_int_or_none(data.get("stage_idx")),
            stage_total=_int_or_none(data.get("stage_total")),
            trace_id=data.get("trace_id"),
            request_id=data.get("request_id"),
            context=context,
            failed_at=str(data.get("failed_at", "")),
        )

    @classmethod
    def from_exception(
        cls,
        exc: "RyotenkAIError",
        *,
        stage_name: str | None = None,
        stage_idx: int | None = None,
        stage_total: int | None = None,
        request_id: str | None = None,
        failed_at: str | None = None,
    ) -> AttemptFailure:
        """Construct from a typed exception.

        ``stage_*`` and ``request_id`` are explicit kwargs because the
        exception itself may not carry them — a pre-stage
        :class:`RyotenkAIError` has no stage info, and ``request_id``
        comes from the contextvar rather than the exception.
        """
        # Allow stage info to flow either from kwargs or from
        # exc.context (H1 stamps it there for stage failures).
        ctx = dict(exc.context) if exc.context else {}
        resolved_stage_name = stage_name or ctx.get("stage_name")
        resolved_stage_idx = stage_idx if stage_idx is not None else _int_or_none(ctx.get("stage_idx"))
        resolved_stage_total = stage_total if stage_total is not None else _int_or_none(ctx.get("stage_total"))
        return cls(
            code=exc.code.value,
            title=exc.title,
            detail=exc.detail,
            stage_name=resolved_stage_name if isinstance(resolved_stage_name, str) else None,
            stage_idx=resolved_stage_idx,
            stage_total=resolved_stage_total,
            trace_id=exc.trace_id,
            request_id=request_id,
            context=ctx or None,
            failed_at=failed_at or utc_now_iso(),
        )

    @classmethod
    def from_legacy_error_string(
        cls,
        error: str,
        *,
        failed_at: str = "",
    ) -> AttemptFailure:
        """Phase H2 migration — back-fill from a legacy ``error: str`` field.

        Pre-H2 ``pipeline_state.json`` files only carry an ``error``
        string and the structured ``failure`` block. To keep resume
        tooling typed-aware across the upgrade we synthesise an
        :class:`AttemptFailure` with ``code="LEGACY_ERROR"`` so the
        consumer can tell apart "real failure record" from "synthesised
        from the plain string".
        """
        return cls(
            code="LEGACY_ERROR",
            title="Legacy attempt failure",
            detail=error,
            failed_at=failed_at,
        )


def _int_or_none(value: Any) -> int | None:
    """Coerce ``value`` to int; return ``None`` if not parseable."""
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


@dataclass(slots=True)
class PipelineAttemptState:
    attempt_id: str
    attempt_no: int
    runtime_name: str
    requested_action: str
    effective_action: str
    restart_from_stage: str | None
    status: str
    started_at: str
    completed_at: str | None = None
    error: str | None = None
    training_critical_config_hash: str = ""
    late_stage_config_hash: str = ""
    model_dataset_config_hash: str = ""
    root_mlflow_run_id: str | None = None
    pipeline_attempt_mlflow_run_id: str | None = None
    training_run_id: str | None = None
    enabled_stage_names: list[str] = field(default_factory=list)
    stage_runs: dict[str, StageRunState] = field(default_factory=dict)
    #: Phase 11.C — provider pod identity for resume / status probing.
    #: ``None`` when the attempt was created before Phase 11.C or
    #: the provider doesn't expose a pod_id (e.g. mock provider).
    pod_metadata: PodMetadata | None = None
    #: Variant 1 — caller-provided tags propagated as MLflow ``meta.*``
    #: tags by the orchestrator. Conventional keys: ``project_id``,
    #: ``actor``, ``config_version_hash``, ``session_id``. ``{}`` for
    #: anonymous runs (e.g. ``ryotenkai run start -c X.yaml`` without
    #: ``--project``).
    metadata: dict[str, Any] = field(default_factory=dict)
    #: Phase H2 — typed failure record. ``None`` for in-flight or
    #: successful attempts. Auto-back-filled from a legacy ``error``
    #: string when loading a pre-H2 state file (see
    #: :meth:`AttemptFailure.from_legacy_error_string`).
    failure: AttemptFailure | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "attempt_id": self.attempt_id,
            "attempt_no": self.attempt_no,
            "runtime_name": self.runtime_name,
            "requested_action": self.requested_action,
            "effective_action": self.effective_action,
            "restart_from_stage": self.restart_from_stage,
            "status": self.status,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "error": self.error,
            "training_critical_config_hash": self.training_critical_config_hash,
            "late_stage_config_hash": self.late_stage_config_hash,
            "model_dataset_config_hash": self.model_dataset_config_hash,
            "root_mlflow_run_id": self.root_mlflow_run_id,
            "pipeline_attempt_mlflow_run_id": self.pipeline_attempt_mlflow_run_id,
            "training_run_id": self.training_run_id,
            "enabled_stage_names": list(self.enabled_stage_names),
            "stage_runs": {name: state.to_dict() for name, state in self.stage_runs.items()},
            # Phase 11.C — None ⇒ omit ``pod_metadata`` from the JSON
            # so legacy diff'ing tools don't see spurious null fields.
            **({"pod_metadata": self.pod_metadata.to_dict()}
               if self.pod_metadata is not None else {}),
            # Variant 1 — empty dict ⇒ omit ``metadata`` from the JSON
            # to keep legacy state files diff-clean.
            **({"metadata": dict(self.metadata)} if self.metadata else {}),
            # Phase H2 — None ⇒ omit ``failure`` from JSON for
            # in-flight / successful attempts.
            **({"failure": self.failure.to_dict()}
               if self.failure is not None else {}),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PipelineAttemptState:
        stage_runs_raw = data.get("stage_runs")
        stage_runs: dict[str, StageRunState] = {}
        if isinstance(stage_runs_raw, dict):
            for stage_name, stage_data in stage_runs_raw.items():
                if isinstance(stage_data, dict):
                    stage_runs[str(stage_name)] = StageRunState.from_dict(stage_data, stage_name=str(stage_name))

        # Phase 11.C — graceful fallback for legacy attempts.
        pod_meta_raw = data.get("pod_metadata")
        pod_metadata: PodMetadata | None = None
        if isinstance(pod_meta_raw, dict):
            pod_metadata = PodMetadata.from_dict(pod_meta_raw)

        # Variant 1 — additive ``metadata`` field. Old state files are
        # missing it; default to ``{}``.
        metadata_raw = data.get("metadata")
        metadata: dict[str, Any] = (
            dict(metadata_raw) if isinstance(metadata_raw, dict) else {}
        )

        # Phase H2 — parse the structured ``failure`` block; back-fill
        # from the legacy ``error: str`` field when missing AND the
        # attempt is in a terminal-failed state. Successful / pending
        # attempts produce ``failure=None``.
        failure_raw = data.get("failure")
        failure: AttemptFailure | None
        if isinstance(failure_raw, dict):
            failure = AttemptFailure.from_dict(failure_raw)
        else:
            failure = None
        legacy_error = data.get("error")
        attempt_status = str(data.get("status", StageRunState.STATUS_PENDING))
        if (
            failure is None
            and isinstance(legacy_error, str)
            and legacy_error
            and attempt_status == StageRunState.STATUS_FAILED
        ):
            failure = AttemptFailure.from_legacy_error_string(
                legacy_error,
                failed_at=str(data.get("completed_at") or ""),
            )

        return cls(
            attempt_id=str(data.get("attempt_id", "")),
            attempt_no=int(data.get("attempt_no", 0) or 0),
            runtime_name=str(data.get("runtime_name", "")),
            requested_action=str(data.get("requested_action", "fresh")),
            effective_action=str(data.get("effective_action", "fresh")),
            restart_from_stage=data.get("restart_from_stage"),
            status=attempt_status,
            started_at=str(data.get("started_at", "")),
            completed_at=data.get("completed_at"),
            error=data.get("error"),
            training_critical_config_hash=str(data.get("training_critical_config_hash", "")),
            late_stage_config_hash=str(data.get("late_stage_config_hash", "")),
            model_dataset_config_hash=str(data.get("model_dataset_config_hash", "")),
            root_mlflow_run_id=data.get("root_mlflow_run_id"),
            pipeline_attempt_mlflow_run_id=data.get("pipeline_attempt_mlflow_run_id"),
            training_run_id=data.get("training_run_id"),
            enabled_stage_names=[str(x) for x in data.get("enabled_stage_names", []) if isinstance(x, str)],
            stage_runs=stage_runs,
            pod_metadata=pod_metadata,
            metadata=metadata,
            failure=failure,
        )


@dataclass(slots=True)
class PipelineState:
    schema_version: int
    logical_run_id: str
    run_directory: str
    config_path: str
    active_attempt_id: str | None
    pipeline_status: str
    training_critical_config_hash: str
    late_stage_config_hash: str
    model_dataset_config_hash: str = ""
    root_mlflow_run_id: str | None = None
    mlflow_runtime_tracking_uri: str | None = None
    mlflow_ca_bundle_path: str | None = None
    attempts: list[PipelineAttemptState] = field(default_factory=list)
    current_output_lineage: dict[str, StageLineageRef] = field(default_factory=dict)
    #: Variant 1 — caller-provided tags (``project_id``, ``actor``,
    #: ``config_version_hash``, …). Mirrored to MLflow as ``meta.*``
    #: tags by the orchestrator. Empty dict for anonymous runs.
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "logical_run_id": self.logical_run_id,
            "run_directory": self.run_directory,
            "config_path": self.config_path,
            "active_attempt_id": self.active_attempt_id,
            "pipeline_status": self.pipeline_status,
            "training_critical_config_hash": self.training_critical_config_hash,
            "late_stage_config_hash": self.late_stage_config_hash,
            "model_dataset_config_hash": self.model_dataset_config_hash,
            "root_mlflow_run_id": self.root_mlflow_run_id,
            "mlflow_runtime_tracking_uri": self.mlflow_runtime_tracking_uri,
            "mlflow_ca_bundle_path": self.mlflow_ca_bundle_path,
            "attempts": [attempt.to_dict() for attempt in self.attempts],
            "current_output_lineage": {
                stage_name: lineage.to_dict() for stage_name, lineage in self.current_output_lineage.items()
            },
            # Variant 1 — empty dict ⇒ omit from JSON so legacy diff
            # tools don't see spurious null fields.
            **({"metadata": dict(self.metadata)} if self.metadata else {}),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PipelineState:
        attempts = [PipelineAttemptState.from_dict(item) for item in data.get("attempts", []) if isinstance(item, dict)]
        lineage_raw = data.get("current_output_lineage")
        current_output_lineage: dict[str, StageLineageRef] = {}
        if isinstance(lineage_raw, dict):
            for stage_name, value in lineage_raw.items():
                if isinstance(value, dict):
                    current_output_lineage[str(stage_name)] = StageLineageRef.from_dict(value)
        # Variant 1 — additive ``metadata`` field. Old state files are
        # missing it; default to ``{}``.
        metadata_raw = data.get("metadata")
        metadata: dict[str, Any] = (
            dict(metadata_raw) if isinstance(metadata_raw, dict) else {}
        )

        return cls(
            schema_version=int(data.get("schema_version", 0) or 0),
            logical_run_id=str(data.get("logical_run_id", "")),
            run_directory=str(data.get("run_directory", "")),
            config_path=str(data.get("config_path", "")),
            active_attempt_id=data.get("active_attempt_id"),
            pipeline_status=str(data.get("pipeline_status", StageRunState.STATUS_PENDING)),
            training_critical_config_hash=str(data.get("training_critical_config_hash", "")),
            late_stage_config_hash=str(data.get("late_stage_config_hash", "")),
            model_dataset_config_hash=str(data.get("model_dataset_config_hash", "")),
            root_mlflow_run_id=data.get("root_mlflow_run_id"),
            mlflow_runtime_tracking_uri=data.get("mlflow_runtime_tracking_uri"),
            mlflow_ca_bundle_path=data.get("mlflow_ca_bundle_path"),
            attempts=attempts,
            current_output_lineage=current_output_lineage,
            metadata=metadata,
        )
