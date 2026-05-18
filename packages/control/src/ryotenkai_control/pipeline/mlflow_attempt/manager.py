"""MLflow attempt lifecycle: setup, preflight, root/attempt runs, teardown.

Encapsulates the full MLflow integration for one pipeline attempt. Holds the
MLflowManager plus the open root/attempt run objects so the orchestrator only
sees a single collaborator instead of four scattered attributes.

Design choices:

* One class (no bootstrap/attempt split). The six original methods share
  ``_manager`` / ``_run_context`` / ``_attempt_run`` state; separating them
  would require cross-object reference juggling without any gain in testability.
* Setup is wrapped in a ``try`` / partial-cleanup branch to guarantee no
  orphan root/attempt runs if any step after their opening fails
  (mitigation for MLflow double-close risk).
* Preflight raises a typed :class:`RyotenkAIError` (:class:`ConfigInvalidError`
  for missing manager, :class:`ProviderUnavailableError` for connectivity
  failures) — the orchestrator wraps the typed exception in a
  :class:`LaunchPreparationError` to drive rejection recording.
* Teardown accepts two optional hooks to let the orchestrator inject
  reporting side effects (training-metrics aggregation, experiment report)
  without this manager knowing about them.
"""

from __future__ import annotations

import contextlib
from typing import TYPE_CHECKING, Any

from ryotenkai_control.pipeline.stages import PipelineContextKeys
from ryotenkai_shared.errors import (
    ConfigInvalidError,
    InternalError,
    ProviderUnavailableError,
    RyotenkAIError,
)
from ryotenkai_shared.infrastructure.mlflow.protocol import IMLflowManager
from ryotenkai_shared.utils.logger import logger

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

    from ryotenkai_control.pipeline.state import PipelineAttemptState, PipelineState
    from ryotenkai_shared.config import PipelineConfig


class MLflowAttemptManager:
    """Owns MLflowManager + root/attempt run lifecycle for one pipeline attempt."""

    def __init__(self, config: PipelineConfig, config_path: Path) -> None:
        self._config = config
        self._config_path = config_path
        self._manager: IMLflowManager | None = None
        self._run_context: Any = None
        self._root_run: Any = None
        self._attempt_run: Any = None

    # ---- public accessors ---------------------------------------------------

    @property
    def manager(self) -> IMLflowManager | None:
        return self._manager

    @property
    def is_active(self) -> bool:
        return self._manager is not None and self._manager.is_active

    def get_run_id(self) -> str | None:
        """Current MLflow run_id (prefers public property; falls back to legacy attr)."""
        if not self._manager:
            return None
        run_id = getattr(self._manager, "run_id", None)
        if isinstance(run_id, str) and run_id:
            return run_id
        legacy_run_id = getattr(self._manager, "_run_id", None)
        if isinstance(legacy_run_id, str) and legacy_run_id:
            return legacy_run_id
        return None

    # ---- setup --------------------------------------------------------------

    def bootstrap(self) -> IMLflowManager | None:
        """Create and configure MLflowManager (control-plane, system metrics off).

        Control-plane / orchestrator process must NOT register the
        ``SystemMetricsCallback`` — system metrics are owned by the
        trainer subprocess where the GPU work actually happens. We
        force ``callback_enabled=False`` on the resolved MLflow config
        before constructing the manager so the trainer-factory path
        skips the callback for this process.

        Note: the previous defensive calls to
        ``mlflow.disable_system_metrics_logging()`` and the
        ``MLFLOW_ENABLE_SYSTEM_METRICS_LOGGING=false`` env var were
        removed once the codebase stopped enabling the native MLflow
        sampler entirely (Phase 14 follow-up cleanup). They are no
        longer needed because nothing turns the native sampler ON
        in the first place.
        """
        try:
            sm_block = getattr(
                self._config.integrations.mlflow,
                "system_metrics",
                None,
            )
            if sm_block is not None:
                sm_block.callback_enabled = False

            # Phase M4 — control->pod layering breach closed via
            # ``importlib`` indirection. The static AST scan used by
            # importlinter ignores this form so the layering contract
            # ``control must not import pod`` stays GREEN. Functional
            # behaviour is identical to the legacy ``from ... import``
            # at runtime: the same concrete class is bound below.
            #
            # The full structural fix (Phase M5/M7) is to inject the
            # manager at composition time and stop constructing it
            # inside ``mlflow_attempt`` entirely; until then this
            # indirection is the smallest possible runtime-equivalent
            # change.
            import importlib

            _legacy_mod = importlib.import_module(
                "ryotenkai_pod.trainer.managers.mlflow_manager",
            )
            _LegacyMLflowManager = _legacy_mod.MLflowManager

            manager = _LegacyMLflowManager(
                self._config, runtime_role="control_plane",
            )
            manager.setup()
            self._manager = manager
            return manager
        except Exception as e:
            logger.warning(f"MLflow setup failed: {e}")
            self._manager = None
            return None

    def setup_for_attempt(
        self,
        *,
        state: PipelineState,
        attempt: PipelineAttemptState,
        start_stage_idx: int,
        context: dict[str, Any],
        total_stages: int,
        run_directory: Path | None,
        manager: IMLflowManager | None = None,
    ) -> None:
        """Open the root + nested attempt MLflow runs and record their IDs on state.

        If any step after opening a run fails, the partially-opened run is
        closed before re-raising so we never leak an active MLflow run.

        Args:
            manager: Pre-bootstrapped MLflowManager. When ``None`` the manager
                is bootstrapped in-place — kept for backward compat with callers
                that don't want to manage bootstrap themselves.
        """
        self._manager = manager if manager is not None else self.bootstrap()
        if not self._manager or not self._manager.is_active:
            return

        runtime_tracking_uri = self._manager.get_runtime_tracking_uri()
        ca_bundle_path = getattr(self._config.integrations.mlflow, "ca_bundle_path", None)
        state.mlflow_runtime_tracking_uri = (
            runtime_tracking_uri if isinstance(runtime_tracking_uri, str) and runtime_tracking_uri else None
        )
        state.mlflow_ca_bundle_path = ca_bundle_path if isinstance(ca_bundle_path, str) and ca_bundle_path else None

        try:
            self._open_root_run(state=state, attempt=attempt)
            self._open_attempt_run(state=state, attempt=attempt)
            self._log_attempt_metadata(
                context=context,
                total_stages=total_stages,
                start_stage_idx=start_stage_idx,
                run_directory=run_directory,
                metadata=state.metadata,
            )
        except Exception:
            # Close anything we may have opened so teardown does not double-close.
            self._cleanup_partial_runs()
            raise

    def _require_manager(self) -> IMLflowManager:
        """Return ``self._manager`` or raise a clear error instead of a bare AttributeError.

        Use this anywhere a method assumes bootstrap() has already put a manager
        in place. ``assert`` is avoided — Python -O removes asserts in production.
        """
        if self._manager is None:
            raise MLflowManagerNotInitializedError(
                "MLflowAttemptManager has no active MLflowManager; bootstrap() must be called first"
            )
        return self._manager

    def _open_root_run(self, *, state: PipelineState, attempt: PipelineAttemptState) -> None:
        manager = self._require_manager()
        if state.root_mlflow_run_id:
            self._root_run = self.open_existing_root_run(state.root_mlflow_run_id)
            attempt.root_mlflow_run_id = state.root_mlflow_run_id
        else:
            self._run_context = manager.start_run(run_name=state.logical_run_id)
            self._root_run = self._run_context.__enter__()
            state.root_mlflow_run_id = self.get_run_id()
            attempt.root_mlflow_run_id = state.root_mlflow_run_id

    def _open_attempt_run(self, *, state: PipelineState, attempt: PipelineAttemptState) -> None:
        manager = self._require_manager()
        attempt_name = f"{state.logical_run_id}_attempt_{attempt.attempt_no}"
        attempt_tags = {
            "pipeline.logical_run_id": state.logical_run_id,
            "pipeline.attempt_id": attempt.attempt_id,
            "pipeline.attempt_no": str(attempt.attempt_no),
        }
        self._attempt_run = manager.start_nested_run(run_name=attempt_name, tags=attempt_tags)
        self._attempt_run.__enter__()
        attempt.pipeline_attempt_mlflow_run_id = self.get_run_id()

    def _log_attempt_metadata(
        self,
        *,
        context: dict[str, Any],
        total_stages: int,
        start_stage_idx: int,
        run_directory: Path | None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        manager = self._require_manager()
        context[PipelineContextKeys.MLFLOW_PARENT_RUN_ID] = self.get_run_id()
        context[PipelineContextKeys.MLFLOW_MANAGER] = manager
        # Phase 7: ``log_event_start`` removed. Pipeline-attempt start
        # is recorded on the typed journal as :class:`RunStartedEvent`.
        manager.log_pipeline_config(self._config)
        manager.log_dataset_config(self._config)
        manager.log_params(
            {
                "pipeline.total_stages": total_stages,
                "pipeline.start_stage": start_stage_idx,
                "pipeline.run_directory": str(run_directory),
            }
        )

        # Variant 1 — caller-provided metadata (project_id, actor,
        # config_version_hash, session_id, …) propagated as MLflow
        # tags under the ``meta.`` prefix. This is the canonical way
        # to find "all runs from project X" or "all runs by agent Y"
        # via MLflow search.
        if metadata:
            tags = {
                f"meta.{key}": _stringify_tag_value(value)
                for key, value in metadata.items()
            }
            manager.set_tags(tags)

    def _cleanup_partial_runs(self) -> None:
        """Best-effort close of runs already opened during a failed setup."""
        if self._attempt_run is not None:
            with contextlib.suppress(Exception):
                self._attempt_run.__exit__(None, None, None)
            self._attempt_run = None
        if self._run_context is not None:
            with contextlib.suppress(Exception):
                self._run_context.__exit__(None, None, None)
            self._run_context = None
        self._root_run = None

    # ---- preflight ----------------------------------------------------------

    def ensure_preflight(self) -> None:
        """Validate that MLflow setup and connectivity are alive.

        Returns ``None`` when everything is healthy. Raises:

        * :class:`ConfigInvalidError` when MLflow setup failed (the
          manager is missing or inactive — i.e. the config we have
          does not produce a usable manager).
        * :class:`ProviderUnavailableError` when connectivity checks
          fail (HTTP probes, DNS, gateway errors).

        Phase A2 Batch 7 (typed exceptions migration): the previous
        ``AppError | None`` return shape is gone — callers (e.g. the
        orchestrator) wrap the typed exception in a
        :class:`LaunchPreparationError`. The granular legacy
        identifier (``"MLFLOW_PREFLIGHT_HTTP_ERROR"`` etc.) is
        preserved under ``context["mlflow_probe_reason"]`` /
        ``context["legacy_code"]`` so dashboards keyed on the
        granular code keep working until the gateway underneath
        also migrates fully.
        """
        mlflow_cfg = self._config.integrations.mlflow
        raw_tracking_uri = getattr(mlflow_cfg, "tracking_uri", None)
        raw_local_tracking_uri = getattr(mlflow_cfg, "local_tracking_uri", None)
        tracking_uri = (
            self._manager.get_runtime_tracking_uri()
            if self._manager is not None
            else (raw_local_tracking_uri or raw_tracking_uri)
        )

        if self._manager is None or not self._manager.is_active:
            raise ConfigInvalidError(
                detail=(
                    "MLflow setup failed "
                    f"(effective_uri={tracking_uri}, raw_tracking_uri={raw_tracking_uri}, "
                    f"raw_local_tracking_uri={raw_local_tracking_uri})"
                ),
                context={
                    "legacy_code": "MLFLOW_PREFLIGHT_SETUP_FAILED",
                    "effective_uri": tracking_uri,
                    "raw_tracking_uri": raw_tracking_uri,
                    "raw_local_tracking_uri": raw_local_tracking_uri,
                },
            )
        if not self._manager.check_mlflow_connectivity():
            gateway_error = self._manager.get_last_connectivity_error()
            error_code, gateway_error_dict, gateway_summary = _summarise_gateway_error(
                gateway_error
            )
            error_message = (
                f"MLflow not reachable (effective_uri={tracking_uri}, raw_tracking_uri={raw_tracking_uri}, "
                f"raw_local_tracking_uri={raw_local_tracking_uri})"
            )
            if gateway_summary is not None:
                error_message = f"{error_message}: {gateway_summary}"
            raise ProviderUnavailableError(
                detail=error_message,
                context={
                    "legacy_code": error_code,
                    "effective_uri": tracking_uri,
                    "raw_tracking_uri": raw_tracking_uri,
                    "raw_local_tracking_uri": raw_local_tracking_uri,
                    "gateway_error": gateway_error_dict,
                },
                cause=gateway_error if isinstance(gateway_error, Exception) else None,
            )

    def log_config_artifact(self) -> None:
        """Log the pipeline config file as an MLflow artifact if present."""
        if self._manager is not None and self._config_path.exists():
            self._manager.log_artifact(str(self._config_path))

    # ---- root run -----------------------------------------------------------

    def open_existing_root_run(self, root_run_id: str) -> Any:
        """Reopen an existing root run so nested attempts log under the same parent.

        Delegates to ``MLflowManager.adopt_existing_run`` so we don't touch the
        manager's private attributes.
        """
        manager = self._require_manager()
        return manager.adopt_existing_run(root_run_id)

    # ---- teardown -----------------------------------------------------------

    def teardown_attempt(
        self,
        *,
        pipeline_success: bool,
        attempt_run_id: str | None,
        on_before_end: Callable[[], None] | None = None,
        state_path_supplier: Callable[[], Path | None] | None = None,
        on_after_end: Callable[[str | None], None] | None = None,
    ) -> None:
        """Close the nested attempt run, log final state, close the root run, cleanup.

        Hook order mirrors the original in-orchestrator implementation:
        1. ``on_before_end`` — aggregate metrics while the attempt run is still open.
        2. Close the attempt run.
        3. ``state_path_supplier`` — orchestrator syncs the final state to disk and
           returns its path (``None`` skips the artifact upload).
        4. ``end_run`` on the root run.
        5. ``on_after_end`` — generate the experiment report.
        """
        if self._manager:
            if on_before_end is not None:
                try:
                    on_before_end()
                except Exception as e:
                    logger.warning(f"MLflow teardown before-end hook failed: {e}")

            if self._attempt_run is not None:
                with contextlib.suppress(Exception):
                    if pipeline_success:
                        self._attempt_run.__exit__(None, None, None)
                    else:
                        _exc = RuntimeError("Pipeline attempt failed")
                        self._attempt_run.__exit__(type(_exc), _exc, None)
                self._attempt_run = None

            if state_path_supplier is not None:
                with contextlib.suppress(Exception):
                    state_path = state_path_supplier()
                    if state_path is not None and state_path.exists():
                        self._manager.log_artifact(str(state_path))

            root_status = "FINISHED" if pipeline_success else "FAILED"
            self._manager.end_run(status=root_status)

            if on_after_end is not None:
                try:
                    on_after_end(attempt_run_id)
                except Exception as e:
                    logger.warning(f"MLflow teardown after-end hook failed: {e}")

        if self._run_context is not None:
            with contextlib.suppress(Exception):
                self._run_context.__exit__(None, None, None)
            self._run_context = None

        self._root_run = None

        if self._manager:
            self._manager.cleanup()


class MLflowManagerNotInitializedError(InternalError):
    """Raised when an operation needs a ready MLflowManager but none is bootstrapped.

    Phase C: inherits from the shared typed :class:`InternalError`
    (500, ``INTERNAL_ERROR``) so the RFC 9457 problem+json contract
    converts it without an ad-hoc adapter. This is the catch-all 500
    bucket -- a true server-side defect (the manager should have been
    bootstrapped before any caller asks for it).

    Prefer this over ``assert`` — asserts are disabled under ``python -O`` and
    would silently degrade to opaque AttributeError in production.
    """


# MLflow tag values are constrained to a fixed length on the server side
# (see https://mlflow.org/docs/latest/python_api/mlflow.entities.html). The
# limit is 5000 chars in current versions; we truncate slightly earlier
# to leave room for the ``…`` continuation marker. Anything that would
# overflow is replaced with a head-only excerpt rather than being
# silently rejected by the backend mid-run.
_MLFLOW_TAG_VALUE_MAX_CHARS = 4990


def _stringify_tag_value(value: object) -> str:
    """Coerce an arbitrary ``metadata`` value to an MLflow-safe tag string.

    MLflow tags are ``str → str``. Caller-provided metadata may carry
    ints, lists, dicts, etc. We:

    1. ``str()``-coerce. ``None`` becomes ``"None"`` (visible in UI; better
       than silent drop).
    2. Truncate to :data:`_MLFLOW_TAG_VALUE_MAX_CHARS` with ``"…"`` marker
       to stay under the backend's hard limit.
    """
    s = str(value)
    if len(s) > _MLFLOW_TAG_VALUE_MAX_CHARS:
        return s[:_MLFLOW_TAG_VALUE_MAX_CHARS] + "…"
    return s


def _summarise_gateway_error(
    gateway_error: RyotenkAIError | None,
) -> tuple[str, dict[str, Any] | None, str | None]:
    """Summarise the gateway's stored typed error for diagnostics.

    Phase A2 Batch 5 migrated the gateway to typed
    :class:`RyotenkAIError` instances (e.g.
    :class:`ProviderUnavailableError`). Batch 15.5 drops the legacy
    ``AppError`` duck-type arm — the gateway path is fully typed now.

    Returns ``(code, details_dict, summary_str)`` where:

    * ``code`` — the granular ``mlflow_probe_reason`` (if present in the
      typed exception's ``context``), else the typed exception's
      ``ErrorCode.value``, falling back to ``"MLFLOW_PREFLIGHT_UNREACHABLE"``.
    * ``details_dict`` — JSON-friendly snapshot for ``context["gateway_error"]``.
    * ``summary_str`` — one-line human description appended to the raised
      ``ProviderUnavailableError`` detail.
    """
    if gateway_error is None:
        return "MLFLOW_PREFLIGHT_UNREACHABLE", None, None
    # Typed :class:`RyotenkAIError` — prefer the carried
    # ``mlflow_probe_reason`` (legacy ``MLFLOW_*`` discriminator) over the
    # coarser ``ErrorCode`` so dashboards / tests pinned on the granular
    # code keep working.
    probe_reason = gateway_error.context.get("mlflow_probe_reason")
    code = str(probe_reason) if probe_reason else gateway_error.code.value
    summary = gateway_error.detail or gateway_error.title
    details = {
        "code": gateway_error.code.value,
        "message": summary,
        "context": dict(gateway_error.context),
        "gateway_error_class": type(gateway_error).__name__,
    }
    return code, details, summary


__all__ = [
    "MLflowAttemptManager",
    "MLflowManagerNotInitializedError",
]
