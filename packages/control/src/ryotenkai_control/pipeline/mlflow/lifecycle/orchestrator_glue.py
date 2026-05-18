"""Glue between :class:`PipelineOrchestrator` and the new MLflow lifecycle.

Phase M7.2 extracts the orchestrator's MLflow setup / preflight /
teardown bodies into this module to:

* Keep ``orchestrator.py`` under its 800-line architectural-guardrail.
* Make the open/close/preflight wiring testable in isolation without
  spinning up the full orchestrator.

The functions here are stateless and take everything they need as
arguments. They never reach back into the orchestrator's instance
fields; the orchestrator is responsible for passing in the collaborator
objects and reading back the results (e.g. ``coord.__enter__``-state
tracking lives on the orchestrator).
"""

from __future__ import annotations

import hashlib
import os
from typing import TYPE_CHECKING, Any

from ryotenkai_control.pipeline.mlflow.lifecycle._tag_utils import (
    stringify_tag_value,
)
from ryotenkai_control.pipeline.stages import PipelineContextKeys
from ryotenkai_shared.errors import RyotenkAIError
from ryotenkai_shared.infrastructure.mlflow.protocols import RunStatus
from ryotenkai_shared.utils.logger import logger

if TYPE_CHECKING:
    from pathlib import Path

    from ryotenkai_control.pipeline.mlflow.lifecycle import (
        ParentRunOpener,
        PreflightConnectivityCheck,
        RunLifecycleCoord,
    )
    from ryotenkai_control.pipeline.mlflow_attempt import MLflowAttemptManager
    from ryotenkai_control.pipeline.state import (
        PipelineAttemptState,
        PipelineState,
    )
    from ryotenkai_shared.infrastructure.mlflow.protocol import IMLflowManager


__all__ = [
    "derive_engine_kind",
    "derive_provider_gpu",
    "derive_provider_kind",
    "open_attempt_with_coord",
    "resolve_journal_for_upload",
    "run_preflight_or_fallback",
    "stamp_state_tracking_uri",
    "teardown_attempt_with_coord",
]


def derive_engine_kind(config: Any) -> str:
    """Derive the ``ryotenkai.engine.kind`` tag from training config.

    Used by :class:`ParentRunOpener` to stamp the root run with the
    canonical engine identifier (``sft|cpt|dpo|grpo|sapo``). Falls
    back to ``"unknown"`` so the tag is always present.
    """
    try:
        strategies = (
            getattr(getattr(config, "training", None), "strategies", []) or []
        )
        if strategies:
            strategy_type = getattr(strategies[0], "strategy_type", None)
            if strategy_type:
                return str(strategy_type).lower()
    except Exception:  # pragma: no cover — defensive
        pass
    return "unknown"


def derive_provider_kind(config: Any) -> str:
    """Derive the ``ryotenkai.provider.kind`` tag from provider config."""
    try:
        provider = getattr(config, "provider", None)
        if provider is None:
            return "unknown"
        return str(getattr(provider, "kind", "unknown") or "unknown").lower()
    except Exception:  # pragma: no cover — defensive
        return "unknown"


def derive_provider_gpu(config: Any) -> str:
    """Derive the ``ryotenkai.provider.gpu`` tag from provider config."""
    try:
        provider = getattr(config, "provider", None)
        if provider is None:
            return "unknown"
        return str(getattr(provider, "gpu_type", None) or "unknown")
    except Exception:  # pragma: no cover — defensive
        return "unknown"


def run_preflight_or_fallback(
    *,
    preflight: PreflightConnectivityCheck | None,
    legacy_mgr: MLflowAttemptManager,
) -> None:
    """Run the narrow preflight, falling back to the legacy probe.

    :raises RyotenkAIError: Wrapped at the caller via
        :func:`_wrap_as_launch_error`. Both narrow and legacy paths
        raise typed :class:`RyotenkAIError` subclasses; the orchestrator
        translates them to :class:`LaunchPreparationError`.
    """
    if preflight is not None:
        # Narrow stack: ping-based, no probe-run side effects.
        preflight.run()
        return
    # Narrow stack absent — keep the legacy ``ensure_preflight`` so a
    # config without ``tracking_uri`` still fails fast.
    legacy_mgr.ensure_preflight()


def stamp_state_tracking_uri(
    *,
    manager: IMLflowManager,
    config: Any,
    state: PipelineState,
) -> None:
    """Persist the runtime tracking URI + CA bundle on ``state``.

    The trainer subprocess re-uses these on resume so it doesn't have
    to re-resolve the URI via the config block.
    """
    runtime_uri = manager.get_runtime_tracking_uri()
    ca_bundle = getattr(config.integrations.mlflow, "ca_bundle_path", None)
    state.mlflow_runtime_tracking_uri = (
        runtime_uri
        if isinstance(runtime_uri, str) and runtime_uri
        else None
    )
    state.mlflow_ca_bundle_path = (
        ca_bundle if isinstance(ca_bundle, str) and ca_bundle else None
    )


def open_attempt_with_coord(
    *,
    coord: RunLifecycleCoord,
    opener: ParentRunOpener,
    manager: IMLflowManager,
    config: Any,
    state: PipelineState,
    attempt: PipelineAttemptState,
    start_stage_idx: int,
    total_stages: int,
    run_directory: Path | None,
    context: dict[str, Any],
) -> None:
    """Open or adopt root + open nested attempt run via the narrow stack.

    Preserves the legacy side effects of
    ``MLflowAttemptManager.setup_for_attempt`` verbatim:

    * ``state.root_mlflow_run_id`` + ``attempt.*_mlflow_run_id`` populated.
    * Context keys ``MLFLOW_PARENT_RUN_ID`` / ``MLFLOW_MANAGER`` set.
    * Wide-manager logging (``log_pipeline_config`` /
      ``log_dataset_config`` / ``log_params``).
    * ``meta.*`` tag bundle from ``state.metadata``.

    On any failure, the coord is finalized with
    :data:`RunStatus.FAILED` and re-raised; the caller is responsible
    for calling ``coord.__exit__``.
    """
    try:
        if state.root_mlflow_run_id:
            root = opener.adopt_root(state.root_mlflow_run_id)
        else:
            experiment_name = getattr(
                config.integrations.mlflow, "experiment_name", ""
            ) or ""
            root = opener.open(
                experiment=experiment_name,
                logical_run_id=state.logical_run_id,
                config_sha256=(
                    getattr(state, "training_critical_config_hash", "") or ""
                ),
                code_commit=os.environ.get(
                    "RYOTENKAI_CODE_COMMIT", "unknown"
                ),
                engine_kind=derive_engine_kind(config),
                provider_kind=derive_provider_kind(config),
                provider_gpu=derive_provider_gpu(config),
            )
        coord.bind_root_run(root)
        state.root_mlflow_run_id = root.run_id
        attempt.root_mlflow_run_id = root.run_id

        attempt_run = opener.open_attempt(
            root_run=root,
            logical_run_id=state.logical_run_id,
            attempt_id=attempt.attempt_id,
            attempt_no=attempt.attempt_no,
        )
        coord.bind_attempt_run(attempt_run)
        attempt.pipeline_attempt_mlflow_run_id = attempt_run.run_id

        # Context propagation (consumed by stages — artifacts/base.py:210,
        # training_launcher.py:585, etc.).
        context[PipelineContextKeys.MLFLOW_PARENT_RUN_ID] = attempt_run.run_id
        context[PipelineContextKeys.MLFLOW_MANAGER] = manager

        # Wide-manager logging (no narrow equivalent for these rich
        # payloads yet).
        try:
            manager.log_pipeline_config(config)
            manager.log_dataset_config(config)
            manager.log_params(
                {
                    "pipeline.total_stages": total_stages,
                    "pipeline.start_stage": start_stage_idx,
                    "pipeline.run_directory": str(run_directory),
                }
            )
        except Exception as e:  # noqa: BLE001 — best-effort logging
            logger.warning(f"MLflow wide-manager logging failed: {e}")

        # ``meta.*`` tags from ``state.metadata``.
        if state.metadata:
            tags = {
                f"meta.{key}": stringify_tag_value(value)
                for key, value in state.metadata.items()
            }
            try:
                manager.set_tags(tags)
            except Exception as e:  # noqa: BLE001 — best-effort logging
                logger.warning(f"MLflow set_tags failed: {e}")
    except Exception:
        # Cleanup partial runs — finalizer's idempotency tag means
        # repeated close is harmless even if e.g. only the root opened
        # before failure.
        try:
            coord.finalize(
                status=RunStatus.FAILED, exit_reason="bootstrap_failed"
            )
        except Exception:  # noqa: BLE001
            pass
        raise


def teardown_attempt_with_coord(
    *,
    coord: RunLifecycleCoord,
    manager: IMLflowManager,
    pipeline_success: bool,
    attempt_run_id: str | None,
    shutdown_signal_name: str | None,
    state_path: Path | None,
    on_save_state: Any,
    on_before_end: Any,
    on_after_end: Any,
    emitter: Any,
) -> None:
    """Drive ``coord.finalize`` with the five legacy hook side-effects.

    Order preserved verbatim from
    :meth:`MLflowAttemptManager.teardown_attempt`:

    1. ``on_before_end`` — aggregate training metrics (best-effort).
    2. ``log_artifact(state.json)`` — persist final state to MLflow.
    3. ``coord.finalize`` — uploads journal + stamps lifecycle tags +
       set_terminated on both attempt and root runs.
    4. ``on_after_end`` — generate experiment report (best-effort).
    5. ``manager.cleanup`` — restore env vars + unregister atexit.

    Caller is responsible for calling ``coord.__exit__`` after this
    returns so the orchestrator can also reset its
    ``_mlflow_coord_entered`` flag.
    """
    # Hook 1: aggregate training metrics (was on_before_end).
    try:
        on_before_end()
    except Exception as e:  # noqa: BLE001 — best-effort
        logger.warning(f"MLflow teardown before-end hook failed: {e}")

    # Hook 2: log state.json artifact (was state_path_supplier).
    try:
        on_save_state()
        if state_path is not None and state_path.exists():
            manager.log_artifact(str(state_path))
    except Exception as e:  # noqa: BLE001 — best-effort
        logger.warning(f"MLflow state.json artifact upload failed: {e}")

    # Hook 3: finalize via coord — never raises.
    journal_path, journal_sha256 = resolve_journal_for_upload(emitter)
    try:
        coord.finalize(
            status=(
                RunStatus.FINISHED if pipeline_success else RunStatus.FAILED
            ),
            journal_path=journal_path,
            journal_sha256=journal_sha256,
            exit_reason=shutdown_signal_name,
        )
    except Exception as e:  # noqa: BLE001 — defensive
        logger.warning(f"MLflow lifecycle coord finalize failed: {e}")

    # Hook 4: generate experiment report (was on_after_end).
    try:
        on_after_end(attempt_run_id)
    except Exception as e:  # noqa: BLE001 — best-effort
        logger.warning(f"MLflow teardown after-end hook failed: {e}")

    # Hook 5: cleanup wide manager.
    try:
        manager.cleanup()
    except Exception as e:  # noqa: BLE001 — best-effort
        logger.warning(f"MLflow wide-manager cleanup failed: {e}")


def resolve_journal_for_upload(
    emitter: Any,
) -> tuple[Path | None, str | None]:
    """Return ``(journal_path, sha256_hex)`` for the coord finalize.

    The orchestrator's events-side coordinator owns the journal handle;
    we read its emitter to fetch the path and hash the file lazily.

    :param emitter: The orchestrator's ``_EventLifecycleCoordinator._emitter``
        (or ``None``). Read via ``.journal.path``.
    :returns: A pair ``(path, sha256)``. Either or both may be ``None``
        if the emitter is missing or the journal file does not exist.
    """
    if emitter is None:
        return None, None
    try:
        journal_path = emitter.journal.path
    except Exception:  # noqa: BLE001 — defensive
        return None, None
    if journal_path is None or not journal_path.exists():
        return journal_path, None
    try:
        h = hashlib.sha256()
        with open(journal_path, "rb") as f:
            while True:
                chunk = f.read(8192)
                if not chunk:
                    break
                h.update(chunk)
        return journal_path, h.hexdigest()
    except Exception:  # noqa: BLE001 — best-effort
        return journal_path, None
