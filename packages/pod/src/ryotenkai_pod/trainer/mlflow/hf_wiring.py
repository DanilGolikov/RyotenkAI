"""HuggingFace MLflow wiring under Pattern A.

Configures :class:`transformers.TrainingArguments` so the HF
:class:`transformers.integrations.MLflowCallback` adopts the parent
MLflow run via environment variables — the pod-trainer subprocess
NEVER calls :func:`mlflow.start_run` or :func:`mlflow.autolog`.

The control plane is responsible for opening the parent run and
exporting the following environment variables before launching the
trainer subprocess (see :mod:`ryotenkai_control.pipeline.stages.managers.deployment.training_launcher`):

* ``MLFLOW_TRACKING_URI``
* ``MLFLOW_RUN_ID`` -- the parent run id to adopt
* ``MLFLOW_NESTED_RUN`` -- must equal the literal ``"TRUE"``
* ``MLFLOW_EXPERIMENT_NAME``
* ``MLFLOW_ENABLE_SYSTEM_METRICS_LOGGING`` -- ``"true"`` to enable
  native MLflow GPU/CPU sampling

:class:`HFMlflowWiring` is a stateless utility -- all knobs are
applied to the supplied :class:`TrainingArguments`-like object and the
native MLflow SDK directly.

See ``docs/plans/vectorized-fluttering-mist.md`` -- Phase M4 for the
full Pattern A migration rationale.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING, Any

from ryotenkai_shared.errors import ConfigInvalidError
from ryotenkai_shared.utils.logger import logger

if TYPE_CHECKING:
    pass


__all__ = ["HFMlflowWiring"]


# Required environment variables exported by ``training_launcher`` for
# Pattern A. Validation is fail-fast so a mis-configured launcher
# surfaces immediately instead of producing an orphan top-level run.
_REQUIRED_ENV_VARS: tuple[str, ...] = (
    "MLFLOW_TRACKING_URI",
    "MLFLOW_RUN_ID",
    "MLFLOW_NESTED_RUN",
    "MLFLOW_EXPERIMENT_NAME",
)


class HFMlflowWiring:
    """Configures HF Trainer with Pattern A env-driven MLflow integration.

    Pattern A: control plane owns the parent run; HF
    :class:`MLflowCallback` opens a structurally-nested child of that
    parent by reading ``MLFLOW_RUN_ID`` + ``MLFLOW_NESTED_RUN=TRUE``
    from the environment.

    This class NEVER:

    * Calls :func:`mlflow.start_run`.
    * Calls :func:`mlflow.autolog`.
    * Mutates ``MLFLOW_RUN_ID`` / ``MLFLOW_NESTED_RUN`` -- both must be
      pre-set by the launcher and consumed read-only here.

    All methods are ``@staticmethod`` because there is no per-instance
    state worth carrying.
    """

    @staticmethod
    def configure_training_args(
        training_args: Any,
        *,
        local_rank: int | None = None,
    ) -> None:
        """Apply Pattern A knobs to a :class:`TrainingArguments`-like object.

        Two adjustments are made:

        1. Force ``report_to=["mlflow"]`` so the HF ``MLflowCallback``
           is the *only* MLflow producer in the trainer subprocess.
        2. Call :func:`mlflow.set_system_metrics_node_id` so multi-GPU
           runs tag system metrics with the local rank (avoids
           overwriting on rank 0 from rank N).

        :param training_args: An instance of
            :class:`transformers.TrainingArguments` (or any object
            exposing a writable ``report_to`` attribute).
        :param local_rank: ``LOCAL_RANK`` from torch distributed. Pass
            ``None`` for single-GPU runs; treated as rank 0.

        .. note::
            ``MLFLOW_*`` env vars are consumed by the HF callback and
            the native MLflow SDK. They MUST be set by the launcher
            BEFORE the trainer subprocess is spawned -- this method
            does not touch them. Call :meth:`validate_env` first.
        """
        # 1) Force HF MLflowCallback as the only MLflow producer.
        try:
            setattr(training_args, "report_to", ["mlflow"])
        except Exception as exc:  # pragma: no cover -- defensive
            logger.warning(
                "[HF_WIRING] failed to set report_to on training_args: %s",
                exc,
            )

        # 2) Lazy mlflow import so unit tests without mlflow installed
        # do not blow up at import time.
        try:
            import mlflow  # type: ignore[import-not-found]

            node_id = int(local_rank) if local_rank is not None else 0
            mlflow.set_system_metrics_node_id(node_id)
            logger.info(
                "[HF_WIRING] system_metrics_node_id=%d (local_rank=%s)",
                node_id,
                local_rank,
            )
        except Exception as exc:  # pragma: no cover -- defensive
            # Failing to set the node id is non-fatal -- system metrics
            # will simply alias on rank 0 across processes. Log loudly
            # but never crash training over it.
            logger.warning(
                "[HF_WIRING] set_system_metrics_node_id failed: %s",
                exc,
            )

    @staticmethod
    def validate_env() -> None:
        """Fail-fast guard for required Pattern A env vars.

        Raises :class:`ConfigInvalidError` when any of the variables in
        :data:`_REQUIRED_ENV_VARS` is missing or empty. Additionally
        asserts ``MLFLOW_NESTED_RUN`` equals the literal ``"TRUE"`` --
        a misconfigured value silently makes HF open a top-level run
        instead of a nested child (R-01).
        """
        missing = [
            key
            for key in _REQUIRED_ENV_VARS
            if not os.environ.get(key, "").strip()
        ]
        if missing:
            raise ConfigInvalidError(
                detail=(
                    "HFMlflowWiring.validate_env: missing required env "
                    f"vars: {sorted(missing)}. Did training_launcher set "
                    "them?"
                ),
                context={
                    "legacy_code": "MLFLOW_ENV_MISSING",
                    "missing_keys": sorted(missing),
                },
            )
        nested = os.environ["MLFLOW_NESTED_RUN"].strip()
        if nested != "TRUE":
            raise ConfigInvalidError(
                detail=(
                    "HFMlflowWiring.validate_env: MLFLOW_NESTED_RUN must "
                    f"equal the literal 'TRUE' (got {nested!r}). HF "
                    "MLflowCallback will silently open a top-level run "
                    "for any other value."
                ),
                context={
                    "legacy_code": "MLFLOW_NESTED_RUN_INVALID",
                    "actual": nested,
                },
            )
        logger.info(
            "[HF_WIRING] env validated: tracking_uri=%s run_id=%s "
            "nested=%s experiment=%s",
            os.environ["MLFLOW_TRACKING_URI"],
            os.environ["MLFLOW_RUN_ID"],
            nested,
            os.environ["MLFLOW_EXPERIMENT_NAME"],
        )
