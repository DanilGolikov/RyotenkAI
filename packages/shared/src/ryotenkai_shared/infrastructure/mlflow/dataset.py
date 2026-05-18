"""Thin :mod:`mlflow.data` HuggingFace-dataset logging facade.

Wraps :func:`mlflow.data.huggingface_dataset.from_huggingface` and
``mlflow.log_input`` so the trainer doesn't have to import MLflow's
``data`` subpackage directly. Centralising the call also gives us one
place to add governance (consent for dataset metadata, redaction of
private revisions, etc.) in the future.

Per ``docs/plans/vectorized-fluttering-mist.md`` §Target architecture —
table row "Dataset linkage (mlflow.data.Dataset log_input)".
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal

from ryotenkai_shared.utils.logger import get_logger

if TYPE_CHECKING:
    from ryotenkai_shared.infrastructure.mlflow.transport import MlflowTransport

logger = get_logger(__name__)


DatasetContext = Literal["training", "validation", "evaluation"]
"""Allowed values for the ``context`` field on ``mlflow.log_input``.

MLflow accepts any string here, but we restrict to the three roles the
RyotenkAI pipeline uses so the UI grouping stays consistent.
"""


class HFDatasetLogger:
    """Log a HuggingFace ``datasets.Dataset`` as a MLflow run input.

    :param client: Configured :class:`MlflowTransport`. The transport's
        ``mlflow.set_tracking_uri`` call has already executed by the
        time the trainer holds this logger, so we can use the fluent
        ``mlflow.log_input`` API safely.
    """

    def __init__(self, client: MlflowTransport) -> None:
        self._client = client

    def log(
        self,
        run_id: str,
        ds: Any,
        *,
        context: DatasetContext,
        name: str,
        path: str,
        revision: str | None = None,
        targets: str | None = None,
    ) -> None:
        """Materialise ``ds`` as ``mlflow.data.huggingface_dataset.HuggingFaceDataset``
        and emit a ``log_input`` event on ``run_id``.

        Uses ``MlflowClient.log_inputs`` (the client-level API) when
        available — that lets us target ``run_id`` explicitly without
        relying on the fluent ``mlflow.active_run()`` context.
        Falls back to the fluent API for older MLflow versions.

        :param run_id: Active MLflow run id to attach the input to.
        :param ds: A ``datasets.Dataset`` (or compatible).
        :param context: Role of this dataset in the run lifecycle.
        :param name: Human-readable name for the dataset.
        :param path: Storage path or HuggingFace dataset identifier
            (e.g. ``"squad"`` or a local path).
        :param revision: Optional dataset revision (commit / tag) for
            reproducibility.
        :param targets: Name of the target column (passed through to
            :func:`mlflow.data.huggingface_dataset.from_huggingface`).

        Logs and swallows MLflow-side errors — dataset-logging failure
        must not crash the training loop. The data lineage is a
        nice-to-have, not a correctness invariant.
        """
        try:
            mlflow_ds = self._build_mlflow_dataset(
                ds,
                name=name,
                path=path,
                revision=revision,
                targets=targets,
            )
        except Exception as exc:  # noqa: BLE001 — boundary
            logger.warning(
                "[DATASET] failed to construct mlflow.data dataset "
                "name=%s path=%s revision=%s: %s",
                name,
                path,
                revision,
                exc,
            )
            return

        try:
            self._log_input(run_id, mlflow_ds, context)
            logger.info(
                "[DATASET] logged run=%s name=%s context=%s path=%s revision=%s",
                run_id,
                name,
                context,
                path,
                revision,
            )
        except Exception as exc:  # noqa: BLE001 — boundary
            logger.warning(
                "[DATASET] log_input failed run=%s name=%s context=%s: %s",
                run_id,
                name,
                context,
                exc,
            )

    # -- internal helpers -------------------------------------------

    @staticmethod
    def _build_mlflow_dataset(
        ds: Any,
        *,
        name: str,
        path: str,
        revision: str | None,
        targets: str | None,
    ) -> Any:
        """Call ``mlflow.data.huggingface_dataset.from_huggingface``.

        Lazy import so this module stays importable without MLflow's
        ``data`` extension wired up.
        """
        from mlflow.data import huggingface_dataset  # noqa: PLC0415

        kwargs: dict[str, Any] = {
            "name": name,
            "path": path,
        }
        if revision is not None:
            kwargs["revision"] = revision
        if targets is not None:
            kwargs["targets"] = targets
        return huggingface_dataset.from_huggingface(ds, **kwargs)

    def _log_input(
        self,
        run_id: str,
        mlflow_ds: Any,
        context: DatasetContext,
    ) -> None:
        """Emit ``log_inputs`` against the specific ``run_id``.

        Prefer the client-level ``MlflowClient.log_inputs`` (target
        run is explicit) over the fluent ``mlflow.log_input`` (uses
        ``mlflow.active_run()``) — the trainer code path runs under
        HF Trainer's MLflowCallback context, where ``active_run()``
        may resolve to the child run rather than our intended target.
        """
        from mlflow.entities import DatasetInput, InputTag  # noqa: PLC0415

        client = self._client.client
        log_inputs = getattr(client, "log_inputs", None)
        # Build a DatasetInput with the context tag — that's how
        # ``mlflow.log_input(dataset, context=...)`` is implemented
        # under the hood in MLflow 3.x.
        dataset_input = DatasetInput(
            dataset=mlflow_ds._to_mlflow_entity()  # noqa: SLF001 — public via API
            if hasattr(mlflow_ds, "_to_mlflow_entity")
            else mlflow_ds,
            tags=[InputTag(key="mlflow.data.context", value=context)],
        )
        if log_inputs is not None:
            log_inputs(run_id, [dataset_input])
            return
        # Fall back to fluent API for older MLflow.
        import mlflow  # noqa: PLC0415

        mlflow.log_input(mlflow_ds, context=context)


__all__ = ["DatasetContext", "HFDatasetLogger"]
