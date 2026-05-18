"""Publish trained models to the MLflow Model Registry via aliases.

Phase M5 of the MLflow redesign replaces the deprecated ``Staging`` /
``Production`` / ``Archived`` stages with MLflow 3.x **aliases**
(``@champion`` / ``@challenger``). :class:`ModelPublisher` is the
single trainer-side entry point for that promotion path.

Usage::

    publisher = ModelPublisher(registry=concrete_registry)
    version = publisher.publish(
        run_id=os.environ["MLFLOW_RUN_ID"],
        artifact_path="model",
        registered_name="ryotenkai/qwen25-05b",
        alias_on_success="challenger",
    )
    logger.info("registered version=%s alias=challenger", version.version)

The publisher only knows about :class:`IModelRegistry`; the concrete
``mlflow.MlflowClient`` wrapping lives in
:class:`ryotenkai_shared.infrastructure.mlflow.registry.MlflowModelRegistry`.

Separation of concerns
----------------------
The publisher owns the **registry** half of the publish lifecycle
(``register_model`` + ``set_alias``). The **artifact-logging** half --
i.e. :func:`mlflow.transformers.log_model` with ``save_pretrained=True``
(R-21) -- is the caller's responsibility and happens BEFORE
:meth:`publish` so the ``runs:/{run_id}/{artifact_path}`` URI is
materialised when registration kicks in. This split keeps the
publisher decoupled from HF Trainer + tokenizer types and from the
mlflow.transformers heavy import.

See ``docs/plans/vectorized-fluttering-mist.md`` -- Phase M5.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from ryotenkai_shared.utils.logger import logger

if TYPE_CHECKING:
    from ryotenkai_shared.infrastructure.mlflow.protocols import (
        IModelRegistry,
        ModelVersion,
    )


__all__ = ["ModelPublisher"]


class ModelPublisher:
    """Publish trained models to the MLflow Model Registry using aliases.

    :param registry: Concrete :class:`IModelRegistry` implementation
        (typically
        :class:`ryotenkai_shared.infrastructure.mlflow.registry.MlflowModelRegistry`
        when wired by the trainer's composition root).

    Aliases over stages
    -------------------
    MLflow's ``Staging`` / ``Production`` / ``Archived`` stages are
    deprecated as of 2.9.0 and removed in 3.0. The supported mechanism
    is registered-model aliases (movable pointers per name). We default
    to ``challenger`` on a successful publish; promotion to
    ``champion`` is a manual gate, executed via
    ``ryotenkai model promote`` (per Q-12 in the plan).
    """

    def __init__(self, registry: IModelRegistry) -> None:
        self._registry = registry

    def publish(
        self,
        run_id: str,
        artifact_path: str,
        registered_name: str,
        alias_on_success: str = "challenger",
    ) -> ModelVersion:
        """Register the model artifact at ``run_id/artifact_path`` and set an alias.

        :param run_id: MLflow run id under which the model artifact
            lives. In Pattern A this is the HF callback's nested child
            run id (read from :attr:`mlflow.active_run`).
        :param artifact_path: Path within the run's artifact tree
            pointing at the model directory (e.g. ``"model"`` for the
            standard
            ``mlflow.transformers.log_model(..., artifact_path="model")``
            convention).
        :param registered_name: Canonical name in the registry
            (template: ``ryotenkai/{experiment}/{model_family}``).
        :param alias_on_success: Movable alias to attach to the new
            version once registration succeeds. Defaults to
            ``"challenger"``; promotion to ``"champion"`` is manual via
            ``ryotenkai model promote``.

        :returns: :class:`ModelVersion` describing the new registered
            version.

        .. note::

            The artifact at ``runs:/{run_id}/{artifact_path}`` MUST
            already exist when this method is called. Logging the
            artifact (``mlflow.transformers.log_model`` with
            ``save_pretrained=True``, R-21) is the caller's
            responsibility -- see :mod:`ryotenkai_pod.trainer.run_training`
            for the canonical wiring sequence.
        """
        # Construct the ``runs:/`` MLflow model URI per the registry
        # contract.
        model_uri = f"runs:/{run_id}/{artifact_path}"
        logger.info(
            "[MODEL_PUBLISHER] registering name=%s uri=%s",
            registered_name,
            model_uri,
        )
        version = self._registry.register(model_uri, registered_name)

        # ``ModelVersion.version`` is a string per the MLflow proto;
        # passing it through unchanged keeps the registry contract
        # simple.
        try:
            self._registry.set_alias(
                registered_name,
                alias_on_success,
                version.version,
            )
            logger.info(
                "[MODEL_PUBLISHER] alias set name=%s alias=%s version=%s",
                registered_name,
                alias_on_success,
                version.version,
            )
        except Exception as exc:
            # The version is already registered; failing to set the
            # alias is non-fatal (operators can attach it manually
            # later). Log loudly and re-raise so the caller decides
            # whether to abort.
            logger.error(
                "[MODEL_PUBLISHER] set_alias failed name=%s alias=%s "
                "version=%s: %s",
                registered_name,
                alias_on_success,
                version.version,
                exc,
            )
            raise

        return version
