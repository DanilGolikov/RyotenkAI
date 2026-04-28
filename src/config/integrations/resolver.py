"""Integration resolver — closes the PR3 follow-up gap.

Project-side YAML carries integration **references** (``MLflowTrackingRef``,
``HuggingFaceHubConfig``) — these only know an integration ``id``. The
runtime engine, however, expects fully-resolved configs (e.g. ``MLflowConfig``
with ``tracking_uri`` populated). This module bridges the gap by reading
the integration's ``current.yaml`` from
``~/.ryotenkai/integrations/<id>/`` and merging it with the project ref.

Pure transformation. Does NOT know about projects, runs, or stages.

Called as the final step of :func:`src.config.pipeline.io.load_config` so
every entry-point (CLI, API, agents constructing configs in-memory)
receives a resolved config.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import yaml

from src.config.integrations.exceptions import (
    IntegrationNotFoundError,
    IntegrationUnresolvedError,
)
from src.config.integrations.mlflow import MLflowConfig, MLflowTrackingRef
from src.config.integrations.mlflow_integration import MLflowIntegrationConfig
from src.workspace.integrations.registry import (
    IntegrationRegistry,
    IntegrationRegistryError,
)
from src.workspace.integrations.store import IntegrationStore

if TYPE_CHECKING:
    from src.config.pipeline.schema import PipelineConfig


def resolve_pipeline_config(
    config: PipelineConfig,
    *,
    registry: IntegrationRegistry | None = None,
) -> PipelineConfig:
    """Resolve every integration ref in ``config`` against the registry.

    The mutation is in-place on the same ``config`` object — the function
    returns it for fluent-style usage. Pydantic ``validate_assignment`` is
    not enabled on :class:`StrictBaseModel`, so swapping a field with a
    different type does not trigger re-validation.

    Args:
        config: validated :class:`PipelineConfig`. May contain unresolved
            refs (the typical YAML-load case).
        registry: optional :class:`IntegrationRegistry` instance. When
            ``None`` the default workspace root (``~/.ryotenkai``) is used.

    Returns:
        The same ``config`` instance, with resolved fields swapped in.

    Raises:
        IntegrationNotFoundError: a referenced integration id is not in
            the registry.
        IntegrationUnresolvedError: a referenced integration exists but
            its ``current.yaml`` is empty / fails schema validation, or
            the integration's declared ``type`` does not match the ref
            type.
    """
    et = config.experiment_tracking
    if et is None:
        return config

    if registry is None:
        registry = IntegrationRegistry()

    # MLflow: ref → resolved MLflowConfig
    if isinstance(et.mlflow, MLflowTrackingRef) and et.mlflow.integration:
        resolved = _resolve_mlflow(et.mlflow, registry)
        # Bypass field-type validation — StrictBaseModel does not have
        # validate_assignment=True, so direct attribute set is safe.
        object.__setattr__(et, "mlflow", resolved)

    # HuggingFace: existence check only. The project-side ref already
    # carries every project-local field (repo_id, private); the
    # integration just needs to exist (its token is fetched at use-time
    # via secrets.get_hf_token(integration_id)).
    if et.huggingface is not None and et.huggingface.integration:
        _verify_integration_exists(et.huggingface.integration, "huggingface", registry)

    return config


# ---------------------------------------------------------------------------
# MLflow resolution
# ---------------------------------------------------------------------------


def _resolve_mlflow(
    ref: MLflowTrackingRef,
    registry: IntegrationRegistry,
) -> MLflowConfig:
    """Read MLflow integration from ``registry``; merge with project ref.

    Merge policy:
    - From integration ``current.yaml`` (validated as :class:`MLflowIntegrationConfig`):
      ``tracking_uri``, ``local_tracking_uri``, ``ca_bundle_path``, ``system_metrics``.
    - From project ref (:class:`MLflowTrackingRef`): ``experiment_name``,
      ``run_description_file``.

    The validator on :class:`MLflowTrackingRef` already enforces
    ``experiment_name`` non-None when ``integration`` is set, so we can
    rely on it here.
    """
    integration_id = cast("str", ref.integration)
    entry = _lookup_integration(integration_id, "mlflow", registry)

    integration_cfg = _load_mlflow_integration_config(integration_id, Path(entry.path))

    return MLflowConfig(
        tracking_uri=integration_cfg.tracking_uri,
        local_tracking_uri=integration_cfg.local_tracking_uri,
        ca_bundle_path=integration_cfg.ca_bundle_path,
        experiment_name=cast("str", ref.experiment_name),  # validated non-None
        run_description_file=ref.run_description_file,
        system_metrics=integration_cfg.system_metrics,
    )


def _load_mlflow_integration_config(
    integration_id: str, integration_root: Path
) -> MLflowIntegrationConfig:
    """Read + validate an MLflow integration's ``current.yaml``."""
    store = IntegrationStore(integration_root)
    yaml_text = store.current_yaml_text()

    if not yaml_text.strip():
        raise IntegrationUnresolvedError(
            integration_id,
            "current.yaml is empty — fill it via the Web UI "
            "(http://localhost:5173/settings/integrations) or CLI",
        )

    try:
        data: dict[str, Any] = yaml.safe_load(yaml_text) or {}
    except yaml.YAMLError as exc:
        raise IntegrationUnresolvedError(
            integration_id, f"current.yaml is not valid YAML: {exc}"
        ) from exc

    if not isinstance(data, dict):
        raise IntegrationUnresolvedError(
            integration_id,
            f"current.yaml must be a mapping at the top level, got "
            f"{type(data).__name__}",
        )

    try:
        return MLflowIntegrationConfig.model_validate(data)
    except Exception as exc:
        raise IntegrationUnresolvedError(
            integration_id,
            f"current.yaml failed MLflowIntegrationConfig schema validation: {exc}",
        ) from exc


# ---------------------------------------------------------------------------
# Generic helpers
# ---------------------------------------------------------------------------


def _lookup_integration(
    integration_id: str,
    expected_type: str,
    registry: IntegrationRegistry,
):
    """Return the registry entry for ``integration_id``.

    Raises :class:`IntegrationNotFoundError` if missing,
    :class:`IntegrationUnresolvedError` if the registered type does not
    match ``expected_type``.
    """
    try:
        entry = registry.resolve(integration_id)
    except IntegrationRegistryError as exc:
        raise IntegrationNotFoundError(integration_id, expected_type) from exc

    if entry.type != expected_type:
        raise IntegrationUnresolvedError(
            integration_id,
            f"registered as type={entry.type!r}, "
            f"but a {expected_type!r}-type integration is expected here",
        )
    return entry


def _verify_integration_exists(
    integration_id: str,
    expected_type: str,
    registry: IntegrationRegistry,
) -> None:
    """Existence-only check (no body merge). Used for HuggingFace refs."""
    _lookup_integration(integration_id, expected_type, registry)


__all__ = [
    "resolve_pipeline_config",
]
