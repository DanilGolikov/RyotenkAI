"""Pre-validation resolver for ``integration: <id>`` shortcuts.

Lives in the UX layer (``src/workspace/integrations/``) because
*integration* is a Settings/registry concept — core's
``PipelineConfig`` schema knows nothing about
``~/.ryotenkai/integrations/``. This module is the bridge: it walks
a raw YAML dict (parsed via ``yaml.safe_load`` but not yet validated
by Pydantic), inlines any ``integration: <id>`` references, and
returns a clean dict ready for ``PipelineConfig(**dict)``.

Workflow:

    raw = yaml.safe_load(path.read_text())
    resolved = resolve_yaml_integrations(raw)   # ← this module
    cfg = PipelineConfig(**resolved)            # core validation

Merge policy: integration-side fields provide defaults; project-side
keys win on conflict. The resolver removes the ``integration:`` key
from the merged output and stamps an ``integration: <id>`` field on
the resolved block (kept as a *secrets-tag* so runtime code can
``secrets.get_provider_token(cfg.integration)``).

Errors raised here:
- :class:`IntegrationNotFoundError` — id not in registry
- :class:`IntegrationUnresolvedError` — found but unfit (empty
  ``current.yaml``, schema mismatch, type mismatch)

The CLI's top-level handler catches these and renders clean
``die()`` messages — same pattern as ``ProjectNotFoundError``.
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import TYPE_CHECKING, Any

import yaml

from src.workspace.integrations.exceptions import (
    IntegrationNotFoundError,
    IntegrationUnresolvedError,
)
from src.workspace.integrations.registry import (
    IntegrationRegistry,
    IntegrationRegistryError,
)
from src.workspace.integrations.store import IntegrationStore

if TYPE_CHECKING:
    from src.workspace.integrations.models import IntegrationRegistryEntry


def resolve_yaml_integrations(
    raw: dict[str, Any],
    *,
    registry: IntegrationRegistry | None = None,
) -> dict[str, Any]:
    """Expand ``integration: <id>`` shortcuts in a raw config dict.

    Walks known paths:
    - ``experiment_tracking.mlflow``: full content merge from the
      integration's ``current.yaml`` (project keys win on conflict).
    - ``experiment_tracking.huggingface``: existence check only;
      ``integration_id`` left in place as a secrets-tag.

    Returns a NEW dict (deep-copied). The input is not mutated, so
    callers can keep showing the original to the user (e.g. in the
    "config you wrote" UI panel) while the orchestrator uses the
    resolved version.

    Raises:
        IntegrationNotFoundError: a referenced ``integration: <id>``
            isn't in the registry.
        IntegrationUnresolvedError: the integration exists but its
            ``current.yaml`` is unusable (empty, malformed, schema
            mismatch, type mismatch).
    """
    if registry is None:
        registry = IntegrationRegistry()

    out = copy.deepcopy(raw)
    et = out.get("experiment_tracking")
    if not isinstance(et, dict):
        return out

    # ---- MLflow: content merge --------------------------------------
    mlflow_block = et.get("mlflow")
    if isinstance(mlflow_block, dict) and mlflow_block.get("integration"):
        et["mlflow"] = _resolve_mlflow_block(mlflow_block, registry)

    # ---- HuggingFace: existence check only --------------------------
    hf_block = et.get("huggingface")
    if isinstance(hf_block, dict) and hf_block.get("integration"):
        _verify_integration(
            integration_id=str(hf_block["integration"]).strip(),
            expected_type="huggingface",
            registry=registry,
        )
        # Project-side ref already carries every project-local field
        # (repo_id, private). Nothing to inline; integration just
        # needs to exist for ``secrets.get_hf_token(integration_id)``
        # to find its ``token.enc``.

    return out


def _resolve_mlflow_block(
    block: dict[str, Any],
    registry: IntegrationRegistry,
) -> dict[str, Any]:
    """Merge ``<project>/configs/current.yaml``'s mlflow block with the
    integration's ``current.yaml``.

    Project keys win on conflict. The ``integration:`` key is
    preserved in the output as a secrets-tag (lets stages call
    ``secrets.get_provider_token(cfg.integration)``).
    """
    integration_id = str(block["integration"]).strip()
    if not integration_id:
        # Empty string after strip — treat like absent and let core
        # validation surface the "tracking_uri required" error.
        return {k: v for k, v in block.items() if k != "integration"}

    entry = _verify_integration(integration_id, "mlflow", registry)
    integration_data = _read_integration_yaml(integration_id, Path(entry.path))

    # Merge: integration provides defaults, project overrides on top.
    merged: dict[str, Any] = {}
    merged.update(integration_data)
    for k, v in block.items():
        if k == "integration":
            # Preserve as secrets-tag in the output dict — the resolver
            # writes it back so MLflowConfig.integration is set.
            continue
        merged[k] = v
    merged["integration"] = integration_id
    return merged


def _read_integration_yaml(
    integration_id: str, integration_root: Path
) -> dict[str, Any]:
    """Read the integration's ``current.yaml`` as a plain dict.

    The schema-strict validation that used to happen here moved to
    ``MLflowConfig`` itself (Pydantic will reject unknown / malformed
    fields when the merged dict is validated). This keeps the
    resolver narrow: filesystem read + parse, nothing more.
    """
    store = IntegrationStore(integration_root)
    yaml_text = store.current_yaml_text()
    if not yaml_text.strip():
        raise IntegrationUnresolvedError(
            integration_id,
            "current.yaml is empty — fill it via the Web UI "
            "(http://localhost:5173/settings/integrations) or CLI",
        )
    try:
        data: Any = yaml.safe_load(yaml_text) or {}
    except yaml.YAMLError as exc:
        raise IntegrationUnresolvedError(
            integration_id, f"current.yaml is not valid YAML: {exc}"
        ) from exc
    if not isinstance(data, dict):
        raise IntegrationUnresolvedError(
            integration_id,
            "current.yaml must be a mapping at the top level, got "
            f"{type(data).__name__}",
        )
    return data


def _verify_integration(
    integration_id: str,
    expected_type: str,
    registry: IntegrationRegistry,
) -> IntegrationRegistryEntry:
    """Existence + type check. Returns the registry entry."""
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


__all__ = [
    "resolve_yaml_integrations",
]
