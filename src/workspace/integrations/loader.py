"""UX-side YAML loader: resolve integrations + validate.

This is the **single entry point** every caller (CLI, project adapter,
Web-API subprocess) uses to turn a YAML file into a validated
:class:`PipelineConfig`. Core's ``PipelineConfig.from_yaml`` stays
pure — it doesn't know about integrations. The resolver pass that
inlines ``integration: <id>`` shortcuts lives here, in the UX layer,
because *integration* is a Settings/registry concept.

Workflow:

    cfg = load_pipeline_config(path)
    PipelineOrchestrator(config=cfg).run()

If the YAML uses ``integration: <id>`` and that id isn't registered,
``IntegrationNotFoundError`` is raised — the CLI top-level handler
turns it into a clean ``die()`` message (no Pydantic traceback).
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import yaml

from src.workspace.integrations.resolver import resolve_yaml_integrations

if TYPE_CHECKING:
    from src.config.pipeline.schema import PipelineConfig
    from src.workspace.integrations.registry import IntegrationRegistry


def load_pipeline_config(
    path: str | Path,
    *,
    registry: IntegrationRegistry | None = None,
) -> PipelineConfig:
    """Load YAML, inline integrations, validate.

    Args:
        path: Path to the project's pipeline YAML.
        registry: Optional :class:`IntegrationRegistry`. Mostly for
            tests — production callers omit and the default
            workspace registry (``~/.ryotenkai``) is used.

    Returns:
        Validated :class:`PipelineConfig` with
        ``_source_path`` / ``_source_root`` set just like
        ``PipelineConfig.from_yaml`` does for the legacy
        non-integration path.

    Raises:
        FileNotFoundError: ``path`` doesn't exist.
        IntegrationNotFoundError: ``integration: <id>`` references an
            unregistered integration.
        IntegrationUnresolvedError: integration exists but its
            ``current.yaml`` is unusable.
        ValidationError: YAML doesn't match :class:`PipelineConfig`
            schema even after integration inline.
    """
    # Lazy import: keep this loader light; PipelineConfig pulls a fair
    # amount of pydantic schema machinery.
    from src.config.pipeline.schema import PipelineConfig

    path_obj = Path(path).expanduser().resolve()
    raw_text = path_obj.read_text(encoding="utf-8")
    raw: dict | None = yaml.safe_load(raw_text)
    if not isinstance(raw, dict):
        raise ValueError(
            f"Pipeline YAML must be a mapping at the top level, got "
            f"{type(raw).__name__}: {path_obj}"
        )

    resolved = resolve_yaml_integrations(raw, registry=registry)
    cfg = PipelineConfig(**resolved)

    # Mirror what PipelineConfig.from_yaml sets — downstream code reads
    # _source_path for the run-state ``config_path`` field, and
    # _source_root to resolve relative dataset paths.
    object.__setattr__(cfg, "_source_path", path_obj)
    root = path_obj.parent
    for parent in path_obj.parents:
        if parent.name == "config":
            root = parent.parent
            break
    object.__setattr__(cfg, "_source_root", root)
    return cfg


__all__ = [
    "load_pipeline_config",
]
