"""YAML loader for pipeline configs.

Single entry point for turning a project YAML into a validated
:class:`PipelineConfig`. Direct ``yaml.safe_load → PipelineConfig`` —
no resolver, no Settings-registry indirection. Project YAMLs carry
their own values inline (tracking_uri, repo_id, etc.).

History: an earlier iteration ran an ``integration: <id>`` resolver
pass here that inlined values from a Settings-managed registry. That
indirection layer is gone — projects own their integration config
directly.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import yaml

if TYPE_CHECKING:
    from src.config.pipeline.schema import PipelineConfig


def load_pipeline_config(path: str | Path) -> PipelineConfig:
    """Load + validate YAML.

    Args:
        path: Path to the project's pipeline YAML.

    Returns:
        Validated :class:`PipelineConfig` with ``_source_path`` /
        ``_source_root`` set so downstream code can resolve relative
        dataset paths.

    Raises:
        FileNotFoundError: ``path`` doesn't exist.
        ValidationError: YAML doesn't match :class:`PipelineConfig` schema.
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

    cfg = PipelineConfig(**raw)

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
