from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import yaml


def load_config(config_path: str | Path, *, resolve_integrations: bool = True):
    """
    Load configuration from YAML file.

    Pipeline order:

    1. Read YAML and validate against :class:`PipelineConfig` schema.
    2. (default) Resolve integration refs against the workspace registry
       — the project ref ``experiment_tracking.mlflow.integration: <id>``
       is replaced with a fully-resolved :class:`MLflowConfig` carrying
       ``tracking_uri`` etc.

    Args:
        config_path: Path to configuration YAML file.
        resolve_integrations: When ``True`` (default), call the resolver
            after schema validation. Set ``False`` for tests that build
            pipeline configs without an integrations registry on disk.

    Returns:
        PipelineConfig: validated and (optionally) resolved configuration.

    Raises:
        FileNotFoundError: config file doesn't exist.
        ValidationError: config fails Pydantic validation.
        IntegrationNotFoundError: a referenced integration id is not
            registered in ``~/.ryotenkai/integrations.json``.
        IntegrationUnresolvedError: a referenced integration exists but
            its ``current.yaml`` is empty / fails schema validation.
    """
    # Local imports to avoid import-time cycles.
    from .schema import PipelineConfig

    cfg = PipelineConfig.from_yaml(config_path)

    if resolve_integrations:
        # Imported lazily because the resolver pulls the workspace
        # registry, which the lightweight schema layer should not
        # depend on at import time.
        from src.config.integrations.resolver import resolve_pipeline_config

        cfg = resolve_pipeline_config(cfg)

    return cfg


class PipelineIOMixin:
    """
    I/O helpers for PipelineConfig.

    NOTE: This mixin assumes the host model defines:
    - self._source_path, self._source_root PrivateAttr
    """

    _source_path: Path | None
    _source_root: Path | None

    def get_source_root(self) -> Path:
        """
        Get root directory for resolving relative paths in config (datasets, etc).

        Rule:
        - If config file is anywhere under a `config/` directory (e.g. `config/pipeline_config.yaml`,
          `config/64_tests/test_1.yaml`), root = parent of that `config/`.
        - Otherwise root = directory containing the config file.
        - If config was created programmatically (no source_path), fallback = cwd.
        """
        if self._source_root is not None:
            return self._source_root
        return Path.cwd()

    def resolve_path(self, path_str: str | None) -> Path | None:
        """
        Resolve a config path string into an absolute local filesystem path.

        - Absolute paths are returned as-is (expanded).
        - Relative paths are resolved against get_source_root().
        """
        if path_str is None:
            return None
        p = Path(path_str).expanduser()
        if p.is_absolute():
            return p
        return (self.get_source_root() / p).resolve()

    @classmethod
    def from_yaml(cls, path: str | Path):
        """Load configuration from YAML file."""
        yaml_mod = cast("Any", yaml)  # PyYAML stubs can confuse IDEs

        path_obj = Path(path).resolve() if isinstance(path, str) else path.resolve()
        with path_obj.open() as f:
            data = yaml_mod.safe_load(f)
        cfg = cls(**data)
        cfg._source_path = path_obj
        # If config is anywhere under a config/ directory, treat its parent as root.
        root = path_obj.parent
        for parent in path_obj.parents:
            if parent.name == "config":
                root = parent.parent
                break
        cfg._source_root = root

        return cfg

    def to_yaml(self, path: str | Path) -> None:
        """Save configuration to YAML file."""
        yaml_mod = cast("Any", yaml)  # PyYAML stubs can confuse IDEs

        path_obj = Path(path) if isinstance(path, str) else path
        with path_obj.open("w") as f:
            yaml_mod.safe_dump(
                cast("Any", self).model_dump(mode="python", exclude_none=True),
                f,
                default_flow_style=False,
                sort_keys=False,
            )


__all__ = [
    "PipelineIOMixin",
    "load_config",
]
