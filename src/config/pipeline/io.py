from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import yaml


def load_config(config_path: str | Path):
    """
    Load configuration from YAML file.

    Args:
        config_path: Path to configuration YAML file

    Returns:
        PipelineConfig: Validated configuration object

    Raises:
        FileNotFoundError: If config file doesn't exist
        ValidationError: If config validation fails
    """
    # Local import to avoid import-time cycles.
    from .schema import PipelineConfig

    return PipelineConfig.from_yaml(config_path)


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
