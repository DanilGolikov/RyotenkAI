"""Constants for the community/ catalogue."""

from __future__ import annotations

from pathlib import Path
from typing import Final

PROJECT_ROOT: Final[Path] = Path(__file__).resolve().parents[2]
COMMUNITY_ROOT: Final[Path] = PROJECT_ROOT / "community"
CACHE_DIR: Final[Path] = COMMUNITY_ROOT / ".cache"

MANIFEST_FILENAME: Final[str] = "manifest.toml"
PRESET_DEFAULT_FILENAME: Final[str] = "preset.yaml"

PLUGIN_KIND_DIRS: Final[dict[str, str]] = {
    "validation": "validation",
    "evaluation": "evaluation",
    "reward": "reward",
    "reports": "reports",
}

PRESET_DIR_NAME: Final[str] = "presets"

ALL_PLUGIN_KINDS: Final[tuple[str, ...]] = tuple(PLUGIN_KIND_DIRS.keys())


__all__ = [
    "ALL_PLUGIN_KINDS",
    "CACHE_DIR",
    "COMMUNITY_ROOT",
    "MANIFEST_FILENAME",
    "PLUGIN_KIND_DIRS",
    "PRESET_DEFAULT_FILENAME",
    "PRESET_DIR_NAME",
    "PROJECT_ROOT",
]
