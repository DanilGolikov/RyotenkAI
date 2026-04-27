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

#: Subdirectory of ``community/`` that hosts shared domain libraries
#: (e.g. ``community/libs/helixql/``). Each direct child with an
#: ``__init__.py`` is preloaded into ``sys.modules`` as
#: ``community_libs.<name>`` by :mod:`src.community.libs` before any
#: plugin is imported, so plugins can do
#: ``from community_libs.helixql.compiler import get_compiler``.
LIBS_DIR_NAME: Final[str] = "libs"

#: ``sys.modules`` namespace under which preloaded libs are exposed.
#: Authors import from this prefix; we never put framework code here.
LIBS_NAMESPACE: Final[str] = "community_libs"

ALL_PLUGIN_KINDS: Final[tuple[str, ...]] = tuple(PLUGIN_KIND_DIRS.keys())


__all__ = [
    "ALL_PLUGIN_KINDS",
    "CACHE_DIR",
    "COMMUNITY_ROOT",
    "LIBS_DIR_NAME",
    "LIBS_NAMESPACE",
    "MANIFEST_FILENAME",
    "PLUGIN_KIND_DIRS",
    "PRESET_DEFAULT_FILENAME",
    "PRESET_DIR_NAME",
    "PROJECT_ROOT",
]
