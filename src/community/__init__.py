"""
Community catalogue: unified registry-agnostic loader for plugins and presets
stored under the repo-root ``community/`` directory.

Each extension (plugin or preset) lives either as a folder or a ZIP archive
and is described by a ``manifest.toml`` that is the single source of truth
for its metadata. Plugin/preset code itself is imported/read lazily.
"""

from src.community.catalog import CommunityCatalog, catalog
from src.community.loader import LoadedPlugin, LoadedPreset

__all__ = ["CommunityCatalog", "LoadedPlugin", "LoadedPreset", "catalog"]
