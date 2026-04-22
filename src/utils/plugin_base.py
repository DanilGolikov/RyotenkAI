"""
BasePlugin — lightweight mixin for all plugin systems in this project.

Metadata (name, priority, version, description, params/thresholds schema,
required secrets) lives in the plugin's ``manifest.toml`` and is attached
to the class at load time by ``src/community/loader.py``. This mixin only
exposes the ClassVar slots that runtime code expects to read.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar

if TYPE_CHECKING:
    from src.community.manifest import PluginManifest


class BasePlugin:
    """Mixin that reserves metadata slots populated by the community loader.

    All plugin systems share three invariants, injected by the loader from
    ``manifest.toml``:

      - ``name``     — unique string key used by registries for lookup.
      - ``priority`` — execution order hint (lower = runs earlier).
      - ``version``  — semver string, useful for compatibility checks.
      - ``_required_secrets`` — tuple of secret keys the plugin needs.
      - ``_community_manifest`` — full :class:`PluginManifest` object, available
        whenever the plugin was loaded via :class:`CommunityCatalog`. Runtime
        helpers (``get_description`` below) read from this so the manifest
        stays the single source of truth for human-readable metadata.

    This class intentionally carries NO abstract methods and NO ``__init__``
    so it can be inserted into any ABC/Protocol hierarchy without MRO conflicts.
    """

    name: ClassVar[str] = ""
    priority: ClassVar[int] = 50
    version: ClassVar[str] = "1.0.0"
    _required_secrets: ClassVar[tuple[str, ...]] = ()
    _community_manifest: ClassVar[PluginManifest | None] = None

    @classmethod
    def get_description(cls) -> str:
        """Return the plugin description.

        Default implementation pulls ``manifest.plugin.description`` — authors
        maintain the text in ``manifest.toml`` and it surfaces identically in
        the UI (via ``GET /plugins/{kind}``) and in runtime reports.

        Override only if you need a *dynamic* description (depending on params
        or state); most plugins should leave this alone.
        """
        manifest = cls._community_manifest
        if manifest is not None:
            return manifest.plugin.description
        return ""


__all__ = ["BasePlugin"]
