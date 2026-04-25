"""
BasePlugin — lightweight mixin for all plugin systems in this project.

Metadata (name, version, description, params/thresholds schema, required
secrets) lives in the plugin's ``manifest.toml`` and is attached to the
class at load time by ``src/community/loader.py``. This mixin only
exposes the ClassVar slots that runtime code expects to read.

Report plugins have their own ordering contract via the ``[reports]``
block's ``order`` field, wired in ``src/reports/plugins/registry.py``.
All other plugin kinds (validation / evaluation / reward) execute in
the order declared in the user's config YAML — we do **not** carry a
global priority field any more.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar

if TYPE_CHECKING:
    from src.community.manifest import PluginManifest


class BasePlugin:
    """Mixin that reserves metadata slots populated by the community loader.

    All plugin systems share these invariants, injected by the loader from
    ``manifest.toml``:

      - ``name``     — unique string key used by registries for lookup.
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
    version: ClassVar[str] = "1.0.0"
    _required_secrets: ClassVar[tuple[str, ...]] = ()
    _community_manifest: ClassVar[PluginManifest | None] = None

    #: Declarative environment-variable contract. Subclasses set this
    #: as a tuple of dicts matching :class:`RequiredEnvSpec`
    #: (``{"name": "...", "description": "...", "optional": False, "secret": True,
    #: "managed_by": "" | "integrations" | "providers"}``).
    #:
    #: The community-manifest loader cross-checks this list against
    #: ``manifest.toml`` ``[[required_env]]`` blocks at load time so
    #: there's exactly one source of truth — code is the contract,
    #: TOML mirrors it. A future ``scripts/sync_plugin_envs.py`` will
    #: regenerate the TOML side from ``REQUIRED_ENV`` automatically.
    REQUIRED_ENV: ClassVar[tuple[dict[str, object], ...]] = ()

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
