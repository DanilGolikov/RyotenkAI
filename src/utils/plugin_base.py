"""
BasePlugin — lightweight mixin for all plugin systems in this project.

Metadata (name, version, description, params/thresholds schema, required
secrets, declarative env contract) lives in the plugin's ``manifest.toml``
and is attached to the class at load time by ``src/community/loader.py``.
This mixin only exposes the ClassVar slots runtime code expects to read.

Report plugins have their own ordering contract via the ``[reports]``
block's ``order`` field, wired in ``src/reports/plugins/registry.py``.
All other plugin kinds (validation / evaluation / reward) execute in
the order declared in the user's config YAML — we do **not** carry a
global priority field any more.

REQUIRED_ENV cross-check (A7 in cozy-booping-walrus):
    A subclass that declares :attr:`REQUIRED_ENV` as a non-empty tuple
    of :class:`RequiredEnvSpec` instances opts into a strict load-time
    check: the loader compares the Python tuple to the manifest's
    ``[[required_env]]`` block element-wise on (name, optional,
    secret, managed_by). Any mismatch raises :class:`ValueError` with
    a precise diff so the author knows which side to fix. Authors who
    prefer to declare the contract only in TOML can leave
    ``REQUIRED_ENV = ()`` (the default) and the check is skipped.

The CLI side of the cross-check (``ryotenkai community sync-envs``)
auto-writes the TOML block from the Python ClassVar so the two sources
never drift in practice.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar

if TYPE_CHECKING:
    from src.community.manifest import PluginManifest, RequiredEnvSpec


class BasePlugin:
    """Mixin that reserves metadata slots populated by the community loader.

    All plugin systems share these invariants, injected by the loader from
    ``manifest.toml``:

      - ``name``     — unique string key used by registries for lookup.
      - ``version``  — semver string, useful for compatibility checks.
      - ``_required_secrets`` — tuple of secret keys the plugin needs
        (derived from the manifest's ``[[required_env]]`` entries with
        ``secret=true, optional=false``).
      - ``_community_manifest`` — full :class:`PluginManifest` object,
        available whenever the plugin was loaded via
        :class:`CommunityCatalog`. Runtime helpers (``get_description``
        below) read from this so the manifest stays the single source
        of truth for human-readable metadata.

    This class intentionally carries NO abstract methods and NO ``__init__``
    so it can be inserted into any ABC/Protocol hierarchy without MRO
    conflicts.
    """

    name: ClassVar[str] = ""
    version: ClassVar[str] = "1.0.0"
    _required_secrets: ClassVar[tuple[str, ...]] = ()
    _community_manifest: ClassVar[PluginManifest | None] = None

    #: Declarative environment-variable contract for the **plugin author**.
    #:
    #: Set to a tuple of :class:`~src.community.manifest.RequiredEnvSpec`
    #: instances (or dicts that match its shape) when you want code to be
    #: the source of truth for the plugin's env requirements. The loader
    #: cross-checks the manifest against this tuple at load time —
    #: mismatch on (name, optional, secret, managed_by) raises with a
    #: precise diff. ``ryotenkai community sync-envs <plugin>`` writes
    #: the manifest side from this declaration.
    #:
    #: Leaving the default empty tuple opts out of the check entirely —
    #: the manifest is the only source of truth in that case. Useful for
    #: third-party plugins that don't depend on ``src.utils.plugin_base``
    #: at all and only declare envs in TOML.
    REQUIRED_ENV: ClassVar[tuple[RequiredEnvSpec, ...]] = ()

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
