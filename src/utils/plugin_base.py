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

import os
from typing import TYPE_CHECKING, Any, ClassVar

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

    #: Names of ``community/libs/<name>/`` packages this plugin imports
    #: from (e.g. ``("helixql",)``). Mirror of ``[plugin].libs`` in
    #: ``manifest.toml``: when both sides are non-empty the loader
    #: cross-checks them and raises on drift. Leaving the default empty
    #: tuple skips the check; the manifest's ``libs`` list is then the
    #: only source of truth (which is fine for plugins that don't
    #: subclass :class:`BasePlugin` or that prefer to keep declarations
    #: in TOML only).
    #:
    #: Order does not matter — comparison is performed on the sorted
    #: set so plugin authors don't have to keep Python and TOML in
    #: lockstep ordering. ``ryotenkai community sync-libs <plugin>``
    #: writes ``manifest.toml``'s ``libs`` from this declaration.
    REQUIRED_LIBS: ClassVar[tuple[str, ...]] = ()

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

    # ------------------------------------------------------------------
    # Runtime accessors — subclasses use these to read envs/secrets
    # injected by the registry. Authors should NOT poke ``self._secrets``
    # or ``os.environ`` directly: these helpers are the contract that
    # gives us a place to add validation, telemetry, and per-test
    # mocking later without touching every plugin.
    # ------------------------------------------------------------------

    def _env(self, name: str, default: str | None = None) -> str | None:
        """Return the value of env var ``name`` for this plugin instance.

        Lookup order:

        1. ``self._injected_env`` — when the registry's ``instantiate``
           call passed ``env=...`` (typically the project's ``env.json``
           dict, plus any test-time overrides). Wins so tests get
           deterministic values without monkey-patching ``os.environ``.
        2. ``os.environ`` — production fallback. The launcher merges
           the project's ``env.json`` on top of process env *before*
           forking, so by the time a plugin runs in-process the
           variables are already there.

        Returns ``default`` (``None`` by default) when the env var is
        unset in both layers. Plugins that treat absence as a hard
        error should declare the env in ``[[required_env]]`` with
        ``optional=false`` — the preflight gate refuses to launch when
        such a key is missing, so by the time ``_env`` is called the
        value is guaranteed present.
        """
        injected = getattr(self, "_injected_env", None)
        if injected and name in injected:
            value = injected[name]
            return value if value is not None else default
        value = os.environ.get(name)
        return value if value is not None and value != "" else default

    def _secret(self, name: str) -> str:
        """Return a required secret value. Raises if not injected.

        Reads from ``self._secrets`` (populated by
        :meth:`PluginRegistry.instantiate` from the per-kind
        ``PluginSecretsResolver`` — the value comes from
        ``secrets.env`` / ``Secrets.model_extra``). The key MUST be
        declared in the manifest's ``[[required_env]]`` with
        ``secret=true, optional=false`` so the loader stamps it onto
        ``cls._required_secrets`` and the registry resolves it at
        instantiate time.

        Raises :class:`KeyError` when the key isn't in the resolved
        dict — that's a programming error in the manifest, not a
        configuration drift, so we surface a clear message instead of
        falling back to ``os.environ``.
        """
        secrets: dict[str, str] | None = getattr(self, "_secrets", None)
        if not secrets:
            raise KeyError(
                f"plugin {type(self).__name__}: no secrets resolved yet. "
                f"Did you declare {name!r} in [[required_env]] with "
                "secret=true, optional=false? Check manifest.toml."
            )
        if name not in secrets:
            raise KeyError(
                f"plugin {type(self).__name__}: secret {name!r} not in "
                f"resolved set {sorted(secrets.keys())!r}. Add it to "
                "[[required_env]] in manifest.toml or fix the access."
            )
        return secrets[name]


__all__ = ["BasePlugin"]
