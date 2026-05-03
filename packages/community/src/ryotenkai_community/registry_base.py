"""Generic plugin registry shared by all four kinds.

Each plugin kind (``validation`` / ``evaluation`` / ``reward`` / ``reports``)
historically grew its own registry with bespoke method names and signatures
(``get_plugin`` / ``get`` / ``create`` / ``build_report_plugins``). The
divergence made cross-cutting concerns — secret injection, schema-version
gates, error handling — duplicate in four places.

This module provides :class:`PluginRegistry` — a generic, instance-based
base that fixes a single public surface:

- ``register_from_community(loaded)`` — populate from
  :class:`~src.community.loader.LoadedPlugin`.
- ``instantiate(plugin_id, *, resolver=None, env=None, **init_kwargs)`` —
  build an instance with uniform secret + env injection.
- ``get_class(plugin_id)`` — bare class lookup (no instantiation).
- ``manifest(plugin_id)`` / ``list_manifests()`` — UI manifest accessors.
- ``list_ids()`` / ``is_registered()`` / ``clear()`` — book-keeping.

Each kind subclasses :class:`PluginRegistry` and overrides
``_make_init_kwargs`` to translate the uniform ``**init_kwargs`` into the
constructor signature its plugins expect. The kind-module then exports a
**module-level singleton** (e.g. ``validation_registry``) that the rest of
the codebase imports — direct class access is reserved for tests that need
to extend behaviour.

Secret injection uses the existing ``_required_secrets`` ClassVar +
:class:`PluginSecretsResolver` contract from
:mod:`src.utils.plugin_secrets` — keys declared in TOML are mirrored onto
the class by the loader and resolved here from the per-kind prefix
(``DTST_`` / ``EVAL_`` / future ``RWRD_`` / ``RPRT_``). Plugins that don't
declare ``_required_secrets`` simply pass through.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, ClassVar, Generic, TypeVar

from src.utils.logger import logger

if TYPE_CHECKING:
    from collections.abc import Mapping

    from src.community.loader import LoadedPlugin
    from src.utils.plugin_secrets import PluginSecretsResolver


T = TypeVar("T")


class PluginRegistry(Generic[T]):
    """One-instance-per-kind registry of community-loaded plugin classes.

    Subclasses set the :attr:`_kind` ClassVar (used in log + error messages)
    and override :meth:`_make_init_kwargs` to adapt the uniform
    ``**init_kwargs`` from :meth:`instantiate` into the constructor
    signature of their plugins.

    The class is intentionally not a singleton at the language level —
    instances are created at module import time and re-used. Tests can
    spin up additional instances for isolation.
    """

    #: Short label used in log lines and error messages (e.g. ``"validation"``).
    _kind: ClassVar[str] = "plugin"

    def __init__(self) -> None:
        self._classes: dict[str, type[T]] = {}
        self._manifests: dict[str, dict[str, Any]] = {}

    # ----- registration -----------------------------------------------------

    def register_from_community(self, loaded: LoadedPlugin) -> None:
        """Add a class loaded by :class:`CommunityCatalog` under its manifest id.

        Re-registering the same class under the same id is idempotent.
        Re-registering a *different* class under an existing id raises
        :class:`ValueError` — the catalog clears the registry before each
        load so this only fires when two community entries claim the same
        plugin id.
        """
        plugin_id = loaded.manifest.plugin.id
        existing = self._classes.get(plugin_id)
        if existing is not None and existing is not loaded.plugin_cls:
            raise ValueError(
                f"{self._kind} plugin id {plugin_id!r} is already registered by "
                f"{existing.__name__!r}; cannot re-register with "
                f"{loaded.plugin_cls.__name__!r}."
            )
        self._classes[plugin_id] = loaded.plugin_cls
        self._manifests[plugin_id] = loaded.manifest.ui_manifest()
        logger.debug(
            "[%s_REGISTRY] Registered plugin: %s",
            self._kind.upper(),
            plugin_id,
        )

    # ----- lookups ----------------------------------------------------------

    def get_class(self, plugin_id: str) -> type[T]:
        """Return the registered class for ``plugin_id`` (no instantiation)."""
        if plugin_id not in self._classes:
            available = sorted(self._classes.keys())
            raise KeyError(
                f"{self._kind} plugin {plugin_id!r} is not registered. "
                f"Available plugins: {available}. "
                "Ensure CommunityCatalog.ensure_loaded() was called."
            )
        return self._classes[plugin_id]

    def manifest(self, plugin_id: str) -> dict[str, Any]:
        """Return the UI manifest dict for ``plugin_id``."""
        if plugin_id not in self._manifests:
            available = sorted(self._manifests.keys())
            raise KeyError(
                f"{self._kind} plugin {plugin_id!r} is not registered. "
                f"Available plugins: {available}."
            )
        return dict(self._manifests[plugin_id])

    def list_ids(self) -> list[str]:
        return list(self._classes.keys())

    def list_manifests(self) -> list[dict[str, Any]]:
        return [dict(manifest) for manifest in self._manifests.values()]

    def is_registered(self, plugin_id: str) -> bool:
        return plugin_id in self._classes

    def get_all(self) -> dict[str, type[T]]:
        return dict(self._classes)

    def clear(self) -> None:
        self._classes.clear()
        self._manifests.clear()

    # ----- instantiation ----------------------------------------------------

    def instantiate(
        self,
        plugin_id: str,
        *,
        resolver: PluginSecretsResolver | None = None,
        env: Mapping[str, str] | None = None,
        **init_kwargs: Any,
    ) -> T:
        """Create a plugin instance with uniform secret + env injection.

        ``init_kwargs`` are passed to :meth:`_make_init_kwargs` for kind-
        specific normalisation (validation / evaluation expect
        ``params=…, thresholds=…``; reward only ``params=…``; reports
        nothing). Subclasses override that hook.

        ``resolver`` — if the plugin class declares ``_required_secrets``
        (mirrored from manifest ``[secrets].required``), this resolver is
        used to fetch the values and inject them as ``instance._secrets``.
        Passing ``resolver=None`` while the plugin requires secrets is a
        :class:`RuntimeError` — fail-fast at the call site that forgot to
        wire up the resolver.

        ``env`` — placeholder for the upcoming ``BasePlugin._env(name)``
        helper (B2 in ``cozy-booping-walrus.md``). Threaded through so
        callers can already pass it; the registry currently sets
        ``instance._injected_env`` to the mapping when provided so plugins
        added in subsequent PRs can read from it.
        """
        plugin_cls = self.get_class(plugin_id)
        normalised_kwargs = self._make_init_kwargs(init_kwargs)
        try:
            instance = plugin_cls(**normalised_kwargs)
        except TypeError as exc:
            # Surface a clearer error than the bare ``__init__`` traceback so
            # plugin authors immediately see which kwargs the registry tried
            # to pass — typical cause is a renamed param in the plugin class.
            raise TypeError(
                f"Failed to instantiate {self._kind} plugin {plugin_id!r} "
                f"with kwargs={list(normalised_kwargs.keys())}: {exc}"
            ) from exc

        self._inject_secrets(instance, plugin_cls, resolver, plugin_id)
        if env is not None:
            object.__setattr__(instance, "_injected_env", dict(env))

        return instance

    # ----- subclass hooks ---------------------------------------------------

    def _make_init_kwargs(self, init_kwargs: dict[str, Any]) -> dict[str, Any]:
        """Translate uniform ``**init_kwargs`` into plugin constructor kwargs.

        Default: pass through unchanged. Subclasses override to enforce
        kind-specific signatures (e.g. validation expects
        ``params, thresholds``; reports take nothing).
        """
        return dict(init_kwargs)

    # ----- internals --------------------------------------------------------

    def _inject_secrets(
        self,
        instance: T,
        plugin_cls: type[T],
        resolver: PluginSecretsResolver | None,
        plugin_id: str,
    ) -> None:
        """Resolve and attach ``instance._secrets`` if the class declares any.

        Centralised here so all four kinds share identical injection
        semantics. Currently driven by the legacy ``_required_secrets``
        ClassVar (mirrored from TOML by the loader). When the migration
        to ``[[required_env]]`` lands (PR6 / C1), this method evolves to
        consult ``REQUIRED_ENV`` for entries with ``secret=True``.
        """
        keys: tuple[str, ...] = getattr(plugin_cls, "_required_secrets", ())
        if not keys:
            return
        if resolver is None:
            raise RuntimeError(
                f"{self._kind} plugin {plugin_id!r} requires secrets "
                f"{list(keys)} but no PluginSecretsResolver was passed to "
                "registry.instantiate(). Wire up the per-kind resolver at "
                "the call site."
            )
        resolved = resolver.resolve(keys)
        object.__setattr__(instance, "_secrets", resolved)
        logger.debug(
            "[%s_REGISTRY] Injected %d secret(s) for plugin %r",
            self._kind.upper(),
            len(resolved),
            plugin_id,
        )


__all__ = ["PluginRegistry"]
