"""Manifest-driven engine registry.

Single source of truth for engine discovery + resolution. Walks
``packages/engines/src/ryotenkai_engines/<id>/engine.toml`` files,
validates each through :class:`EngineManifest`, and exposes typed
access to manifests / runtime classes / config classes.

Mirrors :class:`ryotenkai_providers.registry.ProviderRegistry`:

  * **Defensive loading** — one bad manifest does not topple the whole
    registry. Failures collected as :class:`LoadFailure` and surfaced
    via :meth:`failures` for ``ryotenkai status`` / admin endpoints.

  * **Lazy class resolution** — entry-point classes resolved via
    ``importlib.import_module`` only when ``get_runtime`` /
    ``get_config_class`` is called. Keeps heavy engine SDKs (vLLM
    pulls torch + transformers) out of the import graph for callers
    that don't need them.

  * **Lock-protected singleton** — ``get_registry()`` returns the
    same instance across threads. ``from_filesystem`` may be called
    concurrently; the second caller sees the populated cache.

The registry is consumed by ``ryotenkai_providers`` (for engine
selection in inference providers) and ``ryotenkai_shared.config``
(for the discriminated-union builder). It does not import either —
the engines package is a leaf.
"""

from __future__ import annotations

import importlib
import threading
import tomllib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from ryotenkai_engines.errors import EngineNotRegistered, EngineRegistryError
from ryotenkai_engines.images import resolve_image
from ryotenkai_engines.interfaces import BaseEngineConfig, IInferenceEngine
from ryotenkai_engines.manifest import EngineManifest

# ---------------------------------------------------------------------------
# Public types
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class LoadFailure:
    """One ``engine.toml`` that failed to load.

    Kept on the registry so admin surfaces can list broken installs
    without crashing the rest of the system. Mirrors
    :class:`ryotenkai_providers.registry.LoadFailure`.
    """

    #: Engine id parsed from the directory name (best-effort — may be empty
    #: if the manifest itself was unparseable and the id couldn't be read).
    engine_id: str

    #: Absolute path to the offending ``engine.toml``.
    manifest_path: Path

    #: Human-readable failure reason — first line is the headline.
    reason: str

    #: Original exception type name.
    exc_type: str


# ---------------------------------------------------------------------------
# Discovery roots & filters
# ---------------------------------------------------------------------------


#: Default discovery root — every immediate child folder of
#: ``packages/engines/src/ryotenkai_engines/`` is searched for an
#: ``engine.toml``. Tests pass ``roots=[tmp_path]`` for synthetic engines.
_DEFAULT_ENGINES_ROOT = Path(__file__).resolve().parent

#: Folder names that obviously aren't engines (the engines/ dir hosts the
#: registry/manifest/etc. modules and won't ship engine.toml files).
_NON_ENGINE_DIRS = frozenset({
    "scripts",
    "tests",
    "__pycache__",
})


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


class EngineRegistry:
    """Catalogue of engines loaded from on-disk manifests.

    Construction:
        Use :meth:`from_filesystem` for the standard auto-discovery path.
        Tests can pass ``roots=[tmp_path]`` to load synthetic manifests.
        Module-level :func:`get_registry` returns the lazy singleton.

    Thread-safety:
        Construction is lock-protected. Read-only access (``list``,
        ``get_manifest``, ``get_runtime``, ``get_config_class``) is
        lock-free and safe.
    """

    def __init__(
        self,
        manifests: Mapping[str, EngineManifest],
        failures: Sequence[LoadFailure] = (),
    ) -> None:
        self._manifests: dict[str, EngineManifest] = dict(manifests)
        self._failures: list[LoadFailure] = list(failures)
        # Class-resolution cache — populated lazily on first access.
        self._cls_cache: dict[tuple[str, str], type[Any]] = {}
        self._lock = threading.Lock()

    # ----- discovery ------------------------------------------------------

    @classmethod
    def from_filesystem(
        cls,
        *,
        roots: Sequence[Path] | None = None,
        strict: bool = False,
    ) -> EngineRegistry:
        """Walk ``roots`` and load every ``engine.toml``.

        Args:
            roots: Search roots. Defaults to ``[<engines package>]``
                — the in-tree shipped engines. Test code passes
                ``[tmp_path]`` for synthetic manifests.
            strict: When True, raise on the first :class:`LoadFailure`.
                Used by CI / pre-commit. When False (default), collect
                failures; production must keep going so a broken extra
                engine doesn't crash the CLI.
        """
        roots = list(roots) if roots is not None else [_DEFAULT_ENGINES_ROOT]
        manifests: dict[str, EngineManifest] = {}
        failures: list[LoadFailure] = []

        for root in roots:
            if not root.is_dir():
                continue
            for child in sorted(root.iterdir()):
                if not child.is_dir():
                    continue
                if child.name in _NON_ENGINE_DIRS or child.name.startswith((".", "_")):
                    continue
                manifest_path = child / "engine.toml"
                if not manifest_path.is_file():
                    continue
                result = cls._load_one(manifest_path, strict=strict)
                if isinstance(result, LoadFailure):
                    failures.append(result)
                    continue

                manifest: EngineManifest = result

                # Folder name MUST equal manifest.engine.id.
                if child.name != manifest.engine.id:
                    failure = LoadFailure(
                        engine_id=manifest.engine.id,
                        manifest_path=manifest_path,
                        reason=(
                            f"folder name {child.name!r} does not match "
                            f"engine.id {manifest.engine.id!r}"
                        ),
                        exc_type="EngineIdFolderMismatch",
                    )
                    failures.append(failure)
                    if strict:
                        raise EngineRegistryError(
                            failure.reason,
                            code="ENGINE_ID_FOLDER_MISMATCH",
                            details={"manifest_path": str(manifest_path)},
                        )
                    continue

                if manifest.engine.id in manifests:
                    failure = LoadFailure(
                        engine_id=manifest.engine.id,
                        manifest_path=manifest_path,
                        reason=(
                            f"duplicate engine.id {manifest.engine.id!r} — "
                            f"already loaded earlier"
                        ),
                        exc_type="DuplicateEngineId",
                    )
                    failures.append(failure)
                    if strict:
                        raise EngineRegistryError(
                            failure.reason,
                            code="ENGINE_DUPLICATE",
                            details={"manifest_path": str(manifest_path)},
                        )
                    continue

                manifests[manifest.engine.id] = manifest

        return cls(manifests, failures)

    @staticmethod
    def _load_one(
        manifest_path: Path,
        *,
        strict: bool,
    ) -> EngineManifest | LoadFailure:
        """Read + validate one manifest. Returns either the parsed model or
        a :class:`LoadFailure` describing the problem."""
        engine_id_guess = manifest_path.parent.name
        try:
            data = tomllib.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, tomllib.TOMLDecodeError) as exc:
            failure = LoadFailure(
                engine_id=engine_id_guess,
                manifest_path=manifest_path,
                reason=f"could not read/parse TOML: {exc}",
                exc_type=type(exc).__name__,
            )
            if strict:
                raise EngineRegistryError(
                    failure.reason,
                    code="ENGINE_MANIFEST_PARSE",
                    details={"manifest_path": str(manifest_path)},
                ) from exc
            return failure
        try:
            manifest = EngineManifest.model_validate(data)
        except ValidationError as exc:
            failure = LoadFailure(
                engine_id=engine_id_guess,
                manifest_path=manifest_path,
                reason=f"manifest validation failed:\n{exc}",
                exc_type="ValidationError",
            )
            if strict:
                raise EngineRegistryError(
                    failure.reason,
                    code="ENGINE_MANIFEST_INVALID",
                    details={"manifest_path": str(manifest_path)},
                ) from exc
            return failure
        return manifest

    # ----- read API -------------------------------------------------------

    def list(self) -> tuple[str, ...]:
        """Engine ids, sorted."""
        return tuple(sorted(self._manifests))

    def get_manifest(self, engine_id: str) -> EngineManifest:
        """Get the parsed manifest for ``engine_id``.

        Raises:
            EngineNotRegistered: ``engine_id`` is not known to this registry.
        """
        try:
            return self._manifests[engine_id]
        except KeyError:
            raise EngineNotRegistered(engine_id, known=self.list()) from None

    def failures(self) -> tuple[LoadFailure, ...]:
        """Manifests that didn't load."""
        return tuple(self._failures)

    # ----- class resolution ----------------------------------------------

    def get_runtime(self, engine_id: str) -> type[IInferenceEngine]:
        """Resolve and import the runtime class. Cached.

        Raises:
            EngineNotRegistered: ``engine_id`` is unknown.
            EngineRegistryError: import fails or class isn't an
                ``IInferenceEngine``.
        """
        cls = self._resolve_class(engine_id, role_key="runtime")
        # Runtime-checkable Protocol verification — catches manifests that
        # point at the wrong class entirely.
        if not isinstance(cls, type) or not _is_inference_engine_class(cls):
            raise EngineRegistryError(
                message=(
                    f"engine {engine_id!r} entry_points.runtime points at "
                    f"{cls!r}, which does not satisfy IInferenceEngine."
                ),
                code="ENGINE_RUNTIME_INVALID",
                details={"engine_id": engine_id, "resolved": repr(cls)},
            )
        # Engine_id sanity: drift detector check at runtime — class.engine_id
        # ClassVar must equal manifest.engine.id.
        manifest = self._manifests[engine_id]
        if getattr(cls, "engine_id", None) != manifest.engine.id:
            raise EngineRegistryError(
                message=(
                    f"engine {engine_id!r}: runtime class {cls.__name__} "
                    f"declares engine_id={getattr(cls, 'engine_id', None)!r}, "
                    f"but manifest.engine.id is {manifest.engine.id!r}."
                ),
                code="ENGINE_RUNTIME_ID_DRIFT",
                details={"engine_id": engine_id},
            )
        return cls  # type: ignore[return-value]

    def get_config_class(self, engine_id: str) -> type[BaseEngineConfig]:
        """Resolve and import the config class. Cached.

        Raises:
            EngineNotRegistered: ``engine_id`` is unknown.
            EngineRegistryError: import fails or class isn't a
                ``BaseEngineConfig`` subclass.
        """
        cls = self._resolve_class(engine_id, role_key="config_schema")
        if not (isinstance(cls, type) and issubclass(cls, BaseEngineConfig)):
            raise EngineRegistryError(
                message=(
                    f"engine {engine_id!r} entry_points.config_schema points "
                    f"at {cls!r}, which is not a BaseEngineConfig subclass."
                ),
                code="ENGINE_CONFIG_SCHEMA_INVALID",
                details={"engine_id": engine_id, "resolved": repr(cls)},
            )
        # Discriminator parity: config class's ``kind`` Literal must equal
        # the engine id. Catches drift where author renamed the engine in
        # the manifest but forgot the config class.
        manifest = self._manifests[engine_id]
        kind_literal = _extract_kind_literal(cls)
        if kind_literal is not None and kind_literal != manifest.engine.id:
            raise EngineRegistryError(
                message=(
                    f"engine {engine_id!r}: config class {cls.__name__} has "
                    f"kind={kind_literal!r}, but manifest.engine.id is "
                    f"{manifest.engine.id!r}. The Literal['kind'] field "
                    f"MUST match the engine id."
                ),
                code="ENGINE_CONFIG_KIND_DRIFT",
                details={"engine_id": engine_id},
            )
        return cls

    def _resolve_class(self, engine_id: str, *, role_key: str) -> type[Any]:
        """Lazy class resolution with cache. Raises on import failure."""
        if engine_id not in self._manifests:
            raise EngineNotRegistered(engine_id, known=self.list())
        cache_key = (engine_id, role_key)
        # Read without lock — dict get is atomic; misses re-acquire.
        if (cached := self._cls_cache.get(cache_key)) is not None:
            return cached

        manifest = self._manifests[engine_id]
        if role_key == "runtime":
            ep = manifest.entry_points.runtime
        elif role_key == "config_schema":
            ep = manifest.entry_points.config_schema
        else:
            raise EngineRegistryError(
                message=f"unknown entry-point role_key={role_key!r}",
                code="ENGINE_REGISTRY_BAD_ROLE",
                details={"role_key": role_key},
            )

        try:
            module = importlib.import_module(ep.module)
            cls = getattr(module, ep.class_name)
        except (ImportError, AttributeError) as exc:
            raise EngineRegistryError(
                message=(
                    f"engine {engine_id!r} entry_points.{role_key} locator "
                    f"{ep.module}:{ep.class_name} could not be resolved: {exc}"
                ),
                code="ENGINE_LOCATOR_RESOLVE_FAILED",
                details={"engine_id": engine_id, "role_key": role_key},
            ) from exc

        with self._lock:
            self._cls_cache[cache_key] = cls
        return cls

    # ----- image resolution ----------------------------------------------

    def get_image(
        self,
        engine_id: str,
        *,
        provider_overrides: Mapping[str, Any] | None = None,
        env: Mapping[str, str] | None = None,
    ) -> str:
        """Resolve image name via the override chain (see :mod:`.images`)."""
        manifest = self.get_manifest(engine_id)
        return resolve_image(
            engine_id=engine_id,
            manifest=manifest,
            provider_overrides=provider_overrides,
            env=env,
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _is_inference_engine_class(cls: type[Any]) -> bool:
    """Best-effort runtime check that ``cls`` satisfies ``IInferenceEngine``.

    Pydantic's runtime-checkable Protocol has a known limitation: it only
    inspects instance methods, not ClassVars. We add an explicit check for
    ``engine_id`` and ``config_class`` ClassVars on top of the standard
    ``isinstance`` (which IInferenceEngine being runtime-checkable allows).

    We instantiate a no-arg class to check compliance — engines must have
    a no-arg ``__init__`` (the registry calls ``runtime_cls()`` directly).
    Failures here cause the caller to surface a clean ``ENGINE_RUNTIME_INVALID``.
    """
    if not hasattr(cls, "engine_id"):
        return False
    if not hasattr(cls, "config_class"):
        return False
    # Attempt instantiation; engines are zero-arg constructible.
    try:
        instance = cls()
    except Exception:  # noqa: BLE001 — best-effort, catch anything
        return False
    return isinstance(instance, IInferenceEngine)


def _extract_kind_literal(cls: type[BaseEngineConfig]) -> str | None:
    """Pull out the ``Literal[…]`` value from a config class's ``kind`` field.

    Returns the literal string if exactly one ``Literal[…]`` is found,
    otherwise ``None`` (which the caller treats as "skip drift check").
    """
    field = cls.model_fields.get("kind")
    if field is None:
        return None
    # Pydantic stores Literal as ``Literal["foo"]`` whose ``__args__`` is
    # ``("foo",)``. We inspect via the field's annotation.
    annotation = field.annotation
    args = getattr(annotation, "__args__", None)
    if args is None or len(args) != 1:
        return None
    value = args[0]
    return value if isinstance(value, str) else None


# ---------------------------------------------------------------------------
# Lock-protected singleton
# ---------------------------------------------------------------------------


_singleton_lock = threading.Lock()
_singleton_registry: EngineRegistry | None = None


def get_registry() -> EngineRegistry:
    """Return the lock-protected lazy singleton.

    Production code uses this to avoid re-walking the filesystem on every
    call. Tests should construct via ``EngineRegistry.from_filesystem(roots=[…])``
    directly to keep state isolated, OR call :func:`reset_registry` between
    tests.
    """
    global _singleton_registry
    if _singleton_registry is not None:
        return _singleton_registry
    with _singleton_lock:
        if _singleton_registry is None:
            _singleton_registry = EngineRegistry.from_filesystem()
        return _singleton_registry


def reset_registry() -> None:
    """Clear the singleton — used by test fixtures.

    Not part of the public API; kept module-level for the conftest fixture
    that wraps every test in a fresh registry.
    """
    global _singleton_registry
    with _singleton_lock:
        _singleton_registry = None


__all__ = (
    "LoadFailure",
    "EngineRegistry",
    "get_registry",
    "reset_registry",
)
