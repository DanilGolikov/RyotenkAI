"""Manifest-driven provider registry.

Single source of truth for provider discovery + construction. Replaces
the dual ``GPUProviderFactory`` (training, registry-pattern) + ``InferenceProviderFactory``
(inference, if/elif chain) approach with one declarative registry whose
contents are derived entirely from ``provider.toml`` manifests
co-located with each provider package.

Design (concurrent-gathering-hippo plan §5):

* **Discovery** — :meth:`ProviderRegistry.from_filesystem` walks
  ``packages/providers/src/ryotenkai_providers/<id>/provider.toml``;
  validates each through :class:`ProviderManifest`; collects successes
  and per-file :class:`LoadFailure`. One bad manifest does not topple
  the whole registry — defensive loading mirrors
  :mod:`ryotenkai_community.loader` and the Python.org best-practices
  thread (each plugin in its own try/except).

* **Lazy class resolution** — entry-point classes are resolved via
  :func:`importlib.import_module` only when the corresponding role is
  requested. Single-node-only deployments never import the RunPod SDK;
  the pod sentinel ``test_pod_does_not_import_control_or_providers``
  stays green.

* **Single construction signature** — every provider class takes
  ``__init__(self, ctx: ProviderContext)``. The registry pre-builds
  the context from the pipeline config + secrets and hands it to
  whichever class it instantiates. Eliminates the legacy
  ``(config: dict, secrets)`` vs ``(*, config: PipelineConfig, secrets)``
  signature mismatch.

* **Capability surface backed by manifest** — the registry attaches
  ``_manifest_provider_id`` / ``_manifest_provider_name`` /
  ``_manifest_provider_type`` / ``_manifest_capabilities`` ClassVars
  to every successfully resolved class. ``ProviderBase`` reads these
  in its default property impls. Hand-overrides surface as a drift
  failure in :func:`check_manifests` (PR-2).

* **Result-based API** — every ``create_*`` returns
  :class:`Result[Provider, ProviderError]` instead of raising on
  role mismatch / unknown provider. Caller composes; the resume
  service skips cleanly when a provider doesn't declare a resume
  factory.

This module is consumed by ``ryotenkai_control`` and ``ryotenkai_pod``
but does not import either — the registry lives in the providers
package per the importlinter contract; control / pod call into it
through the existing dependency direction (control → providers,
pod ← providers via importlib at runtime).
"""

from __future__ import annotations

import importlib
import threading
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, ValidationError

from ryotenkai_providers.manifest import (
    LATEST_PROVIDER_SCHEMA_VERSION,
    ProviderManifest,
    ProviderRole,
)
from ryotenkai_shared.utils.logger import logger
from ryotenkai_shared.utils.result import (
    Err,
    Ok,
    ProviderError,
    Result,
)

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from ryotenkai_providers.inference.interfaces import IInferenceProvider
    from ryotenkai_providers.training.interfaces import (
        ICapacityErrorClassifier,
        IGPUProvider,
        IRecoveryProbeProvider,
        ITerminalActionProvider,
        ProviderCapabilities,
    )
    from ryotenkai_shared.config import PipelineConfig, Secrets
    from ryotenkai_shared.infrastructure.lifecycle import IPodLifecycleClient


# ---------------------------------------------------------------------------
# Public types
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ProviderContext:
    """Construction context handed to every provider class.

    Replaces the dual ``(config: dict, secrets)`` vs
    ``(*, config: PipelineConfig, secrets)`` signatures that the legacy
    factories required. Every provider class — training, inference, future
    — accepts a single :class:`ProviderContext`. The registry builds it
    once from the pipeline config + secrets and hands it to whichever
    class it instantiates.

    Frozen + ``slots=True``: the context is read-only. Adding a field is
    a deliberate API extension that requires PR review (the field set is
    documented as the only way the registry hands data to providers, see
    risk R12 in the plan doc).
    """

    #: Canonical id of the provider being constructed (e.g. ``"runpod"``).
    #: Equals ``manifest.provider.id``; the registry passes it explicitly
    #: so providers don't have to grep through ``pipeline_config.providers``.
    provider_id: str

    #: The full :class:`PipelineConfig`. Inference impls need
    #: ``config.inference.engines.vllm.*``; training impls typically use
    #: only ``provider_block`` but get the full config for forward-compat.
    pipeline_config: PipelineConfig

    #: Typed Pydantic-validated provider block from
    #: ``config.providers[provider_id]``. After PR-1.5 + PR-1.9 this is a
    #: ``RunPodProviderConfig`` / ``SingleNodeProviderConfig`` instance with
    #: full type-checking; until then it accepts a raw mapping for the
    #: transitional period (see PR-1.6 / PR-1.7 callers).
    provider_block: Any

    #: Secrets bundle from :func:`ryotenkai_shared.config.secrets.load_secrets`.
    secrets: Secrets


@dataclass(frozen=True, slots=True)
class LoadFailure:
    """One manifest that failed to load. Kept on the registry so admin
    surfaces (`/api/providers`) can list broken installs without crashing
    the rest of the system.

    Mirrors :class:`ryotenkai_community.loader.LoadFailure` for operator-UX
    consistency.
    """

    #: Provider id parsed from the directory name (best-effort — may be
    #: empty when the manifest itself was unparseable and we couldn't
    #: even read the id).
    provider_id: str

    #: Absolute path to the offending ``provider.toml``.
    manifest_path: Path

    #: Human-readable failure reason; first line is the headline, rest
    #: is detail. Surfaced in CLI / API.
    reason: str

    #: Original exception type name (``"ValidationError"`` /
    #: ``"OSError"`` / ``"DuplicateProviderId"``…).
    exc_type: str


class ProviderRegistryError(Exception):
    """Raised when the registry itself can't satisfy a request.

    Distinct from :class:`ProviderError` (which is a dataclass returned
    in :class:`Result.Err` when a provider's backend rejects a call).
    Registry errors are configuration / plumbing issues — caller can't
    recover by retrying, only by fixing the manifest or registering the
    missing provider. They're raised because the registry's contract is
    "you asked for something I have / can do" — invariant breach is a
    bug, not an expected outcome.

    Carries the same ``code`` / ``details`` shape as :class:`AppError`
    so HTTP / CLI error renderers stay uniform.
    """

    def __init__(
        self,
        message: str,
        *,
        code: str = "PROVIDER_REGISTRY_ERROR",
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.code = code
        self.details: dict[str, Any] = dict(details) if details else {}


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


#: Default discovery root — every immediate child folder of
#: ``packages/providers/src/ryotenkai_providers/`` is searched for a
#: ``provider.toml``. Tests pass ``roots=[tmpdir]`` for fabricated cases.
_DEFAULT_PROVIDERS_ROOT = Path(__file__).resolve().parent

#: Folder names that obviously aren't providers (the inference / training
#: namespace dirs hold shared interfaces, not concrete providers).
_NON_PROVIDER_DIRS = frozenset({
    "training",
    "inference",
    "scripts",
    "tests",
    "__pycache__",
})


class ProviderRegistry:
    """Catalogue of providers loaded from on-disk manifests.

    Construction:
        Use :meth:`from_filesystem` for the standard auto-discovery path.
        Tests can pass ``roots=[tmpdir]`` to load fabricated manifests.
        Module-level :func:`get_registry` returns the lazy singleton for
        production code.

    Thread-safety:
        Construction is lock-protected; ``from_filesystem`` may be called
        concurrently and the second caller will return the same registry
        without re-walking. Read-only access (``list``, ``capabilities``,
        ``create_*``) is lock-free and safe.
    """

    def __init__(
        self,
        manifests: Mapping[str, ProviderManifest],
        failures: Sequence[LoadFailure] = (),
    ) -> None:
        self._manifests: dict[str, ProviderManifest] = dict(manifests)
        self._failures: list[LoadFailure] = list(failures)
        # Class-resolution cache — populated lazily on first ``create_*``
        # / ``get_config_class`` call. Keeps RunPod's heavy SDK out of
        # the import graph for single_node-only deployments.
        self._cls_cache: dict[tuple[str, str], type[Any]] = {}
        self._lock = threading.Lock()

    # ----- discovery ------------------------------------------------------

    @classmethod
    def from_filesystem(
        cls,
        *,
        roots: Sequence[Path] | None = None,
        strict: bool = False,
    ) -> ProviderRegistry:
        """Walk ``roots`` and load every ``provider.toml``.

        Args:
            roots: Search roots. Defaults to ``[<providers package>]``
                — the in-tree providers. Test code passes ``[tmpdir]``
                for fabricated manifests.
            strict: When True, raise on the first :class:`LoadFailure`
                (used by CI / pre-commit). When False (default), collect
                failures and surface them via :meth:`failures` —
                production must keep going so a broken manifest in
                ``./extra_providers`` doesn't crash the CLI.
        """
        roots = list(roots) if roots is not None else [_DEFAULT_PROVIDERS_ROOT]
        manifests: dict[str, ProviderManifest] = {}
        failures: list[LoadFailure] = []
        for root in roots:
            if not root.is_dir():
                logger.debug("[PROVIDER_REGISTRY] discovery root missing: %s", root)
                continue
            for child in sorted(root.iterdir()):
                if not child.is_dir():
                    continue
                if child.name in _NON_PROVIDER_DIRS or child.name.startswith("."):
                    continue
                manifest_path = child / "provider.toml"
                if not manifest_path.is_file():
                    continue
                result = cls._load_one(manifest_path, strict=strict)
                if isinstance(result, LoadFailure):
                    failures.append(result)
                    if strict:
                        # _load_one already raised in strict mode; this
                        # branch is defensive — should be unreachable.
                        break
                    continue
                manifest = result
                if manifest.provider.id in manifests:
                    failures.append(
                        LoadFailure(
                            provider_id=manifest.provider.id,
                            manifest_path=manifest_path,
                            reason=(
                                f"duplicate provider.id {manifest.provider.id!r} — "
                                f"already loaded from "
                                f"{manifests[manifest.provider.id]}"
                            ),
                            exc_type="DuplicateProviderId",
                        )
                    )
                    if strict:
                        raise ProviderRegistryError(
                            message=(
                                f"duplicate provider.id {manifest.provider.id!r} "
                                f"in {manifest_path}"
                            ),
                            code="PROVIDER_DUPLICATE",
                            details={"manifest_path": str(manifest_path)},
                        )
                    continue
                manifests[manifest.provider.id] = manifest
        if manifests:
            logger.info(
                "[PROVIDER_REGISTRY] loaded %d provider(s): %s",
                len(manifests),
                ", ".join(sorted(manifests)),
            )
        if failures:
            logger.warning(
                "[PROVIDER_REGISTRY] %d manifest(s) failed to load",
                len(failures),
            )
        return cls(manifests, failures)

    @staticmethod
    def _load_one(
        manifest_path: Path,
        *,
        strict: bool,
    ) -> ProviderManifest | LoadFailure:
        """Read+validate one manifest. Returns either the parsed model or a
        :class:`LoadFailure` describing the problem."""
        provider_id_guess = manifest_path.parent.name
        try:
            data = tomllib.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, tomllib.TOMLDecodeError) as exc:
            failure = LoadFailure(
                provider_id=provider_id_guess,
                manifest_path=manifest_path,
                reason=f"could not read/parse TOML: {exc}",
                exc_type=type(exc).__name__,
            )
            if strict:
                raise ProviderRegistryError(
                    message=failure.reason,
                    code="PROVIDER_MANIFEST_PARSE",
                    details={"manifest_path": str(manifest_path)},
                ) from exc
            return failure
        try:
            manifest = ProviderManifest.model_validate(data)
        except ValidationError as exc:
            failure = LoadFailure(
                provider_id=provider_id_guess,
                manifest_path=manifest_path,
                reason=f"manifest schema validation failed: {exc}",
                exc_type="ValidationError",
            )
            if strict:
                raise ProviderRegistryError(
                    message=failure.reason,
                    code="PROVIDER_MANIFEST_INVALID",
                    details={"manifest_path": str(manifest_path)},
                ) from exc
            return failure
        # provider.id MUST equal the folder name (the audit's
        # "provider_id consistency" invariant — empty string in
        # _manifest_provider_id is dangerous later).
        if manifest.provider.id != provider_id_guess:
            failure = LoadFailure(
                provider_id=provider_id_guess,
                manifest_path=manifest_path,
                reason=(
                    f"provider.id {manifest.provider.id!r} does not match "
                    f"folder name {provider_id_guess!r}. The two MUST match — "
                    f"the discovery walk uses the folder name as the canonical "
                    f"id; a mismatch would silently load the manifest under "
                    f"the wrong key."
                ),
                exc_type="ProviderIdFolderMismatch",
            )
            if strict:
                raise ProviderRegistryError(
                    message=failure.reason,
                    code="PROVIDER_ID_FOLDER_MISMATCH",
                    details={"manifest_path": str(manifest_path)},
                )
            return failure
        return manifest

    # ----- introspection (no class resolution required) -------------------

    def list(
        self, role: ProviderRole | None = None
    ) -> tuple[str, ...]:
        """Sorted ids of every loaded provider.

        Args:
            role: Optionally filter to providers that declare the role.
        """
        ids = sorted(self._manifests)
        if role is None:
            return tuple(ids)
        return tuple(pid for pid in ids if role in self._manifests[pid].provider.roles)

    def has_role(self, provider_id: str, role: ProviderRole) -> bool:
        """Quick predicate: does ``provider_id`` declare ``role``?"""
        manifest = self._manifests.get(provider_id)
        if manifest is None:
            return False
        return role in manifest.provider.roles

    def get_manifest(self, provider_id: str) -> ProviderManifest:
        """Return the parsed manifest. Raises ``KeyError`` for unknown id."""
        try:
            return self._manifests[provider_id]
        except KeyError as exc:
            raise KeyError(
                f"provider {provider_id!r} is not registered. "
                f"Known: {self.list()!r}"
            ) from exc

    def capabilities(self, provider_id: str) -> ProviderCapabilities:
        """Return :class:`ProviderCapabilities` derived from the manifest.

        No class resolution required — pure read of ``[capabilities]``
        block. Used by control-plane code that gates on capability flags
        (``caps.is_local``, ``caps.supports_log_download``) before any
        provider is constructed.
        """
        # Lazy import: avoids circular dependency through training.interfaces.
        from ryotenkai_providers.training.interfaces import ProviderCapabilities, VolumeKind

        manifest = self.get_manifest(provider_id)
        caps_spec = manifest.capabilities
        return ProviderCapabilities(
            provider_type=caps_spec.provider_type,
            is_local=caps_spec.is_local,
            supports_multi_gpu=caps_spec.supports_multi_gpu,
            supports_spot_instances=caps_spec.supports_spot_instances,
            supports_lifecycle_actions=caps_spec.supports_lifecycle_actions,
            has_pause_resume=caps_spec.has_pause_resume,
            supports_recovery_probe=caps_spec.supports_recovery_probe,
            supports_capacity_error_detection=caps_spec.supports_capacity_error_detection,
            supports_log_download=caps_spec.supports_log_download,
            volume_kind=VolumeKind(caps_spec.volume_kind),
            runner_workspace_root=caps_spec.runner_workspace_root,
            max_runtime_hours=caps_spec.max_runtime_hours,
        )

    def required_secrets(
        self,
        provider_id: str,
        *,
        role: ProviderRole | None = None,
    ) -> tuple[str, ...]:
        """Pre-factory secret resolution.

        Replaces ``_resolve_required_secrets_for_provider`` in
        ``startup_validator.py`` (deleted in PR-1.10/1.11). The startup
        validator iterates this tuple and checks each name against the
        ``Secrets`` bundle — missing ⇒ fail-fast at startup.
        """
        return self.get_manifest(provider_id).required_secret_names(role=role)

    def failures(self) -> tuple[LoadFailure, ...]:
        """Manifests that didn't load. Surfaced via ``ryotenkai status``."""
        return tuple(self._failures)

    # ----- config schema -------------------------------------------------

    def get_config_class(self, provider_id: str) -> type[BaseModel]:
        """Pydantic model class for the ``providers.<id>`` YAML block.

        ``PipelineConfig`` validators call ``ConfigCls.model_validate(block)``
        at config load — so YAML typos surface BEFORE any pipeline stage
        runs. Replaces the ``dict[str, Any]`` access pattern from the
        legacy ``PipelineProviderMixin.get_provider_config``.
        """
        cls = self._resolve_class(provider_id, role_key="config_schema")
        if not (isinstance(cls, type) and issubclass(cls, BaseModel)):
            raise ProviderRegistryError(
                message=(
                    f"provider {provider_id!r} entry_points.config_schema points "
                    f"at {cls!r}, which is not a pydantic.BaseModel subclass."
                ),
                code="PROVIDER_CONFIG_SCHEMA_INVALID",
                details={"provider_id": provider_id, "resolved": repr(cls)},
            )
        return cls

    def get_config_json_schema(self, provider_id: str) -> dict[str, Any]:
        """JSON Schema view of the provider's config block.

        Consumed by Web UI for dynamic form rendering — the frontend
        ``GET /api/providers/<id>/config-schema`` returns this dict.
        Replaces the duplicated TypeScript field declarations in
        ``ConfigBuilder/FieldRenderer.tsx``.
        """
        return self.get_config_class(provider_id).model_json_schema()

    # ----- construction --------------------------------------------------

    def create_training(
        self, provider_id: str, ctx: ProviderContext
    ) -> Result[IGPUProvider, ProviderError]:
        """Instantiate the training-side provider.

        Returns ``Err(PROVIDER_ROLE_MISMATCH)`` if the manifest doesn't
        declare ``training``, ``Err(PROVIDER_NOT_REGISTERED)`` for unknown
        ids — never raises for these expected cases. Caller composes
        cleanly (``.unwrap()`` / ``.unwrap_or_else(...)``).
        """
        return self._create_role(provider_id, ctx, role="training")

    def create_inference(
        self, provider_id: str, ctx: ProviderContext
    ) -> Result[IInferenceProvider, ProviderError]:
        """Instantiate the inference-side provider. See :meth:`create_training`."""
        return self._create_role(provider_id, ctx, role="inference")

    def _create_role(
        self,
        provider_id: str,
        ctx: ProviderContext,
        *,
        role: ProviderRole,
    ) -> Result[Any, ProviderError]:
        """Shared body of create_training / create_inference."""
        if provider_id not in self._manifests:
            return Err(
                ProviderError(
                    message=(
                        f"provider {provider_id!r} is not registered. "
                        f"Known: {self.list()!r}."
                    ),
                    code="PROVIDER_NOT_REGISTERED",
                    details={"provider_id": provider_id, "known": list(self.list())},
                )
            )
        manifest = self._manifests[provider_id]
        if role not in manifest.provider.roles:
            return Err(
                ProviderError(
                    message=(
                        f"provider {provider_id!r} does not declare role "
                        f"{role!r}. Declared roles: "
                        f"{list(manifest.provider.roles)!r}."
                    ),
                    code="PROVIDER_ROLE_MISMATCH",
                    details={
                        "provider_id": provider_id,
                        "requested_role": role,
                        "declared_roles": list(manifest.provider.roles),
                    },
                )
            )
        try:
            cls = self._resolve_class(provider_id, role_key=role)
        except ProviderRegistryError as exc:
            return Err(
                ProviderError(
                    message=str(exc),
                    code=exc.code or "PROVIDER_LOCATOR_RESOLVE_FAILED",
                    details=exc.details or {"provider_id": provider_id},
                )
            )
        # Construct — provider classes accept ProviderContext per the
        # unified signature contract.
        try:
            instance = cls(ctx)
        except Exception as exc:
            return Err(
                ProviderError(
                    message=(
                        f"provider {provider_id!r} {role} class "
                        f"{cls.__name__} raised during construction: {exc}"
                    ),
                    code="PROVIDER_CONSTRUCTION_FAILED",
                    details={
                        "provider_id": provider_id,
                        "role": role,
                        "exc_type": type(exc).__name__,
                    },
                )
            )
        return Ok(instance)

    def create_resume_provider(
        self,
        provider_id: str,
        *,
        api_key: str | None = None,
    ) -> Result[ITerminalActionProvider, ProviderError]:
        """Cheap construction for the resume path.

        The resume service needs only the lifecycle methods on a provider
        (``terminate`` / ``pause`` / ``resume``); spinning up the full
        Pydantic-validated training provider is expensive (it parses the
        whole ``providers.<id>`` config block). The provider declares an
        optional ``[entry_points.resume_factory]`` classmethod that takes
        only the credentials it needs — the registry calls it directly,
        no :class:`ProviderContext` involved.

        Returns ``Err(PROVIDER_RESUME_UNAVAILABLE)`` if the manifest
        doesn't declare ``resume_factory`` — clean skip for callers.
        """
        if provider_id not in self._manifests:
            return Err(
                ProviderError(
                    message=f"provider {provider_id!r} is not registered.",
                    code="PROVIDER_NOT_REGISTERED",
                    details={"provider_id": provider_id},
                )
            )
        manifest = self._manifests[provider_id]
        rf = manifest.entry_points.resume_factory
        if rf is None:
            return Err(
                ProviderError(
                    message=(
                        f"provider {provider_id!r} does not declare "
                        f"entry_points.resume_factory — resume bypass "
                        f"is unavailable."
                    ),
                    code="PROVIDER_RESUME_UNAVAILABLE",
                    details={"provider_id": provider_id},
                )
            )
        try:
            module = importlib.import_module(rf.module)
            cls_name, _, method_name = rf.classmethod.partition(".")
            owner_cls = getattr(module, cls_name)
            factory = getattr(owner_cls, method_name)
            instance = factory(api_key=api_key) if api_key is not None else factory()
        except (ImportError, AttributeError) as exc:
            return Err(
                ProviderError(
                    message=(
                        f"provider {provider_id!r} resume_factory locator "
                        f"{rf.module}:{rf.classmethod} could not be resolved: {exc}"
                    ),
                    code="PROVIDER_RESUME_LOCATOR_FAILED",
                    details={"provider_id": provider_id, "exc_type": type(exc).__name__},
                )
            )
        return Ok(instance)

    def resolve_pod_lifecycle_client_cls(
        self, provider_id: str
    ) -> type[IPodLifecycleClient]:
        """Resolve the in-pod lifecycle client class.

        Used by the pod-side bootstrap (PR-1.12 transitional shim,
        PR-2 full projection). Returns the class — pod's lifespan
        constructs it with its own runtime args (``RUNPOD_API_KEY`` etc).
        """
        manifest = self.get_manifest(provider_id)
        if not manifest.capabilities.supports_lifecycle_actions:
            raise ProviderRegistryError(
                message=(
                    f"provider {provider_id!r} declares "
                    f"supports_lifecycle_actions=false — no pod lifecycle "
                    f"client to resolve."
                ),
                code="PROVIDER_NO_LIFECYCLE",
                details={"provider_id": provider_id},
            )
        return self._resolve_class(provider_id, role_key="pod_lifecycle_client")

    # ----- internals ------------------------------------------------------

    def _resolve_class(
        self,
        provider_id: str,
        *,
        role_key: str,
    ) -> type[Any]:
        """Lazy importlib resolution + capability metadata attachment.

        Cache keyed by ``(provider_id, role_key)`` so repeated calls
        (training-then-inference on the same provider) don't redo
        importlib work. First resolution per ``(provider_id, *)`` also
        attaches manifest metadata to the class — that's how
        :class:`ProviderBase` accessors find their data.
        """
        cache_key = (provider_id, role_key)
        cached = self._cls_cache.get(cache_key)
        if cached is not None:
            return cached
        manifest = self.get_manifest(provider_id)
        ep = getattr(manifest.entry_points, role_key, None)
        if ep is None:
            raise ProviderRegistryError(
                message=(
                    f"provider {provider_id!r} has no entry_points.{role_key} "
                    f"declaration."
                ),
                code="PROVIDER_LOCATOR_MISSING",
                details={"provider_id": provider_id, "role_key": role_key},
            )
        try:
            module = importlib.import_module(ep.module)
            cls = getattr(module, ep.class_name)
        except (ImportError, AttributeError) as exc:
            raise ProviderRegistryError(
                message=(
                    f"provider {provider_id!r} entry_points.{role_key} "
                    f"locator {ep.module}:{ep.class_name} could not be "
                    f"resolved: {exc}"
                ),
                code="PROVIDER_LOCATOR_RESOLVE_FAILED",
                details={
                    "provider_id": provider_id,
                    "role_key": role_key,
                    "module": ep.module,
                    "class_name": ep.class_name,
                    "exc_type": type(exc).__name__,
                },
            ) from exc
        self._attach_manifest_metadata(cls, manifest)
        with self._lock:
            self._cls_cache[cache_key] = cls
        return cls

    def _attach_manifest_metadata(
        self, cls: type[Any], manifest: ProviderManifest
    ) -> None:
        """Set the four ``_manifest_*`` ClassVars on ``cls``.

        :class:`ProviderBase` reads these in the default property impls.
        Idempotent — re-attaching the same data is fine; conflicting data
        is caught by the invariant suite (a single class shouldn't be
        pointed at by two manifests, and the registry rejects duplicate
        provider ids at load time).
        """
        # Lazy import — same circular-dependency reason as `capabilities`.
        from ryotenkai_providers.training.interfaces import ProviderBase

        if not isinstance(cls, type):
            return
        if not issubclass(cls, ProviderBase):
            # Inference and pod-lifecycle classes don't have to inherit
            # ProviderBase (they're Protocol-conforming, not base-derived).
            # Skip the attachment in that case — they don't read ClassVars.
            return
        cls._manifest_provider_id = manifest.provider.id
        cls._manifest_provider_name = manifest.provider.name
        cls._manifest_provider_type = manifest.capabilities.provider_type
        cls._manifest_capabilities = self.capabilities(manifest.provider.id)


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------


_REGISTRY_LOCK = threading.Lock()
_REGISTRY: ProviderRegistry | None = None


def get_registry() -> ProviderRegistry:
    """Lazy module-level singleton.

    Production code path. Tests build their own instance via
    :meth:`ProviderRegistry.from_filesystem` with ``roots=[tmpdir]``.

    Thread-safe: the first caller wins; subsequent concurrent calls
    return the same instance without re-walking the filesystem.
    """
    global _REGISTRY
    if _REGISTRY is not None:
        return _REGISTRY
    with _REGISTRY_LOCK:
        if _REGISTRY is None:
            _REGISTRY = ProviderRegistry.from_filesystem()
    return _REGISTRY


def reset_registry() -> None:
    """Drop the cached singleton — for tests only.

    Production has exactly one ``get_registry()`` call per process; tests
    that fabricate manifests in tmpdirs need to clear the cache between
    cases or they'd see leakage from the in-tree manifests of a previous
    test.
    """
    global _REGISTRY
    with _REGISTRY_LOCK:
        _REGISTRY = None


__all__ = [
    "LATEST_PROVIDER_SCHEMA_VERSION",
    "LoadFailure",
    "ProviderContext",
    "ProviderRegistry",
    "ProviderRegistryError",
    "ProviderRole",
    "get_registry",
    "reset_registry",
]
