"""Pydantic models for provider manifests (``provider.toml``).

Each provider ships a ``provider.toml`` next to its package root
(e.g. ``packages/providers/src/ryotenkai_providers/runpod/provider.toml``).
The manifest is the **single source of truth** for:

* identity (id, name, version, stability, author);
* declared roles (training, inference, both);
* capability flags (mirrored 1:1 with capability micro-Protocols on the
  provider class — :class:`IPauseResumeProvider`, :class:`IRecoveryProbeProvider`,
  :class:`ICapacityErrorClassifier`, :class:`ITerminalActionProvider`);
* entry points (training/inference class locators, pod-side lifecycle
  client, optional resume factory, the per-provider Pydantic config
  schema class);
* required environment variables (so the startup validator can fail-fast
  before any provider is instantiated — pre-factory secret resolution).

The manifest is loaded by :class:`ProviderRegistry` at process start.
The schema-level invariants in :meth:`ProviderManifest._validate_invariants`
catch drift the moment the file is parsed; a separate ``check_manifests.py``
CLI runs the same checks plus class-level parity (capability flags vs
Protocol membership) before commit (see ``packages/providers/scripts/``).
"""

from __future__ import annotations

import re
from typing import Any, Literal

from packaging.version import InvalidVersion, Version
from pydantic import BaseModel, ConfigDict, Field, model_validator

#: Provider id format. ``snake_case``, starts with a letter; must equal the
#: package directory name (the loader walks ``packages/providers/src/
#: ryotenkai_providers/<id>/provider.toml``). Stable identifiers are used
#: in env vars (``RYOTENKAI_RUNTIME_PROVIDER``), in user YAML configs, and
#: in operator dashboards — renaming is a contract change.
_PROVIDER_ID_RE = re.compile(r"^[a-z][a-z0-9_]*$")


#: Latest manifest schema version this loader understands. Bumped on
#: breaking changes only (additive optional fields don't bump). A manifest
#: declaring a newer number is rejected with an upgrade hint so operators
#: see the right action.
#:
#: History:
#:   v1 (2026-05) — initial shape. Manifest replaces the dual-factory
#:                  approach (training factory.register + inference if/elif)
#:                  with a single declarative source consumed by
#:                  :class:`ProviderRegistry`.
LATEST_PROVIDER_SCHEMA_VERSION = 1


#: Coarse trustworthiness of the provider implementation. Operator-visible
#: in catalogue listings; the registry refuses to construct an
#: ``experimental`` provider without an explicit opt-in flag at the call
#: site (mirrors the community plugin convention).
Stability = Literal["stable", "beta", "experimental"]


#: ``"local"`` for in-house always-on hosts (single_node), ``"cloud"`` for
#: providers that allocate ephemeral resources on demand (RunPod).
#: Drives capability invariants (e.g. ``supports_capacity_error_detection``
#: requires ``cloud``) and is surfaced to the UI for sorting.
ProviderType = Literal["local", "cloud"]


#: Volume semantics influence the in-pod terminator's decision matrix and
#: the launcher's env builder. ``persistent`` is the RunPod default;
#: ``network`` is RunPod with a network-attached volume; ``local_disk``
#: is single_node (host filesystem).
VolumeKind = Literal["persistent", "network", "local_disk"]


#: Roles a provider may declare in ``[provider].roles``. Multi-valued
#: list — providers that handle BOTH training and inference list both.
#: A separate ``BOTH`` enum value would force every registry lookup to
#: special-case it; a multi-valued list is the cleaner shape (every
#: ``has_role`` predicate just queries set membership).
ProviderRole = Literal["training", "inference"]


# ---------------------------------------------------------------------------
# Sub-models
# ---------------------------------------------------------------------------


class ProviderSpec(BaseModel):
    """Body of the ``[provider]`` block — identity + role declaration."""

    model_config = ConfigDict(extra="forbid")

    id: str = Field(description="Snake_case canonical identifier; equals folder name.")
    name: str = Field(description="Display name for UI / dashboards.")
    version: str = Field(default="1.0.0", description="PEP 440 version string.")
    roles: list[ProviderRole] = Field(
        description=(
            "Non-empty list of roles this provider can fulfil. Each entry "
            "MUST have a corresponding ``entry_points.<role>`` block."
        ),
    )
    description: str = Field(default="", description="One-line summary; surfaces in catalogue.")
    author: str = Field(
        default="",
        description='Free-form attribution, "Name <email>" recommended.',
    )
    stability: Stability = Field(
        default="experimental",
        description=(
            "Coarse maturity marker. Registry refuses to construct "
            "non-stable providers without an explicit opt-in flag."
        ),
    )

    @model_validator(mode="after")
    def _check_id_format(self) -> ProviderSpec:
        if not _PROVIDER_ID_RE.match(self.id):
            raise ValueError(
                f"provider.id must match {_PROVIDER_ID_RE.pattern!r} "
                f"(snake_case identifier matching the folder name under "
                f"packages/providers/src/ryotenkai_providers/); got {self.id!r}"
            )
        return self

    @model_validator(mode="after")
    def _check_version_format(self) -> ProviderSpec:
        try:
            Version(self.version)
        except InvalidVersion as exc:
            raise ValueError(
                f"provider {self.id!r} version {self.version!r} is not a "
                f"valid PEP 440 version: {exc}"
            ) from exc
        return self

    @model_validator(mode="after")
    def _check_roles_non_empty_unique(self) -> ProviderSpec:
        if not self.roles:
            raise ValueError(
                f"provider.roles must be a non-empty subset of "
                f"{{'training', 'inference'}}; got an empty list."
            )
        if len(set(self.roles)) != len(self.roles):
            raise ValueError(
                f"provider.roles contains duplicates: {self.roles!r}. "
                f"Each role may appear at most once."
            )
        return self


class CapabilitiesSpec(BaseModel):
    """Body of the ``[capabilities]`` block.

    Mirrors the runtime :class:`ProviderCapabilities` dataclass in
    ``ryotenkai_providers.training.interfaces`` field-for-field. Each
    boolean flag pairs with either a base Protocol the provider class
    inherits or a behavioural property the orchestrator gates on. The
    flag-to-Protocol parity invariant is enforced by both
    :func:`check_manifests` (pre-commit) and the pytest invariant suite
    (CI) — see ADR row §10.4.
    """

    model_config = ConfigDict(extra="forbid")

    provider_type: ProviderType = Field(description="local | cloud — drives sort/UX.")
    is_local: bool = Field(
        default=False,
        description=(
            "True for always-on local hosts (single_node). Replaces the "
            "pre-Phase-14.D ``provider_name == 'single_node'`` string check "
            "in ``provider_config.py:is_single_node_provider``."
        ),
    )
    supports_multi_gpu: bool = Field(default=False)
    supports_spot_instances: bool = Field(
        default=False,
        description="Cloud-only; spot/preemptible capacity.",
    )
    supports_lifecycle_actions: bool = Field(
        default=False,
        description=(
            "True iff the provider implements :class:`ITerminalActionProvider`. "
            "Pairs 1:1 with Protocol membership; the registry rejects manifests "
            "that flip this flag without updating the class."
        ),
    )
    has_pause_resume: bool = Field(
        default=False,
        description=(
            "Subset of :attr:`supports_lifecycle_actions`: True iff the provider "
            "supports the FULL pause→resume cycle (not just terminate). Implies "
            "the class implements :class:`IPauseResumeProvider`."
        ),
    )
    supports_recovery_probe: bool = Field(
        default=False,
        description=(
            "True iff the provider can probe + recover an existing resource "
            "after the runner connection was lost. Implies the class implements "
            ":class:`IRecoveryProbeProvider`. Replaces the hardcoded "
            "``provider_name != 'runpod'`` skip in ``training_monitor.py``."
        ),
    )
    supports_capacity_error_detection: bool = Field(
        default=False,
        description=(
            "True iff the provider can classify backend error messages as "
            "transient capacity exhaustion vs hard failures. Implies the class "
            "implements :class:`ICapacityErrorClassifier`. Replaces the "
            "``provider == PROVIDER_RUNPOD: is_capacity_error_message`` branch "
            "in ``resume_service.py``."
        ),
    )
    supports_log_download: bool = Field(
        default=False,
        description=(
            "True iff the provider exposes a structured log-download path "
            "(cloud SCP/HTTP-fetch). Single_node = False (logs already on "
            "host filesystem); RunPod = True."
        ),
    )
    volume_kind: VolumeKind = Field(
        default="persistent",
        description="Storage semantics; drives PodTerminator + launcher env.",
    )
    runner_workspace_root: str = Field(
        default="/workspace",
        description=(
            "PYTHONPATH / cwd inside the in-pod runner. Defaults to /workspace "
            "for both shipped providers; future providers mounting elsewhere "
            "(e.g. /data) override here without touching the launcher."
        ),
    )
    max_runtime_hours: int | None = Field(
        default=None,
        description="Hard wall-clock cap; ``null`` means unlimited.",
    )
    supported_engines: tuple[str, ...] = Field(
        default=(),
        description=(
            "Optional whitelist of engine kinds this provider can launch. "
            "Empty tuple (default) means 'no constraint' — the provider "
            "accepts any engine the registry knows about. Populated, the "
            "PipelineConfig cross-validator rejects "
            "(provider, engine) pairs not in this list. Required when "
            "the provider declares the ``inference`` role and ships a "
            "specialized container build."
        ),
    )

    @model_validator(mode="after")
    def _check_invariants(self) -> CapabilitiesSpec:
        # has_pause_resume is a strict subset of supports_lifecycle_actions
        # (you can't pause without being able to terminate).
        if self.has_pause_resume and not self.supports_lifecycle_actions:
            raise ValueError(
                "capabilities.has_pause_resume=true requires "
                "capabilities.supports_lifecycle_actions=true (pause/resume "
                "is a subset of the lifecycle-action contract)."
            )
        # is_local hosts can't have lifecycle actions or persistent volumes —
        # they ARE the volume.
        if self.is_local:
            if self.supports_lifecycle_actions:
                raise ValueError(
                    "capabilities.is_local=true is incompatible with "
                    "supports_lifecycle_actions=true. Local always-on hosts "
                    "have no lifecycle to manage; the orchestrator never "
                    "terminates a local host."
                )
            if self.volume_kind != "local_disk":
                raise ValueError(
                    f"capabilities.is_local=true requires volume_kind='local_disk'; "
                    f"got {self.volume_kind!r}. Local hosts use the host filesystem."
                )
        # Capacity-error classification only makes sense for cloud providers
        # (a local box doesn't run out of GPU "capacity" — there's no fleet).
        if self.supports_capacity_error_detection and self.provider_type != "cloud":
            raise ValueError(
                f"capabilities.supports_capacity_error_detection=true requires "
                f"provider_type='cloud'; got {self.provider_type!r}. Capacity "
                f"errors are a cloud-fleet concept, not a local-host concept."
            )
        # Max-runtime sentinel — explicit None is unlimited; non-positive
        # ints are invalid (caught here so manifests with ``max_runtime_hours = 0``
        # — the legacy sentinel — fail loudly instead of silently meaning
        # "unlimited").
        if self.max_runtime_hours is not None and self.max_runtime_hours <= 0:
            raise ValueError(
                f"capabilities.max_runtime_hours must be a positive integer "
                f"or null; got {self.max_runtime_hours}. Use null for "
                f"unlimited runs (the legacy 0 sentinel is rejected)."
            )
        return self


class ClassEntryPoint(BaseModel):
    """Locator for a runtime-resolved class.

    Loader splits ``module`` + ``class_name`` and uses
    ``importlib.import_module(module).<class_name>`` lazily — only when
    the corresponding role is requested. Keeping locators as strings (not
    direct imports) is what makes ``ryotenkai_providers`` import-safe even
    when one provider's heavy SDK (e.g. RunPod's httpx) is unavailable.
    """

    model_config = ConfigDict(extra="forbid")

    module: str = Field(description="Fully-qualified Python module path.")
    class_name: str = Field(
        alias="class",
        description="Class name within the module.",
    )


class ResumeFactoryEntryPoint(BaseModel):
    """Locator for an optional resume-factory classmethod.

    Phase-14.B resume-service uses this to bypass full provider
    construction (which validates the entire ``providers.<id>`` config
    block) when only the lifecycle methods are needed. Provider declares
    ``RunPodProvider.from_resume_metadata`` here; absence means the
    provider does not support pause→resume bypass.
    """

    model_config = ConfigDict(extra="forbid")

    module: str = Field(description="Fully-qualified Python module path.")
    classmethod: str = Field(
        description="``ClassName.method_name`` reference; loader splits on '.'.",
    )


class EntryPointsSpec(BaseModel):
    """Body of the ``[entry_points.*]`` blocks.

    Each role / capability has its own sub-block; absence of a block means
    the provider does not fulfil that surface. The :class:`ProviderManifest`
    cross-validator enforces role↔block parity.
    """

    model_config = ConfigDict(extra="forbid")

    training: ClassEntryPoint | None = Field(
        default=None,
        description="Locator for the IGPUProvider class. Required if 'training' in roles.",
    )
    inference: ClassEntryPoint | None = Field(
        default=None,
        description="Locator for the IInferenceProvider class. Required if 'inference' in roles.",
    )
    pod_lifecycle_client: ClassEntryPoint | None = Field(
        default=None,
        description=(
            "Locator for the in-pod IPodLifecycleClient class. Required iff "
            "capabilities.supports_lifecycle_actions=true. Used by the pod-side "
            "registry (sub-manifest projection) at lifespan boot."
        ),
    )
    resume_factory: ResumeFactoryEntryPoint | None = Field(
        default=None,
        description=(
            "Optional classmethod for cheap resume-only construction. "
            "Provider may omit this entirely; the registry returns a clean "
            "Err to callers that request it on a non-supporting provider."
        ),
    )
    config_schema: ClassEntryPoint = Field(
        description=(
            "Locator for the per-provider Pydantic config schema "
            "(``RunPodProviderConfig``). The PipelineConfig validator runs "
            "``ConfigCls.model_validate(provider_block)`` at load-time, so "
            "YAML typos surface BEFORE any pipeline stage runs. Required."
        ),
    )


class RequiredEnvSpec(BaseModel):
    """Environment variable / secret a provider needs at runtime.

    Pre-factory: the startup validator reads these without instantiating
    the provider. Replaces the legacy ``_resolve_required_secrets_for_provider``
    string-dispatch helper. Same shape as the community plugin
    ``RequiredEnvSpec`` for operator-UX consistency.

    Fields:
      name               — UPPER_SNAKE_CASE env var.
      description        — what the value is used for.
      optional           — when True, provider runs even if unset.
      secret             — when True, UI renders password-style input.
      required_for_roles — list of roles for which this env is required.
                          Empty list ``[]`` means "required for ANY role
                          this provider fulfils" (e.g. ``RUNPOD_API_KEY``
                          is needed both for training pod creation AND
                          inference REST API — list both, or leave the
                          list empty as a shorthand). When non-empty,
                          only the listed roles trigger the requirement.
    """

    model_config = ConfigDict(extra="forbid")

    name: str = Field(description="UPPER_SNAKE_CASE env var name.")
    description: str = Field(default="", description="Operator-facing description.")
    optional: bool = Field(default=False)
    secret: bool = Field(default=True)
    required_for_roles: list[ProviderRole] = Field(
        default_factory=list,
        description=(
            "Roles for which this env is required. Empty list = required "
            "for any role this provider fulfils. Listing all roles "
            "explicitly is also valid (and more discoverable for the UI)."
        ),
    )

    @model_validator(mode="after")
    def _check_unique_roles(self) -> RequiredEnvSpec:
        if len(set(self.required_for_roles)) != len(self.required_for_roles):
            raise ValueError(
                f"required_env entry {self.name!r} has duplicate values in "
                f"required_for_roles: {self.required_for_roles!r}. Each role "
                f"may appear at most once."
            )
        return self


# ---------------------------------------------------------------------------
# Top-level manifest
# ---------------------------------------------------------------------------


class ProviderManifest(BaseModel):
    """Full manifest loaded from ``packages/providers/.../<id>/provider.toml``.

    This is the **single source of truth** consumed by
    :class:`ProviderRegistry`. The schema-level validators below cover
    every drift class identified in the audit (§10.4 invariants). Loader
    catches the rest (importability, Protocol parity) by walking the
    resolved classes — see ``ProviderRegistry._validate_class_parity``.
    """

    model_config = ConfigDict(extra="forbid")

    schema_version: int = Field(default=LATEST_PROVIDER_SCHEMA_VERSION)
    provider: ProviderSpec
    capabilities: CapabilitiesSpec
    entry_points: EntryPointsSpec
    required_env: list[RequiredEnvSpec] = Field(
        default_factory=list,
        description="Pre-factory secret/env declarations.",
    )

    # ---- whole-document invariants ---------------------------------------

    @model_validator(mode="after")
    def _check_schema_version(self) -> ProviderManifest:
        if self.schema_version < 1:
            raise ValueError(
                f"provider schema_version must be >= 1; got {self.schema_version}"
            )
        if self.schema_version > LATEST_PROVIDER_SCHEMA_VERSION:
            raise ValueError(
                f"provider {self.provider.id!r} declares schema_version="
                f"{self.schema_version}, but this RyotenkAI build supports up "
                f"to v{LATEST_PROVIDER_SCHEMA_VERSION}. Upgrade the host to "
                f"load this manifest, or downgrade the manifest."
            )
        return self

    @model_validator(mode="after")
    def _check_roles_to_entry_points_parity(self) -> ProviderManifest:
        # Every declared role MUST have its entry-point block present —
        # otherwise the registry will crash the moment the role is requested.
        # Catching this at manifest-load time gives a clear error before
        # any pipeline starts.
        if "training" in self.provider.roles and self.entry_points.training is None:
            raise ValueError(
                f"provider {self.provider.id!r} declares role 'training' but "
                f"[entry_points.training] is missing. Either add the block or "
                f"remove 'training' from provider.roles."
            )
        if "inference" in self.provider.roles and self.entry_points.inference is None:
            raise ValueError(
                f"provider {self.provider.id!r} declares role 'inference' but "
                f"[entry_points.inference] is missing. Either add the block or "
                f"remove 'inference' from provider.roles."
            )
        # And the converse: an entry_point without a declared role is dead
        # code (registry never calls it). Reject so the manifest stays honest.
        if (
            self.entry_points.training is not None
            and "training" not in self.provider.roles
        ):
            raise ValueError(
                f"provider {self.provider.id!r} has [entry_points.training] but "
                f"'training' is not in provider.roles. Either add the role or "
                f"remove the unused entry-point block."
            )
        if (
            self.entry_points.inference is not None
            and "inference" not in self.provider.roles
        ):
            raise ValueError(
                f"provider {self.provider.id!r} has [entry_points.inference] but "
                f"'inference' is not in provider.roles. Either add the role or "
                f"remove the unused entry-point block."
            )
        return self

    @model_validator(mode="after")
    def _check_lifecycle_pod_locator_parity(self) -> ProviderManifest:
        # The Mac↔pod parity invariant: declared lifecycle support REQUIRES
        # a pod-side locator. This is the schema-level fix for the audit's
        # "two sources of truth" risk — adding a third provider with
        # ``supports_lifecycle_actions=true`` without a pod entry will fail
        # at manifest LOAD, not 4 hours later in production.
        caps = self.capabilities
        ep = self.entry_points
        if caps.supports_lifecycle_actions and ep.pod_lifecycle_client is None:
            raise ValueError(
                f"provider {self.provider.id!r} declares "
                f"capabilities.supports_lifecycle_actions=true but "
                f"[entry_points.pod_lifecycle_client] is missing. "
                f"Add the pod-side IPodLifecycleClient locator (the in-pod "
                f"registry needs it at runner boot)."
            )
        # Conversely, a pod_lifecycle_client locator without the flag is
        # dead — the pod registry would never resolve it.
        if not caps.supports_lifecycle_actions and ep.pod_lifecycle_client is not None:
            raise ValueError(
                f"provider {self.provider.id!r} has [entry_points.pod_lifecycle_client] "
                f"but capabilities.supports_lifecycle_actions=false. Either "
                f"flip the flag to true or remove the unused entry-point block."
            )
        return self

    @model_validator(mode="after")
    def _check_required_env_role_scopes(self) -> ProviderManifest:
        # Every role in ``required_for_roles`` MUST be one the provider
        # actually declares. Empty list = "any role" — no constraint.
        declared = set(self.provider.roles)
        for spec in self.required_env:
            extra = set(spec.required_for_roles) - declared
            if extra:
                raise ValueError(
                    f"provider {self.provider.id!r} required_env entry "
                    f"{spec.name!r} lists required_for_roles="
                    f"{spec.required_for_roles!r} but provider.roles="
                    f"{list(self.provider.roles)!r} does not declare "
                    f"{sorted(extra)!r}. Either add the role to "
                    f"provider.roles or remove it from required_for_roles."
                )
        return self

    # ---- helpers ---------------------------------------------------------

    def required_secret_names(
        self, *, role: ProviderRole | None = None
    ) -> tuple[str, ...]:
        """Return env names this provider requires for the given role.

        ``role=None`` returns the union across all declared roles. Used
        by :class:`ProviderRegistry.required_secrets` for the pre-factory
        startup-validator path (no provider instantiation needed).

        Filtering rules:
          * ``optional=true`` → never returned.
          * ``secret=false`` → never returned (non-secret envs are
            advisory; the UI surfaces them but startup doesn't block).
          * ``required_for_roles=[]`` → required for every role this
            provider declares (the "shared credential" shorthand).
          * ``required_for_roles=[X, Y]`` → required only when ``role``
            ∈ {X, Y}. ``role=None`` matches any non-empty list.
        """
        out: list[str] = []
        for spec in self.required_env:
            if spec.optional or not spec.secret:
                continue
            if (
                spec.required_for_roles
                and role is not None
                and role not in spec.required_for_roles
            ):
                continue
            out.append(spec.name)
        return tuple(out)

    def ui_manifest(self) -> dict[str, Any]:
        """Flatten into the shape consumed by ``GET /api/providers``.

        The Web UI catalogue card needs identity + capabilities + a list
        of required envs. Keep this method narrow — JSON Schema for the
        config block is fetched separately via
        ``GET /api/providers/<id>/config-schema`` (see §4.4).
        """
        return {
            "schema_version": self.schema_version,
            "id": self.provider.id,
            "name": self.provider.name,
            "version": self.provider.version,
            "roles": list(self.provider.roles),
            "description": self.provider.description,
            "author": self.provider.author,
            "stability": self.provider.stability,
            "capabilities": self.capabilities.model_dump(mode="json"),
            "required_env": [spec.model_dump(mode="json") for spec in self.required_env],
        }


__all__ = [
    "LATEST_PROVIDER_SCHEMA_VERSION",
    "CapabilitiesSpec",
    "ClassEntryPoint",
    "EntryPointsSpec",
    "ProviderManifest",
    "ProviderRole",
    "ProviderSpec",
    "ProviderType",
    "RequiredEnvSpec",
    "ResumeFactoryEntryPoint",
    "Stability",
    "VolumeKind",
]
