"""``IInferenceEngine`` Protocol + supporting types.

The engine plugin contract.

Design principles
-----------------

* **Side-effect free.** Every method on the Protocol returns a value
  (string, dataclass, ``Result``). No HTTP calls, no SSH, no async.
  Provider classes own all execution; engines just describe what to run.

* **Forward-compatible.** ``build_launch_spec`` returns a structured
  :class:`LaunchSpec` (image, args, env, port, volumes), not a shell
  string. Today's docker-via-SSH providers format it into ``docker run …``
  themselves; tomorrow's k8s provider translates the same struct into a
  ``ContainerSpec`` / ``PodSpec`` without touching the engine.

* **Pure config dispatch.** Engine identity flows through Pydantic's
  discriminated union (``BaseEngineConfig.kind``); the runtime class is
  resolved by ``EngineRegistry.get_runtime(kind)``. No string-comparison
  branching outside of Pydantic itself.

The Protocol is ``runtime_checkable`` — sentinel tests assert that every
shipped engine class actually satisfies the surface (drift safety net).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, ClassVar, Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field, model_validator

if TYPE_CHECKING:
    from ryotenkai_engines.capabilities import EngineCapabilities


# ---------------------------------------------------------------------------
# Engine config base — every engine config subclasses this with its own
# Literal[kind].
# ---------------------------------------------------------------------------


class BaseEngineConfig(BaseModel):
    """Common ancestor of every engine-specific config class.

    Subclasses MUST override ``kind`` with a ``Literal["<engine_id>"]``
    matching their ``engine.toml [engine].id``. The Tag-based discriminator
    in :mod:`ryotenkai_engines._config_union` dispatches off this field.

    Example::

        from typing import Literal
        from ryotenkai_engines.interfaces import BaseEngineConfig

        class VLLMEngineConfig(BaseEngineConfig):
            kind: Literal["vllm"] = "vllm"
            tensor_parallel_size: int = 1
            ...
    """

    model_config = ConfigDict(extra="forbid")

    #: Discriminator — every concrete subclass narrows this to a specific
    #: ``Literal["<id>"]``. The default declared on the base is ``str``
    #: only because subclasses MUST override; it's never used directly.
    kind: str


# ---------------------------------------------------------------------------
# LaunchSpec — the structured "what to run" return value of build_launch_spec.
# ---------------------------------------------------------------------------


class LaunchSpec(BaseModel):
    """Structured engine launch description.

    Provider decides how to wrap this:

      * single_node / RunPod (today): format ``docker run --gpus all
        -p {host}:{port}:{port} -v {host_path}:{container_path} {image}
        {args...}``.

      * k8s (future): translate to a ``corev1.Container`` + ``Volume``
        + ``ContainerPort``. ``args`` becomes ``container.args``;
        ``env`` becomes ``container.env``; ``volumes`` become
        ``volumeMounts`` + ``Volume`` entries.

      * systemd (hypothetical): ``ExecStart=/usr/bin/docker run …``
        with the same args.

    Frozen — once an engine has built the spec, providers must NOT
    mutate. If you need to add provider-side env (e.g. ``HF_TOKEN``),
    wrap the spec or build a new one.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    image: str = Field(
        description="Fully-qualified container image (registry/name:tag).",
    )
    container_name: str = Field(
        description="Stable, run-scoped container name (used for docker stop / logs).",
    )
    args: tuple[str, ...] = Field(
        description=(
            "CLI args passed to the engine binary inside the container. "
            "Already split — no shell quoting needed by the provider."
        ),
    )
    env: dict[str, str] = Field(
        default_factory=dict,
        description=(
            "Engine-specific env vars (HF_HOME, model-cache paths, etc.). "
            "Provider may add its own (HF_TOKEN, etc.) on top."
        ),
    )
    port: int = Field(
        ge=1,
        le=65535,
        description="Container-side port the engine listens on.",
    )
    volumes: tuple[tuple[str, str], ...] = Field(
        default=(),
        description=(
            "Bind mounts as (host_path, container_path) tuples. "
            "Provider translates to ``-v host:container`` for docker, "
            "Volume + VolumeMount for k8s."
        ),
    )


# ---------------------------------------------------------------------------
# PrepareStep + PreparePlan — engine-described pre-launch work (PR-16).
#
# Pattern: the engine describes WHAT to run before the inference server
# starts (LoRA merge, GGUF conversion, TensorRT compilation, AWQ
# pre-quant, …). The provider executes — runs the ephemeral container,
# polls logs, verifies artifacts, cleans up. Mirrors the
# LaunchSpec / build_launch_spec / format_docker_run pattern exactly.
#
# Multi-step support from day 1: engines may return a tuple of ordered
# steps that share the workspace volume. tomorrow's llama.cpp engine
# can chain ``convert_gguf → quantize_q4`` without an API change.
# ---------------------------------------------------------------------------


class PrepareStep(BaseModel):
    """One ordered preparation step (e.g. ``"merge_lora"``).

    Pure data. The engine describes; the provider's plan-runner
    executes. No IO inside the engine — providers own SSH, docker,
    log polling, marker checks, artifact verification.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str = Field(
        description=(
            "Stable identifier for this step within the plan "
            "(e.g. 'merge_lora', 'convert_gguf'). Used in logs, MLflow "
            "tags, container names. Must be unique within a PreparePlan."
        ),
    )
    image: str | None = Field(
        default=None,
        description=(
            "Container image. ``None`` = provider uses the engine's "
            "serve image (the common case — vLLM merge runs in the same "
            "image as serve). Set explicitly when the prep step needs a "
            "different toolchain (e.g. TensorRT compilation image)."
        ),
    )
    entrypoint: tuple[str, ...] | None = Field(
        default=None,
        description=(
            "Override the image's ENTRYPOINT (e.g. ``('python3',)`` to "
            "run a Python script inside an image whose default entrypoint "
            "is the inference server). ``None`` = use image default."
        ),
    )
    args: tuple[str, ...] = Field(
        description=(
            "CLI args passed to the entrypoint inside the container. "
            "Already split — no shell quoting needed by the provider."
        ),
    )
    env: dict[str, str] = Field(
        default_factory=dict,
        description=(
            "Engine-specific env vars (cache paths, etc.). Provider may "
            "overlay its own (HF_TOKEN, etc.); on key collision provider "
            "wins (secrets boundary)."
        ),
    )
    volumes: tuple[tuple[str, str], ...] = Field(
        default=(),
        description="Bind mounts as ``(host_path, container_path)`` tuples.",
    )
    inputs: tuple[str, ...] = Field(
        default=(),
        description=(
            "Container paths this step READS. Allows the provider to "
            "verify they exist before running, and lets earlier steps' "
            "outputs flow into later steps' inputs (via the shared "
            "workspace volume — this list is documentary, not piped)."
        ),
    )
    outputs: tuple[str, ...] = Field(
        description=(
            "Container paths this step PRODUCES. Provider verifies the "
            "first one exists post-step (after a successful exit code + "
            "marker check). Empty tuple = step produces no artifacts the "
            "provider needs to verify."
        ),
    )
    success_marker: str | None = Field(
        default=None,
        description=(
            "Optional substring that MUST appear in stdout for the step "
            "to be considered successful. ``None`` = exit-code-only check. "
            "Defense-in-depth against scripts that exit 0 silently."
        ),
    )
    success_artifact: str | None = Field(
        default=None,
        description=(
            "Optional container path that must exist after a successful "
            "step. Provider maps host-side via ``volumes`` and checks "
            "with ``test -f``. ``None`` = skip artifact check."
        ),
    )
    timeout_seconds: int = Field(
        default=3600,
        ge=1,
        description="Hard wall-clock timeout for this step.",
    )


class PreparePlan(BaseModel):
    """Engine's preparation plan — zero-or-more ordered steps.

    Empty plan (``steps=()``) means "no preparation needed" — the
    provider serves the original model_source path directly. Used by
    SGLang and any future engine with native LoRA / no-merge support.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    spec_version: int = Field(
        default=1,
        description=(
            "Bumped on breaking shape changes. Provider rejects unknown "
            "versions with a loud error so engine ↔ provider skew is "
            "caught immediately, not silently."
        ),
    )
    steps: tuple[PrepareStep, ...] = Field(
        default=(),
        description=(
            "Ordered. Provider runs them sequentially via the shared "
            "workspace volume; each step's outputs are visible on disk "
            "to subsequent steps. No parallel execution; fail-fast on "
            "any step error (later steps are NOT run)."
        ),
    )
    final_model_path: str | None = Field(
        default=None,
        description=(
            "Container path the provider passes to ``build_launch_spec("
            "model_path_in_container=...)``. When ``steps`` is non-empty "
            "this MUST be set (validator enforces); usually equals "
            "``steps[-1].outputs[0]``. When ``steps`` is empty, provider "
            "uses its conventional path (whatever ``deploy()`` was given)."
        ),
    )

    @model_validator(mode="after")
    def _validate(self) -> PreparePlan:
        if self.steps and self.final_model_path is None:
            raise ValueError(
                "PreparePlan with steps must set final_model_path. "
                "Empty plan ⇒ final_model_path may be None (provider falls "
                "back to its conventional path)."
            )
        names = [s.name for s in self.steps]
        if len(names) != len(set(names)):
            raise ValueError(
                f"PrepareStep names must be unique within a plan; got: {names}"
            )
        return self

    @classmethod
    def empty(cls) -> PreparePlan:
        """No-op plan factory. Used by ``NoPrepareMixin`` and engines
        that decide at runtime that no prep is needed (e.g. when no
        adapter was supplied)."""
        return cls()


# ---------------------------------------------------------------------------
# IInferenceEngine — the runtime-checkable Protocol every engine implements.
# ---------------------------------------------------------------------------


@runtime_checkable
class IInferenceEngine(Protocol):
    """Engine plugin contract.

    Every engine ships a class that satisfies this Protocol. The class is
    instantiated zero-or-more times per pipeline run via
    ``EngineRegistry.get_runtime(kind)()``. Methods are pure functions of
    the engine config + launch parameters — no IO, no async.

    ClassVars:
      * ``engine_id`` MUST equal ``engine.toml [engine].id``.
      * ``config_class`` MUST be the ``BaseEngineConfig`` subclass for this engine
        (used by the discriminated-union builder to register the variant).
    """

    #: Stable engine id; equals ``engine.toml [engine].id``. Drift detector
    #: checks parity in CI.
    engine_id: ClassVar[str]

    #: The Pydantic config class for this engine. Used by the union builder
    #: at registry construction.
    config_class: ClassVar[type[BaseEngineConfig]]

    def get_capabilities(self) -> EngineCapabilities:
        """Return the engine's declared capabilities.

        MUST exactly match the ``[capabilities]`` block of the engine's
        ``engine.toml``. Drift detector enforces parity.
        """
        ...

    def build_launch_spec(
        self,
        *,
        cfg: BaseEngineConfig,
        image: str,
        container_name: str,
        port: int,
        workspace_host_path: str,
        model_path_in_container: str,
    ) -> LaunchSpec:
        """Build a structured launch description.

        Args:
            cfg: Typed engine config (subclass-specific). Engine MAY assume
                ``isinstance(cfg, self.config_class)`` — type narrowing
                happened at PipelineConfig load.
            image: Fully-qualified container image (resolved by
                :func:`ryotenkai_engines.images.resolve_image`).
            container_name: Stable container name from the provider.
            port: Container-side port the engine should bind to. Usually
                equals ``capabilities.default_port`` but provider may
                override (e.g. multiple engines on one host).
            workspace_host_path: Absolute path on the launcher host that
                will be bind-mounted into the container as ``/workspace``.
                Engine doesn't need to know — just plumbs through to volumes.
            model_path_in_container: Path inside the container where the
                model weights live (typically ``{workspace}/model``).
        """
        ...

    def build_healthcheck_command(
        self,
        *,
        host: str,
        port: int,
    ) -> str:
        """Return a shell snippet whose exit code 0 means "engine is ready".

        Today: a curl probe of the engine's `/v1/models` (or equivalent)
        endpoint. Provider runs this remotely (SSH / docker exec / etc.).
        """
        ...

    def build_default_endpoint_url(
        self,
        *,
        host: str,
        port: int,
    ) -> str:
        """Return the URL that ModelClientFactory will dial.

        For OpenAI-compatible engines: ``f"http://{host}:{port}/v1"``.
        Engines with a non-standard prefix override.
        """
        ...

    def validate_config(
        self,
        cfg: BaseEngineConfig,
    ) -> None:
        """Engine-specific invariant check on the typed config.

        Runs after Pydantic schema validation — catches engine-side rules
        that don't fit cleanly in Pydantic ``@model_validator`` (e.g.
        cross-field rules that depend on the engine's MVP scope).

        Returns ``None`` on success. Raises
        :class:`ryotenkai_shared.errors.EngineConfigInvalidError`
        (status 422, code ``ENGINE_CONFIG_INVALID``) on failure with a
        clear ``detail`` and ``context["reason"]`` subcode. Providers
        let it propagate; HTTP / CLI renderers translate via the
        unified error model.
        """
        ...

    def prepare_model(
        self,
        *,
        cfg: BaseEngineConfig,
        base_model: str,
        adapter_path_in_container: str | None,
        workspace_host_path: str,
        run_id: str,
        trust_remote_code: bool,
    ) -> PreparePlan:
        """Describe pre-launch preparation work (LoRA merge, GGUF conv, …).

        Pure function. Engine returns a :class:`PreparePlan` with zero or
        more ordered :class:`PrepareStep` entries; provider executes them
        sequentially before launching the inference server. Engines
        that need no prep return :meth:`PreparePlan.empty()` — see
        :class:`NoPrepareMixin` for the standard implementation.

        Args:
            cfg: Typed engine config (subclass-specific).
            base_model: HF repo id or container path of the base model.
            adapter_path_in_container: Path to LoRA adapter inside the
                container, or ``None`` when no adapter was supplied.
                The provider has already mapped any host paths into the
                workspace mount before calling.
            workspace_host_path: Host-side absolute path that the provider
                bind-mounts as ``/workspace`` inside every container.
                Engines plumb it through into ``volumes``; never inspect
                or interpret it.
            run_id: Stable run identifier from the pipeline. Used by
                engines for naming output paths within the workspace.
            trust_remote_code: Forward to model loading scripts that
                support it (HF ``trust_remote_code`` semantics).

        Returns:
            The :class:`PreparePlan` — possibly empty (no work needed)
            or carrying ordered steps.

        Raises:
            EngineConfigInvalidError: when the engine can't construct
                a valid plan from the inputs (e.g. config-type mismatch
                that slipped past Pydantic's discriminated union).
                Rare — most config errors fire earlier in
                ``validate_config``.
        """
        ...


# ---------------------------------------------------------------------------
# NoPrepareMixin — default ``prepare_model`` for engines without prep work.
#
# Engines that don't need any preparation (SGLang today, future
# vLLM-with-LiveLoRA) inherit this mixin to get a one-line no-op
# ``prepare_model`` for free. vLLM does NOT use the mixin — it
# overrides with a real implementation.
# ---------------------------------------------------------------------------


class NoPrepareMixin:
    """Default ``prepare_model`` returning an empty plan.

    Pure no-op. Subclasses inherit this BEFORE the concrete engine
    base, e.g.::

        class SGLangEngineRuntime(NoPrepareMixin):
            engine_id: ClassVar[str] = "sglang"
            ...

    The mixin is opt-in (Protocols don't carry default impls), but
    centralizes the boilerplate so every prep-less engine doesn't
    repeat the same one-liner.
    """

    def prepare_model(
        self,
        *,
        cfg: BaseEngineConfig,  # noqa: ARG002 — Protocol-conformant signature
        base_model: str,  # noqa: ARG002
        adapter_path_in_container: str | None,  # noqa: ARG002
        workspace_host_path: str,  # noqa: ARG002
        run_id: str,  # noqa: ARG002
        trust_remote_code: bool,  # noqa: ARG002
    ) -> PreparePlan:
        return PreparePlan.empty()


__all__ = (
    "BaseEngineConfig",
    "IInferenceEngine",
    "LaunchSpec",
    "NoPrepareMixin",
    "PreparePlan",
    "PrepareStep",
)


def _runtime_typing_glue() -> Any:
    """Anchor for runtime imports needed by ``IInferenceEngine`` callers.

    ``EngineCapabilities`` is imported under TYPE_CHECKING above to keep
    the public Protocol surface lean. This trivial helper exists only
    to silence ruff if it ever flags the name as unused — it IS used
    (in the Protocol method signatures), but only in string annotations.
    """
    # Intentional no-op; do not import here to keep TYPE_CHECKING lean.
    return None
