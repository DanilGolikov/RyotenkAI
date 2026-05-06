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

from pydantic import BaseModel, ConfigDict, Field

if TYPE_CHECKING:
    from ryotenkai_engines.capabilities import EngineCapabilities
    from ryotenkai_shared.utils.result import AppError, Result


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
    ) -> Result[None, AppError]:
        """Engine-specific invariant check on the typed config.

        Runs after Pydantic schema validation — catches engine-side rules
        that don't fit cleanly in Pydantic ``@model_validator`` (e.g.
        cross-field rules that depend on the engine's MVP scope).

        Returns ``Ok(None)`` on success, ``Err(AppError)`` with a clear
        message + code on failure. Provider wraps and surfaces the Err.
        """
        ...


__all__ = (
    "BaseEngineConfig",
    "IInferenceEngine",
    "LaunchSpec",
)


def _runtime_typing_glue() -> Any:
    """Anchor for runtime imports needed by ``IInferenceEngine`` callers.

    ``Result`` and ``EngineCapabilities`` are imported under TYPE_CHECKING
    above to keep the public Protocol surface lean. This trivial helper
    exists only to silence ruff if it ever flags those names as unused
    — they ARE used (in the Protocol method signatures), but only in
    string annotations.
    """
    # Intentional no-op; do not import here to keep TYPE_CHECKING lean.
    return None
