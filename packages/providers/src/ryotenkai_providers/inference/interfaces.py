"""
Inference provider interfaces (contracts) and domain models.

Keep this module provider-agnostic and side-effect free.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

if TYPE_CHECKING:
    from pathlib import Path


class PipelineReadinessMode(StrEnum):
    """
    How the pipeline should treat readiness after `deploy()`.

    - WAIT_FOR_HEALTHY: pipeline should actively wait until provider.health_check() is healthy.
    - SKIP: resource is intentionally "parked" after provisioning; readiness is handled by generated scripts.
    """

    WAIT_FOR_HEALTHY = "wait_for_healthy"
    SKIP = "skip"


@dataclass(frozen=True)
class EndpointInfo:
    """Information about a deployed inference endpoint.

    ``endpoint_url`` is ``None`` for tunneled providers (RunPod-pods)
    until ``activate_for_eval`` opens the SSH tunnel. Returning a
    hardcoded ``http://127.0.0.1:port/v1`` pre-tunnel previously caused
    silent eval-garbage runs: the URL looked valid but nothing was
    listening. Consumers must check for ``None`` and refuse to send
    requests until the URL is populated.
    """

    endpoint_url: str | None  # None for tunneled providers until activated
    api_type: str  # openai_compatible | custom
    provider_type: str  # single_node | runpod
    engine: str  # vllm
    model_id: str  # model identifier (base model id or deployed artifact id)

    # Optional URLs
    health_url: str | None = None
    metrics_url: str | None = None
    ui_url: str | None = None

    # Cloud provider resource id, if any
    resource_id: str | None = None
    cost_per_hour: float | None = None


@dataclass(frozen=True)
class InferenceCapabilities:
    """Provider capabilities for future compatibility checks."""

    provider_type: str
    supported_engines: list[str]
    supports_lora: bool = False
    supports_streaming: bool = True
    #: Whether the provider can spin up a live OpenAI-compatible endpoint
    #: on demand for the evaluation stage. ``False`` means evaluation
    #: cannot run with this provider — the pipeline fails fast at the
    #: InferenceDeployer instead of stumbling into a phantom endpoint.
    supports_activate_for_eval: bool = False


@dataclass(frozen=True)
class InferenceArtifactsContext:
    """
    Provider-agnostic context for generating local inference artifacts.

    IMPORTANT:
    - No secrets here. Scripts read secrets from env / secrets.env.
    - Keep it explicit (no untyped dict context) to avoid hidden coupling.
    """

    run_name: str
    mlflow_run_id: str | None
    model_source: str
    endpoint: EndpointInfo


@dataclass(frozen=True)
class InferenceArtifacts:
    """Local inference artifacts generated for a deployment."""

    manifest: dict[str, Any]
    chat_script: str
    readme: str


@runtime_checkable
class IInferenceProvider(Protocol):
    """Unified interface for inference providers.

    Provider implementations inherit
    :class:`ryotenkai_providers.training.interfaces.ProviderBase` for the
    default impl of identity accessors (``provider_id``,
    ``provider_name``, ``provider_type``); they structurally conform to
    this Protocol via the inference-specific methods below.

    Note: the :meth:`get_capabilities` method on this Protocol returns an
    :class:`InferenceCapabilities` (inference-specific surface), distinct
    from :class:`ProviderCapabilities` returned by
    :meth:`IGPUProvider.get_capabilities`. A provider that fulfils both
    roles has separate inference and training classes — each with its
    own ``get_capabilities`` impl returning the role-specific shape.
    """

    @property
    def provider_id(self) -> str:
        """Canonical id from manifest (e.g. ``"runpod"``)."""
        ...

    @property
    def provider_name(self) -> str: ...

    @property
    def provider_type(self) -> str: ...

    def deploy(
        self,
        model_source: str,
        *,
        run_id: str,
        base_model_id: str,
        trust_remote_code: bool = False,
        lora_path: str | None = None,
        keep_running: bool = False,
    ) -> EndpointInfo:
        """Provision the inference endpoint and return its descriptor.

        Raises:
            InferenceUnavailableError: provider provisioning failed
                (transient / permanent).
            ConfigInvalidError: invalid model_source / adapter ref.
        """
        ...

    def get_pipeline_readiness_mode(self) -> PipelineReadinessMode: ...

    def collect_startup_logs(self, *, local_path: Path) -> None: ...

    def build_inference_artifacts(
        self, *, ctx: InferenceArtifactsContext
    ) -> InferenceArtifacts:
        """Render the manifest, chat script, and README content.

        Raises:
            InferenceUnavailableError: provider cannot build artifacts
                (missing pod metadata, etc.).
        """
        ...

    def undeploy(self) -> None:
        """Stop the inference endpoint (best-effort teardown).

        Raises:
            InferenceUnavailableError: backend rejected stop request.
        """
        ...

    def health_check(self) -> bool:
        """Probe endpoint readiness.

        Returns:
            True iff the endpoint is reachable and serving.

        Raises:
            InferenceUnavailableError: provider not deployed / probe
                transport failed.
        """
        ...

    def get_capabilities(self) -> InferenceCapabilities: ...

    def get_endpoint_info(self) -> EndpointInfo | None: ...

    def activate_for_eval(self) -> str:
        """
        Bring up a live inference endpoint specifically for the evaluation stage.

        Called by InferenceDeployer when evaluation.enabled=true, AFTER deploy(),
        only if ``get_capabilities().supports_activate_for_eval`` is ``True``.

        Returns:
            Active OpenAI-compatible base URL ready to receive requests.

        Raises:
            InferenceUnavailableError: startup failed (transient or
                permanent). InferenceDeployer **fails the stage** with
                code ``INFERENCE_ACTIVATION_FAILED`` and explicitly
                calls ``deactivate_after_eval`` to release the
                resource. The pipeline does NOT continue with a
                phantom endpoint.

        Contract:
        - Provider is responsible for all startup details (SSH tunnel, vLLM launch, etc.).
        - The returned URL must be reachable from the pipeline host.
        - For providers where endpoint is already live after deploy() (e.g. single_node),
          this is a no-op that returns the existing URL.
        """
        ...

    def deactivate_after_eval(self) -> None:
        """
        Shut down and clean up the inference endpoint after evaluation completes.

        Called by InferenceDeployer:
        - On stage cleanup() when evaluation.enabled=true (success path).
        - Inline in execute() when activate_for_eval fails, to avoid leaking
          a partially-provisioned pod.

        Policy (per provider):
        - single_node: no-op (endpoint lifecycle is managed externally).
        - runpod_pods: delete the Pod (preserves Network Volume); cost-critical.

        Best-effort: failures are logged but never mask an upstream error.

        Raises:
            InferenceUnavailableError: backend rejected cleanup
                (callers typically swallow on cleanup paths).
        """
        ...
