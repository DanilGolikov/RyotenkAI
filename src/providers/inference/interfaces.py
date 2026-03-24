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

    from src.utils.result import InferenceError, Result


class PipelineReadinessMode(StrEnum):
    """
    How the pipeline should treat readiness after `deploy()`.

    - WAIT_FOR_HEALTHY: pipeline should actively wait until provider.health_check() is healthy.
    - SKIP: resource is intentionally "parked" after provisioning; readiness is handled by generated scripts.
    """

    WAIT_FOR_HEALTHY = "wait_for_healthy"
    SKIP = "skip"


@runtime_checkable
class InferenceEventLogger(Protocol):
    """
    Minimal event logger interface used by inference providers.

    Today this is backed by MLflowManager, but we keep it as a protocol to avoid
    leaking MLflow-specific types into provider interfaces.
    """

    def log_event_start(self, message: str, *, category: str, source: str, **kwargs: Any) -> None: ...

    def log_event_complete(self, message: str, *, category: str, source: str, **kwargs: Any) -> None: ...

    def log_event_error(self, message: str, *, category: str, source: str, **kwargs: Any) -> None: ...


@dataclass(frozen=True)
class EndpointInfo:
    """Information about a deployed inference endpoint."""

    endpoint_url: str  # Client base URL (MVP: via SSH tunnel) e.g. http://127.0.0.1:8000/v1
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
    """Unified interface for inference providers."""

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
        quantization: str | None = None,
        keep_running: bool = False,
    ) -> Result[EndpointInfo, InferenceError]: ...

    def set_event_logger(self, event_logger: InferenceEventLogger | None) -> None: ...

    def get_pipeline_readiness_mode(self) -> PipelineReadinessMode: ...

    def collect_startup_logs(self, *, local_path: Path) -> None: ...

    def build_inference_artifacts(
        self, *, ctx: InferenceArtifactsContext
    ) -> Result[InferenceArtifacts, InferenceError]: ...

    def undeploy(self) -> Result[None, InferenceError]: ...

    def health_check(self) -> Result[bool, InferenceError]: ...

    def get_capabilities(self) -> InferenceCapabilities: ...

    def get_endpoint_info(self) -> EndpointInfo | None: ...

    def activate_for_eval(self) -> Result[str, InferenceError]:
        """
        Bring up a live inference endpoint specifically for the evaluation stage.

        Called by InferenceDeployer when evaluation.enabled=true, AFTER deploy().

        Returns:
            Ok(endpoint_url) — active OpenAI-compatible base URL ready to receive requests.
            Err(InferenceError) — provider does not support eval activation or startup failed;
                                  InferenceDeployer will log a warning and skip evaluation.

        Contract:
        - Provider is responsible for all startup details (SSH tunnel, vLLM launch, etc.).
        - The returned URL must be reachable from the pipeline host.
        - For providers where endpoint is already live after deploy() (e.g. single_node),
          this is a no-op that returns the existing URL.
        """
        ...

    def deactivate_after_eval(self) -> Result[None, InferenceError]:
        """
        Shut down and clean up the inference endpoint after evaluation completes.

        Called by InferenceDeployer.cleanup() when evaluation.enabled=true.

        Policy (per provider):
        - single_node: no-op (endpoint lifecycle is managed externally).
        - runpod_pods: delete the Pod (preserves Network Volume); cost-critical.

        For providers that do not support this, return Err(InferenceError) — InferenceDeployer
        will log a warning but will not fail the pipeline.
        """
        ...
