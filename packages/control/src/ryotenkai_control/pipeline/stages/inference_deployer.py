"""
Stage 7: Inference Deployer

Deploys an OpenAI-compatible inference endpoint and generates local helper scripts
(chat/stop) + manifest-first metadata.
"""

from __future__ import annotations

import json
import time
from typing import TYPE_CHECKING, Any

from ryotenkai_shared.constants import (
    INFERENCE_CHAT_SCRIPT_FILENAME,
    INFERENCE_DIRNAME,
    INFERENCE_MANIFEST_FILENAME,
    INFERENCE_README_FILENAME,
)
from ryotenkai_shared.utils.cancellation import sleep_cancellable
from ryotenkai_control.pipeline.constants import MLFLOW_CATEGORY_INFERENCE
from ryotenkai_control.pipeline.stages.base import PipelineStage
from ryotenkai_control.pipeline.stages.constants import PipelineContextKeys, StageNames
from ryotenkai_shared.errors import (
    InferenceUnavailableError,
    InternalError,
    ModelLoadFailedError,
    RyotenkAIError,
)
from ryotenkai_shared.pipeline_context import RunContext
from ryotenkai_providers.inference.interfaces import (
    EndpointInfo,
    InferenceArtifactsContext,
    InferenceEventLogger,
    PipelineReadinessMode,
)
from ryotenkai_shared.utils.logger import get_run_log_dir, logger

if TYPE_CHECKING:
    from pathlib import Path

    from ryotenkai_providers.inference.interfaces import IInferenceProvider
    from ryotenkai_shared.config import PipelineConfig, Secrets


# Phrases RunPod REST API returns when there is no GPU capacity.
_NO_CAPACITY_PHRASES = (
    "could not find any pods with required specifications",
    "there are no instances currently available",
    "no available datacenter with requested resources",
)


def _is_no_capacity_error(err: object) -> bool:
    """Return True if the deploy error is a transient GPU capacity issue."""
    lower = str(err).lower()
    return any(phrase in lower for phrase in _NO_CAPACITY_PHRASES)


def _make_deferred_endpoint(
    *,
    port: int,
    provider_type: str,
    engine: str,
    model_id: str,
) -> EndpointInfo:
    """Build a placeholder EndpointInfo when pod provisioning is deferred to chat launch time."""
    return EndpointInfo(
        endpoint_url=f"http://127.0.0.1:{port}/v1",
        api_type="openai_compatible",
        provider_type=provider_type,
        engine=engine,
        model_id=model_id,
        health_url=f"http://127.0.0.1:{port}/v1/models",
        resource_id="unknown",
    )


class InferenceDeployer(PipelineStage):
    """Deploy inference endpoint after training and evaluation."""

    def __init__(self, config: PipelineConfig, secrets: Secrets):
        super().__init__(config, StageNames.INFERENCE_DEPLOYER)
        self.secrets = secrets
        self._provider: IInferenceProvider | None = None

    def execute(self, context: dict[str, Any]) -> dict[str, Any]:
        """Deploy an inference endpoint or signal skipped.

        Returns:
            Updated context dict.

        Raises:
            InferenceUnavailableError: pod / activate / health check failed.
            ModelLoadFailedError: model_source unresolved.
            InternalError: missing run context wiring.
        """
        inf_cfg = self.config.inference
        forced_stages = context.get(PipelineContextKeys.FORCED_STAGES, set())
        is_forced = self.stage_name in forced_stages if isinstance(forced_stages, set | list | tuple) else False

        if not inf_cfg.enabled and not is_forced:
            return self.update_context(
                context,
                {
                    "inference_skipped": True,
                    "reason": "inference.enabled=false",
                },
            )

        # Optional event logger from context (if available)
        mlflow_manager = context.get(PipelineContextKeys.MLFLOW_MANAGER)
        event_logger = mlflow_manager if isinstance(mlflow_manager, InferenceEventLogger) else None

        # Resolve model source from ModelRetriever context (raises on miss).
        model_source = self._resolve_model_source(context)

        run = context.get(PipelineContextKeys.RUN)
        if not isinstance(run, RunContext):
            raise InternalError(
                detail="Missing run context: context['run'] must be RunContext (initialized by PipelineOrchestrator)",
                context={"legacy_code": "MISSING_RUN_CONTEXT"},
            )

        run_name = run.name
        base_model_id = self.config.model.name

        # Manifest-driven registry replaces the legacy
        # ``InferenceProviderFactory.create(config, secrets)`` if/elif
        # chain. Batch 12: registry raises typed exceptions directly;
        # callers propagate (or translate to a stage-specific
        # InferenceUnavailableError if context matters).
        from ryotenkai_providers.registry import ProviderContext, get_registry

        provider_id = self.config.inference.provider or ""
        provider_block: object = None
        if provider_id:
            try:
                provider_block = self.config.get_provider_config(provider_id)
            except (KeyError, ValueError):
                # Registry will raise ProviderNotFoundError with the
                # right error message when we call create_inference.
                provider_block = None
        ctx = ProviderContext(
            provider_id=provider_id,
            pipeline_config=self.config,
            provider_block=provider_block,
            secrets=self.secrets,
        )
        try:
            provider = get_registry().create_inference(provider_id, ctx)
        except RyotenkAIError as exc:
            raise InferenceUnavailableError(
                detail=exc.detail or str(exc),
                context={"legacy_code": "INFERENCE_PROVIDER_CREATE_FAILED"},
                cause=exc,
            ) from exc
        self._provider = provider

        # Explicit interface: no provider-private attribute injection
        provider.set_event_logger(event_logger)

        logger.info(
            "🚀 Deploying inference: provider=%s engine=%s",
            inf_cfg.provider,
            inf_cfg.engine.kind,
        )
        eval_cfg = getattr(self.config, "evaluation", None)
        eval_enabled = getattr(eval_cfg, "enabled", False) is True
        # Engine-specific tuning (quantization, max_model_len, etc.) is
        # read by the provider directly from cfg.inference.engine — no
        # longer plumbed through the generic deploy() API. The provider
        # knows the typed engine config via the registry.
        pod_provisioning_failed = False
        try:
            endpoint = provider.deploy(
                model_source=model_source,
                run_id=run_name,  # Canonical run name (single source of truth)
                base_model_id=base_model_id,
                trust_remote_code=self.config.model.trust_remote_code,
                lora_path=None if inf_cfg.common.lora.adapter_path == "auto" else inf_cfg.common.lora.adapter_path,
                # Skip the stop-after-provisioning if eval will activate the pod right after.
                keep_running=eval_enabled,
            )
        except RyotenkAIError as deploy_err:
            # No GPU capacity available (RunPod HTTP 500).
            # Treat as a soft failure: skip health check and generate manifest + chat script
            # anyway — the pod will be created on first chat session launch.
            if _is_no_capacity_error(deploy_err):
                logger.warning(
                    f"⚠️ Inference pod could not be provisioned (no GPU capacity): {deploy_err}\n"
                    "Generating manifest and chat script anyway — "
                    "pod will be created on first chat session."
                )
                pod_provisioning_failed = True
                port = 8000
                try:
                    serve_cfg = getattr(provider, "_serve_cfg", None)
                    if serve_cfg is not None:
                        port = int(serve_cfg.port)
                except Exception:
                    pass
                endpoint = _make_deferred_endpoint(
                    port=port,
                    provider_type=inf_cfg.provider or "unknown",
                    engine=inf_cfg.engine,
                    model_id=base_model_id,
                )
            else:
                raise InferenceUnavailableError(
                    detail=deploy_err.detail or str(deploy_err),
                    context={"legacy_code": "INFERENCE_DEPLOY_FAILED"},
                    cause=deploy_err,
                ) from deploy_err

        # Readiness (provider-defined policy)
        if not pod_provisioning_failed and inf_cfg.common.health_check.enabled:
            readiness = provider.get_pipeline_readiness_mode()
            if readiness == PipelineReadinessMode.WAIT_FOR_HEALTHY:
                try:
                    self._wait_for_healthy(provider, event_logger=event_logger)
                except RyotenkAIError:
                    # Best-effort cleanup on error — swallow secondary errors
                    # so the original failure isn't masked.
                    try:
                        provider.undeploy()
                    except RyotenkAIError as undeploy_exc:
                        logger.debug(
                            f"[INFERENCE] post-failure undeploy errored "
                            f"(non-fatal): {undeploy_exc}"
                        )
                    raise
            elif readiness == PipelineReadinessMode.SKIP:
                logger.info(
                    f"ℹ️ Skipping pipeline health check for {provider.provider_type} "
                    "(readiness is handled by the generated chat script)"
                )

        # Activate for evaluation if needed (BEFORE writing artifacts, so endpoint_url is real).
        # Fail-fast policy: if the provider declares support but activation
        # fails, the stage returns Err. Letting the pipeline continue with
        # a phantom endpoint_url is what produced "successful" eval runs
        # full of empty answers.
        eval_endpoint_url: str | None = None
        if eval_enabled and not pod_provisioning_failed:
            capabilities = provider.get_capabilities()
            if not capabilities.supports_activate_for_eval:
                raise InferenceUnavailableError(
                    detail=(
                        f"evaluation enabled but provider {inf_cfg.provider!r} does not "
                        "support activate_for_eval — set evaluation.enabled=false or "
                        "pick a provider with supports_activate_for_eval=True"
                    ),
                    context={"legacy_code": "INFERENCE_EVAL_NOT_SUPPORTED"},
                )
            try:
                eval_endpoint_url = provider.activate_for_eval()
            except RyotenkAIError as activate_err:
                # Release the partially-provisioned pod inline; relying on
                # stage cleanup() alone is fragile — a downstream stage
                # could still abort the cleanup chain.
                try:
                    provider.deactivate_after_eval()
                except RyotenkAIError as deactivate_err:
                    logger.warning(
                        "[INFERENCE] post-failure deactivate also errored "
                        f"(non-fatal): {deactivate_err}"
                    )
                raise InferenceUnavailableError(
                    detail=(
                        f"failed to activate inference endpoint for evaluation: {activate_err}"
                    ),
                    context={"legacy_code": "INFERENCE_ACTIVATION_FAILED"},
                    cause=activate_err,
                ) from activate_err
            logger.info(f"✅ Inference endpoint ready for evaluation: {eval_endpoint_url}")
        elif eval_enabled and pod_provisioning_failed:
            # Eval was requested but the pod could not be provisioned
            # (NO_GPU_CAPACITY). Refuse rather than emit a manifest that
            # the user thinks is eval-ready.
            raise InferenceUnavailableError(
                detail=(
                    "evaluation enabled but inference pod could not be provisioned "
                    "(no GPU capacity). Disable evaluation or retry later."
                ),
                context={"legacy_code": "INFERENCE_NO_CAPACITY_BLOCKS_EVAL"},
            )

        # Generate manifest + scripts locally (provider supplies content; raises on failure).
        log_dir = get_run_log_dir()
        manifest_path, script_paths = self._write_manifest_and_scripts(
            provider=provider,
            log_dir=log_dir,
            context=context,
            endpoint=endpoint,
            model_source=model_source,
            run_name=run_name,
        )

        return self.update_context(
            context,
            {
                "inference_deployed": not pod_provisioning_failed,
                "inference_pod_deferred": pod_provisioning_failed,  # True → pod will be created on chat launch
                "inference_endpoint_url": endpoint.endpoint_url,
                "inference_model_name": endpoint.model_id,
                # Single source of truth for endpoint URL.
                # When eval is enabled, activate_for_eval guarantees
                # ``eval_endpoint_url`` is populated (or the stage has
                # already raised above).
                "endpoint_url": eval_endpoint_url if eval_enabled else endpoint.endpoint_url,
                "endpoint_info": endpoint.__dict__,
                "inference_manifest_path": str(manifest_path),
                "inference_scripts": {k: str(v) for k, v in script_paths.items()},
            },
        )

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def cleanup(self) -> None:
        """
        Cleanup inference resources after pipeline completion (or failure).

        Policy:
        - evaluation.enabled=true  → call deactivate_after_eval() (provider decides what to stop/delete).
        - evaluation.enabled=false → leave endpoint running for user (they stop it via chat scripts).
        - No provider              → no-op.
        """
        if self._provider is None:
            return

        if self.config.inference.common.keep_inference_after_eval:
            logger.info("[CLEANUP] Inference: keep_inference_after_eval=true — leaving endpoint running")
            return

        eval_cfg = getattr(self.config, "evaluation", None)
        if not getattr(eval_cfg, "enabled", False):
            logger.info("[CLEANUP] Inference: evaluation disabled — leaving endpoint running for user")
            return

        logger.info("[CLEANUP] Inference: evaluation was enabled — calling deactivate_after_eval()")
        try:
            self._provider.deactivate_after_eval()
        except RyotenkAIError as exc:
            logger.warning(f"[CLEANUP] deactivate_after_eval warning: {exc}")
        else:
            logger.info("[CLEANUP] Inference endpoint deactivated successfully")

    def _resolve_model_source(self, context: dict[str, Any]) -> str:
        """Resolve the model identifier the inference provider should serve.

        Raises:
            ModelLoadFailedError: when auto-resolution cannot find a
                concrete model id.
        """
        retriever_ctx = context.get(StageNames.MODEL_RETRIEVER, {})
        hf_repo_id = retriever_ctx.get("hf_repo_id")
        local_model_path = retriever_ctx.get("local_model_path")

        if self.config.inference.common.model_source != "auto":
            return self.config.inference.common.model_source

        if isinstance(hf_repo_id, str) and hf_repo_id:
            return hf_repo_id
        if isinstance(local_model_path, str) and local_model_path:
            return local_model_path

        raise ModelLoadFailedError(
            detail=(
                "Inference model_source=auto but ModelRetriever did not provide "
                "hf_repo_id or local_model_path. Enable HF upload or ensure "
                "model is downloaded locally."
            ),
            context={"legacy_code": "MODEL_SOURCE_NOT_RESOLVED"},
        )

    def _wait_for_healthy(
        self,
        provider: IInferenceProvider,
        *,
        event_logger: InferenceEventLogger | None,
    ) -> None:
        cfg = self.config.inference.common.health_check
        deadline = time.time() + cfg.timeout_seconds
        last_state: str = "not_checked"

        # Setup log collection
        log_dir = get_run_log_dir()
        startup_log_path = log_dir / INFERENCE_DIRNAME / "startup.log"
        startup_log_path.parent.mkdir(parents=True, exist_ok=True)

        logger.info(f"📝 Collecting inference startup logs → {startup_log_path}")

        # Track health check start for event logger (if available)
        health_check_start = time.time()
        if event_logger:
            event_logger.log_event_start(
                "Health check started",
                category=MLFLOW_CATEGORY_INFERENCE,
                source=StageNames.INFERENCE_DEPLOYER,
                timeout_seconds=cfg.timeout_seconds,
            )

        while time.time() < deadline:
            # Collect current logs
            provider.collect_startup_logs(local_path=startup_log_path)

            # Health check
            try:
                ok = provider.health_check()
            except RyotenkAIError as exc:
                last_state = f"error:{exc.detail or exc}"
            else:
                if ok:
                    # Final log collection on success
                    provider.collect_startup_logs(local_path=startup_log_path)
                    health_check_duration = time.time() - health_check_start
                    logger.info("✅ Inference endpoint is healthy")
                    logger.info(f"📥 Startup logs saved: {startup_log_path}")

                    # Log successful health check
                    if event_logger:
                        event_logger.log_event_complete(
                            f"Health check passed ({health_check_duration:.1f}s)",
                            category=MLFLOW_CATEGORY_INFERENCE,
                            source=StageNames.INFERENCE_DEPLOYER,
                            duration_seconds=health_check_duration,
                        )

                    return None
                last_state = "not_ready"

            # Cancel-aware sleep so Ctrl+C during a long health-check
            # window (can be minutes for cold-starting vLLM) wakes the
            # poller immediately instead of waiting out the interval.
            sleep_cancellable(cfg.interval_seconds)

        # Final log collection on timeout (for diagnostics)
        provider.collect_startup_logs(local_path=startup_log_path)
        health_check_duration = time.time() - health_check_start
        logger.error(f"❌ Inference health check timed out. Check logs: {startup_log_path}")

        if event_logger:
            event_logger.log_event_error(
                f"Health check timed out after {cfg.timeout_seconds}s",
                category=MLFLOW_CATEGORY_INFERENCE,
                source=StageNames.INFERENCE_DEPLOYER,
                duration_seconds=health_check_duration,
                timeout_seconds=cfg.timeout_seconds,
                last_state=last_state,
            )

        raise InferenceUnavailableError(
            detail=(
                f"Inference health check timed out after {cfg.timeout_seconds}s "
                f"(last_state={last_state})"
            ),
            context={
                "legacy_code": "INFERENCE_HEALTH_CHECK_TIMEOUT",
                "timeout_seconds": cfg.timeout_seconds,
                "last_state": last_state,
            },
        )

    def _write_manifest_and_scripts(
        self,
        *,
        provider: IInferenceProvider,
        log_dir: Path,
        context: dict[str, Any],
        endpoint: EndpointInfo,
        model_source: str,
        run_name: str,
    ) -> tuple[Path, dict[str, Path]]:
        """Render manifest + helper scripts on disk.

        Raises:
            InferenceUnavailableError: when the provider cannot build artifacts.
        """
        mlflow_run_id = context.get(PipelineContextKeys.MLFLOW_PARENT_RUN_ID)
        if not (isinstance(mlflow_run_id, str) and mlflow_run_id.strip()):
            mlflow_run_id = None

        artifacts_ctx = InferenceArtifactsContext(
            run_name=run_name,
            mlflow_run_id=mlflow_run_id,
            model_source=model_source,
            endpoint=endpoint,
        )
        try:
            artifacts = provider.build_inference_artifacts(ctx=artifacts_ctx)
        except RyotenkAIError as exc:
            raise InferenceUnavailableError(
                detail=exc.detail or str(exc),
                context={"legacy_code": "INFERENCE_ARTIFACTS_BUILD_FAILED"},
                cause=exc,
            ) from exc

        inference_dir = log_dir / INFERENCE_DIRNAME
        inference_dir.mkdir(parents=True, exist_ok=True)

        manifest_path = inference_dir / INFERENCE_MANIFEST_FILENAME
        manifest_path.write_text(json.dumps(artifacts.manifest, ensure_ascii=False, indent=2), encoding="utf-8")

        scripts = {
            "chat": inference_dir / INFERENCE_CHAT_SCRIPT_FILENAME,
        }
        readme_path = inference_dir / INFERENCE_README_FILENAME

        scripts["chat"].write_text(artifacts.chat_script, encoding="utf-8")
        readme_path.write_text(artifacts.readme, encoding="utf-8")
        scripts["readme"] = readme_path

        logger.info(f"🧾 Inference manifest: {manifest_path}")
        logger.info(f"🛠️ Inference scripts: {', '.join(str(p) for p in scripts.values())}")

        return manifest_path, scripts


__all__ = [
    "InferenceDeployer",
]
