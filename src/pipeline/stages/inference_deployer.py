"""
Stage 7: Inference Deployer

Deploys an OpenAI-compatible inference endpoint and generates local helper scripts
(chat/stop) + manifest-first metadata.
"""

from __future__ import annotations

import json
import time
from typing import TYPE_CHECKING, Any

from src.constants import (
    INFERENCE_CHAT_SCRIPT_FILENAME,
    INFERENCE_DIRNAME,
    INFERENCE_MANIFEST_FILENAME,
    INFERENCE_README_FILENAME,
)
from src.pipeline.constants import MLFLOW_CATEGORY_INFERENCE
from src.pipeline.domain import RunContext
from src.pipeline.stages.base import PipelineStage
from src.pipeline.stages.constants import PipelineContextKeys, StageNames
from src.providers.inference.factory import InferenceProviderFactory
from src.providers.inference.interfaces import (
    EndpointInfo,
    InferenceArtifactsContext,
    InferenceEventLogger,
    PipelineReadinessMode,
)
from src.utils.logger import get_run_log_dir, logger
from src.utils.result import AppError, Err, InferenceError, Ok, Result

if TYPE_CHECKING:
    from pathlib import Path

    from src.providers.inference.interfaces import IInferenceProvider
    from src.utils.config import PipelineConfig, Secrets


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

    def execute(self, context: dict[str, Any]) -> Result[dict[str, Any], AppError]:
        inf_cfg = self.config.inference
        forced_stages = context.get(PipelineContextKeys.FORCED_STAGES, set())
        is_forced = self.stage_name in forced_stages if isinstance(forced_stages, set | list | tuple) else False

        if not inf_cfg.enabled and not is_forced:
            return Ok(
                self.update_context(
                    context,
                    {
                        "inference_skipped": True,
                        "reason": "inference.enabled=false",
                    },
                )
            )

        # Optional event logger from context (if available)
        mlflow_manager = context.get(PipelineContextKeys.MLFLOW_MANAGER)
        event_logger = mlflow_manager if isinstance(mlflow_manager, InferenceEventLogger) else None

        # Resolve model source from ModelRetriever context
        model_source_res = self._resolve_model_source(context)
        if model_source_res.is_failure():
            return Err(model_source_res.unwrap_err())  # type: ignore[union-attr]  # already InferenceError
        model_source = model_source_res.unwrap()

        run = context.get(PipelineContextKeys.RUN)
        if not isinstance(run, RunContext):
            return Err(
                InferenceError(
                    message="Missing run context: context['run'] must be RunContext (initialized by PipelineOrchestrator)",
                    code="MISSING_RUN_CONTEXT",
                )
            )

        run_name = run.name
        base_model_id = self.config.model.name

        create_result = InferenceProviderFactory.create(config=self.config, secrets=self.secrets)
        if create_result.is_failure():
            return Err(
                InferenceError(
                    message=str(create_result.unwrap_err()),
                    code="INFERENCE_PROVIDER_CREATE_FAILED",
                )
            )
        provider = create_result.unwrap()
        self._provider = provider

        # Explicit interface: no provider-private attribute injection
        provider.set_event_logger(event_logger)

        logger.info(f"🚀 Deploying inference: provider={inf_cfg.provider} engine={inf_cfg.engine}")
        eval_cfg = getattr(self.config, "evaluation", None)
        eval_enabled = getattr(eval_cfg, "enabled", False) is True
        deploy_res = provider.deploy(
            model_source=model_source,
            run_id=run_name,  # Canonical run name (single source of truth)
            base_model_id=base_model_id,
            trust_remote_code=self.config.model.trust_remote_code,
            lora_path=None if inf_cfg.common.lora.adapter_path == "auto" else inf_cfg.common.lora.adapter_path,
            quantization=inf_cfg.engines.vllm.quantization,
            # Skip the stop-after-provisioning if eval will activate the pod right after.
            keep_running=eval_enabled,
        )

        pod_provisioning_failed = False
        if deploy_res.is_failure():
            deploy_err = deploy_res.unwrap_err()  # type: ignore[union-attr]
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
                    provider_type=inf_cfg.provider,
                    engine=inf_cfg.engine,
                    model_id=base_model_id,
                )
            else:
                return Err(InferenceError(message=str(deploy_err), code="INFERENCE_DEPLOY_FAILED"))
        else:
            endpoint = deploy_res.unwrap()

        # Readiness (provider-defined policy)
        if not pod_provisioning_failed and inf_cfg.common.health_check.enabled:
            readiness = provider.get_pipeline_readiness_mode()
            if readiness == PipelineReadinessMode.WAIT_FOR_HEALTHY:
                healthy = self._wait_for_healthy(provider, event_logger=event_logger)
                if healthy.is_failure():
                    # Best-effort cleanup on error
                    _ = provider.undeploy()
                    return Err(healthy.unwrap_err())  # type: ignore[union-attr]  # already InferenceError
            elif readiness == PipelineReadinessMode.SKIP:
                logger.info(
                    f"ℹ️ Skipping pipeline health check for {provider.provider_type} "
                    "(readiness is handled by the generated chat script)"
                )

        # Activate for evaluation if needed (BEFORE writing artifacts, so endpoint_url is real).
        # If evaluation is enabled but provider doesn't support activation → log warning, skip eval.
        eval_endpoint_url: str | None = None
        if eval_enabled and not pod_provisioning_failed:
            activate_res = provider.activate_for_eval()
            if activate_res.is_success():
                eval_endpoint_url = activate_res.unwrap()
                logger.info(f"✅ Inference endpoint ready for evaluation: {eval_endpoint_url}")
            else:
                activate_err = activate_res.unwrap_err()  # type: ignore[union-attr]
                logger.warning(
                    f"⚠️ Evaluation skipped: provider '{inf_cfg.provider}' does not support "
                    f"activate_for_eval: {activate_err}"
                )

        # Generate manifest + scripts locally (provider supplies content)
        log_dir = get_run_log_dir()
        manifest_res = self._write_manifest_and_scripts(
            provider=provider,
            log_dir=log_dir,
            context=context,
            endpoint=endpoint,
            model_source=model_source,
            run_name=run_name,
        )
        if manifest_res.is_failure():
            return Err(manifest_res.unwrap_err())  # type: ignore[union-attr]  # already InferenceError

        manifest_path, script_paths = manifest_res.unwrap()

        return Ok(
            self.update_context(
                context,
                {
                    "inference_deployed": not pod_provisioning_failed,
                    "inference_pod_deferred": pod_provisioning_failed,  # True → pod will be created on chat launch
                    "inference_endpoint_url": endpoint.endpoint_url,
                    "inference_model_name": endpoint.model_id,
                    # Single source of truth for endpoint URL.
                    # activate_for_eval() overwrites with live URL when evaluation is enabled.
                    "endpoint_url": eval_endpoint_url if eval_endpoint_url is not None else endpoint.endpoint_url,
                    "endpoint_info": endpoint.__dict__,
                    "inference_manifest_path": str(manifest_path),
                    "inference_scripts": {k: str(v) for k, v in script_paths.items()},
                },
            )
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
        result = self._provider.deactivate_after_eval()
        if result.is_failure():
            logger.warning(f"[CLEANUP] deactivate_after_eval warning: {result.unwrap_err()}")
        else:
            logger.info("[CLEANUP] Inference endpoint deactivated successfully")

    def _resolve_model_source(self, context: dict[str, Any]) -> Result[str, InferenceError]:
        retriever_ctx = context.get(StageNames.MODEL_RETRIEVER, {})
        hf_repo_id = retriever_ctx.get("hf_repo_id")
        local_model_path = retriever_ctx.get("local_model_path")

        if self.config.inference.common.model_source != "auto":
            return Ok(self.config.inference.common.model_source)

        if isinstance(hf_repo_id, str) and hf_repo_id:
            return Ok(hf_repo_id)
        if isinstance(local_model_path, str) and local_model_path:
            return Ok(local_model_path)

        return Err(
            InferenceError(
                message="Inference model_source=auto but ModelRetriever did not provide hf_repo_id or local_model_path. "
                "Enable HF upload or ensure model is downloaded locally.",
                code="MODEL_SOURCE_NOT_RESOLVED",
            )
        )

    def _wait_for_healthy(
        self,
        provider: IInferenceProvider,
        *,
        event_logger: InferenceEventLogger | None,
    ) -> Result[None, InferenceError]:
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
            res = provider.health_check()
            if res.is_success():
                ok = res.unwrap()
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

                    return Ok(None)
                last_state = "not_ready"
            else:
                last_state = f"error:{res.unwrap_err()}"  # type: ignore[union-attr]

            time.sleep(cfg.interval_seconds)

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

        return Err(
            InferenceError(
                message=f"Inference health check timed out after {cfg.timeout_seconds}s (last_state={last_state})",
                code="INFERENCE_HEALTH_CHECK_TIMEOUT",
                details={"timeout_seconds": cfg.timeout_seconds, "last_state": last_state},
            )
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
    ) -> Result[tuple[Path, dict[str, Path]], InferenceError]:
        mlflow_run_id = context.get(PipelineContextKeys.MLFLOW_PARENT_RUN_ID)
        if not (isinstance(mlflow_run_id, str) and mlflow_run_id.strip()):
            mlflow_run_id = None

        artifacts_ctx = InferenceArtifactsContext(
            run_name=run_name,
            mlflow_run_id=mlflow_run_id,
            model_source=model_source,
            endpoint=endpoint,
        )
        artifacts_res = provider.build_inference_artifacts(ctx=artifacts_ctx)
        if artifacts_res.is_failure():
            artifacts_err = artifacts_res.unwrap_err()  # type: ignore[union-attr]  # str from provider
            return Err(InferenceError(message=str(artifacts_err), code="INFERENCE_ARTIFACTS_BUILD_FAILED"))
        artifacts = artifacts_res.unwrap()

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

        return Ok((manifest_path, scripts))


__all__ = [
    "InferenceDeployer",
]
