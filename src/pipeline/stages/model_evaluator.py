"""
Stage: Model Evaluator

Evaluates the trained model against an eval dataset using the plugin-based
evaluation architecture. Delegates all evaluation logic to EvaluationRunner.

Pipeline flow:
    InferenceDeployer → [sets context['endpoint_url']] → ModelEvaluator

ModelEvaluator reads:
    - context['endpoint_url']        : live inference endpoint URL
    - context['inference_model_name']: model name for API requests

ModelEvaluator writes to context (under StageNames.MODEL_EVALUATOR):
    - eval_passed          (bool)
    - eval_summary         (dict, JSON-serializable)
    - metrics              (dict[str, float])  — for MLflow logging
    - evaluation_skipped   (bool, optional)
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from src.pipeline.stages.base import PipelineStage
from src.pipeline.stages.constants import PipelineContextKeys, StageNames
from src.utils.logger import logger
from src.utils.result import AppError, Ok, Result

if TYPE_CHECKING:
    from collections.abc import Callable

    from src.config.secrets.model import Secrets
    from src.utils.config import PipelineConfig


@dataclass
class EvaluatorEventCallbacks:
    """
    Callbacks for ModelEvaluator events (SOLID-compliant event collection).

    Used to integrate ModelEvaluator / EvaluationRunner with MLflow or other
    logging systems without coupling the runner to any specific backend.

    Mirrors the pattern of DatasetValidatorEventCallbacks.
    """

    # Evaluation started: dataset_path, sample_count
    on_eval_start: Callable[[str, int], None] | None = None

    # Single plugin started: plugin_id, plugin_name, description
    on_plugin_start: Callable[[str, str, str], None] | None = None

    # Single plugin passed: plugin_id, plugin_name, metrics, duration_ms
    on_plugin_complete: Callable[[str, str, dict, float], None] | None = None

    # Single plugin failed: plugin_id, plugin_name, metrics, errors, recommendations, duration_ms
    on_plugin_failed: Callable[[str, str, dict, list, list, float], None] | None = None

    # Evaluation finished: overall_passed, sample_count, duration_seconds
    on_eval_complete: Callable[[bool, int, float], None] | None = None


class ModelEvaluator(PipelineStage):
    """
    Evaluation stage: runs plugin-based model evaluation via EvaluationRunner.

    Requires:
    - evaluation.enabled=true  in config (enforced by orchestrator — stage is only
      added to the pipeline when evaluation is enabled).
    - context['endpoint_url']  set by InferenceDeployer.execute() (via activate_for_eval).
    """

    def __init__(
        self,
        config: PipelineConfig,
        secrets: Secrets | None = None,
        callbacks: EvaluatorEventCallbacks | None = None,
    ) -> None:
        super().__init__(config, StageNames.MODEL_EVALUATOR)
        self._callbacks = callbacks or EvaluatorEventCallbacks()
        self._secrets = secrets

    def execute(self, context: dict[str, Any]) -> Result[dict[str, Any], AppError]:
        eval_cfg = getattr(self.config, "evaluation", None)
        forced_stages = context.get(PipelineContextKeys.FORCED_STAGES, set())
        is_forced = self.stage_name in forced_stages if isinstance(forced_stages, set | list | tuple) else False

        if (eval_cfg is None or not getattr(eval_cfg, "enabled", False)) and not is_forced:
            return Ok(
                self.update_context(
                    context,
                    {
                        "evaluation_skipped": True,
                        "reason": "evaluation.enabled=false",
                    },
                )
            )

        # Read endpoint URL set by InferenceDeployer.
        # Data is stored under the stage name key (update_context convention),
        # so we look in context[StageNames.INFERENCE_DEPLOYER] first, then fall
        # back to the root for backwards-compatibility with tests that inject directly.
        inference_ctx = context.get(StageNames.INFERENCE_DEPLOYER, {})
        endpoint_url: str | None = inference_ctx.get("endpoint_url") or context.get("endpoint_url")
        if not endpoint_url:
            logger.warning(
                "[EVAL] No endpoint_url in context — evaluation skipped. "
                "InferenceDeployer may have failed to activate the endpoint."
            )
            return Ok(
                self.update_context(
                    context,
                    {
                        "evaluation_skipped": True,
                        "reason": "endpoint_url not available (provider does not support activate_for_eval)",
                    },
                )
            )

        # Model name for API requests.
        # vLLM serves the merged model under its directory path by default,
        # not under config.model.name. Mirror the chat_inference.py approach:
        # ask /v1/models and take the first id; fall back to config.model.name.
        static_model_name: str = (
            inference_ctx.get("inference_model_name") or context.get("inference_model_name") or self.config.model.name
        )
        model_name = self._resolve_model_name(endpoint_url=endpoint_url, fallback=static_model_name)
        engine: str = getattr(self.config.inference, "engine", "vllm")

        logger.info(f"[EVAL] Starting evaluation — endpoint: {endpoint_url}, " f"model: {model_name}, engine: {engine}")
        start_time = time.time()

        # Build inference client via factory (DIP: depends on IModelInference, not concrete class)
        from src.evaluation.model_client.factory import ModelClientFactory

        system_prompt = self._load_system_prompt(context)
        if system_prompt:
            logger.info("[EVAL] System prompt loaded (inference.llm config)")
        else:
            logger.debug("[EVAL] No system prompt configured — requests will be sent without system message")

        inference_client = ModelClientFactory.create(
            engine=engine,
            base_url=endpoint_url,
            model=model_name,
            timeout_seconds=60,
            max_tokens=512,  # noqa: WPS432
            temperature=0.0,
            system_prompt=system_prompt,
        )

        # Run evaluation
        from typing import cast

        from src.evaluation.plugins.secrets import SecretsResolver
        from src.evaluation.runner import EvaluationRunner

        secrets_resolver = SecretsResolver(self._secrets) if self._secrets is not None else None
        runner = EvaluationRunner(
            cast(Any, eval_cfg),
            callbacks=self._callbacks,
            secrets_resolver=secrets_resolver,
        )
        summary = runner.run(inference_client)

        duration = time.time() - start_time
        logger.info(
            f"[EVAL] Evaluation complete in {duration:.1f}s — "
            f"passed={summary.overall_passed}, plugins={list(summary.plugin_results.keys())}"
        )

        verdict = "PASSED" if summary.overall_passed else "FAILED"
        logger.info(f"[EVAL] Overall verdict: {verdict}")
        if not summary.overall_passed:
            for plugin_name, result in summary.plugin_results.items():
                if not result.passed:
                    logger.warning(f"[EVAL] Plugin '{plugin_name}' FAILED: {result.errors}")

        return Ok(
            self.update_context(
                context,
                {
                    "eval_passed": summary.overall_passed,
                    "eval_summary": summary.to_dict(),
                    "evaluation_completed_at": time.time(),
                },
            )
        )

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    @staticmethod
    def _resolve_model_name(*, endpoint_url: str, fallback: str) -> str:
        """
        Ask /v1/models and return the first model id reported by the server.

        vLLM serves the merged model under its local directory path by default.
        The chat_inference.py script works around this by dynamically fetching
        the actual name — we do the same so eval requests don't get 404.

        Falls back to `fallback` (config.model.name) on any error.
        """
        import urllib.request

        try:
            # endpoint_url is either "http://host/v1" or "http://host" — normalise to /v1/models
            base = endpoint_url.rstrip("/")
            if not base.endswith("/v1"):
                base = base + "/v1"
            models_url = base + "/models"
            with urllib.request.urlopen(models_url, timeout=5) as resp:
                data = json.loads(resp.read().decode())
                models = data.get("data") or []
                if models:
                    resolved = str(models[0]["id"])
                    if resolved != fallback:
                        logger.info(f"[EVAL] Resolved model name from /v1/models: {resolved!r} (config: {fallback!r})")
                    return resolved
        except Exception as exc:
            logger.debug(f"[EVAL] /v1/models lookup failed, using fallback {fallback!r}: {exc}")
        return fallback

    @staticmethod
    def _build_mlflow_metrics(plugin_results: dict[str, Any]) -> dict[str, float]:
        """Flat plugin metrics dict — kept for backward compatibility with tests."""
        metrics: dict[str, float] = {}
        for plugin_name, result in plugin_results.items():
            for metric_key, value in result.metrics.items():
                if isinstance(value, int | float):
                    metrics[f"eval.{plugin_name}.{metric_key}"] = float(value)
            metrics[f"eval.{plugin_name}.passed"] = 1.0 if result.passed else 0.0
        return metrics

    def _load_system_prompt(self, context: dict) -> str | None:
        """
        Resolve system prompt via SystemPromptLoader.

        Reads InferenceLLMConfig from the active provider config and delegates
        all source-selection logic to SystemPromptLoader.load().

        Returns:
            System prompt text, or None if not configured / not resolvable.
        """
        try:
            provider_cfg_raw: dict = self.config.get_provider_config()
        except (KeyError, ValueError):
            return None

        from src.config.inference.common import InferenceLLMConfig
        from src.evaluation.system_prompt import SystemPromptLoader

        llm_raw = provider_cfg_raw.get("inference", {}).get("llm", {})
        if not isinstance(llm_raw, dict):
            return None
        llm_cfg = InferenceLLMConfig.model_validate(llm_raw)

        mlflow_cfg = getattr(getattr(self.config, "experiment_tracking", None), "mlflow", None)

        # Use gateway from mlflow_manager when available (provides timeout + URI normalization)
        mlflow_manager = context.get("mlflow_manager") if context else None
        gateway = getattr(mlflow_manager, "_gateway", None) if mlflow_manager is not None else None

        try:
            result = SystemPromptLoader.load(llm_cfg, mlflow_cfg=mlflow_cfg, gateway=gateway)
        except ValueError as exc:
            logger.error(f"[EVAL] System prompt configuration error: {exc}")
            return None

        if result:
            return result.text
        return None
