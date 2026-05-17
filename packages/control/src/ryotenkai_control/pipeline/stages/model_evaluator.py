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
from typing import TYPE_CHECKING, Any

import httpx

from ryotenkai_shared.errors import InferenceUnavailableError
from ryotenkai_shared.events import UNKNOWN_OFFSET
from ryotenkai_shared.events.types.control_evaluation import (
    EvaluationCompletedEvent,
    EvaluationCompletedPayload,
    EvaluationStartedEvent,
    EvaluationStartedPayload,
)
from ryotenkai_shared.utils.cancellation import sleep_cancellable
from ryotenkai_control.pipeline.stages.base import PipelineStage
from ryotenkai_control.pipeline.stages.constants import PipelineContextKeys, StageNames
from ryotenkai_shared.utils.logger import logger

# Pre-flight probe parameters. Strict by design: a healthy endpoint
# answers within ~1 s; two attempts are enough to absorb a single
# packet loss without spending real time on a long-poll.
_PREFLIGHT_TIMEOUT_S = 5.0
_PREFLIGHT_RETRIES = 1
_PREFLIGHT_RETRY_BACKOFF_S = 1.0

# Source URI for envelopes emitted from this stage.
_STAGE_SOURCE = "control://orchestrator/model_evaluator"

if TYPE_CHECKING:
    from ryotenkai_shared.config.secrets.model import Secrets
    from ryotenkai_shared.config import PipelineConfig
    from ryotenkai_shared.events import IEventEmitter


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
        *,
        emitter: IEventEmitter | None = None,
    ) -> None:
        super().__init__(config, StageNames.MODEL_EVALUATOR)
        self._emitter = emitter
        self._secrets = secrets

    def set_emitter(self, emitter: IEventEmitter) -> None:
        self._emitter = emitter

    @staticmethod
    def _resolve_run_id(context: dict[str, Any]) -> str:
        run_obj = context.get(PipelineContextKeys.RUN)
        run_name = getattr(run_obj, "name", None)
        if isinstance(run_name, str) and run_name:
            return run_name
        return "unknown"

    def execute(self, context: dict[str, Any]) -> dict[str, Any]:
        """Run plugin-based evaluation against the configured endpoint.

        Returns:
            Updated pipeline context.

        Raises:
            InferenceUnavailableError: when ``context['endpoint_url']`` is
                missing or the endpoint pre-flight probe fails.
        """
        eval_cfg = getattr(self.config, "evaluation", None)
        forced_stages = context.get(PipelineContextKeys.FORCED_STAGES, set())
        is_forced = self.stage_name in forced_stages if isinstance(forced_stages, set | list | tuple) else False

        if (eval_cfg is None or not getattr(eval_cfg, "enabled", False)) and not is_forced:
            return self.update_context(
                context,
                {
                    "evaluation_skipped": True,
                    "reason": "evaluation.enabled=false",
                },
            )

        # Read endpoint URL set by InferenceDeployer.
        # Data is stored under the stage name key (update_context convention),
        # so we look in context[StageNames.INFERENCE_DEPLOYER] first, then fall
        # back to the root for backwards-compatibility with tests that inject directly.
        inference_ctx = context.get(StageNames.INFERENCE_DEPLOYER, {})
        endpoint_url: str | None = inference_ctx.get("endpoint_url") or context.get("endpoint_url")
        if not endpoint_url:
            raise InferenceUnavailableError(
                detail=(
                    "evaluation requires a live inference endpoint, but "
                    "context['endpoint_url'] is missing — InferenceDeployer "
                    "either did not run or did not activate the endpoint"
                ),
                context={"legacy_code": "EVAL_ENDPOINT_MISSING"},
            )

        # Pre-flight: prove the endpoint is actually reachable before
        # firing N samples at it. activate_for_eval should have left it
        # healthy; this catches the gap between stages — pod auto-stop,
        # SSH tunnel torn down, network glitch. Without this, an
        # unreachable endpoint produces 31× Connection refused that the
        # eval code handles per-sample (empty answer), surfacing as
        # "successful" eval with garbage results.
        _preflight_check_endpoint(endpoint_url)

        # Model name for API requests.
        # vLLM serves the merged model under its directory path by default,
        # not under config.model.name. Mirror the chat_inference.py approach:
        # ask /v1/models and take the first id; fall back to config.model.name.
        static_model_name: str = (
            inference_ctx.get("inference_model_name") or context.get("inference_model_name") or self.config.model.name
        )
        model_name = self._resolve_model_name(endpoint_url=endpoint_url, fallback=static_model_name)
        # Engine kind comes from the discriminated union — no fallback
        # default. ``cfg.inference.engine.kind`` is the single source of truth.
        engine: str = self.config.inference.engine.kind

        logger.info(f"[EVAL] Starting evaluation — endpoint: {endpoint_url}, " f"model: {model_name}, engine: {engine}")
        start_time = time.time()

        # Build inference client via factory (DIP: depends on IModelInference, not concrete class)
        from ryotenkai_control.evaluation.model_client.factory import ModelClientFactory

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

        from ryotenkai_control.evaluation.plugins.secrets import SecretsResolver
        from ryotenkai_control.evaluation.runner import EvaluationRunner

        run_id = self._resolve_run_id(context)
        # Emit a single typed ``evaluation_started`` envelope before
        # the runner spins up; per-plugin envelopes are emitted from
        # within the runner via the shared emitter.
        if self._emitter is not None:
            plugin_names: list[str] = []
            try:
                plugin_names = [p.id for p in eval_cfg.evaluators.plugins]
            except Exception:
                plugin_names = []
            self._emitter.emit(
                EvaluationStartedEvent(
                    source=_STAGE_SOURCE,
                    run_id=run_id,
                    offset=UNKNOWN_OFFSET,
                    payload=EvaluationStartedPayload(
                        plugin_names=plugin_names,
                        model_path=str(model_name or ""),
                    ),
                ),
            )

        secrets_resolver = SecretsResolver(self._secrets) if self._secrets is not None else None
        runner = EvaluationRunner(
            cast("Any", eval_cfg),
            emitter=self._emitter,
            run_id=run_id,
            secrets_resolver=secrets_resolver,
        )
        summary = runner.run(inference_client)

        duration = time.time() - start_time
        logger.info(
            f"[EVAL] Evaluation complete in {duration:.1f}s — "
            f"passed={summary.overall_passed}, plugins={list(summary.plugin_results.keys())}"
        )

        # Typed evaluation_completed envelope: aggregated metrics +
        # total wall-clock duration.
        if self._emitter is not None:
            aggregated: dict[str, float] = {}
            for plugin_id, result in summary.plugin_results.items():
                for k, v in result.metrics.items():
                    if isinstance(v, (int, float)):
                        aggregated[f"{plugin_id}.{k}"] = float(v)
                aggregated[f"{plugin_id}.passed"] = 1.0 if result.passed else 0.0
            self._emitter.emit(
                EvaluationCompletedEvent(
                    source=_STAGE_SOURCE,
                    run_id=run_id,
                    offset=UNKNOWN_OFFSET,
                    payload=EvaluationCompletedPayload(
                        aggregated_metrics=aggregated,
                        total_duration_s=float(duration),
                    ),
                ),
            )

        verdict = "PASSED" if summary.overall_passed else "FAILED"
        logger.info(f"[EVAL] Overall verdict: {verdict}")
        if not summary.overall_passed:
            for plugin_name, result in summary.plugin_results.items():
                if not result.passed:
                    logger.warning(f"[EVAL] Plugin '{plugin_name}' FAILED: {result.errors}")

        return self.update_context(
            context,
            {
                "eval_passed": summary.overall_passed,
                "eval_summary": summary.to_dict(),
                "evaluation_completed_at": time.time(),
            },
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
            provider_cfg_raw = self.config.get_provider_config()
        except (KeyError, ValueError):
            return None

        from pydantic import BaseModel

        from ryotenkai_shared.config.inference.common import InferenceLLMConfig
        from ryotenkai_shared.infrastructure.mlflow.system_prompt import SystemPromptLoader

        # Provider block may be a typed Pydantic schema (post validator)
        # or a raw dict (modular runtime / tests).
        if isinstance(provider_cfg_raw, BaseModel):
            inference_obj = getattr(provider_cfg_raw, "inference", None)
            llm_obj = getattr(inference_obj, "llm", None) if inference_obj else None
            if isinstance(llm_obj, InferenceLLMConfig):
                llm_cfg = llm_obj
            elif isinstance(llm_obj, BaseModel):
                llm_cfg = InferenceLLMConfig.model_validate(llm_obj.model_dump(mode="json"))
            else:
                return None
        elif isinstance(provider_cfg_raw, dict):
            llm_raw = provider_cfg_raw.get("inference", {}).get("llm", {})
            if not isinstance(llm_raw, dict):
                return None
            llm_cfg = InferenceLLMConfig.model_validate(llm_raw)
        else:
            return None

        mlflow_cfg = getattr(getattr(self.config, "integrations", None), "mlflow", None)

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


# ---------------------------------------------------------------------------
# Pre-flight helpers
# ---------------------------------------------------------------------------


def _preflight_check_endpoint(
    endpoint_url: str,
    *,
    timeout_s: float = _PREFLIGHT_TIMEOUT_S,
    retries: int = _PREFLIGHT_RETRIES,
) -> None:
    """One-shot reachability probe: ``GET {endpoint_url}/models``.

    Two attempts (``1 + retries``) cover a single packet loss without
    dragging the wait into long-poll territory — providers must deliver
    a live endpoint by the time ``activate_for_eval`` returns Ok.

    Raises:
        InferenceUnavailableError: when all attempts fail or any 5xx.
    """
    base = endpoint_url.rstrip("/")
    url = f"{base}/models"
    last_err: str = "unknown"
    for attempt in range(retries + 1):
        try:
            response = httpx.get(url, timeout=timeout_s)
        except (httpx.ConnectError, httpx.TimeoutException) as exc:
            last_err = type(exc).__name__
        except httpx.HTTPError as exc:
            last_err = f"{type(exc).__name__}: {exc}"
        else:
            if response.status_code < 500:
                logger.info(
                    "[EVAL] pre-flight OK (%s → HTTP %d)",
                    url,
                    response.status_code,
                )
                return None
            last_err = f"HTTP {response.status_code}"
        if attempt < retries:
            sleep_cancellable(_PREFLIGHT_RETRY_BACKOFF_S)
    raise InferenceUnavailableError(
        detail=(
            f"inference endpoint pre-flight failed: GET {url} → {last_err} "
            f"(after {retries + 1} attempts). Endpoint is unreachable; "
            "evaluation aborted to avoid garbage results."
        ),
        context={"legacy_code": "EVAL_ENDPOINT_UNREACHABLE"},
    )
