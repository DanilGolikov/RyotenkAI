"""End-to-end combinatorial tests covering the deploy → activate → eval
chain, demonstrating that the new fail-fast behaviour stops the
pipeline before garbage results can be produced.

Matrix (deploy, activate, preflight) → expected stage outcomes:

| deploy | activate | preflight | InferenceDeployer | ModelEvaluator |
|--------|----------|-----------|-------------------|----------------|
| Ok     | Err      | n/a       | Err(ACTIVATION)   | NOT INVOKED    |  ← original bug (silent garbage) is gone
| Ok     | Ok       | Err       | Ok                | Err(UNREACHABLE)|  ← gap-between-stages catch
| Ok     | Ok       | Ok        | Ok                | Ok              |  ← happy path
"""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from src.pipeline.stages.constants import StageNames
from src.pipeline.stages.inference_deployer import InferenceDeployer
from src.pipeline.stages.model_evaluator import ModelEvaluator
from src.pipeline.state import RunContext
from src.providers.inference.interfaces import (
    EndpointInfo,
    InferenceCapabilities,
)
from src.utils.config import Secrets
from src.utils.result import Failure, InferenceError, Ok, Success


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------


class _FakeInferenceProvider:
    """Configurable fake. Each test wires the failure modes it cares about."""

    def __init__(
        self,
        *,
        deploy_result=None,
        activate_result=None,
        deactivate_result=None,
        capabilities=None,
    ):
        self._deploy_result = deploy_result or Ok(EndpointInfo(
            endpoint_url=None,  # tunneled provider, real URL via activate_for_eval
            api_type="openai_compatible",
            provider_type="fake",
            engine="vllm",
            model_id="test-model/test-llm",
            resource_id="pod-x",
        ))
        self._activate_result = activate_result or Ok("http://127.0.0.1:8000/v1")
        self._deactivate_result = deactivate_result or Ok(None)
        self._capabilities = capabilities or InferenceCapabilities(
            provider_type="fake",
            supported_engines=["vllm"],
            supports_activate_for_eval=True,
        )
        self.activate_calls = 0
        self.deactivate_calls = 0

    @property
    def provider_name(self) -> str:
        return "fake"

    @property
    def provider_type(self) -> str:
        return "fake"

    def deploy(self, model_source: str, **kwargs):
        return self._deploy_result

    def set_event_logger(self, _logger) -> None:
        return None

    def get_pipeline_readiness_mode(self):
        from src.providers.inference.interfaces import PipelineReadinessMode
        return PipelineReadinessMode.SKIP

    def collect_startup_logs(self, *, local_path: Path) -> None:
        return None

    def build_inference_artifacts(self, *, ctx):
        from src.providers.inference.interfaces import InferenceArtifacts
        return Ok(InferenceArtifacts(
            manifest={"endpoint": {"client_base_url": "http://127.0.0.1:8000/v1"}},
            chat_script="# stub\n",
            readme="stub",
        ))

    def undeploy(self):
        return Ok(None)

    def health_check(self):
        return Ok(True)

    def get_capabilities(self):
        return self._capabilities

    def get_endpoint_info(self):
        return None

    def activate_for_eval(self):
        self.activate_calls += 1
        return self._activate_result

    def deactivate_after_eval(self):
        self.deactivate_calls += 1
        return self._deactivate_result


def _mk_inference_cfg(*, eval_enabled: bool = True) -> MagicMock:
    cfg = MagicMock()
    cfg.inference.enabled = True
    cfg.inference.provider = "fake"
    cfg.inference.engine = "vllm"
    cfg.inference.common.health_check.enabled = False
    cfg.inference.common.lora.adapter_path = "auto"
    cfg.inference.common.keep_inference_after_eval = False
    cfg.inference.common.model_source = "auto"
    cfg.inference.engines.vllm.quantization = None
    cfg.evaluation = MagicMock()
    cfg.evaluation.enabled = eval_enabled
    cfg.model.name = "test-model"
    cfg.model.trust_remote_code = False
    cfg.integrations.mlflow = None
    return cfg


def _mk_eval_cfg() -> MagicMock:
    cfg = MagicMock()
    cfg.model.name = "test-model"
    cfg.evaluation = MagicMock()
    cfg.evaluation.enabled = True
    cfg.inference.engine = "vllm"
    cfg.integrations.mlflow = None
    return cfg


def _initial_context() -> dict[str, Any]:
    return {
        StageNames.MODEL_RETRIEVER: {"hf_repo_id": "test/repo"},
        "mlflow_parent_run_id": None,
        "run": RunContext(
            name="run_test",
            created_at_utc=datetime(2026, 1, 20, 12, 34, 56, tzinfo=timezone.utc),
        ),
    }


# ---------------------------------------------------------------------------
# Combinatorial cases
# ---------------------------------------------------------------------------


def test_deploy_ok_activate_err_aborts_pipeline_before_eval(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The original bug repro: activate_for_eval fails → pipeline must
    STOP at InferenceDeployer. ModelEvaluator never runs, so no garbage."""
    fake = _FakeInferenceProvider(
        activate_result=Failure(InferenceError(
            message="POD_SSH_READY_TIMEOUT", code="RUNPOD_EVAL_ACTIVATE_FAILED",
        )),
    )

    import src.pipeline.stages.inference_deployer as deployer_mod
    monkeypatch.setattr(
        deployer_mod.InferenceProviderFactory, "create",
        lambda **kwargs: Success(fake),
    )
    monkeypatch.setattr(deployer_mod, "get_run_log_dir", lambda: tmp_path)
    monkeypatch.setattr(deployer_mod.time, "sleep", lambda s: None)

    deployer = InferenceDeployer(_mk_inference_cfg(), Secrets(HF_TOKEN="hf_test"))
    res = deployer.execute(_initial_context())

    assert res.is_failure()
    assert res.unwrap_err().code == "INFERENCE_ACTIVATION_FAILED"
    # Inline cleanup must have run (D5 of plan)
    assert fake.deactivate_calls == 1


def test_deploy_ok_activate_ok_preflight_err_stops_evaluator(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Endpoint died between activate and eval (pod auto-stopped, network
    glitch). Pre-flight catches it → evaluator returns Err, no garbage."""
    import httpx as _httpx

    fake = _FakeInferenceProvider()  # all defaults = success
    import src.pipeline.stages.inference_deployer as deployer_mod
    monkeypatch.setattr(
        deployer_mod.InferenceProviderFactory, "create",
        lambda **kwargs: Success(fake),
    )
    monkeypatch.setattr(deployer_mod, "get_run_log_dir", lambda: tmp_path)
    monkeypatch.setattr(deployer_mod.time, "sleep", lambda s: None)

    deploy_res = InferenceDeployer(
        _mk_inference_cfg(), Secrets(HF_TOKEN="hf_test"),
    ).execute(_initial_context())

    assert deploy_res.is_success()
    out = deploy_res.unwrap()[StageNames.INFERENCE_DEPLOYER]
    # Real URL flowed from activate_for_eval into context
    assert out["endpoint_url"] == "http://127.0.0.1:8000/v1"

    # Now run evaluator — endpoint dies between stages.
    evaluator = ModelEvaluator(_mk_eval_cfg())
    with patch("src.pipeline.stages.model_evaluator.httpx.get") as mock_get, \
         patch("src.pipeline.stages.model_evaluator.time.sleep"), \
         patch("src.evaluation.runner.EvaluationRunner") as MockRunner:
        mock_get.side_effect = _httpx.ConnectError("connection refused")
        eval_res = evaluator.execute(deploy_res.unwrap())
        # The eval runner must NOT have been instantiated.
        MockRunner.assert_not_called()

    assert eval_res.is_err()
    assert eval_res.unwrap_err().code == "EVAL_ENDPOINT_UNREACHABLE"


def test_deploy_ok_activate_ok_preflight_ok_runs_evaluation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Happy path: all three steps succeed, eval actually runs."""
    fake = _FakeInferenceProvider()
    import src.pipeline.stages.inference_deployer as deployer_mod
    monkeypatch.setattr(
        deployer_mod.InferenceProviderFactory, "create",
        lambda **kwargs: Success(fake),
    )
    monkeypatch.setattr(deployer_mod, "get_run_log_dir", lambda: tmp_path)
    monkeypatch.setattr(deployer_mod.time, "sleep", lambda s: None)

    deploy_res = InferenceDeployer(
        _mk_inference_cfg(), Secrets(HF_TOKEN="hf_test"),
    ).execute(_initial_context())
    assert deploy_res.is_success()

    from src.evaluation.runner import RunSummary

    summary = RunSummary(overall_passed=True, sample_count=10, duration_seconds=1.0)
    summary.plugin_results = {}

    evaluator = ModelEvaluator(_mk_eval_cfg())
    with patch("src.pipeline.stages.model_evaluator._preflight_check_endpoint",
               return_value=Ok(None)), \
         patch("src.evaluation.runner.EvaluationRunner") as MockRunner, \
         patch(
             "src.evaluation.model_client.factory.ModelClientFactory.create",
             return_value=MagicMock(),
         ):
        mock_runner = MagicMock()
        mock_runner.run.return_value = summary
        MockRunner.return_value = mock_runner
        eval_res = evaluator.execute(deploy_res.unwrap())

    assert eval_res.is_success()
    out = eval_res.unwrap()[StageNames.MODEL_EVALUATOR]
    assert out["eval_passed"] is True


def test_provider_without_capability_aborts_at_inference_deployer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Provider declares supports_activate_for_eval=False + eval enabled
    → InferenceDeployer Err(INFERENCE_EVAL_NOT_SUPPORTED) without any
    activate_for_eval call."""
    fake = _FakeInferenceProvider(
        capabilities=InferenceCapabilities(
            provider_type="hosted_only",
            supported_engines=["vllm"],
            supports_activate_for_eval=False,
        ),
    )

    import src.pipeline.stages.inference_deployer as deployer_mod
    monkeypatch.setattr(
        deployer_mod.InferenceProviderFactory, "create",
        lambda **kwargs: Success(fake),
    )
    monkeypatch.setattr(deployer_mod, "get_run_log_dir", lambda: tmp_path)
    monkeypatch.setattr(deployer_mod.time, "sleep", lambda s: None)

    res = InferenceDeployer(
        _mk_inference_cfg(), Secrets(HF_TOKEN="hf_test"),
    ).execute(_initial_context())

    assert res.is_failure()
    assert res.unwrap_err().code == "INFERENCE_EVAL_NOT_SUPPORTED"
    # activate_for_eval was never invoked — fail-fast on capability check.
    assert fake.activate_calls == 0
