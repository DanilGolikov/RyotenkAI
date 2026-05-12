"""
E2E tests for Pipeline Stages Integration.

Tests that stages work together correctly:
- Context propagation between stages
- Stage dependencies
- Error handling across stages
"""

from __future__ import annotations

from typing import Any, Callable

import pytest

from ryotenkai_shared.utils.result import Err, Ok, Result


class _FakeStage:
    """Tiny test double for an arbitrary pipeline stage callable.

    These E2E tests originally used ``MagicMock(spec=PipelineStage)`` and
    invoked ``stage.run(config, secrets, context)``, which is **not** the
    real :class:`PipelineStage.run(context)` signature. The fixtures were
    really testing a sequence-of-callable-stages workflow, not the
    ``PipelineStage`` ABC. This fake captures that intent without pretending
    to satisfy a class spec that the call shape doesn't actually match.

    WHY not a real stage subclass: these stages return ``Ok``/``Err`` with
    domain-specific payloads keyed by stage name (``runpod_deployer``,
    ``training_monitor`` etc.) and the test wires them by sharing a
    ``context`` dict. Building real stages would require provider configs,
    secrets, and pod fixtures that this E2E intentionally elides.
    """

    def __init__(
        self,
        stage_name: str,
        side_effect: Callable[[Any, Any, dict[str, Any]], Result[dict[str, Any], str]],
    ) -> None:
        self.stage_name = stage_name
        self._side_effect = side_effect
        self.call_count = 0

    def run(self, config: Any, secrets: Any, context: dict[str, Any]) -> Result[dict[str, Any], str]:
        self.call_count += 1
        return self._side_effect(config, secrets, context)

    def set_run_return(self, value: Result[dict[str, Any], str]) -> None:
        """Override run() to return the given value on subsequent calls.

        Used by error-path tests that flip a happy stage to a failing one.
        """
        self._side_effect = lambda _c, _s, _ctx: value

# =============================================================================
# FIXTURES
# =============================================================================


@pytest.fixture
def mock_dataset_validator() -> _FakeStage:
    """Create fake DatasetValidator stage."""

    def run_validator(config, secrets, context):  # noqa: ARG001
        return Ok(
            {
                "dataset_validator": {
                    "sample_count": 1000,
                    "avg_length": 150,
                    "empty_ratio": 0.01,
                    "diversity_score": 0.85,
                    "valid": True,
                }
            }
        )

    return _FakeStage("Dataset Validator", run_validator)


@pytest.fixture
def mock_runpod_deployer() -> _FakeStage:
    """Create fake RunPodDeployer stage."""

    def run_deployer(config, secrets, context):  # noqa: ARG001
        return Ok(
            {
                "runpod_deployer": {
                    "pod_id": "pod_test_123456",
                    "pod_ip": "185.1.2.3",
                    "ssh_port": 22169,
                    "gpu_type": "NVIDIA A40",
                    "status": "RUNNING",
                }
            }
        )

    return _FakeStage("RunPod Deployer", run_deployer)


@pytest.fixture
def mock_training_monitor() -> _FakeStage:
    """Create fake TrainingMonitor stage."""

    def run_monitor(config, secrets, context):  # noqa: ARG001
        pod_id = context.get("runpod_deployer", {}).get("pod_id")
        if not pod_id:
            return Err("No pod_id in context")

        return Ok(
            {
                "training_monitor": {
                    "pod_id": pod_id,
                    "training_complete": True,
                    "final_loss": 1.2,
                    "epochs_completed": 3,
                }
            }
        )

    return _FakeStage("Training Monitor", run_monitor)


@pytest.fixture
def mock_model_retriever() -> _FakeStage:
    """Create fake ModelRetriever stage."""

    def run_retriever(config, secrets, context):  # noqa: ARG001
        training_complete = context.get("training_monitor", {}).get("training_complete")
        if not training_complete:
            return Err("Training not complete")

        return Ok(
            {
                "model_retriever": {
                    "model_path": "/tmp/retrieved_model",
                    "model_size_gb": 14.5,
                    "retrieval_time_sec": 120,
                }
            }
        )

    return _FakeStage("Model Retriever", run_retriever)


@pytest.fixture
def mock_model_evaluator() -> _FakeStage:
    """Create fake ModelEvaluator stage."""

    def run_evaluator(config, secrets, context):  # noqa: ARG001
        model_path = context.get("model_retriever", {}).get("model_path")
        if not model_path:
            return Err("No model path in context")

        return Ok(
            {
                "model_evaluator": {
                    "model_path": model_path,
                    "perplexity": 3.5,
                    "bleu_score": 0.75,
                    "evaluation_passed": True,
                }
            }
        )

    return _FakeStage("Model Evaluator", run_evaluator)


# =============================================================================
# TEST CLASS: Context Propagation
# =============================================================================


@pytest.mark.e2e
@pytest.mark.integration
class TestContextPropagation:
    """Tests for context propagation between stages."""

    def test_context_flows_between_all_stages(
        self,
        mock_config,
        mock_secrets,
        mock_dataset_validator,
        mock_runpod_deployer,
        mock_training_monitor,
        mock_model_retriever,
        mock_model_evaluator,
    ):
        """
        Given: Full pipeline with 5 stages
        When: Pipeline runs
        Then: Context accumulates data from each stage
        """
        stages = [
            mock_dataset_validator,
            mock_runpod_deployer,
            mock_training_monitor,
            mock_model_retriever,
            mock_model_evaluator,
        ]

        context = {}

        # Run each stage and merge context
        for stage in stages:
            result = stage.run(mock_config, mock_secrets, context)
            assert result.is_success()
            context.update(result.unwrap())

        # Verify all stages contributed to context
        assert "dataset_validator" in context
        assert "runpod_deployer" in context
        assert "training_monitor" in context
        assert "model_retriever" in context
        assert "model_evaluator" in context

    def test_stage_uses_data_from_previous_stage(
        self,
        mock_config,
        mock_secrets,
        mock_runpod_deployer,
        mock_training_monitor,
    ):
        """
        Given: TrainingMonitor depends on RunPodDeployer
        When: TrainingMonitor runs after RunPodDeployer
        Then: TrainingMonitor uses pod_id from context
        """
        context = {}

        # Run deployer first
        deployer_result = mock_runpod_deployer.run(mock_config, mock_secrets, context)
        context.update(deployer_result.unwrap())

        # Run monitor
        monitor_result = mock_training_monitor.run(mock_config, mock_secrets, context)

        assert monitor_result.is_success()
        monitor_data = monitor_result.unwrap()["training_monitor"]
        assert monitor_data["pod_id"] == "pod_test_123456"

    def test_stage_fails_without_required_context(
        self,
        mock_config,
        mock_secrets,
        mock_training_monitor,
    ):
        """
        Given: TrainingMonitor without prior context
        When: TrainingMonitor runs
        Then: Returns error
        """
        context = {}  # Empty - no pod_id

        result = mock_training_monitor.run(mock_config, mock_secrets, context)

        assert result.is_failure()
        assert "pod_id" in result.error


# =============================================================================
# TEST CLASS: Stage Dependencies
# =============================================================================


@pytest.mark.e2e
@pytest.mark.integration
class TestStageDependencies:
    """Tests for stage dependency handling."""

    def test_dataset_validator_to_deployer(
        self,
        mock_config,
        mock_secrets,
        mock_dataset_validator,
        mock_runpod_deployer,
    ):
        """
        Given: DatasetValidator produces valid=True
        When: RunPodDeployer runs
        Then: Deployment proceeds
        """
        context = {}

        # Validate dataset
        val_result = mock_dataset_validator.run(mock_config, mock_secrets, context)
        context.update(val_result.unwrap())

        # Deploy (can proceed because dataset is valid)
        deploy_result = mock_runpod_deployer.run(mock_config, mock_secrets, context)

        assert deploy_result.is_success()

    def test_retriever_depends_on_training_complete(
        self,
        mock_config,
        mock_secrets,
        mock_model_retriever,
    ):
        """
        Given: Training not complete in context
        When: ModelRetriever runs
        Then: Returns error
        """
        context = {
            "training_monitor": {
                "training_complete": False,  # Not complete
            }
        }

        result = mock_model_retriever.run(mock_config, mock_secrets, context)

        assert result.is_failure()
        assert "not complete" in result.error.lower()

    def test_evaluator_depends_on_model_path(
        self,
        mock_config,
        mock_secrets,
        mock_model_evaluator,
    ):
        """
        Given: No model_path in context
        When: ModelEvaluator runs
        Then: Returns error
        """
        context = {}  # No model_retriever data

        result = mock_model_evaluator.run(mock_config, mock_secrets, context)

        assert result.is_failure()
        assert "model path" in result.error.lower()


# =============================================================================
# TEST CLASS: Error Handling Across Stages
# =============================================================================


@pytest.mark.e2e
@pytest.mark.integration
class TestErrorHandlingAcrossStages:
    """Tests for error handling across pipeline stages."""

    def test_early_stage_failure_stops_pipeline(
        self,
        mock_config,
        mock_secrets,
        mock_dataset_validator,
        mock_runpod_deployer,
    ):
        """
        Given: DatasetValidator fails
        When: Pipeline runs
        Then: Subsequent stages don't run after failure
        """
        # Make validator fail - override the side_effect
        mock_dataset_validator.set_run_return(Err("Dataset too small"))

        context = {}
        stages_run = []
        failed = False

        for stage in [mock_dataset_validator, mock_runpod_deployer]:
            result = stage.run(mock_config, mock_secrets, context)
            stages_run.append(stage.stage_name)

            if result.is_failure():
                failed = True
                break

            context.update(result.unwrap())

        # Pipeline should have stopped at first stage
        assert failed is True
        assert "Dataset Validator" in stages_run

    def test_middle_stage_failure_preserves_context(
        self,
        mock_config,
        mock_secrets,
        mock_dataset_validator,
        mock_runpod_deployer,
        mock_training_monitor,
    ):
        """
        Given: TrainingMonitor fails
        When: Pipeline runs
        Then: Context from successful stages is preserved
        """
        # Make monitor fail - override the side_effect
        mock_training_monitor.set_run_return(Err("Training timeout"))

        context = {}

        # Run stages
        val_result = mock_dataset_validator.run(mock_config, mock_secrets, context)
        context.update(val_result.unwrap())

        deploy_result = mock_runpod_deployer.run(mock_config, mock_secrets, context)
        context.update(deploy_result.unwrap())

        # This will fail
        monitor_result = mock_training_monitor.run(mock_config, mock_secrets, context)

        assert monitor_result.is_failure()
        # But previous context is preserved
        assert "dataset_validator" in context
        assert "runpod_deployer" in context

    def test_cleanup_called_on_failure(
        self,
        mock_config,
        mock_secrets,
        mock_runpod_deployer,
        mock_training_monitor,
    ):
        """
        Given: Stage fails after pod creation
        When: Error handling runs
        Then: Pod cleanup should be possible
        """
        context = {}

        # Deploy successfully
        deploy_result = mock_runpod_deployer.run(mock_config, mock_secrets, context)
        context.update(deploy_result.unwrap())

        # Training fails - override the side_effect
        mock_training_monitor.set_run_return(Err("Training OOM"))
        monitor_result = mock_training_monitor.run(mock_config, mock_secrets, context)

        assert monitor_result.is_failure()
        # Pod ID is in context for cleanup
        assert context["runpod_deployer"]["pod_id"] == "pod_test_123456"


# =============================================================================
# TEST CLASS: Stage Output Validation
# =============================================================================


@pytest.mark.e2e
@pytest.mark.integration
class TestStageOutputValidation:
    """Tests for stage output format and validation."""

    def test_dataset_validator_output_format(
        self,
        mock_config,
        mock_secrets,
        mock_dataset_validator,
    ):
        """
        Given: DatasetValidator runs
        When: Output is checked
        Then: Contains required metrics
        """
        result = mock_dataset_validator.run(mock_config, mock_secrets, {})

        assert result.is_success()
        data = result.unwrap()["dataset_validator"]

        assert "sample_count" in data
        assert "avg_length" in data
        assert "empty_ratio" in data
        assert "diversity_score" in data
        assert "valid" in data

    def test_runpod_deployer_output_format(
        self,
        mock_config,
        mock_secrets,
        mock_runpod_deployer,
    ):
        """
        Given: RunPodDeployer runs
        When: Output is checked
        Then: Contains required pod info
        """
        result = mock_runpod_deployer.run(mock_config, mock_secrets, {})

        assert result.is_success()
        data = result.unwrap()["runpod_deployer"]

        assert "pod_id" in data
        assert "pod_ip" in data
        assert "ssh_port" in data
        assert "status" in data

    def test_training_monitor_output_format(
        self,
        mock_config,
        mock_secrets,
        mock_training_monitor,
    ):
        """
        Given: TrainingMonitor runs with valid context
        When: Output is checked
        Then: Contains training status
        """
        context = {"runpod_deployer": {"pod_id": "pod_test_123"}}

        result = mock_training_monitor.run(mock_config, mock_secrets, context)

        assert result.is_success()
        data = result.unwrap()["training_monitor"]

        assert "training_complete" in data
        assert "final_loss" in data

    def test_model_evaluator_output_format(
        self,
        mock_config,
        mock_secrets,
        mock_model_evaluator,
    ):
        """
        Given: ModelEvaluator runs with valid context
        When: Output is checked
        Then: Contains evaluation metrics
        """
        context = {"model_retriever": {"model_path": "/tmp/model"}}

        result = mock_model_evaluator.run(mock_config, mock_secrets, context)

        assert result.is_success()
        data = result.unwrap()["model_evaluator"]

        assert "perplexity" in data
        assert "evaluation_passed" in data
