"""
E2E tests for full pipeline execution.

Tests the complete pipeline flow:
DatasetValidator → RunPodDeployer → TrainingMonitor → ModelRetriever → [InferenceDeployer] → [ModelEvaluator]

All tests use mocks for GPU and RunPod - no real resources required.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.pipeline.orchestrator import PipelineOrchestrator
from src.pipeline.stages.base import PipelineStage
from src.utils.result import AppError, Err, Ok

# =============================================================================
# FIXTURES
# =============================================================================


@pytest.fixture
def mock_stages():
    """Create mock stages that succeed."""
    stages = []
    stage_names = [
        "Dataset Validator",
        "RunPod Deployer",
        "Training Monitor",
        "Model Retriever",
        "Model Evaluator",
    ]

    for name in stage_names:
        stage = MagicMock(spec=PipelineStage)
        stage.stage_name = name
        stage.run.return_value = Ok({name: {"status": "success"}})
        stages.append(stage)

    return stages


@pytest.fixture
def mock_orchestrator_with_stages(mock_config, mock_stages):
    """Create orchestrator with mock stages."""
    with (
        patch("src.pipeline.bootstrap.pipeline_bootstrap.load_config") as mock_load_config,
        patch("src.pipeline.bootstrap.pipeline_bootstrap.load_secrets") as mock_load_secrets,
    ):
        mock_load_config.return_value = mock_config
        mock_load_secrets.return_value = MagicMock()

        orchestrator = PipelineOrchestrator(Path("test_config.yaml"))
        orchestrator.stages = mock_stages

        return orchestrator


@pytest.fixture(autouse=True)
def bypass_mlflow_preflight(monkeypatch: pytest.MonkeyPatch) -> None:
    """E2E flow tests mock pipeline stages, so MLflow connectivity is out of scope here."""
    monkeypatch.setattr(
        PipelineOrchestrator,
        "_ensure_mlflow_preflight",
        lambda self, *, state: None,
    )


# =============================================================================
# TEST CLASS: Full Pipeline E2E
# =============================================================================


@pytest.mark.e2e
class TestFullPipelineE2E:
    """E2E tests for complete pipeline execution."""

    def test_pipeline_with_all_stages_success(self, mock_orchestrator_with_stages):
        """
        Given: All stages are configured to succeed
        When: Pipeline runs
        Then: All 5 stages execute successfully
        """
        orchestrator = mock_orchestrator_with_stages

        result = orchestrator.run()

        assert result.is_success()
        # Verify all stages were called
        for stage in orchestrator.stages:
            stage.run.assert_called_once()

    def test_pipeline_context_propagation(self, mock_orchestrator_with_stages):
        """
        Given: Pipeline with 5 stages
        When: Pipeline runs
        Then: Context is passed between stages
        """
        orchestrator = mock_orchestrator_with_stages

        result = orchestrator.run()

        assert result.is_success()
        context = result.unwrap()

        # Each stage should have added its data to context
        assert "Dataset Validator" in context
        assert "RunPod Deployer" in context
        assert "Training Monitor" in context
        assert "Model Retriever" in context
        assert "Model Evaluator" in context

    def test_pipeline_fails_on_first_stage_error(self, mock_orchestrator_with_stages):
        """
        Given: First stage (DatasetValidator) fails
        When: Pipeline runs
        Then: Pipeline stops immediately and returns error
        """
        orchestrator = mock_orchestrator_with_stages

        # Make first stage fail
        orchestrator.stages[0].run.return_value = Err(AppError(message="Dataset validation failed", code="TEST_ERROR"))

        result = orchestrator.run()

        assert result.is_failure()
        assert "Dataset Validator" in str(result.error)

        # Only first stage should be called
        orchestrator.stages[0].run.assert_called_once()
        orchestrator.stages[1].run.assert_not_called()

    def test_pipeline_fails_on_middle_stage_error(self, mock_orchestrator_with_stages):
        """
        Given: TrainingMonitor (stage 3) fails
        When: Pipeline runs
        Then: First 2 stages run, then pipeline stops
        """
        orchestrator = mock_orchestrator_with_stages

        # Make stage 3 (Training Monitor) fail
        orchestrator.stages[2].run.return_value = Err(AppError(message="Training failed", code="TEST_ERROR"))

        result = orchestrator.run()

        assert result.is_failure()
        assert "Training Monitor" in str(result.error)

        # First 3 stages called, rest not called
        orchestrator.stages[0].run.assert_called_once()
        orchestrator.stages[1].run.assert_called_once()
        orchestrator.stages[2].run.assert_called_once()
        orchestrator.stages[3].run.assert_not_called()

# =============================================================================
# TEST CLASS: Pipeline Cleanup
# =============================================================================


@pytest.mark.e2e
class TestPipelineCleanup:
    """Tests for pipeline cleanup on error."""

    def test_cleanup_called_on_error(self, mock_orchestrator_with_stages):
        """
        Given: Pipeline with RunPod pod created
        When: A stage fails
        Then: Cleanup is called to terminate pod
        """
        orchestrator = mock_orchestrator_with_stages

        # Simulate pod creation in stage 1
        orchestrator.stages[1].run.return_value = Ok(
            {"RunPod Deployer": {"pod_id": "pod_test_123", "status": "success"}}
        )

        # Make stage 3 fail
        orchestrator.stages[2].run.return_value = Err("Training failed")

        # Mock cleanup
        with patch.object(orchestrator, "_cleanup_resources") as mock_cleanup:
            result = orchestrator.run()

        assert result.is_failure()
        mock_cleanup.assert_called_once()

    def test_cleanup_called_on_success(self, mock_orchestrator_with_stages):
        """
        Given: Pipeline completes successfully
        When: All stages succeed
        Then: Cleanup is still called (in finally block)
        """
        orchestrator = mock_orchestrator_with_stages

        with patch.object(orchestrator, "_cleanup_resources") as mock_cleanup:
            result = orchestrator.run()

        assert result.is_success()
        mock_cleanup.assert_called_once()


# =============================================================================
# TEST CLASS: Stage Discovery
# =============================================================================


@pytest.mark.e2e
class TestStageDiscovery:
    """Tests for stage listing and discovery."""

    def test_list_stages(self, mock_orchestrator_with_stages):
        """
        Given: Orchestrator with stages
        When: list_stages() is called
        Then: Returns list of stage names
        """
        orchestrator = mock_orchestrator_with_stages

        stages = orchestrator.list_stages()

        assert len(stages) == 5
        assert "Dataset Validator" in stages
        assert "RunPod Deployer" in stages
        assert "Training Monitor" in stages
        assert "Model Retriever" in stages
        assert "Model Evaluator" in stages

    def test_get_stage_by_name(self, mock_orchestrator_with_stages):
        """
        Given: Orchestrator with stages
        When: get_stage_by_name() is called
        Then: Returns correct stage
        """
        orchestrator = mock_orchestrator_with_stages

        stage = orchestrator.get_stage_by_name("Training Monitor")

        assert stage is not None
        assert stage.stage_name == "Training Monitor"

    def test_get_stage_by_name_not_found(self, mock_orchestrator_with_stages):
        """
        Given: Orchestrator with stages
        When: get_stage_by_name() with invalid name
        Then: Returns None
        """
        orchestrator = mock_orchestrator_with_stages

        stage = orchestrator.get_stage_by_name("NonExistent Stage")

        assert stage is None


# =============================================================================
# TEST CLASS: Real Config Loading (Integration)
# =============================================================================


@pytest.mark.e2e
@pytest.mark.integration
class TestRealConfigLoading:
    """Tests with real config file loading."""

    def test_load_test_config(self, test_fixtures_dir):
        """
        Given: Test config file
        When: Orchestrator loads config
        Then: Config is loaded correctly
        """
        config_path = test_fixtures_dir / "configs" / "test_pipeline.yaml"

        if not config_path.exists():
            pytest.skip("Test config file not found")

        with patch("src.pipeline.bootstrap.pipeline_bootstrap.load_secrets") as mock_secrets:
            mock_secrets.return_value = MagicMock()

            orchestrator = PipelineOrchestrator(config_path)

            assert orchestrator.config is not None
            provider_config = orchestrator.config.get_provider_config()
            # provider_config is a dict
            mock_mode = (provider_config.get("training") or {}).get("mock_mode", False)
            assert mock_mode is True
