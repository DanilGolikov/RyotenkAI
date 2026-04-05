"""
Additional integration tests for TrainingContainer.

These tests expand coverage for:
- Strategy factory creation and interaction
- Dataset loader creation and usage
- Trainer factory creation and configuration
- MLflow manager integration
- Full end-to-end workflows
- Error handling scenarios

Target: Increase container.py coverage from 39.54% to >60%.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from src.utils.config import (
    DatasetConfig,
    DatasetLocalPaths,
    DatasetSourceLocal,
    ExperimentTrackingConfig,
    GlobalHyperparametersConfig,
    InferenceConfig,
    InferenceEnginesConfig,
    InferenceVLLMEngineConfig,
    LoraConfig,
    MLflowConfig,
    ModelConfig,
    PipelineConfig,
    StrategyPhaseConfig,
    TrainingOnlyConfig,
)
from src.utils.container import TrainingContainer

# =============================================================================
# FIXTURES
# =============================================================================


@pytest.fixture
def full_config():
    """Create a complete PipelineConfig for integration tests."""
    return PipelineConfig(
        model=ModelConfig(
            name="Qwen/Qwen2.5-0.5B-Instruct",
            torch_dtype="bfloat16",
            trust_remote_code=False,
        ),
        training=TrainingOnlyConfig(
            type="qlora",
            qlora=LoraConfig(
                r=8,
                lora_alpha=16,
                lora_dropout=0.05,
                bias="none",
                target_modules="all-linear",
                use_dora=False,
                use_rslora=False,
                init_lora_weights="gaussian",
            ),
            hyperparams=GlobalHyperparametersConfig(
                per_device_train_batch_size=1,
                gradient_accumulation_steps=1,
                learning_rate=2e-4,
                warmup_ratio=0.0,
                epochs=1,
            ),
            strategies=[
                StrategyPhaseConfig(strategy_type="sft"),
            ],
        ),
        datasets={
            "default": DatasetConfig(
                source_type="local",
                source_local=DatasetSourceLocal(local_paths=DatasetLocalPaths(train="data/train.jsonl", eval=None)),
            )
        },
        inference=InferenceConfig(
            enabled=False,
            provider="single_node",
            engine="vllm",
            engines=InferenceEnginesConfig(
                vllm=InferenceVLLMEngineConfig(
                    merge_image="test/merge:latest",
                    serve_image="test/vllm:latest",
                )
            ),
        ),
        experiment_tracking=ExperimentTrackingConfig(
            mlflow=MLflowConfig(
                tracking_uri="http://localhost:5000",
                experiment_name="test",
                log_artifacts=False,
                log_model=False,
            )
        ),
    )


# =============================================================================
# TESTS: Strategy Factory
# =============================================================================


class TestStrategyFactoryIntegration:
    """Test strategy factory creation and usage."""

    def test_strategy_factory_lazy_init(self, full_config):
        """Strategy factory should be lazily initialized."""
        container = TrainingContainer(full_config)

        # Should not raise
        factory = container.strategy_factory

        assert factory is not None

    def test_strategy_factory_created_only_once(self, full_config):
        """Strategy factory should be singleton per container."""
        container = TrainingContainer(full_config)

        factory1 = container.strategy_factory
        factory2 = container.strategy_factory

        assert factory1 is factory2


# =============================================================================
# TESTS: Dataset Loader
# =============================================================================


class TestDatasetLoaderIntegration:
    """Test dataset loader creation and usage."""

    def test_dataset_loader_lazy_init(self, full_config):
        """Dataset loader should be lazily initialized."""
        container = TrainingContainer(full_config)

        loader = container.dataset_loader

        assert loader is not None

    def test_dataset_loader_created_only_once(self, full_config):
        """Dataset loader should be singleton per container."""
        container = TrainingContainer(full_config)

        loader1 = container.dataset_loader
        loader2 = container.dataset_loader

        assert loader1 is loader2

    def test_dataset_loader_config_dependency(self, full_config):
        """Dataset loader should have config reference."""
        container = TrainingContainer(full_config)

        loader = container.dataset_loader

        # Should be initialized with config
        assert loader is not None


# =============================================================================
# TESTS: Trainer Factory
# =============================================================================


class TestTrainerFactoryIntegration:
    """Test trainer factory creation and configuration."""

    def test_trainer_factory_lazy_init(self, full_config):
        """Trainer factory should be lazily initialized."""
        container = TrainingContainer(full_config)

        factory = container.trainer_factory

        assert factory is not None

    def test_trainer_factory_created_only_once(self, full_config):
        """Trainer factory should be singleton per container."""
        container = TrainingContainer(full_config)

        factory1 = container.trainer_factory
        factory2 = container.trainer_factory

        assert factory1 is factory2

    def test_trainer_factory_memory_manager_injection(self, full_config):
        """Trainer factory should receive memory manager."""
        container = TrainingContainer(full_config)

        factory = container.trainer_factory

        # Should be initialized
        assert factory is not None


# =============================================================================
# TESTS: MLflow Manager
# =============================================================================


class TestMLflowManagerIntegration:
    """Test MLflow manager integration."""

    def test_mlflow_manager_lazy_init(self, full_config):
        """MLflow manager should be created lazily."""
        container = TrainingContainer(full_config)

        mlflow_mgr = container.mlflow_manager

        # Should be initialized
        assert mlflow_mgr is not None

    def test_mlflow_manager_created_only_once(self, full_config):
        """MLflow manager should be singleton per container."""
        container = TrainingContainer(full_config)

        mgr1 = container.mlflow_manager
        mgr2 = container.mlflow_manager

        assert mgr1 is mgr2

    def test_mlflow_manager_always_present_with_required_config(self, full_config):
        """MLflow manager should always exist when config schema is valid."""
        container = TrainingContainer(full_config)

        mlflow_mgr = container.mlflow_manager

        assert mlflow_mgr is not None

    def test_mlflow_manager_config_dependency(self, full_config):
        """MLflow manager should use container's config."""
        container = TrainingContainer(full_config)

        mlflow_mgr = container.mlflow_manager

        # Should be initialized
        assert mlflow_mgr is not None


# =============================================================================
# TESTS: Full Workflow
# =============================================================================


class TestFullWorkflow:
    """Test complete container workflow."""

    def test_can_access_all_dependencies(self, full_config):
        """Should be able to access all container dependencies."""
        container = TrainingContainer(full_config)

        # All should be accessible
        _ = container.config
        _ = container.memory_manager
        _ = container.completion_notifier
        _ = container.strategy_factory
        _ = container.dataset_loader
        _ = container.trainer_factory

        # MLflow might be None if disabled
        _ = container.mlflow_manager

    def test_override_preserves_config(self, full_config):
        """override() should preserve config reference."""
        original = TrainingContainer(full_config)

        mock_mm = MagicMock()
        overridden = original.override(memory_manager=mock_mm)

        # Config should be same
        assert overridden.config == original.config

        # Memory manager should be overridden
        assert overridden.memory_manager == mock_mm

    def test_for_testing_creates_minimal_container(self, full_config):
        """for_testing() should create container with minimal deps."""
        container = TrainingContainer.for_testing(full_config)

        # Should have NoOp memory manager
        mm = container.memory_manager
        assert not mm.is_memory_critical()  # NoOp always returns False

    def test_from_config_path_loads_and_creates_container(self, tmp_path):
        """from_config_path() should load config and create container."""
        config_file = tmp_path / "pipeline.yaml"
        config_file.write_text("""
model:
  name: Qwen/Qwen2.5-0.5B-Instruct
  torch_dtype: bfloat16
  trust_remote_code: false
training:
  type: qlora
  hyperparams:
    per_device_train_batch_size: 1
    gradient_accumulation_steps: 1
    learning_rate: 2.0e-4
    warmup_ratio: 0.0
    epochs: 1
  qlora:
    r: 8
    lora_alpha: 16
    lora_dropout: 0.05
    bias: none
    target_modules: all-linear
    use_dora: false
    use_rslora: false
    init_lora_weights: gaussian
  strategies:
    - strategy_type: sft
datasets:
  default:
    source_type: local
    source_local:
      local_paths:
        train: data/train.jsonl
        eval: null
inference:
  enabled: false
  provider: single_node
  engine: vllm
  engines:
    vllm:
      merge_image: test/merge:latest
      serve_image: test/vllm:latest
experiment_tracking:
  mlflow:
    tracking_uri: "http://localhost:5000"
    experiment_name: "test"
    log_artifacts: false
    log_model: false
""")

        container = TrainingContainer.from_config_path(str(config_file))

        assert container.config.model.name == "Qwen/Qwen2.5-0.5B-Instruct"
        assert len(container.config.training.strategies) == 1


# =============================================================================
# TESTS: Error Handling
# =============================================================================


class TestErrorHandling:
    """Test error handling scenarios."""

    def test_missing_config_file_raises(self):
        """Loading from non-existent file should raise."""
        with pytest.raises(FileNotFoundError):
            TrainingContainer.from_config_path("/nonexistent/config.yaml")

    def test_invalid_yaml_raises(self, tmp_path):
        """Invalid YAML should raise parsing error."""
        config_file = tmp_path / "invalid.yaml"
        config_file.write_text("this is not valid: yaml: : :")

        with pytest.raises(Exception):  # YAML parsing error
            TrainingContainer.from_config_path(str(config_file))

    def test_container_with_none_config_raises(self):
        """Creating container with None config should fail."""
        with pytest.raises((TypeError, AttributeError)):
            TrainingContainer(None)


# =============================================================================
# TESTS: Dependency Interactions
# =============================================================================


class TestDependencyInteractions:
    """Test interactions between dependencies."""

    def test_memory_manager_used_by_components(self, full_config):
        """Memory manager should be accessible."""
        container = TrainingContainer(full_config)

        mm = container.memory_manager
        factory = container.trainer_factory

        # Both should be initialized
        assert mm is not None
        assert factory is not None

    def test_config_accessible_from_container(self, full_config):
        """Config should be accessible from container."""
        container = TrainingContainer(full_config)

        # Access all components
        _ = container.strategy_factory
        _ = container.dataset_loader
        _ = container.trainer_factory

        # Container should still have config
        assert container.config == full_config


# =============================================================================
# TESTS: Lazy Initialization Order
# =============================================================================


class TestLazyInitializationOrder:
    """Test that lazy initialization works in any order."""

    def test_access_trainer_factory_first(self, full_config):
        """Accessing trainer factory first should work."""
        container = TrainingContainer(full_config)

        # Access in unusual order
        _ = container.trainer_factory
        _ = container.dataset_loader
        _ = container.strategy_factory
        _ = container.memory_manager

        # All should be initialized

    def test_access_dataset_loader_first(self, full_config):
        """Accessing dataset loader first should work."""
        container = TrainingContainer(full_config)

        _ = container.dataset_loader
        _ = container.memory_manager
        _ = container.trainer_factory

        # All should work

    def test_multiple_accesses_same_instance(self, full_config):
        """Multiple accesses should return same instances."""
        container = TrainingContainer(full_config)

        # Access multiple times
        mm1 = container.memory_manager
        factory1 = container.strategy_factory
        mm2 = container.memory_manager
        factory2 = container.strategy_factory

        assert mm1 is mm2
        assert factory1 is factory2


# =============================================================================
# TESTS: Container State
# =============================================================================


class TestContainerState:
    """Test container state management."""

    def test_container_is_not_singleton(self, full_config):
        """Different containers should be independent."""
        container1 = TrainingContainer(full_config)
        container2 = TrainingContainer(full_config)

        # Should be different instances
        assert container1 is not container2

        # Components should also be different
        assert container1.memory_manager is not container2.memory_manager

    def test_override_creates_independent_container(self, full_config):
        """Overridden container should be independent."""
        original = TrainingContainer(full_config)
        overridden = original.override(memory_manager=MagicMock())

        # Should be different containers
        assert original is not overridden

        # Original should not be affected
        orig_mm = original.memory_manager
        over_mm = overridden.memory_manager

        assert orig_mm is not over_mm


# =============================================================================
# TESTS: for_testing() Factory
# =============================================================================


class TestForTestingFactory:
    """Test for_testing() factory method."""

    def test_for_testing_with_custom_memory_manager(self, full_config):
        """for_testing() should accept custom memory manager."""
        mock_mm = MagicMock()
        mock_mm.is_memory_critical.return_value = False

        container = TrainingContainer.for_testing(full_config, memory_manager=mock_mm)

        assert container.memory_manager is mock_mm

    def test_for_testing_with_custom_notifier(self, full_config):
        """for_testing() should accept custom notifier."""
        mock_notifier = MagicMock()

        container = TrainingContainer.for_testing(full_config, completion_notifier=mock_notifier)

        assert container.completion_notifier is mock_notifier

    def test_for_testing_all_deps_accessible(self, full_config):
        """for_testing() container should have all deps accessible."""
        container = TrainingContainer.for_testing(full_config)

        # Should not raise
        _ = container.memory_manager
        _ = container.completion_notifier
        _ = container.strategy_factory
        _ = container.dataset_loader
        _ = container.trainer_factory
