"""
Unit tests for TrainingContainer.

Tests:
- Container initialization
- Dependency injection
- Mock substitution for testing
- Interface compliance
"""

from __future__ import annotations

from contextlib import nullcontext
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from src.utils.container import (
    ICompletionNotifier,
    IMemoryManager,
    TrainingContainer,
)

# =============================================================================
# FIXTURES
# =============================================================================


@pytest.fixture
def mock_config():
    """Create a minimal mock PipelineConfig."""
    config = MagicMock()
    config.model.name = "test-model"
    config.training.type = "qlora"
    config.training.get_effective_load_in_4bit = MagicMock(return_value=True)
    config.training.lora = None
    config.training.get_strategy_chain.return_value = []
    return config


@pytest.fixture
def mock_memory_manager() -> IMemoryManager:
    """Create a mock MemoryManager."""

    class MockMemoryManager:
        """Mock memory manager for testing."""

        def __init__(self):
            self.operations: list[str] = []

        def get_memory_stats(self) -> None:
            return None

        def is_memory_critical(self) -> bool:
            return False

        def clear_cache(self) -> int:
            self.operations.append("clear_cache")
            return 100

        def aggressive_cleanup(self) -> int:
            self.operations.append("aggressive_cleanup")
            return 200

        def get_training_recommendations(self) -> dict[str, Any]:
            return {
                "gpu_name": "MockGPU",
                "gpu_tier": "testing",
                "total_vram_gb": 8,
            }

        def safe_operation(self, operation_name: str, context: dict[str, Any] | None = None):
            _ = context
            self.operations.append(f"safe_operation:{operation_name}")
            return nullcontext()

        @property
        def gpu_info(self):
            return None

        @property
        def preset(self):
            return None

    return MockMemoryManager()


@pytest.fixture
def mock_notifier() -> ICompletionNotifier:
    """Create a mock notifier."""

    class MockNotifier:
        """Mock notifier for testing."""

        def __init__(self):
            self.complete_calls: list[dict] = []
            self.failed_calls: list[tuple[str, dict]] = []

        def notify_complete(self, data: dict[str, Any]) -> None:
            self.complete_calls.append(data)

        def notify_failed(self, error: str, data: dict[str, Any]) -> None:
            self.failed_calls.append((error, data))

    return MockNotifier()


# =============================================================================
# TESTS: Container Initialization
# =============================================================================


class TestContainerInitialization:
    """Tests for container creation."""

    def test_create_container_with_config(self, mock_config):
        """Container should be created with config."""
        container = TrainingContainer(mock_config)

        assert container.config == mock_config

    def test_create_container_with_injected_memory_manager(self, mock_config, mock_memory_manager):
        """Container should use injected MemoryManager."""
        container = TrainingContainer(
            mock_config,
            _memory_manager=mock_memory_manager,
        )

        # Should return injected instance, not create new one
        mm = container.memory_manager
        assert mm == mock_memory_manager

    def test_for_testing_factory(self, mock_config, mock_memory_manager):
        """for_testing() should create container with test dependencies."""
        container = TrainingContainer.for_testing(
            mock_config,
            memory_manager=mock_memory_manager,
        )

        assert container.memory_manager == mock_memory_manager

    def test_from_config_path(self, tmp_path):
        """from_config_path() should load config from file."""
        # Create a minimal config file
        config_file = tmp_path / "test_config.yaml"
        config_file.write_text("""
model:
  name: Qwen/Qwen2.5-0.5B-Instruct
  tokenizer_name: Qwen/Qwen2.5-0.5B-Instruct
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
      dataset: default
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
""")

        container = TrainingContainer.from_config_path(str(config_file))

        assert container.config.model.name == "Qwen/Qwen2.5-0.5B-Instruct"


# =============================================================================
# TESTS: Dependency Injection
# =============================================================================


class TestDependencyInjection:
    """Tests for dependency injection behavior."""

    def test_memory_manager_lazy_init(self, mock_config):
        """MemoryManager should be lazily initialized."""
        with patch("src.utils.memory_manager.MemoryManager") as MockMM:
            mock_instance = MagicMock()
            MockMM.auto_configure.return_value = mock_instance

            container = TrainingContainer(mock_config)

            # Should not be created yet
            MockMM.auto_configure.assert_not_called()

            # Access triggers creation
            _ = container.memory_manager

            MockMM.auto_configure.assert_called_once()

    def test_injected_memory_manager_bypasses_creation(self, mock_config, mock_memory_manager):
        """Injected MemoryManager should bypass auto-configuration."""
        with patch("src.utils.memory_manager.MemoryManager") as MockMM:
            container = TrainingContainer(
                mock_config,
                _memory_manager=mock_memory_manager,
            )

            _ = container.memory_manager

            # Should NOT call auto_configure because we injected
            MockMM.auto_configure.assert_not_called()

    def test_completion_notifier_default(self, mock_config):
        """Default notifier should be MarkerFileNotifier."""
        container = TrainingContainer(mock_config)

        notifier = container.completion_notifier

        from src.training.notifiers.marker_file import MarkerFileNotifier

        assert isinstance(notifier, MarkerFileNotifier)

    def test_completion_notifier_injected(self, mock_config, mock_notifier):
        """Injected notifier should be used."""
        container = TrainingContainer(
            mock_config,
            _completion_notifier=mock_notifier,
        )

        assert container.completion_notifier == mock_notifier


# =============================================================================
# TESTS: Helper Methods
# =============================================================================


class TestHelperMethods:
    """Tests for container helper methods."""

    def test_override_creates_new_container(self, mock_config, mock_memory_manager, mock_notifier):
        """override() should create new container with overrides."""
        original = TrainingContainer(mock_config)

        overridden = original.override(
            memory_manager=mock_memory_manager,
            completion_notifier=mock_notifier,
        )

        # Should be different container
        assert overridden is not original

        # Overrides should be applied
        assert overridden.memory_manager == mock_memory_manager
        assert overridden.completion_notifier == mock_notifier


# =============================================================================
# TESTS: Interface Compliance
# =============================================================================


class TestInterfaceCompliance:
    """Tests for Protocol interface compliance."""

    def test_mock_memory_manager_is_imemorymanager(self, mock_memory_manager):
        """Mock should implement IMemoryManager."""
        assert isinstance(mock_memory_manager, IMemoryManager)

    def test_mock_notifier_is_icompletionnotifier(self, mock_notifier):
        """Mock should implement ICompletionNotifier."""
        assert isinstance(mock_notifier, ICompletionNotifier)

    def test_noop_memory_manager_is_imemorymanager(self, mock_config):
        """NoOpMemoryManager from for_testing should implement IMemoryManager."""
        container = TrainingContainer.for_testing(mock_config)

        mm = container.memory_manager
        assert isinstance(mm, IMemoryManager)


# =============================================================================
# TESTS: Integration with Notifier
# =============================================================================


class TestNotifierIntegration:
    """Tests for notifier usage."""

    def test_notifier_notify_complete(self, mock_config, mock_notifier):
        """Notifier should record complete calls."""
        container = TrainingContainer(
            mock_config,
            _completion_notifier=mock_notifier,
        )

        container.completion_notifier.notify_complete(
            {
                "output_path": "/tmp/model",
                "model_name": "test",
            }
        )

        assert len(mock_notifier.complete_calls) == 1
        assert mock_notifier.complete_calls[0]["output_path"] == "/tmp/model"

    def test_notifier_notify_failed(self, mock_config, mock_notifier):
        """Notifier should record failed calls."""
        container = TrainingContainer(
            mock_config,
            _completion_notifier=mock_notifier,
        )

        container.completion_notifier.notify_failed(
            "Test error",
            {"error_type": "TestError"},
        )

        assert len(mock_notifier.failed_calls) == 1
        assert mock_notifier.failed_calls[0][0] == "Test error"


# =============================================================================
# TESTS: For Testing Scenarios
# =============================================================================


class TestForTestingScenarios:
    """Tests demonstrating testing scenarios."""

    def test_can_test_train_v2_without_gpu(self, mock_config, mock_memory_manager):
        """Should be able to create container for testing without GPU."""
        container = TrainingContainer.for_testing(
            mock_config,
            memory_manager=mock_memory_manager,
        )

        # Memory manager should work without GPU
        recs = container.memory_manager.get_training_recommendations()
        assert "gpu_name" in recs
        assert not container.memory_manager.is_memory_critical()

    def test_safe_operation_works_with_mock(self, mock_config, mock_memory_manager):
        """safe_operation should work with mock."""
        container = TrainingContainer(
            mock_config,
            _memory_manager=mock_memory_manager,
        )

        # This should not raise
        with container.memory_manager.safe_operation("test_op"):
            pass

        assert "safe_operation:test_op" in mock_memory_manager.operations
