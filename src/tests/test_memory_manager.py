"""
Tests for MemoryManager - OOM protection utilities.

Tests cover:
- Memory stats retrieval
- Cache clearing
- Safe operation context manager
- OOM retry decorator
- GPU detection and classification
- GPU presets and auto-configuration
"""

from unittest.mock import patch

import pytest

from src.utils.memory_manager import (
    GPUInfo,
    GPUPreset,
    GPUTier,
    MemoryManager,
    MemoryStats,
    OOMRecoverableError,
    get_memory_manager,
    reset_memory_manager,
)


class TestMemoryStats:
    """Test MemoryStats dataclass."""

    def test_critical_threshold(self):
        """Test critical memory detection at >90%."""
        stats = MemoryStats(total_mb=8000, free_mb=500, used_mb=7500, utilization_percent=93.75)
        assert stats.is_critical is True
        assert stats.is_warning is True

    def test_warning_threshold(self):
        """Test warning memory detection at >80%."""
        stats = MemoryStats(total_mb=8000, free_mb=1500, used_mb=6500, utilization_percent=81.25)
        assert stats.is_critical is False
        assert stats.is_warning is True

    def test_normal_memory(self):
        """Test normal memory levels."""
        stats = MemoryStats(total_mb=8000, free_mb=4000, used_mb=4000, utilization_percent=50.0)
        assert stats.is_critical is False
        assert stats.is_warning is False


class TestMemoryManager:
    """Test MemoryManager core functionality."""

    def test_init_defaults(self):
        """Test default initialization."""
        mm = MemoryManager()
        assert mm.memory_margin_mb == MemoryManager.DEFAULT_MEMORY_MARGIN_MB
        assert mm.max_retries == MemoryManager.DEFAULT_MAX_RETRIES
        assert mm.device == "cuda"

    def test_init_custom_params(self):
        """Test custom initialization."""
        mm = MemoryManager(device="cuda:1", memory_margin_mb=1000, max_retries=5)
        assert mm.device == "cuda:1"
        assert mm.memory_margin_mb == 1000
        assert mm.max_retries == 5

    @patch("src.utils.memory_manager.MemoryManager.cuda_available", False)
    def test_get_memory_stats_no_cuda(self):
        """Test memory stats when CUDA unavailable."""
        mm = MemoryManager()
        mm._cuda_available = False
        stats = mm.get_memory_stats()
        assert stats is None

    @patch("src.utils.memory_manager.MemoryManager.cuda_available", True)
    def test_get_available_memory_no_cuda(self):
        """Test available memory returns 0 when no CUDA."""
        mm = MemoryManager()
        mm._cuda_available = False
        assert mm.get_available_memory_mb() == 0

    def test_clear_cache_runs_gc(self):
        """Test cache clear triggers garbage collection."""
        mm = MemoryManager()
        mm._cuda_available = False  # Skip CUDA operations

        # Should not raise
        freed = mm.clear_cache()
        assert isinstance(freed, int)

    def test_aggressive_cleanup(self):
        """Test aggressive cleanup."""
        mm = MemoryManager()
        mm._cuda_available = False

        freed = mm.aggressive_cleanup()
        assert isinstance(freed, int)

    def test_checkpoint_tracking(self):
        """Test checkpoint creation."""
        mm = MemoryManager()
        mm._cuda_available = False

        mm.checkpoint("step_1")
        mm.checkpoint("step_2")

        assert len(mm._checkpoints) == 2
        assert "step_1" in mm._checkpoints
        assert "step_2" in mm._checkpoints


class TestOOMRecoverableError:
    """Test OOM error class."""

    def test_error_message(self):
        """Test error contains operation name."""
        error = OOMRecoverableError("training_step")
        assert "training_step" in str(error)
        assert error.operation == "training_step"

    def test_error_with_memory_info(self):
        """Test error includes memory info."""
        error = OOMRecoverableError("forward", {"free_mb": 100, "used_mb": 7900})
        assert "forward" in str(error)
        assert error.memory_info["free_mb"] == 100


class TestSafeOperation:
    """Test safe_operation context manager."""

    def test_successful_operation(self):
        """Test normal operation completes."""
        mm = MemoryManager()
        mm._cuda_available = False

        result = None
        with mm.safe_operation("test_op"):
            result = 42

        assert result == 42

    def test_oom_converted_to_recoverable(self):
        """Test OOM RuntimeError converted to OOMRecoverableError."""
        mm = MemoryManager()
        mm._cuda_available = False

        with pytest.raises(OOMRecoverableError) as exc_info, mm.safe_operation("failing_op"):
            raise RuntimeError("CUDA out of memory")

        assert exc_info.value.operation == "failing_op"

    def test_non_oom_error_propagates(self):
        """Test non-OOM errors propagate unchanged."""
        mm = MemoryManager()
        mm._cuda_available = False

        with pytest.raises(ValueError), mm.safe_operation("test_op"):
            raise ValueError("Not an OOM error")


class TestMemoryProtectionDecorator:
    """Test with_memory_protection decorator."""

    def test_successful_call(self):
        """Test decorated function works normally."""
        mm = MemoryManager()
        mm._cuda_available = False

        @mm.with_memory_protection("test_func")
        def add(a, b):
            return a + b

        assert add(2, 3) == 5

    def test_retry_on_oom(self):
        """Test automatic retry on OOM."""
        mm = MemoryManager(max_retries=3)
        mm._cuda_available = False

        call_count = 0

        @mm.with_memory_protection("retry_test")
        def flaky_function():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise RuntimeError("CUDA out of memory")
            return "success"

        result = flaky_function()
        assert result == "success"
        assert call_count == 3

    def test_exhausted_retries(self):
        """Test error raised after all retries exhausted."""
        mm = MemoryManager(max_retries=2)
        mm._cuda_available = False

        @mm.with_memory_protection("always_fail")
        def always_fails():
            raise RuntimeError("CUDA out of memory")

        with pytest.raises(OOMRecoverableError):
            always_fails()


class TestBatchSizeEstimation:
    """Test memory estimation utilities."""

    def test_estimate_batch_memory(self):
        """Test batch memory estimation."""
        mm = MemoryManager()

        # 7B model, batch 4, seq 2048, FP16
        estimate = mm.estimate_batch_memory(
            model_params=7_000_000_000,
            batch_size=4,
            seq_length=2048,
            dtype_bytes=2,
        )

        # Should be substantial for 7B model
        assert estimate > 10000  # >10GB estimate


class TestGPUTier:
    """Test GPU tier classification."""

    def test_tier_values(self):
        """Test all tier values exist."""
        assert GPUTier.MINIMAL.value == "minimal"
        assert GPUTier.CONSUMER_LOW.value == "consumer_low"
        assert GPUTier.CONSUMER_MID.value == "consumer_mid"
        assert GPUTier.CONSUMER_HIGH.value == "consumer_high"
        assert GPUTier.PROFESSIONAL.value == "professional"
        assert GPUTier.DATACENTER.value == "datacenter"


class TestGPUInfo:
    """Test GPUInfo dataclass."""

    def test_gpu_info_creation(self):
        """Test GPUInfo creation."""
        info = GPUInfo(
            name="NVIDIA GeForce RTX 4060",
            total_memory_mb=8192,
            tier=GPUTier.CONSUMER_LOW,
        )
        assert info.name == "NVIDIA GeForce RTX 4060"
        assert info.total_memory_mb == 8192
        assert info.tier == GPUTier.CONSUMER_LOW

    def test_total_memory_gb(self):
        """Test GB conversion."""
        info = GPUInfo(name="Test", total_memory_mb=24576, tier=GPUTier.CONSUMER_HIGH)
        assert info.total_memory_gb == 24.0

    def test_str_representation(self):
        """Test string representation."""
        info = GPUInfo(name="RTX 4090", total_memory_mb=24576, tier=GPUTier.CONSUMER_HIGH)
        assert "RTX 4090" in str(info)
        assert "24.0GB" in str(info)


class TestGPUPreset:
    """Test GPU presets."""

    def test_preset_for_minimal(self):
        """Test preset for minimal GPUs."""
        preset = GPUPreset.for_tier(GPUTier.MINIMAL)
        assert preset.memory_margin_mb == 300
        assert preset.max_retries == 5  # More retries for constrained GPUs

    def test_preset_for_consumer_low(self):
        """Test preset for 8GB GPUs."""
        preset = GPUPreset.for_tier(GPUTier.CONSUMER_LOW)
        assert preset.memory_margin_mb == 500
        assert preset.critical_threshold == 90.0

    def test_preset_for_datacenter(self):
        """Test preset for datacenter GPUs."""
        preset = GPUPreset.for_tier(GPUTier.DATACENTER)
        assert preset.memory_margin_mb == 4000
        assert preset.critical_threshold == 95.0  # Higher threshold for big GPUs


class TestGPUClassification:
    """Test GPU classification logic."""

    def test_classify_rtx_4060(self):
        """Test RTX 4060 classification."""
        tier = MemoryManager._classify_gpu_by_name("NVIDIA GeForce RTX 4060")
        assert tier == GPUTier.CONSUMER_LOW

    def test_classify_rtx_4090(self):
        """Test RTX 4090 classification."""
        tier = MemoryManager._classify_gpu_by_name("NVIDIA GeForce RTX 4090")
        assert tier == GPUTier.CONSUMER_HIGH

    def test_classify_a100(self):
        """Test A100 classification."""
        tier = MemoryManager._classify_gpu_by_name("NVIDIA A100-SXM4-80GB")
        assert tier == GPUTier.DATACENTER

    def test_classify_h100(self):
        """Test H100 classification."""
        tier = MemoryManager._classify_gpu_by_name("NVIDIA H100 PCIe")
        assert tier == GPUTier.DATACENTER

    def test_classify_a6000(self):
        """Test A6000 classification."""
        tier = MemoryManager._classify_gpu_by_name("NVIDIA RTX A6000")
        assert tier == GPUTier.PROFESSIONAL

    def test_classify_by_memory_8gb(self):
        """Test classification by memory - 8GB."""
        tier = MemoryManager._classify_gpu_by_memory(8192)
        assert tier == GPUTier.CONSUMER_LOW

    def test_classify_by_memory_24gb(self):
        """Test classification by memory - 24GB."""
        tier = MemoryManager._classify_gpu_by_memory(24576)
        assert tier == GPUTier.CONSUMER_HIGH

    def test_classify_by_memory_80gb(self):
        """Test classification by memory - 80GB."""
        tier = MemoryManager._classify_gpu_by_memory(81920)
        assert tier == GPUTier.DATACENTER

    def test_classify_unknown_gpu(self):
        """Test unknown GPU falls back."""
        tier = MemoryManager._classify_gpu_by_name("Unknown GPU XYZ-9000")
        assert tier == GPUTier.UNKNOWN


class TestAutoConfiguration:
    """Test auto-configuration functionality."""

    def test_for_gpu_rtx4060(self):
        """Test creating manager for specific GPU."""
        mm = MemoryManager.for_gpu("RTX 4060")
        assert mm.memory_margin_mb == 500
        assert mm.max_retries == 3

    def test_for_gpu_a100(self):
        """Test creating manager for A100."""
        mm = MemoryManager.for_gpu("A100")
        assert mm.memory_margin_mb == 4000
        assert mm.max_retries == 2  # Fewer retries needed

    def test_custom_thresholds(self):
        """Test custom threshold override."""
        mm = MemoryManager(memory_margin_mb=1000, max_retries=5)
        assert mm.memory_margin_mb == 1000
        assert mm.max_retries == 5


class TestSingleton:
    """Test singleton pattern."""

    def test_get_memory_manager_returns_instance(self):
        """Test singleton getter works."""
        reset_memory_manager()  # Ensure clean state
        mm = get_memory_manager(auto_configure=False)
        assert isinstance(mm, MemoryManager)

    def test_get_memory_manager_returns_same_instance(self):
        """Test singleton returns same instance."""
        reset_memory_manager()
        mm1 = get_memory_manager(auto_configure=False)
        mm2 = get_memory_manager(auto_configure=False)
        assert mm1 is mm2

    def test_reset_memory_manager(self):
        """Test singleton reset."""
        reset_memory_manager()
        mm1 = get_memory_manager(auto_configure=False)
        reset_memory_manager()
        mm2 = get_memory_manager(auto_configure=False)
        assert mm1 is not mm2


class TestIntegrationScenarios:
    """Integration-style tests for realistic scenarios."""

    def test_training_step_protection(self):
        """Simulate protected training step."""
        mm = MemoryManager(max_retries=2)
        mm._cuda_available = False

        # Simulate training
        losses = []

        @mm.with_memory_protection("training")
        def train_step(batch_idx):
            # Simulate occasional OOM on first batch
            if batch_idx == 0:
                mm._checkpoints.append("recovered")
            return 0.5 - (batch_idx * 0.1)

        for i in range(5):
            loss = train_step(i)
            losses.append(loss)

        assert len(losses) == 5
        assert losses[0] > losses[-1]  # Loss decreased

    def test_memory_monitoring_workflow(self):
        """Test typical memory monitoring workflow."""
        mm = MemoryManager()
        mm._cuda_available = False

        # Workflow: check → clear if needed → operate → log
        if mm.is_memory_critical():
            mm.clear_cache()

        mm.checkpoint("before_training")

        with mm.safe_operation("training"):
            pass  # Training would happen here

        mm.log_memory_status("After training: ")

        assert "before_training" in mm._checkpoints

    def test_multi_gpu_workflow(self):
        """Test workflow for multiple GPUs."""
        # Create managers for different GPUs
        mm_4060 = MemoryManager.for_gpu("RTX 4060")
        mm_a100 = MemoryManager.for_gpu("A100")

        # Different thresholds
        assert mm_4060.memory_margin_mb < mm_a100.memory_margin_mb
        assert mm_4060._critical_threshold <= mm_a100._critical_threshold
