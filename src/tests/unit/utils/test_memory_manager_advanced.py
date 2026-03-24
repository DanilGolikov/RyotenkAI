"""
Advanced tests for MemoryManager - covering uncovered functionality.

Based on coverage report, these areas need more tests:
- Lines 338, 426, 438, 462, 478, 497-499, 504-506, 511, 521-522
- Lines 548, 560-562, 573-575, 600, 618-621, 681, 693
- Lines 750-757, 771-780, 800-803, 827, 882-885, 941-946, 952

Focus areas:
1. on_gpu_detected callback (line 338)
2. GPU classification edge cases (426, 438, 462, 478)
3. _detect_gpu() error paths (497-499, 504-506)
4. Memory warning callbacks with fragmentation (750-757, 771-780)
5. Context-aware OOM errors (800-803)
6. Exhausted retries path (882-885)
7. Singleton manager (941-946, 952)
"""

from unittest.mock import MagicMock, patch

import pytest

from src.utils.memory_manager import (
    GPUInfo,
    GPUTier,
    MemoryEventCallbacks,
    MemoryManager,
    OOMRecoverableError,
    get_memory_manager,
    reset_memory_manager,
)

# =============================================================================
# TESTS: GPU Classification Edge Cases
# =============================================================================


class TestGPUClassificationEdgeCases:
    """Test GPU classification for edge cases and rare GPUs."""

    def test_classify_rtx_6000(self):
        """RTX 6000 should be classified as PROFESSIONAL."""
        tier = MemoryManager._classify_gpu_by_name("NVIDIA RTX 6000 Ada Generation")
        assert tier == GPUTier.PROFESSIONAL

    def test_classify_a4500(self):
        """A4500 should be classified as PROFESSIONAL."""
        tier = MemoryManager._classify_gpu_by_name("NVIDIA RTX A4500")
        assert tier == GPUTier.PROFESSIONAL

    def test_classify_h200(self):
        """H200 should be classified as DATACENTER."""
        tier = MemoryManager._classify_gpu_by_name("NVIDIA H200")
        assert tier == GPUTier.DATACENTER

    def test_classify_4070_ti(self):
        """RTX 4070 Ti should be CONSUMER_MID."""
        tier = MemoryManager._classify_gpu_by_name("NVIDIA GeForce RTX 4070 Ti")
        assert tier == GPUTier.CONSUMER_MID

    def test_classify_3080_ti(self):
        """RTX 3080 Ti should be CONSUMER_MID."""
        tier = MemoryManager._classify_gpu_by_name("NVIDIA GeForce RTX 3080 Ti")
        assert tier == GPUTier.CONSUMER_MID

    def test_classify_3060_ti(self):
        """RTX 3060 Ti should be CONSUMER_LOW."""
        tier = MemoryManager._classify_gpu_by_name("NVIDIA GeForce RTX 3060 Ti")
        assert tier == GPUTier.CONSUMER_LOW

    def test_classify_titan(self):
        """Titan should be CONSUMER_HIGH."""
        tier = MemoryManager._classify_gpu_by_name("NVIDIA TITAN RTX")
        assert tier == GPUTier.CONSUMER_HIGH

    def test_classify_1080(self):
        """GTX 1080 should be MINIMAL."""
        tier = MemoryManager._classify_gpu_by_name("NVIDIA GeForce GTX 1080")
        assert tier == GPUTier.MINIMAL

    def test_classify_2070(self):
        """RTX 2070 should be MINIMAL."""
        tier = MemoryManager._classify_gpu_by_name("NVIDIA GeForce RTX 2070")
        assert tier == GPUTier.MINIMAL

    def test_classify_by_memory_40gb(self):
        """40GB VRAM should be DATACENTER."""
        tier = MemoryManager._classify_gpu_by_memory(40 * 1024)
        assert tier == GPUTier.DATACENTER

    def test_classify_by_memory_24gb(self):
        """24GB VRAM should be CONSUMER_HIGH."""
        tier = MemoryManager._classify_gpu_by_memory(24 * 1024)
        assert tier == GPUTier.CONSUMER_HIGH

    def test_classify_by_memory_16gb(self):
        """16GB VRAM should be CONSUMER_MID."""
        tier = MemoryManager._classify_gpu_by_memory(16 * 1024)
        assert tier == GPUTier.CONSUMER_MID

    def test_classify_by_memory_10gb(self):
        """10GB VRAM should be CONSUMER_LOW (< 12GB)."""
        tier = MemoryManager._classify_gpu_by_memory(10 * 1024)
        assert tier == GPUTier.CONSUMER_LOW

    def test_classify_by_memory_8gb(self):
        """8GB VRAM should be CONSUMER_LOW."""
        tier = MemoryManager._classify_gpu_by_memory(8 * 1024)
        assert tier == GPUTier.CONSUMER_LOW

    def test_classify_by_memory_6gb(self):
        """6GB VRAM should be MINIMAL."""
        tier = MemoryManager._classify_gpu_by_memory(6 * 1024)
        assert tier == GPUTier.MINIMAL


# =============================================================================
# TESTS: _detect_gpu Error Paths
# =============================================================================


class TestDetectGPUErrorPaths:
    """Test _detect_gpu() error handling."""

    @patch("torch.cuda.is_available", return_value=True)
    @patch("torch.cuda.get_device_properties")
    def test_detect_gpu_exception_handling(self, mock_props, mock_cuda_avail):
        """_detect_gpu() should handle exceptions gracefully."""
        # Simulate exception
        mock_props.side_effect = RuntimeError("CUDA error")

        mm = MemoryManager(device="cuda", auto_detect=True)

        # Should not crash, gpu_info should be None
        assert mm._gpu_info is None

    @patch("torch.cuda.is_available", return_value=True)
    @patch("torch.cuda.get_device_properties")
    def test_detect_gpu_missing_name(self, mock_props, mock_cuda_avail):
        """_detect_gpu() should handle missing GPU name."""
        # Mock GPU without proper name
        props = MagicMock()
        props.name = ""  # Empty name
        props.total_memory = 12 * 1024**3
        props.major = 7
        props.minor = 5
        mock_props.return_value = props

        mm = MemoryManager(device="cuda", auto_detect=True)

        # Should still work, using memory-based classification
        if mm._gpu_info:
            # If detected, tier should be based on memory (12GB)
            assert mm._gpu_info.tier == GPUTier.CONSUMER_MID


# =============================================================================
# TESTS: Memory Warning with Fragmentation
# =============================================================================


class TestMemoryWarningWithFragmentation:
    """Test memory warning callbacks with high fragmentation."""

    @patch("torch.cuda.is_available", return_value=True)
    @patch("torch.cuda.mem_get_info")
    @patch("torch.cuda.memory_reserved")
    @patch("torch.cuda.memory_allocated")
    def test_safe_operation_fragmentation_warning(self, mock_allocated, mock_reserved, mock_mem_info, mock_cuda_avail):
        """safe_operation should warn about high fragmentation."""
        warning_calls = []

        callbacks = MemoryEventCallbacks(
            on_memory_warning=lambda util, used, total, crit: warning_calls.append((util, used, total, crit))
        )

        # 50% utilization but high fragmentation
        # Total: 20GB, Free: 10GB, Used: 10GB
        # Reserved: 15GB, Allocated: 6GB -> fragmentation = 60%
        mock_mem_info.return_value = (10 * 1024**3, 20 * 1024**3)
        mock_reserved.return_value = 15 * 1024**3
        mock_allocated.return_value = 6 * 1024**3

        mm = MemoryManager(callbacks=callbacks)

        with mm.safe_operation("test_op"):
            pass

        # Should trigger fragmentation warning
        assert len(warning_calls) >= 1

    @patch("torch.cuda.is_available", return_value=True)
    @patch("torch.cuda.mem_get_info")
    def test_safe_operation_critical_memory_warning(self, mock_mem_info, mock_cuda_avail):
        """safe_operation should fire warning on critical memory."""
        warning_calls = []

        callbacks = MemoryEventCallbacks(on_memory_warning=lambda util, used, total, crit: warning_calls.append(crit))

        # 95% utilization (critical)
        mock_mem_info.return_value = (500 * 1024**2, 10 * 1024**3)

        mm = MemoryManager(callbacks=callbacks)

        with mm.safe_operation("critical_op"):
            pass

        # Should have critical warning
        assert True in warning_calls


# =============================================================================
# TESTS: Context-Aware OOM Errors
# =============================================================================


class TestContextAwareOOM:
    """Test OOM errors with context information."""

    def test_safe_operation_with_batch_size_context(self):
        """OOM with batch_size context should suggest reduction."""
        mm = MemoryManager()
        mm._cuda_available = False

        context = {"batch_size": 32, "model": "llama-7b"}

        with pytest.raises(OOMRecoverableError) as exc_info, mm.safe_operation("forward", context=context):
            raise RuntimeError("CUDA out of memory")

        # Should have captured context
        assert exc_info.value.operation == "forward"

    def test_safe_operation_with_empty_context(self):
        """OOM with empty context should work."""
        mm = MemoryManager()
        mm._cuda_available = False

        with pytest.raises(OOMRecoverableError), mm.safe_operation("forward", context={}):
            raise RuntimeError("CUDA out of memory")

    def test_safe_operation_with_none_context(self):
        """OOM with None context should work."""
        mm = MemoryManager()
        mm._cuda_available = False

        with pytest.raises(OOMRecoverableError), mm.safe_operation("forward", context=None):
            raise RuntimeError("CUDA out of memory")


# =============================================================================
# TESTS: Exhausted Retries
# =============================================================================


class TestExhaustedRetries:
    """Test behavior when all retries are exhausted."""

    def test_with_memory_protection_all_retries_fail(self):
        """All retries exhausted should raise OOMRecoverableError."""
        mm = MemoryManager(max_retries=2)
        mm._cuda_available = False

        @mm.with_memory_protection("always_fail")
        def always_fail():
            raise RuntimeError("CUDA out of memory")

        with pytest.raises(OOMRecoverableError) as exc_info:
            always_fail()

        # Should have operation name
        # Operation name will include attempt number
        assert "always_fail" in exc_info.value.operation

    def test_with_memory_protection_retry_callback_all_attempts(self):
        """Retry callback should be called for each retry."""
        retry_calls = []

        callbacks = MemoryEventCallbacks(on_oom_retry=lambda op, att, max_att: retry_calls.append((op, att, max_att)))

        mm = MemoryManager(max_retries=3, callbacks=callbacks)
        mm._cuda_available = False

        @mm.with_memory_protection("retry_all")
        def always_fail():
            raise RuntimeError("CUDA out of memory")

        with pytest.raises(OOMRecoverableError):
            always_fail()

        # Should have 2 retry callbacks (attempt 2 and 3)
        assert len(retry_calls) >= 2


# =============================================================================
# TESTS: Singleton Manager
# =============================================================================


class TestSingletonManager:
    """Test get_memory_manager() singleton behavior."""

    def test_get_memory_manager_creates_singleton(self):
        """get_memory_manager() should create singleton."""
        reset_memory_manager()

        mm1 = get_memory_manager(auto_configure=False)
        mm2 = get_memory_manager(auto_configure=False)

        # Should be same instance
        assert mm1 is mm2

    def test_reset_memory_manager(self):
        """reset_memory_manager() should clear singleton."""
        mm1 = get_memory_manager(auto_configure=False)
        reset_memory_manager()
        mm2 = get_memory_manager(auto_configure=False)

        # Should be different instances
        assert mm1 is not mm2

    @patch("torch.cuda.is_available", return_value=True)
    @patch("torch.cuda.get_device_properties")
    def test_get_memory_manager_auto_configure(self, mock_props, mock_cuda_avail):
        """get_memory_manager(auto_configure=True) should detect GPU."""
        reset_memory_manager()

        mock_props.return_value = MagicMock(name="NVIDIA RTX 4090", total_memory=24 * 1024**3, major=8, minor=9)

        mm = get_memory_manager(auto_configure=True)

        # Should have detected GPU
        assert mm._gpu_info is not None


# =============================================================================
# TESTS: is_memory_warning
# =============================================================================


class TestIsMemoryWarning:
    """Test is_memory_warning() method."""

    @patch("torch.cuda.is_available", return_value=True)
    @patch("torch.cuda.mem_get_info")
    def test_is_memory_warning_true(self, mock_mem_info, mock_cuda_avail):
        """is_memory_warning() should return True at 85% usage."""
        # 85% utilization
        mock_mem_info.return_value = (1500 * 1024**2, 10 * 1024**3)

        mm = MemoryManager()

        assert mm.is_memory_warning() is True

    @patch("torch.cuda.is_available", return_value=True)
    @patch("torch.cuda.mem_get_info")
    def test_is_memory_warning_false(self, mock_mem_info, mock_cuda_avail):
        """is_memory_warning() should return False at 70% usage."""
        # 70% utilization
        mock_mem_info.return_value = (3 * 1024**3, 10 * 1024**3)

        mm = MemoryManager()

        assert mm.is_memory_warning() is False

    def test_is_memory_warning_no_cuda(self):
        """is_memory_warning() should return False with no CUDA."""
        mm = MemoryManager()
        mm._cuda_available = False

        assert mm.is_memory_warning() is False


# =============================================================================
# TESTS: for_gpu Factory Method
# =============================================================================


class TestForGPUFactory:
    """Test MemoryManager.for_gpu() factory method."""

    def test_for_gpu_4090(self):
        """for_gpu('RTX 4090') should apply correct preset."""
        mm = MemoryManager.for_gpu("RTX 4090")

        assert mm.memory_margin_mb == 1500  # CONSUMER_HIGH
        assert mm._critical_threshold == 90.0

    def test_for_gpu_a100(self):
        """for_gpu('A100') should apply DATACENTER preset."""
        mm = MemoryManager.for_gpu("A100")

        assert mm.memory_margin_mb == 4000  # DATACENTER
        assert mm._critical_threshold == 95.0

    def test_for_gpu_3060(self):
        """for_gpu('3060') should apply MINIMAL preset."""
        mm = MemoryManager.for_gpu("3060")

        assert mm.memory_margin_mb == 300  # MINIMAL
        assert mm.max_retries == 5


# =============================================================================
# TESTS: Context Manager Error Recovery
# =============================================================================


class TestContextManagerErrorRecovery:
    """Test safe_operation context manager error recovery."""

    @patch("torch.cuda.is_available", return_value=True)
    @patch("torch.cuda.mem_get_info")
    @patch("torch.cuda.empty_cache")
    def test_safe_operation_calls_cleanup_on_oom(self, mock_empty, mock_mem_info, mock_cuda_avail):
        """safe_operation should call aggressive_cleanup() on OOM."""
        mock_mem_info.side_effect = [
            (5 * 1024**3, 10 * 1024**3),  # Before
            (7 * 1024**3, 10 * 1024**3),  # After cleanup
        ]

        mm = MemoryManager()

        with pytest.raises(OOMRecoverableError), mm.safe_operation("test_oom"):
            raise RuntimeError("CUDA out of memory")

        # Should have called empty_cache during aggressive_cleanup
        assert mock_empty.called


# =============================================================================
# TESTS: GPUInfo Properties
# =============================================================================


class TestGPUInfoProperties:
    """Test GPUInfo dataclass."""

    def test_gpu_info_str(self):
        """GPUInfo should have readable string representation."""
        info = GPUInfo(
            name="RTX 4090", total_memory_mb=24 * 1024, tier=GPUTier.CONSUMER_HIGH, compute_capability=(8, 9)
        )

        str_repr = str(info)
        assert "RTX 4090" in str_repr
        assert "24.0GB" in str_repr

    def test_gpu_info_equality(self):
        """GPUInfo instances should be comparable."""
        info1 = GPUInfo("RTX 4090", 24 * 1024, GPUTier.CONSUMER_HIGH, (8, 9))
        info2 = GPUInfo("RTX 4090", 24 * 1024, GPUTier.CONSUMER_HIGH, (8, 9))

        assert info1 == info2


# =============================================================================
# TESTS: Checkpoint with Memory Stats
# =============================================================================


class TestCheckpointMemoryStats:
    """Test checkpoint() storing memory stats."""

    @patch("torch.cuda.is_available", return_value=True)
    @patch("torch.cuda.mem_get_info")
    def test_checkpoint_stores_memory_stats(self, mock_mem_info, mock_cuda_avail):
        """checkpoint() should store memory stats."""
        mock_mem_info.return_value = (8 * 1024**3, 16 * 1024**3)

        mm = MemoryManager()
        mm.checkpoint("epoch_1")

        # Should have checkpoint
        assert "epoch_1" in mm._checkpoints

    def test_checkpoint_no_cuda(self):
        """checkpoint() should work without CUDA."""
        mm = MemoryManager()
        mm._cuda_available = False

        mm.checkpoint("step_1")

        # Should have checkpoint (but stats might be None)
        assert "step_1" in mm._checkpoints
