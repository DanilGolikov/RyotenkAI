"""
Memory Manager - Graceful OOM Handling

Provides memory management utilities for training on ANY GPU.
Auto-detects GPU and adapts memory thresholds accordingly.

Supported GPU classes:
- Consumer: RTX 4060 (8GB), RTX 4070 (12GB), RTX 4080 (16GB), RTX 4090 (24GB)
- Professional: RTX A4000 (16GB), RTX A5000 (24GB), RTX A6000 (48GB)
- Datacenter: A100 (40/80GB), H100 (80GB), L40S (48GB)
- Legacy: RTX 3090 (24GB), RTX 3080 (10GB), etc.

Features:
- Auto GPU detection and classification
- Dynamic memory thresholds based on VRAM
- Graceful OOM recovery with auto-retry
- Memory checkpoints before heavy operations
- CUDA cache management

Based on ideas from Miles/Slime project memory patterns.

Example:
    mm = MemoryManager.auto_configure()  # Auto-detect GPU
    with mm.safe_operation("training_step"):
        trainer.train()  # Protected from OOM crashes
"""

from __future__ import annotations

import gc
from collections.abc import Callable
from contextlib import contextmanager, suppress
from dataclasses import dataclass
from enum import Enum
from functools import wraps
from typing import Any, ParamSpec, TypeVar

from src.utils.logger import logger

_DEVICE_CUDA = "cuda"
FRAG_HIGH_RATIO = 0.3
FRAG_WARN_RATIO = 0.4
UTIL_CRITICAL_PCT = 90.0
UTIL_WARNING_PCT = 80.0

# Type variables for generic decorator with proper type preservation
P = ParamSpec("P")
R = TypeVar("R")
F = TypeVar("F", bound=Callable[..., Any])


# GPU tier classification thresholds (GB)
_GPU_TIER_DATACENTER_GB = 40
_GPU_TIER_CONSUMER_HIGH_GB = 24
_GPU_TIER_CONSUMER_MID_GB = 12
_GPU_TIER_CONSUMER_LOW_GB = 8


class GPUTier(Enum):
    """GPU classification by VRAM capacity."""

    MINIMAL = "minimal"  # <8GB (GTX 1060, etc.) - very constrained
    CONSUMER_LOW = "consumer_low"  # 8GB (RTX 4060, 3070)
    CONSUMER_MID = "consumer_mid"  # 12-16GB (RTX 4070, 4080, 3080 Ti)
    CONSUMER_HIGH = "consumer_high"  # 24GB (RTX 4090, 3090)
    PROFESSIONAL = "professional"  # 24-48GB (RTX A5000, A6000)
    DATACENTER = "datacenter"  # 40-80GB (A100, H100)
    UNKNOWN = "unknown"


@dataclass
class GPUInfo:
    """Information about detected GPU."""

    name: str
    total_memory_mb: int
    tier: GPUTier
    compute_capability: tuple[int, int] | None = None
    device_index: int = 0

    @property
    def total_memory_gb(self) -> float:
        return self.total_memory_mb / 1024

    def __str__(self) -> str:
        return f"{self.name} ({self.total_memory_gb:.1f}GB, {self.tier.value})"


@dataclass
class GPUPreset:
    """Memory management preset for a GPU tier."""

    tier: GPUTier
    memory_margin_mb: int  # Minimum free memory to maintain
    memory_margin_percent: float  # Alternative: % of total VRAM
    critical_threshold: float  # % utilization considered critical
    warning_threshold: float  # % utilization considered warning
    max_retries: int  # OOM retry attempts

    @classmethod
    def for_tier(cls, tier: GPUTier) -> GPUPreset:
        """Get preset for a GPU tier."""
        #  tier           │ margin_mb │ margin_% │ critical │ warning │ retries
        #  ─────────────────────────────────────────────────────────────────────
        #  MINIMAL        │    300    │   5.0    │   92.0   │  85.0   │   5
        #  CONSUMER_LOW   │    500    │   6.0    │   90.0   │  80.0   │   3
        #  CONSUMER_MID   │    800    │   5.0    │   90.0   │  80.0   │   3
        #  CONSUMER_HIGH  │   1500    │   6.0    │   90.0   │  80.0   │   3
        #  PROFESSIONAL   │   2000    │   5.0    │   92.0   │  82.0   │   2
        #  DATACENTER     │   4000    │   5.0    │   95.0   │  85.0   │   2
        #  UNKNOWN        │    500    │   6.0    │   90.0   │  80.0   │   3
        presets = {
            GPUTier.MINIMAL: cls(
                tier=tier,
                memory_margin_mb=300,  # noqa: WPS432
                memory_margin_percent=5.0,  # noqa: WPS432
                critical_threshold=92.0,  # noqa: WPS432
                warning_threshold=85.0,  # noqa: WPS432
                max_retries=5,  # noqa: WPS432
            ),
            GPUTier.CONSUMER_LOW: cls(
                tier=tier,
                memory_margin_mb=500,  # noqa: WPS432
                memory_margin_percent=6.0,  # noqa: WPS432
                critical_threshold=90.0,  # noqa: WPS432
                warning_threshold=80.0,  # noqa: WPS432
                max_retries=3,  # noqa: WPS432
            ),
            GPUTier.CONSUMER_MID: cls(
                tier=tier,
                memory_margin_mb=800,  # noqa: WPS432
                memory_margin_percent=5.0,  # noqa: WPS432
                critical_threshold=90.0,  # noqa: WPS432
                warning_threshold=80.0,  # noqa: WPS432
                max_retries=3,  # noqa: WPS432
            ),
            GPUTier.CONSUMER_HIGH: cls(
                tier=tier,
                memory_margin_mb=1500,  # noqa: WPS432
                memory_margin_percent=6.0,  # noqa: WPS432
                critical_threshold=90.0,  # noqa: WPS432
                warning_threshold=80.0,  # noqa: WPS432
                max_retries=3,  # noqa: WPS432
            ),
            GPUTier.PROFESSIONAL: cls(
                tier=tier,
                memory_margin_mb=2000,  # noqa: WPS432
                memory_margin_percent=5.0,  # noqa: WPS432
                critical_threshold=92.0,  # noqa: WPS432
                warning_threshold=82.0,  # noqa: WPS432
                max_retries=2,  # noqa: WPS432
            ),
            GPUTier.DATACENTER: cls(
                tier=tier,
                memory_margin_mb=4000,  # noqa: WPS432
                memory_margin_percent=5.0,  # noqa: WPS432
                critical_threshold=95.0,  # noqa: WPS432
                warning_threshold=85.0,  # noqa: WPS432
                max_retries=2,  # noqa: WPS432
            ),
            GPUTier.UNKNOWN: cls(
                tier=tier,
                memory_margin_mb=500,  # noqa: WPS432
                memory_margin_percent=6.0,  # noqa: WPS432
                critical_threshold=90.0,  # noqa: WPS432
                warning_threshold=80.0,  # noqa: WPS432
                max_retries=3,  # noqa: WPS432
            ),
        }
        return presets.get(tier, presets[GPUTier.UNKNOWN])


# =============================================================================
# EVENT CALLBACKS
# =============================================================================


@dataclass
class MemoryEventCallbacks:
    """
    Callbacks for memory events (SOLID-compliant event collection).

    Used to integrate MemoryManager with MLflow or other logging systems
    without creating direct dependencies.

    Example:
        callbacks = MemoryEventCallbacks(
            on_gpu_detected=lambda name, vram, tier: print(f"GPU: {name}"),
            on_cache_cleared=lambda freed: print(f"Freed: {freed}MB"),
        )
        mm = MemoryManager(callbacks=callbacks)
    """

    # GPU detection event
    on_gpu_detected: Callable[[str, float, str], None] | None = None
    # Args: gpu_name, vram_gb, tier

    # Cache cleared event
    on_cache_cleared: Callable[[int], None] | None = None
    # Args: freed_mb

    # Memory warning/critical event
    on_memory_warning: Callable[[float, int, int, bool], None] | None = None
    # Args: utilization_percent, used_mb, total_mb, is_critical

    # OOM error event
    on_oom: Callable[[str, int | None], None] | None = None
    # Args: operation, free_mb (or None)

    # OOM retry event
    on_oom_retry: Callable[[str, int, int], None] | None = None
    # Args: operation, attempt, max_attempts


class OOMRecoverableError(Exception):
    """
    OOM error that can be recovered from.

    Contains information about the operation that failed
    and recovery suggestions.
    """

    def __init__(self, operation: str, memory_info: dict[str, int] | None = None):
        self.operation = operation
        self.memory_info = memory_info or {}
        super().__init__(f"OOM during '{operation}'. Recovery possible.")

    def __str__(self) -> str:
        msg = f"OOM during '{self.operation}'"
        if self.memory_info:
            msg += f" (free: {self.memory_info.get('free_mb', '?')}MB)"
        return msg


@dataclass
class MemoryStats:
    """Current memory statistics."""

    total_mb: int
    free_mb: int
    used_mb: int
    utilization_percent: float
    gpu_name: str | None = None
    reserved_mb: int = 0  # PyTorch reserved memory
    allocated_mb: int = 0  # PyTorch allocated memory

    @property
    def fragmentation_ratio(self) -> float:
        """
        Calculate memory fragmentation ratio.

        Ratio = 1.0 - (allocated / reserved)
        Example: If 10GB reserved but only 5GB allocated, fragmentation is 0.5 (50%).
        This means 5GB is held by caching allocator but not used by tensors.
        """
        if self.reserved_mb <= 0:
            return 0.0
        return 1.0 - (self.allocated_mb / self.reserved_mb)

    @property
    def is_critical(self) -> bool:
        """Check if memory utilization is critical (>90%)."""
        return self.utilization_percent > UTIL_CRITICAL_PCT

    @property
    def is_warning(self) -> bool:
        """Check if memory utilization is warning level (>80%)."""
        return self.utilization_percent > UTIL_WARNING_PCT

    def is_critical_for_threshold(self, threshold: float) -> bool:
        """Check if utilization exceeds custom threshold."""
        return self.utilization_percent > threshold

    def is_warning_for_threshold(self, threshold: float) -> bool:
        """Check if utilization exceeds custom warning threshold."""
        return self.utilization_percent > threshold


class MemoryManager:
    """
    Graceful OOM handling for ANY GPU.

    Provides:
    - Auto GPU detection and classification
    - Dynamic memory thresholds
    - Safe operation context managers
    - Auto-retry with cache clearing
    - Memory checkpoints for recovery

    Works on any CUDA device from GTX 1060 to H100.

    Example:
        # Auto-configure for detected GPU
        mm = MemoryManager.auto_configure()
        print(f"Detected: {mm.gpu_info}")

        # Or manual configuration
        mm = MemoryManager(memory_margin_mb=500)

        # Check memory before operation
        if mm.is_memory_critical():
            mm.clear_cache()

        # Protected operation
        with mm.safe_operation("forward_pass"):
            output = model(input_ids)

        # Or use decorator
        @mm.with_memory_protection("training")
        def train_step():
            ...
    """

    # Default memory margin to keep free (MB)
    DEFAULT_MEMORY_MARGIN_MB = 500

    # Number of retry attempts on OOM
    DEFAULT_MAX_RETRIES = 3

    def __init__(
        self,
        device: str = _DEVICE_CUDA,
        memory_margin_mb: int | None = None,
        max_retries: int | None = None,
        preset: GPUPreset | None = None,
        auto_detect: bool = False,
        callbacks: MemoryEventCallbacks | None = None,
    ):
        """
        Initialize MemoryManager.

        Args:
            device: CUDA device ("cuda", "cuda:0", etc.)
            memory_margin_mb: Minimum free memory to maintain (MB).
                              If None and auto_detect=True, will be auto-configured.
            max_retries: Number of OOM retry attempts.
                         If None and auto_detect=True, will be auto-configured.
            preset: Optional GPUPreset for manual configuration
            auto_detect: If True, auto-detect GPU and configure thresholds
            callbacks: Optional event callbacks for MLflow integration
        """
        self.device = device
        self._cuda_available: bool | None = None
        self._checkpoints: list[str] = []
        self._gpu_info: GPUInfo | None = None
        self._preset: GPUPreset | None = preset
        self._callbacks = callbacks or MemoryEventCallbacks()

        # Auto-detect GPU if requested
        if auto_detect:
            self._gpu_info = self._detect_gpu()
            if self._gpu_info:
                logger.debug(
                    f"[MM:GPU_DETECTED] name={self._gpu_info.name}, "
                    f"vram={self._gpu_info.total_memory_gb:.1f}GB, "
                    f"compute_cap={self._gpu_info.compute_capability}"
                )
                logger.debug(f"[MM:TIER_CLASSIFIED] tier={self._gpu_info.tier.value}")

                # Fire GPU detected callback
                if self._callbacks.on_gpu_detected:
                    self._callbacks.on_gpu_detected(
                        self._gpu_info.name,
                        self._gpu_info.total_memory_gb,
                        self._gpu_info.tier.value,
                    )

            if self._gpu_info and not preset:
                self._preset = GPUPreset.for_tier(self._gpu_info.tier)

        # Apply preset or defaults
        if self._preset:
            self.memory_margin_mb = memory_margin_mb or self._preset.memory_margin_mb
            self.max_retries = max_retries if max_retries is not None else self._preset.max_retries
            self._critical_threshold = self._preset.critical_threshold
            self._warning_threshold = self._preset.warning_threshold
            logger.debug(
                f"[MM:PRESET_APPLIED] tier={self._preset.tier.value}, "
                f"margin={self.memory_margin_mb}MB, "
                f"critical={self._critical_threshold}%, "
                f"warning={self._warning_threshold}%"
            )
        else:
            self.memory_margin_mb = memory_margin_mb or self.DEFAULT_MEMORY_MARGIN_MB
            self.max_retries = max_retries if max_retries is not None else self.DEFAULT_MAX_RETRIES
            self._critical_threshold = UTIL_CRITICAL_PCT
            self._warning_threshold = UTIL_WARNING_PCT
            logger.debug(f"[MM:DEFAULTS_APPLIED] margin={self.memory_margin_mb}MB, retries={self.max_retries}")

        # Log initialization
        if self._gpu_info:
            logger.info(
                f"MemoryManager initialized for {self._gpu_info} "
                f"(margin={self.memory_margin_mb}MB, retries={self.max_retries})"
            )
        else:
            logger.info(f"MemoryManager initialized (margin={self.memory_margin_mb}MB, retries={self.max_retries})")

    @classmethod
    def auto_configure(
        cls,
        device: str = _DEVICE_CUDA,
        callbacks: MemoryEventCallbacks | None = None,
    ) -> MemoryManager:
        """
        Create MemoryManager with auto-detected GPU configuration.

        Automatically detects GPU and applies optimal thresholds.

        Args:
            device: CUDA device
            callbacks: Optional event callbacks for MLflow integration

        Returns:
            Configured MemoryManager

        Example:
            mm = MemoryManager.auto_configure()
            print(f"GPU: {mm.gpu_info}")
            print(f"Tier: {mm.gpu_info.tier}")
            print(f"Margin: {mm.memory_margin_mb}MB")
        """
        return cls(device=device, auto_detect=True, callbacks=callbacks)

    @classmethod
    def for_gpu(cls, gpu_name: str, device: str = _DEVICE_CUDA) -> MemoryManager:
        """
        Create MemoryManager for a specific GPU model.

        Useful for testing or when auto-detection doesn't work.

        Args:
            gpu_name: GPU name (e.g., "RTX 4060", "A100")
            device: CUDA device

        Returns:
            Configured MemoryManager
        """
        tier = cls._classify_gpu_by_name(gpu_name)
        preset = GPUPreset.for_tier(tier)
        return cls(device=device, preset=preset)

    @staticmethod
    def _classify_gpu_by_name(name: str) -> GPUTier:
        """Classify GPU tier by name."""
        name_lower = name.lower()

        # Datacenter GPUs
        if any(x in name_lower for x in ["h100", "a100", "h200"]):
            return GPUTier.DATACENTER

        # Professional GPUs
        if any(x in name_lower for x in ["a6000", "a5000", "a4500", "l40", "rtx 6000"]):
            return GPUTier.PROFESSIONAL

        # Consumer High (24GB)
        if any(x in name_lower for x in ["4090", "3090", "titan"]):
            return GPUTier.CONSUMER_HIGH

        # Consumer Mid (12-16GB)
        if any(x in name_lower for x in ["4080", "4070 ti", "3080 ti", "3080"]):
            return GPUTier.CONSUMER_MID

        # Consumer Low (8GB)
        if any(x in name_lower for x in ["4060", "4070", "3070", "3060 ti", "2080"]):
            return GPUTier.CONSUMER_LOW

        # Minimal (<8GB)
        if any(x in name_lower for x in ["3060", "2070", "2060", "1080", "1070", "1060"]):
            return GPUTier.MINIMAL

        return GPUTier.UNKNOWN

    @staticmethod
    def _classify_gpu_by_memory(total_mb: int) -> GPUTier:
        """Classify GPU tier by total VRAM."""
        total_gb = total_mb / 1024

        if total_gb >= _GPU_TIER_DATACENTER_GB:
            return GPUTier.DATACENTER
        elif total_gb >= _GPU_TIER_CONSUMER_HIGH_GB:
            return GPUTier.CONSUMER_HIGH  # or PROFESSIONAL
        elif total_gb >= _GPU_TIER_CONSUMER_MID_GB:
            return GPUTier.CONSUMER_MID
        elif total_gb >= _GPU_TIER_CONSUMER_LOW_GB:
            return GPUTier.CONSUMER_LOW
        else:
            return GPUTier.MINIMAL

    def _detect_gpu(self) -> GPUInfo | None:
        """Detect GPU and return info."""
        if not self.cuda_available:
            return None

        try:
            import torch as torch_module

            torch: Any = torch_module

            # Get device index
            if self.device == _DEVICE_CUDA:
                device_idx = 0
            else:
                device_idx = int(self.device.split(":")[-1])

            # Get GPU properties
            props = torch.cuda.get_device_properties(device_idx)
            total_mb = props.total_memory // (1024 * 1024)

            # Classify by name first, then by memory as fallback
            tier = self._classify_gpu_by_name(props.name)
            if tier == GPUTier.UNKNOWN:
                tier = self._classify_gpu_by_memory(total_mb)

            return GPUInfo(
                name=props.name,
                total_memory_mb=total_mb,
                tier=tier,
                compute_capability=(props.major, props.minor),
                device_index=device_idx,
            )

        except Exception as e:
            logger.warning(f"GPU detection failed: {e}")
            return None

    @property
    def gpu_info(self) -> GPUInfo | None:
        """Get detected GPU info."""
        if self._gpu_info is None:
            self._gpu_info = self._detect_gpu()
        return self._gpu_info

    @property
    def preset(self) -> GPUPreset | None:
        """Get active GPU preset."""
        return self._preset

    @property
    def cuda_available(self) -> bool:
        """Check if CUDA is available (cached)."""
        if self._cuda_available is None:
            try:
                import torch as _torch_module
            except ImportError:
                _torch_module = None  # type: ignore[assignment]
                self._cuda_available = False
            else:
                torch: Any = _torch_module
                self._cuda_available = torch.cuda.is_available()
        return bool(self._cuda_available)

    def get_memory_stats(self) -> MemoryStats | None:
        """
        Get current GPU memory statistics.

        Returns:
            MemoryStats or None if CUDA unavailable
        """
        if not self.cuda_available:
            return None

        try:
            import torch as torch_module

            torch: Any = torch_module

            free, total = torch.cuda.mem_get_info(self.device)
            free_mb = free // (1024 * 1024)
            total_mb = total // (1024 * 1024)
            used_mb = total_mb - free_mb
            utilization = (used_mb / total_mb) * 100 if total_mb > 0 else 0

            # Get GPU name if available
            gpu_name = None
            with suppress(Exception):
                # Prefer simple API: tests patch get_device_name; also avoids get_device_properties pitfalls
                gpu_name = torch.cuda.get_device_name(self.device)

            # Get detailed PyTorch memory stats for fragmentation analysis
            reserved_mb = 0
            allocated_mb = 0
            try:
                reserved = torch.cuda.memory_reserved(self.device)
                allocated = torch.cuda.memory_allocated(self.device)
                reserved_mb = reserved // (1024 * 1024)
                allocated_mb = allocated // (1024 * 1024)
            except Exception:
                # Fallback if memory_reserved/allocated fails (e.g. older pytorch)
                pass

            return MemoryStats(
                total_mb=total_mb,
                free_mb=free_mb,
                used_mb=used_mb,
                utilization_percent=utilization,
                gpu_name=gpu_name,
                reserved_mb=reserved_mb,
                allocated_mb=allocated_mb,
            )
        except Exception as e:
            logger.warning(f"Failed to get memory stats: {e}")
            return None

    def get_available_memory_mb(self) -> int:
        """Get available GPU memory in MB."""
        stats = self.get_memory_stats()
        return stats.free_mb if stats else 0

    def is_memory_critical(self) -> bool:
        """
        Check if memory is at critical level.

        Uses GPU-specific thresholds if auto-configured.

        Returns:
            True if free memory < margin or utilization > threshold
        """
        stats = self.get_memory_stats()
        if stats is None:
            return False

        is_critical = stats.free_mb < self.memory_margin_mb or stats.is_critical_for_threshold(self._critical_threshold)
        if is_critical:
            # Check for high fragmentation
            frag_msg = ""
            if stats.fragmentation_ratio > FRAG_HIGH_RATIO:
                frag_msg = f", frag={stats.fragmentation_ratio:.2f}"

            logger.debug(
                f"[MM:MEMORY_CRITICAL] free={stats.free_mb}MB, "
                f"used={stats.used_mb}MB, "
                f"util={stats.utilization_percent:.1f}%{frag_msg}"
            )
        return is_critical

    def is_memory_warning(self) -> bool:
        """
        Check if memory is at warning level.

        Uses GPU-specific thresholds if auto-configured.

        Returns:
            True if utilization > warning threshold
        """
        stats = self.get_memory_stats()
        if stats is None:
            return False
        return stats.is_warning_for_threshold(self._warning_threshold)

    def clear_cache(self) -> int:
        """
        Clear CUDA cache and run garbage collection.

        Returns:
            Freed memory in MB (approximate)
        """
        before = self.get_available_memory_mb()

        # Python garbage collection first
        gc.collect()

        if self.cuda_available:
            try:
                import torch as torch_module

                torch: Any = torch_module

                torch.cuda.empty_cache()
                torch.cuda.synchronize(self.device)
            except Exception as e:
                logger.warning(f"Cache clear warning: {e}")

        after = self.get_available_memory_mb()
        freed = after - before

        if freed > 0:
            logger.debug(f"Cache cleared: +{freed}MB freed")
            # Fire callback
            if self._callbacks.on_cache_cleared:
                self._callbacks.on_cache_cleared(freed)

        return freed

    def aggressive_cleanup(self) -> int:
        """
        Aggressive memory cleanup for OOM recovery.

        Multiple GC passes + cache clear.

        Returns:
            Freed memory in MB
        """
        before = self.get_available_memory_mb()

        # Multiple GC passes (catches cycles)
        for _ in range(3):
            gc.collect()

        if self.cuda_available:
            try:
                import torch as torch_module

                torch: Any = torch_module

                # Clear cache
                torch.cuda.empty_cache()

                # Reset peak memory stats
                torch.cuda.reset_peak_memory_stats(self.device)

                # Synchronize
                torch.cuda.synchronize(self.device)
            except Exception as e:
                logger.warning(f"Aggressive cleanup warning: {e}")

        after = self.get_available_memory_mb()
        freed = after - before

        logger.info(f"Aggressive cleanup: +{freed}MB freed")
        logger.debug(f"[MM:CACHE_CLEARED] freed={freed}MB, available_now={after}MB")

        # Fire callback (use same as cache_cleared for simplicity)
        if freed > 0:
            cb = self._callbacks.on_cache_cleared
            if cb is not None:
                cb(freed)

        return freed

    def checkpoint(self, name: str) -> None:
        """
        Create memory checkpoint for potential rollback.

        Args:
            name: Checkpoint identifier
        """
        stats = self.get_memory_stats()
        self._checkpoints.append(name)
        logger.debug(f"Memory checkpoint '{name}': {stats.used_mb}MB used" if stats else f"Checkpoint '{name}'")

    def log_memory_status(self, prefix: str = "") -> None:
        """Log current memory status."""
        stats = self.get_memory_stats()
        if stats:
            status = "🔴 CRITICAL" if stats.is_critical else ("🟡 WARNING" if stats.is_warning else "🟢 OK")
            logger.info(
                f"{prefix}Memory: {stats.used_mb}/{stats.total_mb}MB ({stats.utilization_percent:.1f}%) {status}"
            )

    @contextmanager
    def safe_operation(
        self,
        operation_name: str,
        context: dict[str, Any] | None = None,
    ):
        """
        Context manager for memory-safe operations.

        Automatically:
        - Checks memory before operation
        - Clears cache if needed
        - Catches OOM and provides recovery info with context

        Args:
            operation_name: Name for logging/debugging
            context: Optional context for error reporting (e.g. {"batch_size": 1, "seq_len": 4096})

        Yields:
            None

        Raises:
            OOMRecoverableError: If OOM occurs (can be caught for retry)

        Example:
            with mm.safe_operation("forward_pass", context={"batch_size": 4}):
                output = model(input_ids)
        """
        # Pre-operation memory check
        stats_before = self.get_memory_stats()

        # Log fragmentation warning if high (>40%) even if not critical yet
        if stats_before and stats_before.fragmentation_ratio > FRAG_WARN_RATIO:
            logger.warning(
                f"[MEMORY_V2] High memory fragmentation ({stats_before.fragmentation_ratio:.1%}) before '{operation_name}'. "
                f"Reserved: {stats_before.reserved_mb}MB, Allocated: {stats_before.allocated_mb}MB. "
                "Consider setting PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True"
            )
            # Fire warning callback for fragmentation
            if self._callbacks.on_memory_warning:
                self._callbacks.on_memory_warning(
                    stats_before.utilization_percent,
                    stats_before.used_mb,
                    stats_before.total_mb,
                    False,  # Not critical OOM yet, just warning
                )

        logger.debug(
            f"[MM:SAFE_OP_START] operation={operation_name}, "
            f"used={stats_before.used_mb if stats_before else 0}MB, "
            f"free={stats_before.free_mb if stats_before else 0}MB"
        )

        if self.is_memory_critical():
            logger.warning(f"Low memory before '{operation_name}', clearing cache...")
            # Fire memory warning callback
            if stats_before and self._callbacks.on_memory_warning:
                self._callbacks.on_memory_warning(
                    stats_before.utilization_percent,
                    stats_before.used_mb,
                    stats_before.total_mb,
                    True,  # is_critical
                )
            self.clear_cache()

        self.checkpoint(operation_name)

        try:
            yield
            logger.debug(f"[MM:SAFE_OP_SUCCESS] operation={operation_name}")
        except RuntimeError as e:
            error_msg = str(e).lower()
            if "out of memory" in error_msg or (_DEVICE_CUDA in error_msg and "memory" in error_msg):
                stats = self.get_memory_stats()
                memory_info = {
                    "free_mb": stats.free_mb if stats else 0,
                    "used_mb": stats.used_mb if stats else 0,
                    "fragmentation": stats.fragmentation_ratio if stats else 0.0,
                }

                # Enhance error message with context if available
                context_str = ""
                if context:
                    context_str = f" Context: {context}"
                    # Provide actionable advice based on context
                    if "batch_size" in context and context["batch_size"] > 1:
                        context_str += f" -> TRY REDUCING batch_size (current: {context['batch_size']})"

                logger.error(f"OOM during '{operation_name}'{context_str}")
                logger.debug(
                    f"[MM:OOM_TRIGGERED] operation={operation_name}, memory_info={memory_info}, context={context}"
                )
                self.log_memory_status("   ")

                # Fire OOM callback
                if self._callbacks.on_oom:
                    self._callbacks.on_oom(operation_name, stats.free_mb if stats else None)

                # Attempt recovery
                self.aggressive_cleanup()

                # Cast memory_info values to int to match OOMRecoverableError signature
                # stats.free_mb is int, fragmentation is float
                oom_info: dict[str, int] | None = {
                    "free_mb": int(stats.free_mb) if stats else 0,
                    "used_mb": int(stats.used_mb) if stats else 0,
                    # We skip fragmentation here as OOMRecoverableError expects dict[str, int]
                }

                raise OOMRecoverableError(operation_name, oom_info) from e
            raise

    def with_memory_protection(
        self,
        operation_name: str,
        max_retries: int | None = None,
        context: dict[str, Any] | None = None,
        context_factory: Callable[..., dict[str, Any]] | None = None,
    ) -> Callable[[Callable[P, R]], Callable[P, R]]:
        """
        Decorator for automatic OOM retry with enhanced features.

        Wraps function with:
        - Memory check before call
        - Automatic retry on OOM
        - Progressive cleanup between retries
        - Dynamic context extraction (NEW)
        - Enhanced logging with emojis (NEW)

        This is the RECOMMENDED way to protect training operations from OOM.

        Args:
            operation_name: Name for logging
            max_retries: Override default retry count (None = use self.max_retries)
            context: Optional static context for error reporting
            context_factory: Optional callable that receives (*args, **kwargs) and returns
                           dynamic context dict. Useful for extracting runtime info like batch_size.

        Returns:
            Decorated function with preserved type signature

        Example (basic):
            @mm.with_memory_protection("training_step")
            def train_step(batch):
                return model(batch)

        Example (with static context):
            @mm.with_memory_protection("forward", context={"model": "gpt2"})
            def forward(x):
                return model(x)

        Example (with dynamic context factory):
            def extract_context(self, trainer, checkpoint):
                return {"batch_size": trainer.args.per_device_train_batch_size}

            @mm.with_memory_protection("train_phase", context_factory=extract_context)
            def _run_training(self, trainer, checkpoint):
                trainer.train(resume_from_checkpoint=checkpoint)

        Note:
            - If both context and context_factory provided, they are merged (factory wins)
            - attempt_number is automatically added to context
            - Works with bound methods (self/cls are passed to context_factory)
        """
        retries = max_retries if max_retries is not None else self.max_retries

        def decorator(func: Callable[P, R]) -> Callable[P, R]:
            @wraps(func)
            def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
                last_error: Exception | None = None

                for attempt in range(retries + 1):
                    # Build context for this attempt
                    attempt_context: dict[str, Any] = {"attempt_number": attempt}

                    # Add static context if provided
                    if context:
                        attempt_context.update(context)

                    # Call context_factory if provided (dynamic context)
                    if context_factory:
                        try:
                            dynamic_ctx = context_factory(*args, **kwargs)
                            attempt_context.update(dynamic_ctx)
                        except Exception as e:
                            logger.warning(f"[MM:CONTEXT_FACTORY_FAILED] operation={operation_name}, error={e}")

                    try:
                        with self.safe_operation(f"{operation_name}_attempt_{attempt}", context=attempt_context):
                            return func(*args, **kwargs)

                    except OOMRecoverableError as e:
                        last_error = e
                        logger.debug(
                            f"[MM:RETRY_ATTEMPT] operation={operation_name}, "
                            f"attempt={attempt + 1}/{retries + 1}, "
                            f"free_mb={e.memory_info.get('free_mb', '?')}, "
                            f"context={attempt_context}"
                        )

                        if attempt < retries:
                            logger.warning(
                                f"🔄 OOM detected, retry {attempt + 1}/{retries} for '{operation_name}' "
                                f"(free: {e.memory_info.get('free_mb', '?')}MB)"
                            )
                            # Fire retry callback
                            if self._callbacks.on_oom_retry:
                                self._callbacks.on_oom_retry(operation_name, attempt + 1, retries)

                            # Progressive cleanup
                            freed = self.aggressive_cleanup()
                            logger.info(f"   💾 Cleanup freed {freed}MB, retrying...")
                        else:
                            logger.error(
                                f"❌ All {retries} retries exhausted for '{operation_name}'. "
                                f"Last free memory: {e.memory_info.get('free_mb', '?')}MB. "
                                f"Context: {attempt_context}"
                            )

                # All retries failed
                if last_error:
                    raise last_error
                # Should not reach here
                raise RuntimeError(f"Unexpected: no error but no success for '{operation_name}'")

            return wrapper

        return decorator

    @staticmethod
    def estimate_batch_memory(
        model_params: int,
        batch_size: int,
        seq_length: int,
        dtype_bytes: int = 2,  # FP16 default
    ) -> int:
        """
        Estimate memory required for a batch (approximate).

        Args:
            model_params: Number of model parameters
            batch_size: Batch size
            seq_length: Sequence length
            dtype_bytes: Bytes per parameter (2 for FP16)

        Returns:
            Estimated memory in MB
        """
        # Model weights
        model_mb = (model_params * dtype_bytes) // (1024 * 1024)

        # Activations (rough estimate: 4x model size for training)
        activations_mb = (batch_size * seq_length * dtype_bytes * 4) // (1024 * 1024)

        # Gradients (same as model)
        gradients_mb = model_mb

        # Optimizer states (2x for Adam)
        optimizer_mb = model_mb * 2

        total = model_mb + activations_mb + gradients_mb + optimizer_mb
        return total


# Singleton instance for convenience
_default_manager: MemoryManager | None = None


def get_memory_manager(auto_configure: bool = True) -> MemoryManager:
    """
    Get or create default MemoryManager instance.

    Args:
        auto_configure: If True (default), auto-detect GPU on first call

    Returns:
        Singleton MemoryManager instance
    """
    global _default_manager
    if _default_manager is None:
        if auto_configure:
            _default_manager = MemoryManager.auto_configure()
        else:
            _default_manager = MemoryManager()
    return _default_manager


def reset_memory_manager() -> None:
    """Reset singleton instance (useful for testing)."""
    global _default_manager
    _default_manager = None


__all__ = [
    "GPUInfo",
    "GPUPreset",
    "GPUTier",
    "MemoryEventCallbacks",
    "MemoryManager",
    "MemoryStats",
    "OOMRecoverableError",
    "get_memory_manager",
    "reset_memory_manager",
]
