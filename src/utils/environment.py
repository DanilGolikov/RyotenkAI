"""
Environment Reporter for Training Reproducibility.

Collects and reports environment information critical for reproducibility:
- Python version
- CUDA/PyTorch versions
- ML library versions (transformers, peft, trl, accelerate)
- GPU information
- Git commit hash (if available)
- OS information

Usage:
    from src.utils.environment import EnvironmentReporter
    reporter = EnvironmentReporter.collect()
    reporter.log_summary()  # Print to console
    reporter.to_dict()      # For W&B/checkpoint

Example output:
    🔧 Environment:
       Python: 3.11.5
       PyTorch: 2.1.0+cu118
       Transformers: 4.35.0
       CUDA: 11.8
       GPU: NVIDIA RTX 4090 (24GB)
       Git: a1b2c3d
"""

from __future__ import annotations

import importlib.metadata as importlib_metadata
import platform
import subprocess
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from src.utils.logger import logger

_VERSION_UNAVAILABLE = "N/A"


@dataclass(frozen=True)
class EnvironmentSnapshot:
    """
    Immutable snapshot of environment for reproducibility.

    Captures all relevant library versions and system information
    at training start time.

    Attributes:
        python_version: Python version string
        torch_version: PyTorch version
        cuda_version: CUDA version (or _VERSION_UNAVAILABLE)
        transformers_version: Transformers library version
        peft_version: PEFT library version
        trl_version: TRL library version
        accelerate_version: Accelerate library version
        bitsandbytes_version: BitsAndBytes version (for QLoRA)
        gpu_name: GPU name (or "CPU")
        gpu_vram_gb: GPU VRAM in GB (or 0)
        os_info: OS name and version
        git_commit: Git commit hash (or _VERSION_UNAVAILABLE)
        timestamp: ISO timestamp when snapshot was created

    Example:
        snapshot = EnvironmentSnapshot.collect()
        print(snapshot.torch_version)
        # "2.1.0+cu118"
    """

    python_version: str
    torch_version: str
    cuda_version: str
    transformers_version: str
    peft_version: str
    trl_version: str
    accelerate_version: str
    bitsandbytes_version: str
    gpu_name: str
    gpu_vram_gb: float
    os_info: str
    git_commit: str
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    @classmethod
    def collect(cls) -> EnvironmentSnapshot:
        """
        Collect current environment information.

        Attempts to import all libraries and gracefully handles missing ones.

        Returns:
            EnvironmentSnapshot: Captured environment state
        """
        return cls(
            python_version=_get_python_version(),
            torch_version=_get_package_version("torch"),
            cuda_version=_get_cuda_version(),
            transformers_version=_get_package_version("transformers"),
            peft_version=_get_package_version("peft"),
            trl_version=_get_package_version("trl"),
            accelerate_version=_get_package_version("accelerate"),
            bitsandbytes_version=_get_package_version("bitsandbytes"),
            gpu_name=_get_gpu_name(),
            gpu_vram_gb=_get_gpu_vram_gb(),
            os_info=_get_os_info(),
            git_commit=_get_git_commit(),
        )

    def to_dict(self) -> dict[str, Any]:
        """
        Convert to dictionary for W&B/checkpoint storage.

        Returns:
            Dictionary with all environment information
        """
        return {
            "python_version": self.python_version,
            "torch_version": self.torch_version,
            "cuda_version": self.cuda_version,
            "transformers_version": self.transformers_version,
            "peft_version": self.peft_version,
            "trl_version": self.trl_version,
            "accelerate_version": self.accelerate_version,
            "bitsandbytes_version": self.bitsandbytes_version,
            "gpu_name": self.gpu_name,
            "gpu_vram_gb": self.gpu_vram_gb,
            "os_info": self.os_info,
            "git_commit": self.git_commit,
            "timestamp": self.timestamp,
        }


class EnvironmentReporter:
    """
    Reports environment information for training reproducibility.

    Collects environment snapshot and provides formatted logging
    and serialization for W&B and checkpoint metadata.

    Example:
        >>> reporter = EnvironmentReporter.collect()
        >>> reporter.log_summary()
        🔧 Environment:
           Python: 3.11.5
           ...

        >>> # For checkpoint metadata
        >>> checkpoint_meta = {"environment": reporter.to_dict()}
    """

    def __init__(self, snapshot: EnvironmentSnapshot):
        """
        Initialize reporter with environment snapshot.

        Args:
            snapshot: Pre-collected environment snapshot
        """
        self.snapshot = snapshot

    @classmethod
    def collect(cls) -> EnvironmentReporter:
        """
        Collect current environment and create reporter.

        Returns:
            EnvironmentReporter: Ready to log/export
        """
        snapshot = EnvironmentSnapshot.collect()
        return cls(snapshot)

    def log_summary(self) -> None:
        """
        Log environment summary to console.

        Formats all environment information in a readable format.
        """
        s = self.snapshot

        logger.info("🔧 Environment:")
        logger.info(f"   Python: {s.python_version}")
        logger.info(f"   PyTorch: {s.torch_version}")
        logger.info(f"   CUDA: {s.cuda_version}")
        logger.info(f"   Transformers: {s.transformers_version}")
        logger.info(f"   PEFT: {s.peft_version}")
        logger.info(f"   TRL: {s.trl_version}")
        logger.info(f"   Accelerate: {s.accelerate_version}")
        logger.info(f"   BitsAndBytes: {s.bitsandbytes_version}")
        logger.info(f"   GPU: {s.gpu_name} ({s.gpu_vram_gb:.1f}GB)")
        logger.info(f"   OS: {s.os_info}")
        if s.git_commit != _VERSION_UNAVAILABLE:
            logger.info(f"   Git: {s.git_commit}")

    def log_debug(self) -> None:
        """Log detailed environment info at DEBUG level."""
        s = self.snapshot
        logger.debug(f"[ENV:SNAPSHOT] {s.to_dict()}")

    def to_dict(self) -> dict[str, Any]:
        """
        Get environment as dictionary.

        Returns:
            Dictionary for checkpoint metadata
        """
        return self.snapshot.to_dict()

    def get_short_summary(self) -> str:
        """
        Get one-line summary.

        Returns:
            Short summary string like "Python 3.11, PyTorch 2.1.0, CUDA 11.8"
        """
        s = self.snapshot
        parts = [
            f"Python {s.python_version.split()[0]}",
            f"PyTorch {s.torch_version.split('+')[0]}",
            f"CUDA {s.cuda_version}",
        ]
        return ", ".join(parts)


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================


def _get_python_version() -> str:
    """Get Python version string."""
    return platform.python_version()


def _get_package_version(package_name: str) -> str:
    """
    Get package version or 'N/A' if not installed.

    Args:
        package_name: Name of package to check

    Returns:
        Version string or 'N/A'
    """
    try:
        return importlib_metadata.version(package_name)
    except importlib_metadata.PackageNotFoundError:
        return _VERSION_UNAVAILABLE


def _get_cuda_version() -> str:
    """Get CUDA version from PyTorch or 'N/A'."""
    try:
        import torch as torch_module

        torch: Any = torch_module

        if torch.cuda.is_available():
            return torch.version.cuda or _VERSION_UNAVAILABLE
        return "N/A (CPU mode)"
    except ImportError:
        return _VERSION_UNAVAILABLE


def _get_gpu_name() -> str:
    """Get GPU name or 'CPU'."""
    try:
        import torch as torch_module

        torch: Any = torch_module

        if torch.cuda.is_available():
            return torch.cuda.get_device_name(0)
        return "CPU"
    except ImportError:
        return "CPU"


def _get_gpu_vram_gb() -> float:
    """Get GPU VRAM in GB or 0."""
    try:
        import torch as torch_module

        torch: Any = torch_module

        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(0)
            return props.total_memory / (1024**3)
        return 0.0
    except ImportError:
        return 0.0


def _get_os_info() -> str:
    """Get OS name and version."""
    return f"{platform.system()} {platform.release()}"


def _get_git_commit() -> str:
    """Get current git commit hash or 'N/A'."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
            cwd=Path(__file__).parent.parent.parent,
        )
        if result.returncode == 0:
            return result.stdout.strip()
        return _VERSION_UNAVAILABLE
    except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
        return _VERSION_UNAVAILABLE


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    "EnvironmentReporter",
    "EnvironmentSnapshot",
]
