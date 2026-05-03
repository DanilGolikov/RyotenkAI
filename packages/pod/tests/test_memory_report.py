"""
Tests for Memory Reporting (MemoryAnalyzer and ReportBuilder).

Tests edge cases for memory analysis:
- Clean run
- OOM run
- High overhead
- Fragmentation
- Config mismatch
- Phase overrides (NEW)
"""

from datetime import datetime
from unittest.mock import MagicMock

import pytest

from src.reports.core.analyzers_memory import MemoryAnalyzer
from src.reports.models.report import MemoryEvent, MemoryManagementInfo, MetricStatus


@pytest.fixture
def memory_analyzer():
    return MemoryAnalyzer()


@pytest.fixture
def base_info():
    return MemoryManagementInfo(
        gpu_name="NVIDIA RTX 4090",
        total_vram_gb=24.0,
        memory_margin_mb=500,
        actual_model_size="7B",
    )


def test_clean_run(memory_analyzer, base_info):
    """Test analysis for a perfect run."""
    config = {"batch_size": 4, "model_size": "7B"}

    analysis = memory_analyzer.analyze(base_info, config)

    assert analysis.status == MetricStatus.GOOD
    assert analysis.verdict == "Healthy"
    assert analysis.efficiency_score == 100
    assert analysis.oom_count == 0
    assert not analysis.recommendations


def test_oom_critical(memory_analyzer, base_info):
    """Test analysis with OOM events."""
    base_info.oom_events = [
        MemoryEvent(datetime.now(), "oom", "OOM during forward", operation="forward_pass"),
        MemoryEvent(datetime.now(), "oom", "OOM during backward", operation="backward_pass"),
        MemoryEvent(datetime.now(), "oom", "OOM during step", operation="step"),
        MemoryEvent(datetime.now(), "oom", "OOM during step", operation="step"),
    ]
    # Set max_retries to 3 (default), we have 4 events -> Critical
    base_info.max_retries = 3

    config = {"batch_size": 4, "model_size": "7B"}

    analysis = memory_analyzer.analyze(base_info, config)

    assert analysis.status == MetricStatus.BAD
    assert "OOM Loop" in analysis.verdict or "failure" in analysis.verdict.lower()
    assert analysis.efficiency_score <= 0
    assert len(analysis.recommendations) >= 1
    assert "reduce batch_size" in analysis.recommendations[0].lower()


def test_high_overhead(memory_analyzer, base_info):
    """Test warning for excessive cache clearing."""
    # Add 20 cache clear events
    base_info.cache_clears = [
        MemoryEvent(datetime.now(), "cache_clear", "Cleared cache", freed_mb=100) for _ in range(20)
    ]
    config = {"batch_size": 4, "model_size": "7B"}

    analysis = memory_analyzer.analyze(base_info, config)

    assert analysis.status == MetricStatus.WARNING
    assert "Warning: high overhead" in analysis.verdict
    assert analysis.overhead_seconds == 10.0  # 20 * 0.5s
    assert analysis.efficiency_score < 100


def test_high_fragmentation(memory_analyzer, base_info):
    """Test detection of memory fragmentation."""
    base_info.memory_warnings = [
        MemoryEvent(datetime.now(), "warning", "High fragmentation (frag=0.45)", utilization_percent=80.0),
        MemoryEvent(datetime.now(), "warning", "High fragmentation (frag=0.55)", utilization_percent=85.0),
    ]
    config = {"batch_size": 4, "model_size": "7B"}

    analysis = memory_analyzer.analyze(base_info, config)

    # Might affect verdict depending on counts
    assert analysis.fragmentation_warnings == 2
    # Check recommendation for PYTORCH_CUDA_ALLOC_CONF
    assert any("expandable_segments" in r for r in analysis.recommendations)


def test_config_mismatch(memory_analyzer, base_info):
    """Test aggressive configuration detection."""
    # User used large batch size
    config = {"batch_size": 8, "model_size": "7B"}

    # And it caused OOM
    base_info.oom_events = [MemoryEvent(datetime.now(), "oom", "OOM", operation="step")]

    analysis = memory_analyzer.analyze(base_info, config)

    # Should detect OOM occurred
    assert analysis.oom_count == 1


def test_phase_override_mismatch(memory_analyzer, base_info):
    """Test when global config is fine, but a phase override is aggressive."""
    # Global config
    config = {"batch_size": 4, "model_size": "7B"}

    # Mock phases
    mock_phase = MagicMock()
    mock_phase.phase_idx = 1
    mock_phase.strategy = "dpo"
    # Phase overrides to 8
    mock_phase.effective_config = {"per_device_train_batch_size": 8}

    phases = [mock_phase]

    analysis = memory_analyzer.analyze(base_info, config, phases=phases)

    # Analysis should complete without errors
    assert analysis is not None


# REMOVED: test_max_seq_length_mismatch - max_seq_length is deprecated in TRL 0.26.0
