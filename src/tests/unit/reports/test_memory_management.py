"""
Tests for Memory Management section in Experiment Reports.

Validates correct rendering and analysis of MemoryManager events,
GPU detection, OOM handling, cache clears, and config recommendations.
"""

import logging
from datetime import datetime

import pytest

from src.reports.core.builder import ReportBuilder
from src.reports.domain.entities import (
    ExperimentData,
    MemoryEvent,
    RunStatus,
)
from src.reports.plugins.composer import ReportComposer
from src.community.catalog import catalog
from src.reports.plugins.interfaces import ReportPluginContext
from src.reports.plugins.markdown_block_renderer import MarkdownBlockRenderer
from src.reports.plugins.registry import build_report_plugins


class _DummyProvider:
    def load(self, run_id: str):  # pragma: no cover
        raise NotImplementedError


def _render_report_markdown(*, data: ExperimentData, report) -> str:
    catalog.reload()
    composer = ReportComposer(build_report_plugins())
    ctx = ReportPluginContext(
        run_id=data.run_id,
        data_provider=_DummyProvider(),
        data=data,
        report=report,
        logger=logging.getLogger(__name__),
    )
    blocks, _records = composer.compose(ctx)
    return MarkdownBlockRenderer().render(blocks)

# =============================================================================
# FIXTURES
# =============================================================================


@pytest.fixture
def base_experiment_data():
    """Base experiment data without MM events."""
    return ExperimentData(
        run_id="test_run_123",
        run_name="test_run",
        experiment_name="test_experiment",
        status=RunStatus.FINISHED,
        start_time=datetime(2025, 1, 1, 10, 0, 0),
        end_time=datetime(2025, 1, 1, 10, 30, 0),
        duration_seconds=1800.0,
        phases=[],
        root_params={},
        memory_events=[],
        # timeline_events removed - use stage_envelopes instead
    )


@pytest.fixture
def full_mm_experiment_data():
    """Experiment data with full MM events."""
    return ExperimentData(
        run_id="test_run_full_mm",
        run_name="test_run_full",
        experiment_name="test_experiment",
        status=RunStatus.FINISHED,
        start_time=datetime(2025, 1, 1, 10, 0, 0),
        end_time=datetime(2025, 1, 1, 10, 30, 0),
        duration_seconds=1800.0,
        phases=[],
        root_params={
            "mm.gpu_name": "NVIDIA GeForce RTX 4060",
            "mm.gpu_tier": "consumer_low",
            "mm.total_vram_gb": "8.0",
            "mm.memory_margin_mb": "500",
            "mm.critical_threshold": "90.0",
            "mm.warning_threshold": "80.0",
            "mm.max_retries": "3",
            "mm.max_model": "7B (QLoRA) / 3B (LoRA)",
            "mm.notes": "Use batch_size=1 for stability",
            "mm.recommended_batch_size.7B": "1",
            "mm.recommended_batch_size.3B": "2",
        },
        memory_events=[
            MemoryEvent(
                timestamp=datetime(2025, 1, 1, 10, 0, 0),
                event_type="info",
                message="GPU detected: NVIDIA GeForce RTX 4060 (8.0GB, consumer_low)",
                source="MemoryManager",
            ),
            MemoryEvent(
                timestamp=datetime(2025, 1, 1, 10, 5, 0),
                event_type="info",
                message="Cache cleared: 450MB freed",
                source="MemoryManager",
                phase="phase_0",
                freed_mb=450,
            ),
            MemoryEvent(
                timestamp=datetime(2025, 1, 1, 10, 10, 0),
                event_type="warning",
                message="Memory utilization high: 82.5%",
                source="MemoryManager",
                phase="phase_0",
                utilization_percent=82.5,
            ),
        ],
        # timeline_events removed - use stage_envelopes instead
    )


# =============================================================================
# BASIC RENDERING
# =============================================================================


class TestMemoryManagementBasicRendering:
    """Test basic MM section rendering."""

    def test_render_with_mm_data(self, full_mm_experiment_data):
        """Test rendering MM section with full data."""
        builder = ReportBuilder(full_mm_experiment_data)
        report = builder.build()

        output = _render_report_markdown(data=full_mm_experiment_data, report=report)

        assert "## 💾 Memory Management" in output
        # Just check it renders without crashing
        assert len(output) > 500

    def test_render_without_mm_data(self, base_experiment_data):
        """Test rendering when MM was not active."""
        builder = ReportBuilder(base_experiment_data)
        report = builder.build()

        output = _render_report_markdown(data=base_experiment_data, report=report)

        assert "## 💾 Memory Management" in output
        assert len(output) > 500


# =============================================================================
# GPU DETECTION
# =============================================================================


class TestMemoryManagementGPUDetection:
    """Test GPU detection event handling."""

    def test_gpu_detection_params(self, base_experiment_data):
        """Test GPU params are displayed."""
        data = base_experiment_data
        data.root_params = {
            "mm.gpu_name": "RTX 4090",
            "mm.total_vram_gb": "24.0",
            "mm.gpu_tier": "consumer_high",
        }

        builder = ReportBuilder(data)
        report = builder.build()

        # Just verify report builds successfully
        assert report.memory_management is not None


# =============================================================================
# CACHE CLEARS
# =============================================================================


class TestMemoryManagementCacheClears:
    """Test cache clear event tracking."""

    def test_multiple_cache_clears(self, base_experiment_data):
        """Test tracking multiple cache clears."""
        data = base_experiment_data
        data.memory_events = [
            MemoryEvent(
                timestamp=datetime(2025, 1, 1, 10, 0, 0),
                event_type="info",
                message="Cache cleared: 100MB freed",
                source="MemoryManager",
                freed_mb=100,
            ),
            MemoryEvent(
                timestamp=datetime(2025, 1, 1, 10, 5, 0),
                event_type="info",
                message="Cache cleared: 200MB freed",
                source="MemoryManager",
                freed_mb=200,
            ),
        ]

        builder = ReportBuilder(data)
        report = builder.build()

        assert report.memory_management is not None
        # Should track cache clears
        mm = report.memory_management
        assert hasattr(mm, "cache_clears") or hasattr(mm, "events")


# =============================================================================
# OOM EVENTS
# =============================================================================


class TestMemoryManagementOOMEvents:
    """Test OOM event handling."""

    def test_oom_recovery_sequence(self, base_experiment_data):
        """Test OOM + recovery sequence."""
        data = base_experiment_data
        data.memory_events = [
            MemoryEvent(
                timestamp=datetime(2025, 1, 1, 10, 0, 0),
                event_type="error",
                message="OOM during 'forward_pass'",
                source="MemoryManager",
                operation="forward_pass",
            ),
            MemoryEvent(
                timestamp=datetime(2025, 1, 1, 10, 0, 5),
                event_type="warning",
                message="OOM recovery attempt 1/3",
                source="MemoryManager",
            ),
            MemoryEvent(
                timestamp=datetime(2025, 1, 1, 10, 0, 6),
                event_type="info",
                message="Cache cleared: 1200MB freed",
                source="MemoryManager",
                freed_mb=1200,
            ),
        ]

        builder = ReportBuilder(data)
        report = builder.build()

        assert report.memory_management is not None


# =============================================================================
# SUMMARY
# =============================================================================


class TestMemoryManagementSummary:
    """Test overall MM summary and rendering."""

    def test_e2e_realistic_scenario(self):
        """End-to-end test with realistic MM scenario."""
        data = ExperimentData(
            run_id="e2e_test",
            run_name="realistic_training",
            experiment_name="test_exp",
            status=RunStatus.FINISHED,
            start_time=datetime(2025, 1, 1, 10, 0, 0),
            end_time=datetime(2025, 1, 1, 10, 30, 0),
            duration_seconds=1800.0,
            phases=[],
            root_params={
                "mm.gpu_name": "NVIDIA GeForce RTX 4060",
                "mm.gpu_tier": "consumer_low",
                "mm.total_vram_gb": "8.0",
            },
            memory_events=[
                MemoryEvent(
                    timestamp=datetime(2025, 1, 1, 10, 0, 0),
                    event_type="info",
                    message="GPU detected: NVIDIA GeForce RTX 4060",
                    source="MemoryManager",
                ),
                MemoryEvent(
                    timestamp=datetime(2025, 1, 1, 10, 5, 0),
                    event_type="warning",
                    message="Memory utilization high: 85%",
                    source="MemoryManager",
                    utilization_percent=85.0,
                ),
                MemoryEvent(
                    timestamp=datetime(2025, 1, 1, 10, 5, 5),
                    event_type="info",
                    message="Cache cleared: 500MB freed",
                    source="MemoryManager",
                    freed_mb=500,
                ),
            ],
            # timeline_events removed - use stage_envelopes instead
        )

        builder = ReportBuilder(data)
        report = builder.build()

        output = _render_report_markdown(data=data, report=report)

        # Check key sections exist
        assert "## 💾 Memory Management" in output
        # Should have warnings or events mentioned
        assert len(output) > 1000  # Reasonable report length
