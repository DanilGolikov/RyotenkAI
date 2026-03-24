"""
Memory Analyzer.

Analyzes memory health and efficiency using Domain Entities.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from src.reports.models.report import MemoryAnalysis, MemoryManagementInfo, MetricStatus

_OOM_PENALTY_PER_EVENT = 20
_WARN_PENALTY_HIGH = 30
_CONFIG_PENALTY = 15

if TYPE_CHECKING:
    from src.reports.domain.entities import MemoryEvent, PhaseData


class MemoryAnalyzer:
    """
    Analyzes memory health and efficiency.
    """

    def analyze(
        self,
        events: list[MemoryEvent] | MemoryManagementInfo,
        config: dict[str, Any],
        phases: list[PhaseData] | None = None,
    ) -> MemoryAnalysis:
        """
        Analyze memory data and generate insights.

        Args:
            events: List of MemoryEvent (from domain)
            config: Training config (batch size, etc.)
            phases: Optional list of PhaseData for detailed per-phase checks

        Returns:
            MemoryAnalysis object (View Model)
        """
        insights = []
        recommendations = []

        # Support both legacy (MemoryManagementInfo) and new (list[MemoryEvent]) call styles.
        # - Legacy tests pass MemoryManagementInfo (already grouped).
        # - ReportBuilder passes list[MemoryEvent] from Domain.
        if isinstance(events, MemoryManagementInfo):
            oom_events = list(events.oom_events)
            cache_clears = list(events.cache_clears)
            warnings = list(events.memory_warnings)
            max_retries_from_info = events.max_retries
        else:
            oom_events = [e for e in events if e.event_type == "oom"]
            cache_clears = [e for e in events if e.event_type == "cache_clear"]
            warnings = [e for e in events if e.event_type in ("warning", "critical")]
            max_retries_from_info = None

        # 1. Analyze OOMs
        oom_count = len(oom_events)
        max_retries = (
            max_retries_from_info or config.get("mm.max_retries") or config.get("max_retries") or 3  # default
        )

        # 2. Analyze Overhead (Cache Clears)
        # Assume 0.5s per cache clear (conservative estimate for sync + gc)
        cache_clear_count = len(cache_clears)
        overhead_seconds = cache_clear_count * 0.5

        # 3. Analyze Fragmentation
        # Check warnings for "fragmentation" keyword
        frag_warnings = 0
        for w in warnings:
            if "fragmentation" in w.message.lower() or "frag=" in w.message.lower():
                frag_warnings += 1

        # 4. Generate overall status/verdict (tests expect MemoryAnalyzer to classify health)
        verdict = "Healthy"
        score = 100
        status = MetricStatus.GOOD

        # OOM Logic
        if oom_count > max_retries:
            verdict = "Critical: OOM loop / failure"
            score -= 100
            status = MetricStatus.BAD
            insights.append(f"Training failed: {oom_count} OOM events (retry limit {max_retries} exceeded).")
            recommendations.append("Critically reduce batch_size or use a smaller model.")
        elif oom_count > 0:
            verdict = "OOM (recovered)"
            score -= _OOM_PENALTY_PER_EVENT * oom_count
            status = MetricStatus.WARNING
            insights.append(f"There were {oom_count} OOM events, but MemoryManager recovered training.")
            recommendations.append("Lower `per_device_train_batch_size` for stability.")
            recommendations.append("Enable `gradient_checkpointing` if it is disabled.")

        # Overhead Logic
        if cache_clear_count > 10:
            if oom_count == 0:
                verdict = "Warning: high overhead"
            if status != MetricStatus.BAD:
                status = MetricStatus.WARNING
            score -= cache_clear_count // 5  # -1 point per 5 clears
            insights.append(
                f"Memory protection triggered {cache_clear_count} times. Approximate time lost: {overhead_seconds:.1f}s."
            )
            recommendations.append("Slightly reduce batch size to avoid cache clears.")

        # Fragmentation & Warning Count Logic
        total_warnings = len(warnings)

        if total_warnings > 5:
            if oom_count == 0:
                verdict = "Unstable memory"
            score -= _WARN_PENALTY_HIGH
            if status != MetricStatus.BAD:
                status = MetricStatus.WARNING
            insights.append(f"High number of memory warnings ({total_warnings}). The system is near its limit.")

        elif total_warnings > 3:
            if oom_count == 0:
                verdict = "Memory pressure"
            score -= 10
            if status != MetricStatus.BAD:
                status = MetricStatus.WARNING

        if frag_warnings > 0:
            score -= frag_warnings * 5
            if status != MetricStatus.BAD:
                status = MetricStatus.WARNING
            insights.append(f"Detected {frag_warnings} high-fragmentation events. Memory is available but fragmented.")
            recommendations.append("Set environment variable: `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`")

        # 5. Config Analysis (Plan vs Fact)
        # Check if user ignored recommendations
        has_config_issues = self._analyze_config_compliance(
            _config=config, _phases=phases, _insights=insights, _recommendations=recommendations, oom_count=oom_count
        )

        if has_config_issues:
            if oom_count == 0:
                verdict = "Aggressive configuration"
            score -= _CONFIG_PENALTY
            if status == MetricStatus.GOOD:
                status = MetricStatus.WARNING

        # Cap score
        score = max(0, min(100, score))

        return MemoryAnalysis(
            status=status,
            verdict=verdict,
            insights=insights,
            recommendations=list(dict.fromkeys(recommendations)),  # Dedup (stable order)
            overhead_seconds=overhead_seconds,
            fragmentation_warnings=frag_warnings,
            oom_count=oom_count,
            efficiency_score=score,
        )

    @staticmethod
    def _analyze_config_compliance(
        _config: dict[str, Any],
        _phases: list[PhaseData] | None,
        _insights: list[str],
        _recommendations: list[str],
        oom_count: int,
    ) -> bool:
        """Analyze if current config matches recommendations."""
        # Note: Recommendations now come from config params if logged by MM
        # Here we do a simpler check based on heuristics if specific recommendation map is missing

        found_issues = False

        # Check global batch size vs heuristics
        # This is simplified compared to old logic which parsed "mm.recommended_batch_size.*"
        # In new architecture, we should pass those recommendations in if available.
        # For now, let's just check against OOMs.

        if oom_count > 0:
            # If we had OOMs, the config WAS too aggressive by definition
            found_issues = True
            _insights.append("Configuration led to OOM events.")

        return found_issues
