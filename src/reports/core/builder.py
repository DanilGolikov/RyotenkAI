"""
Report Builder.

Transforms clean Domain Data (ExperimentData) into View Model (ExperimentReport).
Focuses on business logic, aggregation, and presentation rules.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from src.reports.core.analyzers import (
    MetricAnalyzer,
    PercentileCalculator,
)
from src.reports.core.analyzers_memory import MemoryAnalyzer
from src.reports.core.health_policy import DEFAULT_HEALTH_POLICY, HealthPolicy
from src.reports.core.metrics_registry import get_metrics_for_strategy
from src.reports.domain.entities import (
    DatasetValidation,
    EvalPluginResult,
    EvaluationReport,
    ExperimentData,
    MetricHistory,
    MetricTrend,
    PhaseData,
    ValidationPluginResults,
    ValidationReport,
)
from src.reports.models.report import (
    ConfigInfo,
    ExperimentHealth,
    ExperimentReport,
    Issue,
    MemoryManagementInfo,
    MemoryPhaseStats,
    MetricStatus,
    ModelInfo,
    PercentileStats,
    PhaseInfo,
    ReportSummary,
    ResourcesInfo,
    TimelineEvent,
)
from src.utils.logger import get_logger

logger = get_logger(__name__)

BUILDER_MS_CAP = 30000
KEY_ATTRIBUTES = "attributes"
KEY_FAILED = "failed"
KEY_LOADED = "loaded"
KEY_MESSAGE = "message"
KEY_NAME = "name"
KEY_PASSED = "passed"
KEY_SAMPLE_COUNT = "sample_count"
KEY_STATUS = "status"
KEY_UNKNOWN = "Unknown"
KEY_UNKNOWN_LOWER = "unknown"
KEY_WARN = "WARN"
SYS_GPU_UTIL = "system/gpu_0_utilization_percentage"
SYS_GPU_MEM_MB = "system/gpu_0_memory_usage_megabytes"
SYS_GPU_MEM_PCT = "system/gpu_0_memory_usage_percentage"
SYS_CPU_UTIL = "system/cpu_utilization_percentage"
SYS_MEM_MB = "system/system_memory_usage_megabytes"
SYS_MEM_PCT = "system/system_memory_usage_percentage"


def _scalar(value: Any, default: Any = 0) -> Any:
    """Return value as-is unless it is a list/tuple, in which case take the last element.

    MLflow sometimes stores metric history as a list instead of a scalar
    (e.g. when aggregate_metrics copies history arrays into run.data.metrics).
    This guard prevents int()/float() from crashing on list inputs.
    """
    if isinstance(value, list | tuple):
        return value[-1] if value else default
    return value if value is not None else default


def _failed_samples_count(value: Any) -> int:
    """Return the number of failed samples.

    failed_samples in EvalResult is stored as list[int] (indices of failed samples).
    Builder needs just the count for display purposes.
    """
    if isinstance(value, list):
        return len(value)
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


class ReportBuilder:
    """
    Builds ExperimentReport from ExperimentData.
    """

    def __init__(self, data: ExperimentData):
        self._data: ExperimentData = data

        # Analyzers
        self._metric_analyzer = MetricAnalyzer()
        self._memory_analyzer = MemoryAnalyzer()
        self._percentile_calc = PercentileCalculator()

    def build(self) -> ExperimentReport:
        """Build complete experiment report."""
        logger.debug(f"[BUILDER] Building report for {self._data.run_id[:8]}")

        # 1. Build Phases (and run analysis per phase)
        phases = self._build_phases()

        # 2. Build Memory Info (and run analysis)
        mm_info = self._build_memory_management(phases)

        # 3. Build Validation (BEFORE issues, to include validation warnings in issues)
        validation = self._build_validation()

        # 4. Build Evaluation (BEFORE issues)
        evaluation = self._build_evaluation()

        # 5. Build Issues (including global errors + validation warnings)
        issues = self._build_issues(phases, mm_info, validation, evaluation)

        # 6. Build Summary (using detected issues for health)
        summary = self._build_summary(phases, mm_info, issues)

        return ExperimentReport(
            generated_at=self._data.end_time or self._data.start_time or datetime.now(),
            summary=summary,
            model=self._build_model(),
            phases=phases,
            resources=self._build_resources(),
            timeline=self._build_timeline(),
            issues=issues,
            config=self._build_config(),
            memory_management=mm_info,
            validation=validation,
            evaluation=evaluation,
        )

    # =========================================================================
    # SUMMARY
    # =========================================================================

    def _build_summary(
        self, phases: list[PhaseInfo], _mm_info: MemoryManagementInfo | None, issues: list[Issue]
    ) -> ReportSummary:
        """Build high-level summary."""
        d = self._data

        # Aggregate totals
        total_epochs = int(sum((p.epochs if p.epochs is not None else 0.0) for p in phases))
        total_steps = int(sum((p.steps if p.steps is not None else 0) for p in phases))

        # Calculate rates
        sps_values = [p.samples_per_second for p in phases if p.samples_per_second]
        avg_sps = sum(sps_values) / len(sps_values) if sps_values else None

        # Final loss (from last phase)
        final_loss = phases[-1].final_loss if phases else None

        # Health (NEW: via HealthPolicy)
        health, health_explanation = self._determine_health(issues)

        return ReportSummary(
            run_id=d.run_id,
            run_name=d.run_name,
            experiment_name=d.experiment_name,
            status=d.status,  # RunStatus matches
            health=health,
            health_explanation=health_explanation,
            start_time=d.start_time,
            end_time=d.end_time,
            duration_total_seconds=d.duration_seconds,
            duration_formatted=self._format_duration(d.duration_seconds),
            final_loss=final_loss,
            total_epochs=total_epochs,
            total_steps=total_steps,
            samples_per_second=avg_sps,
            steps_per_second=None,  # TODO: Add to domain if needed
            strategy_chain=[p.strategy for p in phases],
        )

    def _determine_health(self, issues: list[Issue]) -> tuple[ExperimentHealth, str]:
        """
        Determine overall experiment health via HealthPolicy.

        Args:
            issues: List of all detected issues

        Returns:
            (ExperimentHealth, explanation)
        """
        policy = HealthPolicy(settings=DEFAULT_HEALTH_POLICY)
        health, explanation = policy.evaluate(
            run_status=self._data.status,
            issues=issues,
        )
        return health, explanation

    # =========================================================================
    # PHASES
    # =========================================================================

    def _build_phases(self) -> list[PhaseInfo]:
        """Convert Domain Phases to View Phases with analysis."""
        view_phases = []

        for p in self._data.phases:
            # 1. Ensure resource history is populated (slice from global if missing)
            self._ensure_phase_resource_history(p)

            # 2. Calculate Trends
            loss_trend = self._calc_trend(p.history.get("loss")) or self._calc_trend(p.history.get("train_loss"))

            # 3. Run Metric Analysis
            metrics_analysis = []
            metrics_to_check = get_metrics_for_strategy(p.strategy)

            for key in metrics_to_check:
                hist = p.history.get(key)
                if hist:
                    trend = self._calc_trend(hist)
                    if trend:
                        analysis = self._metric_analyzer.analyze(key, trend)
                        if analysis:
                            metrics_analysis.append(analysis)

            # 4. Resources (Per Phase) - Calculate avg for PhaseInfo
            gpu_util = self._percentile_calc.calculate(
                p.history.get(SYS_GPU_UTIL, MetricHistory("", [], [], [])).values
            )
            gpu_mem_mb = self._percentile_calc.calculate(
                p.history.get(SYS_GPU_MEM_MB, MetricHistory("", [], [], [])).values
            )
            gpu_mem_pct = self._percentile_calc.calculate(
                p.history.get(SYS_GPU_MEM_PCT, MetricHistory("", [], [], [])).values
            )
            cpu_util = self._percentile_calc.calculate(
                p.history.get(SYS_CPU_UTIL, MetricHistory("", [], [], [])).values
            )
            sys_mem_mb = self._percentile_calc.calculate(
                p.history.get(SYS_MEM_MB, MetricHistory("", [], [], [])).values
            )
            sys_mem_pct = self._percentile_calc.calculate(
                p.history.get(SYS_MEM_PCT, MetricHistory("", [], [], [])).values
            )

            view_phases.append(
                PhaseInfo(
                    phase_idx=p.idx,
                    run_id=self._data.run_id,  # Link back to parent for now
                    run_name=p.name,
                    strategy=p.strategy,
                    status=p.status,
                    duration_seconds=p.duration_seconds,
                    final_loss=_scalar(p.metrics.get("train_loss") or p.metrics.get("loss")),
                    loss_trend=loss_trend or MetricTrend(),
                    train_loss_trend=loss_trend or MetricTrend(),  # Duplicate for safety
                    metrics_analysis=metrics_analysis,
                    epochs=_scalar(p.metrics.get("epoch")),
                    steps=int(_scalar(p.metrics.get("global_step"), 0)),
                    samples_per_second=_scalar(p.metrics.get("train_samples_per_second")),
                    effective_config=p.config,
                    # Resource Averages
                    gpu_utilization=gpu_util.avg,
                    gpu_memory_mb=gpu_mem_mb.avg,
                    gpu_memory_percent=gpu_mem_pct.avg,
                    cpu_utilization=cpu_util.avg,
                    system_memory_mb=sys_mem_mb.avg,
                    system_memory_percent=sys_mem_pct.avg,
                )
            )

        return view_phases

    def _ensure_phase_resource_history(self, phase: PhaseData):
        """
        Populate phase resource history by slicing global history if local history is empty.
        Crucial for short phases where MLflow didn't capture a sample attached to the phase run.
        """
        resource_keys = [
            SYS_GPU_UTIL,
            SYS_GPU_MEM_MB,
            SYS_GPU_MEM_PCT,
            SYS_CPU_UTIL,
            SYS_MEM_MB,
            SYS_MEM_PCT,
        ]

        for key in resource_keys:
            if key not in phase.history or not phase.history[key].values:
                # Try to slice from global history
                sliced = self._slice_resource_history(phase.start_time, phase.end_time, key)
                if sliced:
                    phase.history[key] = sliced

    def _slice_resource_history(self, start: datetime | None, end: datetime | None, key: str) -> MetricHistory | None:
        """Slice global history for a specific time window."""
        global_hist = self._data.resource_history.get(key)
        if not global_hist or not global_hist.timestamps:
            return None

        if not start:
            return None  # Cannot slice without start time

        start_ts = start.timestamp() * 1000
        end_ts = (end.timestamp() * 1000) if end else (datetime.now().timestamp() * 1000)

        # 1. Filter points within window
        indices = [i for i, t in enumerate(global_hist.timestamps) if start_ts <= t <= end_ts]

        if indices:
            return MetricHistory(
                key=key,
                values=[global_hist.values[i] for i in indices],
                steps=[global_hist.steps[i] for i in indices],
                timestamps=[global_hist.timestamps[i] for i in indices],
            )

        # 2. If no points in window (short phase), find nearest point
        # This prevents "zero" stats for short phases
        # Find point with minimum distance to center of phase
        center_ts = (start_ts + end_ts) / 2
        nearest_idx = min(range(len(global_hist.timestamps)), key=lambda i: abs(global_hist.timestamps[i] - center_ts))

        # Only use nearest if it's reasonably close (e.g. within 30s)
        if abs(global_hist.timestamps[nearest_idx] - center_ts) < BUILDER_MS_CAP:
            return MetricHistory(
                key=key,
                values=[global_hist.values[nearest_idx]],
                steps=[global_hist.steps[nearest_idx]],
                timestamps=[global_hist.timestamps[nearest_idx]],
            )

        return None

    # =========================================================================
    # MEMORY
    # =========================================================================

    def _build_memory_management(self, phases: list[PhaseInfo]) -> MemoryManagementInfo:
        """Build Memory Info."""
        source = self._data.source_config
        params = self._data.root_params

        # Helper to extract mm params
        def get_mm(key: str):
            val = params.get(key) or params.get(f"mm.{key}") or source.get(key) or source.get(f"mm.{key}")
            if val is None:
                return None
            # Handle string conversions if needed (though adapter should have parsed)
            try:
                if "threshold" in key:
                    return float(val)
                if "mb" in key or "retries" in key:
                    return int(float(val))  # float->int safety
            except (ValueError, TypeError):
                pass
            return val

        # Build Stats per phase
        phase_stats: list[MemoryPhaseStats] = []
        domain_phases = self._data.phases
        for idx, p_info in enumerate(phases):
            if idx >= len(domain_phases):
                break
            p_data = domain_phases[idx]

            # Helper to calc stats from PhaseData history
            def calc_stats(key: str, phase_data: PhaseData = p_data) -> PercentileStats:  # Bind loop variable
                hist = phase_data.history.get(key)
                if hist and hist.values:
                    return self._percentile_calc.calculate(hist.values)
                return PercentileStats()

            mem_mb = calc_stats(SYS_GPU_MEM_MB)
            mem_pct = calc_stats(SYS_GPU_MEM_PCT)

            phase_stats.append(
                MemoryPhaseStats(
                    phase_idx=p_info.phase_idx,
                    strategy=p_info.strategy,
                    peak_mb=int(mem_mb.max_val or 0),
                    peak_percent=mem_pct.max_val or 0.0,
                    avg_mb=int(mem_mb.avg or 0),
                    avg_percent=mem_pct.avg or 0.0,
                    p95_mb=int(mem_mb.p95 or 0),
                    p99_mb=int(mem_mb.p99 or 0),
                )
            )

        # Create Info object
        mm_margin = get_mm("memory_margin_mb")
        mm_max_retries = get_mm("max_retries")
        mm_max_model = get_mm("max_model")
        mm_actual_model_size = get_mm("actual_model_size")
        mm_notes = get_mm("notes")

        info = MemoryManagementInfo(
            gpu_name=self._data.gpu_info.get(KEY_NAME),
            gpu_tier=self._data.gpu_info.get("tier"),
            total_vram_gb=self._data.gpu_info.get("total_vram_gb_raw"),
            memory_margin_mb=int(mm_margin) if mm_margin is not None else None,
            critical_threshold=get_mm("critical_threshold"),
            warning_threshold=get_mm("warning_threshold"),
            max_retries=int(mm_max_retries) if mm_max_retries is not None else None,
            max_model=str(mm_max_model) if mm_max_model is not None else None,
            actual_model_size=str(mm_actual_model_size) if mm_actual_model_size is not None else None,
            notes=str(mm_notes) if mm_notes is not None else None,
            oom_events=[e for e in self._data.memory_events if e.event_type == "oom"],
            cache_clears=[e for e in self._data.memory_events if e.event_type == "cache_clear"],
            memory_warnings=[e for e in self._data.memory_events if e.event_type in ("warning", "critical")],
            phase_stats=phase_stats,
        )

        # Run Analysis
        # We pass PhaseData (domain) to analyzer.
        # But here we have View Phases. We need to pass Domain Phases.
        # Ideally, Analyzer should accept View Models OR Domain Models.
        # Since we refactored Analyzer to accept Domain Models, let's pass self._data.phases

        # Merge all configs for analysis context
        full_config = self._data.source_config.copy()
        if domain_phases:
            full_config.update(domain_phases[0].config)

        info.analysis = self._memory_analyzer.analyze(
            events=self._data.memory_events, config=full_config, phases=domain_phases
        )

        return info

    # =========================================================================
    # CONFIG & MODEL
    # =========================================================================

    def _build_config(self) -> ConfigInfo:
        """Build config info."""
        # Prefer first phase config if available, else source config
        effective_conf = self._data.phases[0].config if self._data.phases else {}
        source_conf = self._data.source_config
        root_params = self._data.root_params

        # Merge for full config reconstruction: Root Params (structure) + Phase (actuals)
        combined_conf = root_params.copy()
        combined_conf.update(effective_conf)

        # Merger for specific fields
        def get(key: str) -> Any:
            return effective_conf.get(key) or root_params.get(key) or source_conf.get(key)

        return ConfigInfo(
            batch_size=get("batch_size"),
            learning_rate=get("learning_rate"),
            grad_accum=get("grad_accum"),  # Added
            scheduler=get("scheduler"),  # Added
            warmup_ratio=get("warmup_ratio"),  # Added
            num_epochs=get("num_epochs") or get("epochs"),
            max_steps=get("max_steps"),
            optimizer=get("optim") or get("optimizer"),
            weight_decay=get("weight_decay"),
            fp16=get("fp16"),
            bf16=get("bf16"),
            gradient_checkpointing=get("gradient_checkpointing"),
            full_config=source_conf,
            source_config=source_conf,
            params_config=combined_conf,
        )

    def _build_model(self) -> ModelInfo:
        """Build model info."""
        conf = self._data.phases[0].config if self._data.phases else {}
        source = self._data.source_config
        params = self._data.root_params

        # Try to find name in various places
        name = (
            conf.get("model_name")
            or source.get("model", {}).get(KEY_NAME)
            or params.get("model_name")
            or params.get("config.model.name")
            or KEY_UNKNOWN
        )

        # Build strategy chain from all phases
        if self._data.phases:
            strategies = []
            for phase in self._data.phases:
                strategy = phase.config.get("strategy_type")
                if strategy and strategy not in strategies:  # Avoid duplicates
                    strategies.append(strategy.upper())
            training_type = " → ".join(strategies) if strategies else conf.get("strategy_type", KEY_UNKNOWN)
        else:
            training_type = conf.get("strategy_type", KEY_UNKNOWN)

        return ModelInfo(
            name=name,
            training_type=training_type,
            trainable_parameters=params.get("trainable_parameters") or source.get("trainable_parameters"),
            total_parameters=params.get("total_parameters") or source.get("total_parameters"),
            trainable_percent=params.get("trainable_percent") or source.get("trainable_percent"),
            loading_time_seconds=params.get("model_loading_time_seconds"),
            model_size_mb=params.get("model_size_mb"),
            lora_rank=params.get("lora_r"),
            lora_alpha=params.get("lora_alpha"),
        )

    # =========================================================================
    # RESOURCES & TIMELINE
    # =========================================================================

    def _build_resources(self) -> ResourcesInfo:
        """Build resource stats from global history."""
        # Prefer global history from Training Run (continuous timeline)
        hist = self._data.resource_history

        if hist:
            gpu_util = hist.get(SYS_GPU_UTIL)
            gpu_mem_mb = hist.get(SYS_GPU_MEM_MB)
            gpu_mem_pct = hist.get(SYS_GPU_MEM_PCT)
            cpu_util = hist.get(SYS_CPU_UTIL)
            sys_mem_mb = hist.get(SYS_MEM_MB)
            sys_mem_pct = hist.get(SYS_MEM_PCT)

            # Helper to calculate stats
            def calc(h):
                if not h:
                    return PercentileStats()
                return self._percentile_calc.calculate(h.values)

            return ResourcesInfo(
                gpu_name=self._data.gpu_info.get(KEY_NAME),
                gpu_tier=self._data.gpu_info.get("tier"),
                gpu_vram_gb=self._data.gpu_info.get("vram") or self._data.gpu_info.get("total_vram"),
                gpu_utilization=calc(gpu_util),
                gpu_memory_mb=calc(gpu_mem_mb),
                gpu_memory_percent=calc(gpu_mem_pct),
                cpu_utilization=calc(cpu_util),
                system_memory_mb=calc(sys_mem_mb),
                system_memory_percent=calc(sys_mem_pct),
            )

        # Fallback: Aggregate history from all phases
        gpu_utils: list[float] = []
        gpu_mem_mbs: list[float] = []
        gpu_mem_pcts: list[float] = []
        cpu_utils: list[float] = []
        sys_mem_mbs: list[float] = []
        sys_mem_pcts: list[float] = []

        for p in self._data.phases:
            # Collect from phase history if available
            self._collect_metric(p, SYS_GPU_UTIL, gpu_utils)
            self._collect_metric(p, SYS_GPU_MEM_MB, gpu_mem_mbs)
            self._collect_metric(p, SYS_GPU_MEM_PCT, gpu_mem_pcts)
            self._collect_metric(p, SYS_CPU_UTIL, cpu_utils)
            self._collect_metric(p, SYS_MEM_MB, sys_mem_mbs)
            self._collect_metric(p, SYS_MEM_PCT, sys_mem_pcts)

        return ResourcesInfo(
            gpu_name=self._data.gpu_info.get(KEY_NAME),
            gpu_tier=self._data.gpu_info.get("tier"),
            gpu_vram_gb=self._data.gpu_info.get("vram") or self._data.gpu_info.get("total_vram"),
            gpu_utilization=self._percentile_calc.calculate(gpu_utils),
            gpu_memory_mb=self._percentile_calc.calculate(gpu_mem_mbs),
            gpu_memory_percent=self._percentile_calc.calculate(gpu_mem_pcts),
            cpu_utilization=self._percentile_calc.calculate(cpu_utils),
            system_memory_mb=self._percentile_calc.calculate(sys_mem_mbs),
            system_memory_percent=self._percentile_calc.calculate(sys_mem_pcts),
        )

    @staticmethod
    def _collect_metric(phase: PhaseData, key: str, target: list[float]):
        """Helper to collect metric values from phase history."""
        hist = phase.history.get(key)
        if hist and hist.values:
            target.extend(hist.values)
        else:
            # Fallback to single value metric if available
            val = phase.metrics.get(key)
            if val is not None:
                target.append(val)

    def _build_timeline(self) -> list[TimelineEvent]:
        """Build timeline from per-stage envelopes.

        Each stage envelope provides: stage name, status, started_at, duration_seconds.
        Returns one TimelineEvent per stage (sorted by started_at).

        Falls back to an empty list when no stage_envelopes are available
        (e.g. old runs loaded for backward compat by MLflowAdapter).
        """
        if not self._data.stage_envelopes:
            return []

        timeline = []
        for envelope in self._data.stage_envelopes:
            ts = self._parse_ts(envelope.started_at) if envelope.started_at else None
            timeline.append(
                TimelineEvent(
                    timestamp=ts,
                    event_type="stage",
                    message=f"{envelope.stage}: {envelope.status} ({envelope.duration_seconds:.1f}s)",
                    source=envelope.stage,
                    origin="pipeline",
                    category="pipeline",
                    severity="ERROR" if envelope.status == "failed" else "INFO",
                    attributes={
                        "stage": envelope.stage,
                        "status": envelope.status,
                        "duration_seconds": envelope.duration_seconds,
                        "error": envelope.error,
                    },
                )
            )

        timeline.sort(key=lambda x: x.timestamp or datetime.min)
        return timeline

    def _build_issues(
        self,
        phases: list[PhaseInfo],
        mm_info: MemoryManagementInfo,
        validation: ValidationReport | None,
        evaluation: EvaluationReport | None = None,
    ) -> list[Issue]:
        """Aggregate issues from stage envelopes, metrics analysis, memory and validation."""
        issues = []

        # 1. From Metrics
        for p in phases:
            for m in p.metrics_analysis:
                if m.status in (MetricStatus.BAD, MetricStatus.WARNING):
                    issues.append(
                        Issue(
                            severity="ERROR" if m.status == MetricStatus.BAD else KEY_WARN,
                            message=f"{m.display_name}: {m.verdict}",
                            context=p.display_name,
                        )
                    )

        # 2. From Memory
        if mm_info.analysis:
            for insight in mm_info.analysis.insights:
                issues.append(Issue(severity=KEY_WARN, message=insight, context="Memory"))

        # 3. From Stage Envelopes
        for envelope in self._data.stage_envelopes:
            if envelope.status == "failed":
                issues.append(
                    Issue(
                        severity="ERROR",
                        message=envelope.error or f"Stage '{envelope.stage}' failed",
                        context=envelope.stage,
                    )
                )
            elif envelope.status == "interrupted":
                issues.append(
                    Issue(
                        severity=KEY_WARN,
                        message=f"Stage '{envelope.stage}' was interrupted",
                        context=envelope.stage,
                    )
                )

        # 4. From Missing Artifacts
        # Only training_events.json is tracked here (connection issue with remote PC).
        # Stage artifact absence means the stage did not run — not an issue.
        if self._data.missing_artifacts:
            for artifact in self._data.missing_artifacts:
                if artifact == "training_events.json":
                    issues.append(
                        Issue(
                            severity=KEY_WARN,
                            message=(
                                "Training events (training_events.json) were not found in MLflow artifacts. "
                                "This may indicate network connectivity issues to the MLflow server during training. "
                                "Check: (1) MLflow is reachable at the configured tracking_uri from the remote PC, "
                                "(2) IP addresses in the configuration match the current network."
                            ),
                            context="Report Generation",
                        )
                    )
                else:
                    # Corrupted/unreadable artifact (parse failure)
                    issues.append(
                        Issue(
                            severity=KEY_WARN,
                            message=f"Artifact could not be parsed: {artifact}",
                            context="Report Generation",
                        )
                    )

        # 5. From Dataset Validation
        if validation:
            for dataset_val in validation.datasets:
                if dataset_val.status == KEY_FAILED:
                    plugin_names = [p.id for p in dataset_val.plugin_results if not p.passed]
                    failed_count = dataset_val.failed_plugins
                    issues.append(
                        Issue(
                            severity=KEY_WARN,
                            message=f"Dataset '{dataset_val.dataset_name}': {failed_count} validation plugin(s) failed ({', '.join(plugin_names)})",
                            context="Dataset Validation",
                        )
                    )
                elif dataset_val.has_partial_failure:
                    plugin_names = [p.id for p in dataset_val.plugin_results if not p.passed]
                    issues.append(
                        Issue(
                            severity=KEY_WARN,
                            message=f"Dataset '{dataset_val.dataset_name}': partial failure ({dataset_val.failed_plugins}/{dataset_val.total_plugins} plugins failed: {', '.join(plugin_names)}, below critical threshold of {dataset_val.critical_failures})",
                            context="Dataset Validation",
                        )
                    )

        # 6. From Evaluation Plugins
        if evaluation:
            for plugin in evaluation.plugins:
                if not plugin.passed:
                    issues.append(
                        Issue(
                            severity=KEY_WARN,
                            message=f"Evaluation plugin '{plugin.name}' failed: {', '.join(plugin.errors) or 'see metrics'}",
                            context="Model Evaluation",
                        )
                    )

        return issues

    # =========================================================================
    # VALIDATION
    # =========================================================================

    def _build_validation(self) -> ValidationReport | None:
        """Build dataset validation report from per-stage artifact data.

        Reads from data.validation_results (ValidationArtifactData) populated
        by MLflowAdapter from dataset_validator_results.json.

        Falls back to the legacy event-based parsing when validation_results
        is not available (old runs that still have pipeline_events.json).
        """
        # --- New path: read from structured artifact ---
        validation_data = self._data.validation_results
        if validation_data and isinstance(validation_data.get("datasets"), list):
            dataset_validations: list[DatasetValidation] = []
            for ds in validation_data["datasets"]:
                if not isinstance(ds, dict):
                    continue
                plugins_raw = ds.get("plugins", [])
                plugin_results = []
                for p in plugins_raw:
                    if not isinstance(p, dict):
                        continue
                    plugin_results.append(
                        ValidationPluginResults(
                            id=p.get("id", KEY_UNKNOWN_LOWER),
                            plugin_name=p.get("plugin_name", ""),
                            status=KEY_PASSED if p.get("passed") else KEY_FAILED,
                            duration_ms=float(p.get("duration_ms", 0.0)),
                            description=p.get("description", ""),
                            metrics=p.get("metrics", {}),
                            params=p.get("params", {}),
                            thresholds=p.get("thresholds", {}),
                            errors=p.get("errors", []),
                            recommendations=p.get("recommendations", []),
                        )
                    )
                total = len(plugin_results)
                passed_count = sum(1 for p in plugin_results if p.passed)
                failed_count = total - passed_count
                dataset_validations.append(
                    DatasetValidation(
                        dataset_name=ds.get("name", KEY_UNKNOWN_LOWER),
                        dataset_path=ds.get("path"),
                        sample_count=ds.get(KEY_SAMPLE_COUNT),
                        status=ds.get("status", KEY_UNKNOWN_LOWER),
                        total_plugins=total,
                        passed_plugins=passed_count,
                        failed_plugins=failed_count,
                        critical_failures=int(ds.get("critical_failures", 0)),
                        plugin_results=plugin_results,
                    )
                )
            if dataset_validations:
                return ValidationReport(datasets=dataset_validations)
            return None

        # --- Legacy path: no artifact → return None (old behavior) ---
        return None

    def _build_evaluation(self) -> EvaluationReport | None:
        """Build evaluation report from per-stage artifact data.

        Reads from data.evaluation_results (EvalArtifactData) populated
        by MLflowAdapter from evaluation_results.json.
        """
        eval_data = self._data.evaluation_results
        if not eval_data or not isinstance(eval_data, dict):
            return None

        plugins_raw = eval_data.get("plugins", {})
        plugins: list[EvalPluginResult] = []
        if isinstance(plugins_raw, dict):
            for name, p in plugins_raw.items():
                if not isinstance(p, dict):
                    continue
                plugins.append(
                    EvalPluginResult(
                        name=name,
                        plugin_name=p.get("plugin_name", ""),
                        passed=bool(p.get("passed", False)),
                        description=p.get("description", ""),
                        params=p.get("params", {}),
                        thresholds=p.get("thresholds", {}),
                        metrics={k: float(v) for k, v in p.get("metrics", {}).items() if isinstance(v, int | float)},
                        errors=p.get("errors", []),
                        recommendations=p.get("recommendations", []),
                        sample_count=int(_scalar(p.get(KEY_SAMPLE_COUNT, 0))),
                        failed_samples=_failed_samples_count(p.get("failed_samples", 0)),
                    )
                )

        return EvaluationReport(
            overall_passed=bool(eval_data.get("overall_passed", False)),
            sample_count=int(eval_data.get(KEY_SAMPLE_COUNT, 0)),
            duration_seconds=float(eval_data.get("duration_seconds", 0.0)),
            skipped_plugins=eval_data.get("skipped_plugins", []),
            errors=eval_data.get("errors", []),
            plugins=plugins,
        )

    # =========================================================================
    # HELPERS
    # =========================================================================

    @staticmethod
    def _calc_trend(history: MetricHistory | None) -> MetricTrend | None:
        """Calculate trend from history."""
        if not history or not history.values:
            return None

        first = history.values[0]
        last = history.values[-1]
        change_pct = ((last - first) / first * 100) if first != 0 else 0

        direction = "stable"
        if change_pct < -5:
            direction = "decreased"
        elif change_pct > 5:
            direction = "increased"

        return MetricTrend(
            first=first,
            last=last,
            min_val=min(history.values),
            max_val=max(history.values),
            change_pct=change_pct,
            direction=direction,
            data_points=len(history.values),
        )

    @staticmethod
    def _format_duration(seconds: float) -> str:
        if seconds < 60:
            return f"{seconds:.1f}s"
        minutes = int(seconds // 60)
        return f"{minutes}m {int(seconds % 60)}s"

    @staticmethod
    def _parse_ts(ts: str | None) -> Any:
        if not ts:
            return None
        try:
            return datetime.fromisoformat(ts.replace("Z", "+00:00"))
        except (TypeError, ValueError):
            return None
