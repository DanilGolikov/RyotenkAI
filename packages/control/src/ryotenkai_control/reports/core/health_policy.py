"""
Health Policy — centralized logic for determining report status.

Takes normalized signals (issues) and applies configurable rules to determine
ExperimentHealth (GREEN/YELLOW/RED).
"""

from __future__ import annotations

from dataclasses import dataclass

from src.reports.domain.entities import RunStatus
from src.reports.models.report import ExperimentHealth, Issue

_REASONS_JOIN = "; "


@dataclass(frozen=True)
class HealthPolicySettings:
    """
    Health policy settings.

    All thresholds and trigger lists are defined in code (not in YAML/JSON).
    """

    # Thresholds for WARN-level events/issues
    warn_to_yellow: int = 3
    warn_to_red: int = 5

    # Critical triggers (any match → RED)
    critical_run_statuses: frozenset[RunStatus] = frozenset({RunStatus.FAILED})
    critical_issue_severities: frozenset[str] = frozenset({"ERROR"})
    critical_event_types: frozenset[str] = frozenset({"error"})
    critical_signal_codes: frozenset[str] = frozenset(
        {
            "PIPELINE_STAGE_FAILED",
            "TRAINING_FAILED",
            "MEMORY_OOM_LOOP",
        }
    )

    # Yellow triggers (any match → at least YELLOW)
    yellow_signal_codes: frozenset[str] = frozenset(
        {
            "MEMORY_OOM_RECOVERED",
            "MEMORY_PRESSURE",
        }
    )

    # What counts as WARNs for thresholds (usually only "WARN")
    count_issue_severities: frozenset[str] = frozenset({"WARN"})


# Global default settings
DEFAULT_HEALTH_POLICY = HealthPolicySettings()


class HealthPolicy:
    """
    Determines overall experiment health based on rules.

    Algorithm:
    1. Check critical triggers → RED
    2. Check yellow triggers → at least YELLOW
    3. Count WARN issues: >=5 → RED, >=3 → YELLOW
    4. Return (health, explanation)
    """

    def __init__(self, settings: HealthPolicySettings = DEFAULT_HEALTH_POLICY):
        self.settings = settings

    def evaluate(
        self,
        run_status: RunStatus,
        issues: list[Issue],
    ) -> tuple[ExperimentHealth, str]:
        """
        Assess overall experiment health.

        Args:
            run_status: MLflow run status
            issues: List of detected issues (from metrics, memory, timeline)

        Returns:
            (ExperimentHealth, explanation)
        """
        reasons = []

        # 1. Check critical triggers
        if run_status in self.settings.critical_run_statuses:
            reasons.append(f"Run status: {run_status.value}")
            return ExperimentHealth.RED, _REASONS_JOIN.join(reasons)

        # Check ERROR issues
        error_issues = [i for i in issues if i.severity in self.settings.critical_issue_severities]
        if error_issues:
            reasons.append(f"{len(error_issues)} ERROR issue(s)")
            return ExperimentHealth.RED, _REASONS_JOIN.join(reasons)

        # 2. Count WARN issues
        warn_issues = [i for i in issues if i.severity in self.settings.count_issue_severities]
        warn_count = len(warn_issues)

        # 3. Check WARN thresholds
        if warn_count >= self.settings.warn_to_red:
            reasons.append(f"{warn_count} WARN issues (>= {self.settings.warn_to_red})")
            return ExperimentHealth.RED, _REASONS_JOIN.join(reasons)

        if warn_count >= self.settings.warn_to_yellow:
            reasons.append(f"{warn_count} WARN issues (>= {self.settings.warn_to_yellow})")
            return ExperimentHealth.YELLOW, _REASONS_JOIN.join(reasons)

        # 4. Check yellow triggers (optional, if we add signal_codes to Issue)
        # Skip for now, as Issue doesn't have 'code' field

        # 5. All good
        return ExperimentHealth.GREEN, "No critical issues detected"


__all__ = [
    "DEFAULT_HEALTH_POLICY",
    "HealthPolicy",
    "HealthPolicySettings",
]
