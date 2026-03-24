"""
Unit tests for HealthPolicy.

Validates ExperimentHealth (RED/YELLOW/GREEN) from configurable rules and thresholds.
"""

from src.reports.core.health_policy import (
    DEFAULT_HEALTH_POLICY,
    HealthPolicy,
    HealthPolicySettings,
)
from src.reports.domain.entities import RunStatus
from src.reports.models.report import ExperimentHealth, Issue


class TestHealthPolicy:
    """Tests for HealthPolicy."""

    def test_green_no_issues(self):
        """GREEN: no issues."""
        policy = HealthPolicy()
        health, explanation = policy.evaluate(
            run_status=RunStatus.FINISHED,
            issues=[],
        )
        assert health == ExperimentHealth.GREEN
        assert "No critical issues" in explanation

    def test_red_run_failed(self):
        """RED: run status = FAILED."""
        policy = HealthPolicy()
        health, explanation = policy.evaluate(
            run_status=RunStatus.FAILED,
            issues=[],
        )
        assert health == ExperimentHealth.RED
        assert "failed" in explanation.lower()

    def test_red_error_issue(self):
        """RED: ERROR issue present."""
        policy = HealthPolicy()
        health, explanation = policy.evaluate(
            run_status=RunStatus.FINISHED,
            issues=[
                Issue(severity="ERROR", message="Critical failure", context="Phase 1"),
            ],
        )
        assert health == ExperimentHealth.RED
        assert "ERROR issue" in explanation

    def test_yellow_3_warnings(self):
        """YELLOW: 3 WARN issues (threshold >= 3)."""
        policy = HealthPolicy()
        health, explanation = policy.evaluate(
            run_status=RunStatus.FINISHED,
            issues=[
                Issue(severity="WARN", message="Warning 1"),
                Issue(severity="WARN", message="Warning 2"),
                Issue(severity="WARN", message="Warning 3"),
            ],
        )
        assert health == ExperimentHealth.YELLOW
        assert "3 WARN" in explanation

    def test_green_2_warnings(self):
        """GREEN: 2 WARN issues (below threshold 3)."""
        policy = HealthPolicy()
        health, explanation = policy.evaluate(
            run_status=RunStatus.FINISHED,
            issues=[
                Issue(severity="WARN", message="Warning 1"),
                Issue(severity="WARN", message="Warning 2"),
            ],
        )
        assert health == ExperimentHealth.GREEN

    def test_red_5_warnings(self):
        """RED: 5 WARN issues (threshold >= 5)."""
        policy = HealthPolicy()
        health, explanation = policy.evaluate(
            run_status=RunStatus.FINISHED,
            issues=[Issue(severity="WARN", message=f"Warning {i}") for i in range(5)],
        )
        assert health == ExperimentHealth.RED
        assert "5 WARN" in explanation

    def test_yellow_4_warnings(self):
        """YELLOW: 4 WARN issues (>= 3 but < 5)."""
        policy = HealthPolicy()
        health, explanation = policy.evaluate(
            run_status=RunStatus.FINISHED,
            issues=[Issue(severity="WARN", message=f"Warning {i}") for i in range(4)],
        )
        assert health == ExperimentHealth.YELLOW
        assert "4 WARN" in explanation

    def test_red_priority_over_warn(self):
        """RED: ERROR takes priority over WARN."""
        policy = HealthPolicy()
        health, explanation = policy.evaluate(
            run_status=RunStatus.FINISHED,
            issues=[
                Issue(severity="ERROR", message="Critical"),
                Issue(severity="WARN", message="Warning"),
            ],
        )
        assert health == ExperimentHealth.RED
        assert "ERROR" in explanation

    def test_custom_settings_warn_threshold(self):
        """Custom warn thresholds."""
        settings = HealthPolicySettings(
            warn_to_yellow=2,
            warn_to_red=4,
        )
        policy = HealthPolicy(settings=settings)

        # 2 WARN -> YELLOW
        health, _ = policy.evaluate(
            run_status=RunStatus.FINISHED,
            issues=[
                Issue(severity="WARN", message="W1"),
                Issue(severity="WARN", message="W2"),
            ],
        )
        assert health == ExperimentHealth.YELLOW

        # 4 WARN -> RED
        health, _ = policy.evaluate(
            run_status=RunStatus.FINISHED,
            issues=[Issue(severity="WARN", message=f"W{i}") for i in range(4)],
        )
        assert health == ExperimentHealth.RED

    def test_ignore_info_issues(self):
        """INFO issues do not affect health."""
        policy = HealthPolicy()
        health, _ = policy.evaluate(
            run_status=RunStatus.FINISHED,
            issues=[
                Issue(severity="INFO", message="Info 1"),
                Issue(severity="INFO", message="Info 2"),
                Issue(severity="INFO", message="Info 3"),
            ],
        )
        assert health == ExperimentHealth.GREEN

    def test_mixed_severities(self):
        """Mixed severities: INFO ignored, WARN counted."""
        policy = HealthPolicy()
        health, _ = policy.evaluate(
            run_status=RunStatus.FINISHED,
            issues=[
                Issue(severity="WARN", message="W1"),
                Issue(severity="INFO", message="Info"),
                Issue(severity="WARN", message="W2"),
                Issue(severity="WARN", message="W3"),
            ],
        )
        assert health == ExperimentHealth.YELLOW  # 3 WARN

    def test_default_policy_settings(self):
        """DEFAULT_HEALTH_POLICY values."""
        assert DEFAULT_HEALTH_POLICY.warn_to_yellow == 3
        assert DEFAULT_HEALTH_POLICY.warn_to_red == 5
        assert RunStatus.FAILED in DEFAULT_HEALTH_POLICY.critical_run_statuses
        assert "ERROR" in DEFAULT_HEALTH_POLICY.critical_issue_severities
