"""Tests for validation data models."""

import pytest
from pydantic import ValidationError

from src.data.validation.base import ValidationErrorGroup, ValidationPlugin
from src.data.validation.models import AggregatedValidationResult, PluginConfig


class TestPluginConfig:
    """Test suite for PluginConfig model."""

    def test_create_with_plugin_only(self):
        """Test: Create PluginConfig with id and plugin."""
        config = PluginConfig(id="min_samples_main", plugin="min_samples")

        assert config.id == "min_samples_main"
        assert config.plugin == "min_samples"
        assert config.params == {}

    def test_create_with_params(self):
        """Test: Create PluginConfig with parameters."""
        config = PluginConfig(id="avg_length_main", plugin="avg_length", params={"min": 50, "max": 2048})

        assert config.id == "avg_length_main"
        assert config.plugin == "avg_length"
        assert config.params["min"] == 50
        assert config.params["max"] == 2048

    def test_missing_plugin_fields_raise_error(self):
        """Test: Missing id/plugin fields raises ValidationError."""
        with pytest.raises(ValidationError):
            PluginConfig()

    def test_model_serialization(self):
        """Test: Model can be serialized to dict."""
        config = PluginConfig(id="dedup_main", plugin="deduplication", params={"max_ratio": 0.1})

        data = config.model_dump()

        assert data["id"] == "dedup_main"
        assert data["plugin"] == "deduplication"
        assert data["params"]["max_ratio"] == 0.1

    def test_model_deserialization(self):
        """Test: Model can be created from dict."""
        data = {"id": "diversity_main", "plugin": "diversity_score", "params": {"min_score": 0.7}}

        config = PluginConfig(**data)

        assert config.id == "diversity_main"
        assert config.plugin == "diversity_score"
        assert config.params["min_score"] == 0.7

    def test_empty_params_default(self):
        """Test: params defaults to empty dict."""
        config = PluginConfig(id="test_plugin_main", plugin="test_plugin")

        assert isinstance(config.params, dict)
        assert len(config.params) == 0


class TestAggregatedValidationResult:
    """Test suite for AggregatedValidationResult model."""

    def test_create_passed_result(self):
        """Test: Create passed aggregated result."""
        result = AggregatedValidationResult(
            passed=True,
            total_plugins=2,
            failed_plugins=0,
            total_errors=0,
            total_warnings=1,
            plugin_results=[
                {"plugin": "min_samples", "passed": True},
                {"plugin": "avg_length", "passed": True},
            ],
            recommendations=[],
        )

        assert result.passed is True
        assert result.total_plugins == 2
        assert result.failed_plugins == 0
        assert result.total_errors == 0
        assert result.total_warnings == 1
        assert len(result.plugin_results) == 2

    def test_create_failed_result(self):
        """Test: Create failed aggregated result."""
        result = AggregatedValidationResult(
            passed=False,
            total_plugins=2,
            failed_plugins=1,
            total_errors=2,
            total_warnings=0,
            plugin_results=[
                {"plugin": "min_samples", "passed": False},
                {"plugin": "avg_length", "passed": True},
            ],
            recommendations=["Add more samples", "Check data quality"],
        )

        assert result.passed is False
        assert result.failed_plugins == 1
        assert result.total_errors == 2
        assert len(result.recommendations) == 2

    def test_all_fields_required(self):
        """Test: All fields are required."""
        with pytest.raises(ValidationError):
            AggregatedValidationResult(
                passed=True,
                # Missing required fields
            )

    def test_model_serialization(self):
        """Test: Model can be serialized to dict."""
        result = AggregatedValidationResult(
            passed=True,
            total_plugins=1,
            failed_plugins=0,
            total_errors=0,
            total_warnings=0,
            plugin_results=[{"test": "data"}],
            recommendations=[],
        )

        data = result.model_dump()

        assert data["passed"] is True
        assert data["total_plugins"] == 1
        assert "plugin_results" in data

    def test_model_deserialization(self):
        """Test: Model can be created from dict."""
        data = {
            "passed": False,
            "total_plugins": 3,
            "failed_plugins": 2,
            "total_errors": 5,
            "total_warnings": 3,
            "plugin_results": [
                {"plugin": "test1", "passed": False},
                {"plugin": "test2", "passed": False},
                {"plugin": "test3", "passed": True},
            ],
            "recommendations": ["Fix errors", "Review data"],
        }

        result = AggregatedValidationResult(**data)

        assert result.passed is False
        assert result.total_plugins == 3
        assert result.failed_plugins == 2
        assert len(result.recommendations) == 2

    def test_consistency_passed_with_no_failures(self):
        """Test: Passed=True should have failed_plugins=0."""
        result = AggregatedValidationResult(
            passed=True,
            total_plugins=5,
            failed_plugins=0,
            total_errors=0,
            total_warnings=2,
            plugin_results=[],
            recommendations=[],
        )

        assert result.passed is True
        assert result.failed_plugins == 0

    def test_consistency_failed_with_failures(self):
        """Test: Passed=False should have failed_plugins>0."""
        result = AggregatedValidationResult(
            passed=False,
            total_plugins=5,
            failed_plugins=2,
            total_errors=3,
            total_warnings=1,
            plugin_results=[],
            recommendations=["Fix issues"],
        )

        assert result.passed is False
        assert result.failed_plugins > 0

    def test_empty_plugin_results(self):
        """Test: Can have empty plugin_results list."""
        result = AggregatedValidationResult(
            passed=True,
            total_plugins=0,
            failed_plugins=0,
            total_errors=0,
            total_warnings=0,
            plugin_results=[],
            recommendations=[],
        )

        assert len(result.plugin_results) == 0

    def test_complex_plugin_results(self):
        """Test: Plugin results can contain complex data."""
        result = AggregatedValidationResult(
            passed=True,
            total_plugins=1,
            failed_plugins=0,
            total_errors=0,
            total_warnings=0,
            plugin_results=[
                {
                    "name": "min_samples",
                    "passed": True,
                    "metrics": {"count": 100, "threshold": 50},
                    "execution_time_ms": 1.5,
                }
            ],
            recommendations=[],
        )

        assert len(result.plugin_results) == 1
        assert result.plugin_results[0]["name"] == "min_samples"
        assert result.plugin_results[0]["metrics"]["count"] == 100


class TestValidationErrorRendering:
    def test_render_error_groups_without_ellipsis_for_truncated_list(self):
        lines = ValidationPlugin.render_error_groups(
            [
                ValidationErrorGroup(
                    error_type="parse_error",
                    sample_indices=[56, 57, 58, 59, 60],
                    total_count=522,
                )
            ]
        )

        assert lines == ["parse_error: [56, 57, 58, 59, 60]"]

    def test_render_error_groups_keeps_full_list_when_all_indices_present(self):
        lines = ValidationPlugin.render_error_groups(
            [
                ValidationErrorGroup(
                    error_type="validation_error",
                    sample_indices=[160, 161, 162],
                    total_count=3,
                )
            ]
        )

        assert lines == ["validation_error: [160, 161, 162]"]
