"""
Test InferenceHealthCheckConfig validation (regression for retries limit).
"""

import pytest
from pydantic import ValidationError

from src.utils.config import InferenceHealthCheckConfig


def test_health_check_retries_within_limit():
    """Valid: retries within 1-20 range."""
    config = InferenceHealthCheckConfig(
        enabled=True,
        timeout_seconds=120,
        interval_seconds=5,
        retries=20,  # Max valid value
    )
    assert config.retries == 20


def test_health_check_retries_min_value():
    """Valid: retries = 1 (minimum)."""
    config = InferenceHealthCheckConfig(
        enabled=True,
        timeout_seconds=120,
        interval_seconds=5,
        retries=1,
    )
    assert config.retries == 1


def test_health_check_retries_exceeds_limit():
    """Invalid: retries > 20 should raise ValidationError."""
    with pytest.raises(ValidationError) as exc_info:
        InferenceHealthCheckConfig(
            enabled=True,
            timeout_seconds=120,
            interval_seconds=5,
            retries=24,  # INVALID: exceeds max (20)
        )

    errors = exc_info.value.errors()
    assert len(errors) == 1
    assert errors[0]["loc"] == ("retries",)
    assert errors[0]["type"] == "less_than_equal"
    assert errors[0]["ctx"]["le"] == 20


def test_health_check_retries_below_min():
    """Invalid: retries = 0 should raise ValidationError."""
    with pytest.raises(ValidationError) as exc_info:
        InferenceHealthCheckConfig(
            enabled=True,
            timeout_seconds=120,
            interval_seconds=5,
            retries=0,  # INVALID: below min (1)
        )

    errors = exc_info.value.errors()
    assert len(errors) == 1
    assert errors[0]["loc"] == ("retries",)
    assert errors[0]["type"] == "greater_than_equal"


def test_health_check_defaults():
    """Test default values."""
    config = InferenceHealthCheckConfig()
    assert config.enabled is True
    assert config.timeout_seconds == 120
    assert config.interval_seconds == 5
    assert config.retries == 3  # Default


def test_health_check_timeout_validation():
    """Test timeout_seconds validation (5-3600)."""
    # Valid: within range
    config = InferenceHealthCheckConfig(timeout_seconds=3600)
    assert config.timeout_seconds == 3600

    # Invalid: exceeds max
    with pytest.raises(ValidationError) as exc_info:
        InferenceHealthCheckConfig(timeout_seconds=4000)

    errors = exc_info.value.errors()
    assert errors[0]["loc"] == ("timeout_seconds",)
    assert errors[0]["type"] == "less_than_equal"


def test_health_check_interval_validation():
    """Test interval_seconds validation (1-120)."""
    # Valid: within range
    config = InferenceHealthCheckConfig(interval_seconds=120)
    assert config.interval_seconds == 120

    # Invalid: exceeds max
    with pytest.raises(ValidationError) as exc_info:
        InferenceHealthCheckConfig(interval_seconds=150)

    errors = exc_info.value.errors()
    assert errors[0]["loc"] == ("interval_seconds",)
    assert errors[0]["type"] == "less_than_equal"
