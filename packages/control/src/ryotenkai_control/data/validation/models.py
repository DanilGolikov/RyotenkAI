"""
Data models for validation system.

Provides Pydantic models for configuration and results.
"""

from typing import Any

from pydantic import BaseModel, Field


class PluginConfig(BaseModel):
    """
    Configuration for a single validation plugin instance.

    Attributes:
        id: Unique instance id
        plugin: Plugin name (must be registered)
        params: Plugin runtime parameters (optional)
        thresholds: Plugin pass/fail criteria (optional)
    """

    id: str = Field(..., description="Plugin instance id")
    plugin: str = Field(..., description="Registered plugin name")
    params: dict[str, Any] = Field(
        default_factory=dict,
        description="Plugin runtime parameters (optional, uses defaults if not specified)",
    )
    thresholds: dict[str, Any] = Field(
        default_factory=dict,
        description="Plugin pass/fail criteria (optional, uses defaults if not specified)",
    )


class AggregatedValidationResult(BaseModel):
    """
    Aggregated results from all validation plugins.

    Attributes:
        passed: Whether all validations passed
        total_plugins: Total number of plugins executed
        failed_plugins: Number of plugins that failed
        total_errors: Total error count across all plugins
        total_warnings: Total warning count
        plugin_results: List of individual plugin results
        recommendations: Aggregated recommendations from failed plugins
    """

    passed: bool
    total_plugins: int
    failed_plugins: int
    total_errors: int
    total_warnings: int
    plugin_results: list[dict[str, Any]]
    recommendations: list[str]


__all__ = ["AggregatedValidationResult", "PluginConfig"]
