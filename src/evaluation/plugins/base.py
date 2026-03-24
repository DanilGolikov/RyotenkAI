"""
Core data models and base class for evaluator plugins.

Design:
- EvalSample: immutable data unit passed to each plugin.
- EvalResult: output from a plugin (pass/fail + metrics + recommendations).
- EvaluatorPlugin: ABC that every evaluator plugin must implement.

Flow:
    EvaluationRunner:
        1. Load JSONL dataset → list of (question, expected_answer)
        2. Collect model answers via IModelInference → list[EvalSample]
        3. For each plugin (sorted by priority):
               result = plugin.evaluate(samples)
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, ClassVar

from src.utils.plugin_base import BasePlugin


@dataclass
class EvalSample:
    """
    A single evaluation unit.

    Populated by EvaluationRunner:
        - question + expected_answer: loaded from JSONL dataset
        - model_answer: collected via IModelInference before plugins run
        - metadata: arbitrary extra fields from the dataset row (all keys except
          the reserved ones: question, expected_answer, answer, context, messages).
          Plugins can read any key via sample.metadata.get("key").
    """

    question: str
    model_answer: str
    expected_answer: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class EvalResult:
    """
    Output of a single evaluator plugin run.

    Attributes:
        plugin_name:      Unique plugin identifier (matches EvaluatorPlugin.name).
        passed:           True if all thresholds are satisfied.
        metrics:          Numeric/string metrics computed by the plugin.
        errors:           Human-readable reasons for failure (empty if passed).
        recommendations:  Actionable suggestions when passed=False.
        sample_count:     Total number of samples evaluated.
        failed_samples:   Indices of samples that failed (for debugging).
    """

    plugin_name: str
    passed: bool
    metrics: dict[str, Any] = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)
    recommendations: list[str] = field(default_factory=list)
    sample_count: int = 0
    failed_samples: list[int] = field(default_factory=list)


class EvaluatorPlugin(BasePlugin, ABC):
    """
    Base class for all evaluator plugins.

    Subclass contract:
    - Override class variables: name, priority.
    - Implement get_description(), evaluate(), get_recommendations().
    - Optionally override `_validate_contract()` for config validation at init.

    Plugin receives ALREADY-COLLECTED samples (with model_answer filled in).
    It does NOT call inference itself — that is EvaluationRunner's responsibility.
    """

    # ----------------------------------------------------------------
    # Metadata — name / priority / version inherited from BasePlugin
    # ----------------------------------------------------------------
    requires_expected_answer: ClassVar[bool] = False

    def __init__(self, params: dict[str, Any], thresholds: dict[str, Any]) -> None:
        self.params = dict(params)
        self.thresholds = dict(thresholds)
        self._validate_contract()

    @classmethod
    @abstractmethod
    def get_description(cls) -> str:
        """Return a human-readable description of what this plugin checks."""
        ...

    @abstractmethod
    def evaluate(self, samples: list[EvalSample]) -> EvalResult:
        """
        Evaluate the list of samples and return a result.

        Called by EvaluationRunner after model answers are collected.
        All samples have model_answer populated.

        Args:
            samples: Complete EvalSample list with model_answer filled.

        Returns:
            EvalResult with pass/fail verdict, metrics, and recommendations.
        """
        ...

    @abstractmethod
    def get_recommendations(self, result: EvalResult) -> list[str]:
        """
        Return actionable recommendations based on the evaluation result.

        Called automatically by evaluate() — implementations typically delegate here.
        Can also be called by orchestration layer for reporting.
        """
        ...

    def _validate_contract(self) -> None:
        """
        Optional config contract validation at plugin init time.

        Raise ValueError with a descriptive message if params/thresholds are invalid.
        """

    def _param(self, key: str, default: Any = None) -> Any:
        return self.params.get(key, default)

    def _threshold(self, key: str, default: Any = None) -> Any:
        return self.thresholds.get(key, default)


__all__ = [
    "EvalResult",
    "EvalSample",
    "EvaluatorPlugin",
]
