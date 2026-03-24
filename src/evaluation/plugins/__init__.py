"""Evaluation plugin infrastructure."""

from .base import EvalResult, EvalSample, EvaluatorPlugin
from .discovery import ensure_evaluation_plugins_discovered
from .registry import EvaluatorPluginRegistry

__all__ = [
    "EvalResult",
    "EvalSample",
    "EvaluatorPlugin",
    "EvaluatorPluginRegistry",
    "ensure_evaluation_plugins_discovered",
]
