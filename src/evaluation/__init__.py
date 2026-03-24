"""
Evaluation module.

Public API:
    from src.evaluation import EvaluationRunner
    from src.evaluation.model_client import ModelClientFactory, IModelInference
    from src.evaluation.plugins.base import EvalSample, EvalResult, EvaluatorPlugin
    from src.evaluation.plugins.registry import EvaluatorPluginRegistry
"""

from .runner import EvaluationRunner

__all__ = [
    "EvaluationRunner",
]
