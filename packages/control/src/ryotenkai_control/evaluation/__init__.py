"""
Evaluation module.

Public API:
    from ryotenkai_control.evaluation import EvaluationRunner
    from ryotenkai_control.evaluation.model_client import ModelClientFactory, IModelInference
    from ryotenkai_control.evaluation.plugins.base import EvalSample, EvalResult, EvaluatorPlugin
    from ryotenkai_control.evaluation.plugins.registry import EvaluatorPluginRegistry
"""

from .runner import EvaluationRunner

__all__ = [
    "EvaluationRunner",
]
