"""
IModelInference — interface for collecting model answers during evaluation.

Used exclusively by EvaluationRunner to populate EvalSample.model_answer.
Individual plugins do NOT use this interface directly.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class IModelInference(Protocol):
    """
    Minimal contract for getting a text response from the trained model.

    Implementations:
    - OpenAICompatibleInferenceClient: calls any OpenAI-compatible endpoint (vLLM, etc.)
    - MockInferenceClient: returns predictable responses for unit tests.
    """

    def generate(self, prompt: str) -> str:
        """
        Send a prompt to the model and return its response.

        Args:
            prompt: Raw user prompt (question from the eval dataset).

        Returns:
            Model response as a plain string.

        Raises:
            RuntimeError: if the request fails and the caller should handle the error.
        """
        ...


__all__ = ["IModelInference"]
