"""
IJudgeProvider interface and JudgeResponse dataclass.

Design:
- IJudgeProvider is a Protocol (structural typing) — implementors don't need to inherit.
- JudgeResponse carries the raw integer score (1–5), a reasoning string (CoT), and the
  raw JSON response for debug logging.
- CerebrasProvider implements IJudgeProvider.
- MockJudgeProvider (in tests/conftest.py) implements IJudgeProvider for unit tests.

Any future judge provider (OpenAI, Mistral, local Ollama, ...) only needs to implement
the two-line `judge()` contract — no changes to plugins or EvaluationRunner required.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable


@dataclass
class JudgeResponse:
    """
    Result returned by an IJudgeProvider.judge() call.

    Attributes:
        score:        Integer quality score from the judge (1 = worst, 5 = best).
        reasoning:    Chain-of-thought explanation from the judge model.
        raw_response: Raw JSON string returned by the API (for debug logging).
    """

    score: int
    reasoning: str
    raw_response: str


@runtime_checkable
class IJudgeProvider(Protocol):
    """
    Minimal contract for an LLM-as-judge provider.

    Implementations:
        CerebrasProvider  — production Cerebras API
        MockJudgeProvider — test double (in tests/conftest.py)

    The provider is responsible for:
    - Building the prompt (system + user messages)
    - Calling the external API
    - Parsing the response JSON and returning JudgeResponse

    It does NOT handle retries at the plugin level — retries are implemented
    inside the provider itself (max_retries param).
    """

    def judge(self, question: str, expected: str, model_answer: str) -> JudgeResponse:
        """
        Evaluate a single sample and return a JudgeResponse.

        Args:
            question:     The input question/task given to the model.
            expected:     The ground-truth / reference answer.
            model_answer: The answer produced by the model under evaluation.

        Returns:
            JudgeResponse with score (1–5), reasoning, and raw API response.

        Raises:
            RuntimeError: on unrecoverable API or parsing errors.
        """
        ...


__all__ = [
    "IJudgeProvider",
    "JudgeResponse",
]
