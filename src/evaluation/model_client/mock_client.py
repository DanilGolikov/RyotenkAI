"""
MockInferenceClient — deterministic stub for unit tests.

Returns a fixed or callable response without making any HTTP calls.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Callable


class MockInferenceClient:
    """
    Inference client that returns predictable answers for unit tests.

    Usage:
        # Return the same response for every prompt:
        client = MockInferenceClient(response="SELECT * FROM nodes")

        # Return a dynamic response based on the prompt:
        client = MockInferenceClient(response=lambda prompt: f"answer to: {prompt}")
    """

    def __init__(self, response: str | Callable[[str], str] = "mock_answer") -> None:
        self._response = response

    def generate(self, prompt: str) -> str:
        if callable(self._response):
            return self._response(prompt)
        return self._response


__all__ = ["MockInferenceClient"]
