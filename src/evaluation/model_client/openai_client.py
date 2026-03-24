"""
OpenAI-compatible inference client for evaluation.

Works with any OpenAI-compatible endpoint (vLLM, OpenAI API, etc.).
Used by EvaluationRunner to collect model_answer for each eval sample.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from typing import Any


class OpenAICompatibleInferenceClient:
    """
    Minimal HTTP client for OpenAI-compatible /v1/chat/completions endpoints.

    Intentionally avoids the `openai` SDK to keep dependencies light.
    Uses only stdlib (urllib) — no external packages required.
    """

    def __init__(
        self,
        base_url: str,
        model: str,
        *,
        timeout_seconds: int = 60,
        max_tokens: int = 512,
        temperature: float = 0.0,
        system_prompt: str | None = None,
        api_key: str = "not-required",
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._model = model
        self._timeout = timeout_seconds
        self._max_tokens = max_tokens
        self._temperature = temperature
        self._system_prompt = system_prompt
        self._api_key = api_key

    def generate(self, prompt: str) -> str:
        """
        Send a chat completion request and return the assistant's response text.

        Raises:
            RuntimeError: on HTTP errors, JSON decode failures, or unexpected response shape.
        """
        messages: list[dict[str, str]] = []
        if self._system_prompt:
            messages.append({"role": "system", "content": self._system_prompt})
        messages.append({"role": "user", "content": prompt})

        payload: dict[str, Any] = {
            "model": self._model,
            "messages": messages,
            "max_tokens": self._max_tokens,
            "temperature": self._temperature,
        }

        url = f"{self._base_url}/chat/completions"
        body = json.dumps(payload).encode("utf-8")
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self._api_key}",
        }

        req = urllib.request.Request(url, data=body, headers=headers, method="POST")

        try:
            with urllib.request.urlopen(req, timeout=self._timeout) as resp:
                response_data: dict[str, Any] = json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            body_text = e.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"Inference endpoint returned HTTP {e.code}: {body_text}") from e
        except urllib.error.URLError as e:
            raise RuntimeError(f"Failed to connect to inference endpoint {url}: {e.reason}") from e

        try:
            return str(response_data["choices"][0]["message"]["content"])
        except (KeyError, IndexError, TypeError) as e:
            raise RuntimeError(f"Unexpected response shape from inference endpoint: {response_data}") from e


__all__ = ["OpenAICompatibleInferenceClient"]
