"""CerebrasProvider — ``IJudgeProvider`` impl over the Cerebras chat-completions API.

Uses only stdlib (urllib) — no extra dependencies. Handles retries with
exponential backoff on HTTP 429 / 5xx.
"""

from __future__ import annotations

import contextlib
import json
import re
import ssl
import time
import urllib.error
import urllib.request
from typing import Any

from src.utils.logger import logger

from .interface import JudgeResponse

_CEREBRAS_API_URL = "https://api.cerebras.ai/v1/chat/completions"
_SCORE_MIN = 1
_SCORE_MAX = 5

_SYSTEM_PROMPT = """\
You are an expert evaluator. Rate how accurately the model's answer matches the expected answer.

Criteria:
- Factual accuracy: does the model's answer contain correct information?
- Completeness: are all key aspects covered?
- Relevance: is there extraneous content?

Scale:
1 = Completely wrong
2 = Mostly wrong, with minor correct elements
3 = Partially correct, with significant gaps
4 = Mostly correct, with minor shortcomings
5 = Fully correct and complete

Examples:
Score 5: expected="QUERY A() => N<User> RETURN N", model="QUERY A() => N<User> RETURN N"
Score 1: expected="QUERY A() => N<User> RETURN N", model="I don't know how to answer"

Do not reward verbosity. A short correct answer is better than a long partially correct one.
Return only JSON in English: {"reasoning": "<your analysis in English>", "score": <1-5>}\
"""

DEFAULT_MODEL = "llama3.1-8b"
DEFAULT_TEMPERATURE = 0.0
DEFAULT_MAX_TOKENS = 512
DEFAULT_MAX_RETRIES = 3
RETRY_BASE_SLEEP = 2.0


def _make_ssl_context() -> ssl.SSLContext:
    try:
        import certifi

        return ssl.create_default_context(cafile=certifi.where())
    except ImportError:
        pass

    try:
        return ssl.create_default_context()
    except ssl.SSLError:
        pass

    logger.warning(
        "[CEREBRAS] SSL certificate verification disabled — system CA bundle not found. "
        "Run 'pip install certifi' or '/Applications/Python*/Install Certificates.command' to fix."
    )
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE
    return ctx


class CerebrasProvider:
    """``IJudgeProvider`` implementation over the Cerebras API."""

    def __init__(
        self,
        api_key: str,
        model: str = DEFAULT_MODEL,
        temperature: float = DEFAULT_TEMPERATURE,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        max_retries: int = DEFAULT_MAX_RETRIES,
    ) -> None:
        self._api_key = api_key
        self._model = model
        self._temperature = temperature
        self._max_tokens = max_tokens
        self._max_retries = max_retries
        self._ssl_context: ssl.SSLContext = _make_ssl_context()

    def judge(self, question: str, expected: str, model_answer: str) -> JudgeResponse:
        messages = self._build_messages(
            question=question, expected=expected, model_answer=model_answer
        )
        raw = self._call_api_with_retries(messages)
        return self._parse_response(raw)

    def _build_messages(
        self, *, question: str, expected: str, model_answer: str
    ) -> list[dict[str, str]]:
        user_content = (
            f"Question: {question}\n\n"
            f"Expected answer: {expected}\n\n"
            f"Model answer: {model_answer}"
        )
        return [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ]

    def _call_api_with_retries(self, messages: list[dict[str, str]]) -> str:
        last_error: Exception | None = None
        for attempt in range(self._max_retries):
            try:
                return self._call_api(messages)
            except RuntimeError as exc:
                last_error = exc
                is_retryable = "429" in str(exc) or "5" in str(exc)[:3]
                if not is_retryable or attempt == self._max_retries - 1:
                    raise
                sleep_secs = RETRY_BASE_SLEEP * (2**attempt)
                logger.warning(
                    f"[CEREBRAS] API error (attempt {attempt + 1}/{self._max_retries}), "
                    f"retrying in {sleep_secs:.0f}s: {exc}"
                )
                time.sleep(sleep_secs)
        raise RuntimeError(
            f"Cerebras API failed after {self._max_retries} retries"
        ) from last_error

    def _call_api(self, messages: list[dict[str, str]]) -> str:
        payload: dict[str, Any] = {
            "model": self._model,
            "messages": messages,
            "max_tokens": self._max_tokens,
            "temperature": self._temperature,
            "response_format": {"type": "json_object"},
        }
        body = json.dumps(payload).encode("utf-8")
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self._api_key}",
            "User-Agent": "ryotenkai-eval/1.0",
        }
        req = urllib.request.Request(_CEREBRAS_API_URL, data=body, headers=headers, method="POST")
        try:
            with urllib.request.urlopen(req, timeout=60, context=self._ssl_context) as resp:
                data: dict[str, Any] = json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            body_text = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"Cerebras API HTTP {exc.code}: {body_text}") from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(f"Cerebras API connection error: {exc.reason}") from exc

        try:
            return str(data["choices"][0]["message"]["content"])
        except (KeyError, IndexError, TypeError) as exc:
            raise RuntimeError(f"Unexpected Cerebras API response shape: {data}") from exc

    def _parse_response(self, raw: str) -> JudgeResponse:
        parsed: dict[str, Any] | None = None

        with contextlib.suppress(json.JSONDecodeError):
            parsed = json.loads(raw)

        if parsed is None:
            match = re.search(r"\{[^}]+\}", raw, re.DOTALL)
            if match:
                with contextlib.suppress(json.JSONDecodeError):
                    parsed = json.loads(match.group())

        score: int | None = None
        reasoning: str = ""

        if parsed is not None:
            raw_score = parsed.get("score")
            reasoning = str(parsed.get("reasoning", ""))
            if isinstance(raw_score, int | float):
                score = int(raw_score)

        if score is None:
            score_match = re.search(r'"score"\s*:\s*([1-5])', raw)
            if score_match:
                score = int(score_match.group(1))

        if score is None:
            raise RuntimeError(
                f"Could not extract score from Cerebras response. Raw response: {raw[:200]!r}"
            )

        score = max(_SCORE_MIN, min(_SCORE_MAX, score))
        return JudgeResponse(score=score, reasoning=reasoning, raw_response=raw)


__all__ = [
    "DEFAULT_MAX_RETRIES",
    "DEFAULT_MAX_TOKENS",
    "DEFAULT_MODEL",
    "DEFAULT_TEMPERATURE",
    "CerebrasProvider",
]
