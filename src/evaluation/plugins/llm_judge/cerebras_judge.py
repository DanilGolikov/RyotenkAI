"""
Cerebras LLM-as-judge plugin.

Two classes:
- CerebrasProvider  — IJudgeProvider impl using Cerebras chat completions API (urllib only).
- CerebrasJudgePlugin — EvaluatorPlugin that uses CerebrasProvider to score each sample.

Scoring:
    Judge returns integer 1–5 in JSON mode with a 'reasoning' field (CoT).
    Plugin normalizes: normalized = (score - 1) / 4  →  [0, 1]
    EvalResult.metrics: mean_score, p50_score, score_distribution (1..5), sample_count.

Configuration (YAML):
    evaluation:
      evaluators:
        - id: judge_main
          plugin: cerebras_judge
          enabled: true
          params:
            model: "llama-3.3-70b"
            max_samples: 50
            temperature: 0.0
            max_tokens: 512
            max_retries: 3
          thresholds:
            min_mean_score: 0.6

Secret:
    EVAL_CEREBRAS_API_KEY must be set in secrets.env.
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

from src.evaluation.plugins.base import EvalResult, EvalSample, EvaluatorPlugin
from src.evaluation.plugins.llm_judge.interface import JudgeResponse
from src.evaluation.plugins.registry import EvaluatorPluginRegistry
from src.evaluation.plugins.secrets import requires_secrets
from src.evaluation.plugins.utils import PluginReportRow, aggregate_scores, save_plugin_report
from src.utils.logger import logger

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_CEREBRAS_API_URL = "https://api.cerebras.ai/v1/chat/completions"
_SCORE_MIN = 1
_SCORE_MAX = 5
_SCORE_RANGE = _SCORE_MAX - _SCORE_MIN  # 4

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

_DEFAULT_MODEL = "llama3.1-8b"
_DEFAULT_TEMPERATURE = 0.0
_DEFAULT_MAX_TOKENS = 512
_DEFAULT_MAX_RETRIES = 3
_DEFAULT_MAX_SAMPLES = 50
_DEFAULT_MIN_MEAN_SCORE = 0.6
_RETRY_BASE_SLEEP = 2.0


def _make_ssl_context() -> ssl.SSLContext:
    """
    Build an SSL context for Cerebras API requests.

    Tries certifi bundle first (most reliable). Falls back to the default
    system context. If both fail (e.g. macOS without 'Install Certificates'),
    returns an unverified context and logs a one-time warning — this is a
    developer-machine convenience, not a security bypass in production.
    """
    # 1. Try certifi (installed alongside most ML stacks)
    try:
        import certifi

        ctx = ssl.create_default_context(cafile=certifi.where())
        return ctx
    except ImportError:
        pass

    # 2. Try the default system context
    try:
        ctx = ssl.create_default_context()
        # Quick smoke-test: if the system bundle is broken this raises on first use
        return ctx
    except ssl.SSLError:
        pass

    # 3. Last resort: skip verification and warn
    logger.warning(
        "[CEREBRAS] SSL certificate verification disabled — system CA bundle not found. "
        "Run 'pip install certifi' or '/Applications/Python*/Install Certificates.command' to fix."
    )
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE
    return ctx


# ---------------------------------------------------------------------------
# CerebrasProvider
# ---------------------------------------------------------------------------


class CerebrasProvider:
    """
    IJudgeProvider implementation for Cerebras API.

    Uses only stdlib (urllib) — no external packages required.
    Implements retry with exponential backoff for transient API errors.
    """

    def __init__(
        self,
        api_key: str,
        model: str = _DEFAULT_MODEL,
        temperature: float = _DEFAULT_TEMPERATURE,
        max_tokens: int = _DEFAULT_MAX_TOKENS,
        max_retries: int = _DEFAULT_MAX_RETRIES,
    ) -> None:
        self._api_key = api_key
        self._model = model
        self._temperature = temperature
        self._max_tokens = max_tokens
        self._max_retries = max_retries
        self._ssl_context: ssl.SSLContext = _make_ssl_context()

    # ------------------------------------------------------------------
    # IJudgeProvider contract
    # ------------------------------------------------------------------

    def judge(self, question: str, expected: str, model_answer: str) -> JudgeResponse:
        """
        Call Cerebras chat completions and return a scored JudgeResponse.

        Retries up to max_retries times on HTTP 429 / 5xx errors.

        Raises:
            RuntimeError: if all retries are exhausted or parsing fails fatally.
        """
        messages = self._build_messages(question=question, expected=expected, model_answer=model_answer)
        raw = self._call_api_with_retries(messages)
        return self._parse_response(raw)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_messages(self, *, question: str, expected: str, model_answer: str) -> list[dict[str, str]]:
        user_content = f"Question: {question}\n\nExpected answer: {expected}\n\nModel answer: {model_answer}"
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
                sleep_secs = _RETRY_BASE_SLEEP * (2**attempt)
                logger.warning(
                    f"[CEREBRAS] API error (attempt {attempt + 1}/{self._max_retries}), "
                    f"retrying in {sleep_secs:.0f}s: {exc}"
                )
                time.sleep(sleep_secs)
        raise RuntimeError(f"Cerebras API failed after {self._max_retries} retries") from last_error

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
        """
        Parse judge JSON response.

        Primary: json.loads() on the raw string.
        Fallback: regex extraction of 'score': N — handles edge cases where
                  the model wraps JSON in markdown fences or adds extra text.

        Raises RuntimeError only if score cannot be extracted at all.
        """
        parsed: dict[str, Any] | None = None

        with contextlib.suppress(json.JSONDecodeError):
            parsed = json.loads(raw)

        # Fallback: extract first JSON object from the string
        if parsed is None:
            match = re.search(r"\{[^}]+\}", raw, re.DOTALL)
            if match:
                with contextlib.suppress(json.JSONDecodeError):
                    parsed = json.loads(match.group())

        # Last resort: extract score integer directly
        score: int | None = None
        reasoning: str = ""

        if parsed is not None:
            raw_score = parsed.get("score")
            reasoning = str(parsed.get("reasoning", ""))
            if isinstance(raw_score, int | float):
                score = int(raw_score)

        if score is None:
            # Regex fallback on the raw string
            score_match = re.search(r'"score"\s*:\s*([1-5])', raw)
            if score_match:
                score = int(score_match.group(1))

        if score is None:
            raise RuntimeError(f"Could not extract score from Cerebras response. Raw response: {raw[:200]!r}")

        # Clamp to valid range
        score = max(_SCORE_MIN, min(_SCORE_MAX, score))
        return JudgeResponse(score=score, reasoning=reasoning, raw_response=raw)


# ---------------------------------------------------------------------------
# CerebrasJudgePlugin
# ---------------------------------------------------------------------------


@EvaluatorPluginRegistry.register
@requires_secrets("EVAL_CEREBRAS_API_KEY")
class CerebrasJudgePlugin(EvaluatorPlugin):
    """
    LLM-as-judge evaluation plugin using Cerebras API.

    Evaluates model answers by asking a Cerebras judge model to score each
    sample (1–5). Scores are normalized to [0, 1] and aggregated into
    mean_score, p50_score, and score_distribution metrics.

    Requires secret: EVAL_CEREBRAS_API_KEY in secrets.env.
    """

    name = "cerebras_judge"
    priority = 60
    requires_expected_answer = True

    # Injected by EvaluationRunner via @requires_secrets mechanism
    _secrets: dict[str, str]

    @classmethod
    def get_description(cls) -> str:
        return (
            "LLM-as-judge: uses Cerebras to score model answers on a 1–5 scale "
            "against expected answers. Normalizes scores to [0, 1]."
        )

    def evaluate(self, samples: list[EvalSample]) -> EvalResult:
        api_key: str = self._secrets["EVAL_CEREBRAS_API_KEY"]

        model: str = str(self.params.get("model", _DEFAULT_MODEL))
        temperature: float = float(self.params.get("temperature", _DEFAULT_TEMPERATURE))
        max_tokens: int = int(self.params.get("max_tokens", _DEFAULT_MAX_TOKENS))
        max_retries: int = int(self.params.get("max_retries", _DEFAULT_MAX_RETRIES))
        max_samples: int = int(self.params.get("max_samples", _DEFAULT_MAX_SAMPLES))

        provider = CerebrasProvider(
            api_key=api_key,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            max_retries=max_retries,
        )

        # Limit samples to avoid cost overrun
        eval_samples = samples[:max_samples]
        if len(samples) > max_samples:
            logger.warning(
                f"[CEREBRAS] Limiting evaluation to {max_samples} of {len(samples)} samples (max_samples={max_samples})"
            )

        normalized_scores: list[float] = []
        raw_scores: list[int] = []
        failed_indices: list[int] = []
        rows: list[PluginReportRow] = []

        for idx, sample in enumerate(eval_samples):
            expected = sample.expected_answer or ""
            try:
                response = provider.judge(
                    question=sample.question,
                    expected=expected,
                    model_answer=sample.model_answer,
                )
                raw_score = response.score
                normalized = (raw_score - _SCORE_MIN) / _SCORE_RANGE
                normalized_scores.append(normalized)
                raw_scores.append(raw_score)
                rows.append(
                    PluginReportRow(
                        idx=idx,
                        question=sample.question,
                        model_answer=sample.model_answer,
                        expected_answer=sample.expected_answer,
                        score=normalized,
                        raw_score=raw_score,
                        note=response.reasoning,
                    )
                )
                logger.debug(
                    f"[CEREBRAS] Sample {idx}: score={raw_score}/{_SCORE_MAX}, "
                    f"reasoning={response.reasoning[:80]!r}"
                )
            except RuntimeError as exc:
                logger.warning(f"[CEREBRAS] Failed to judge sample {idx}: {exc}")
                failed_indices.append(idx)
                rows.append(
                    PluginReportRow(
                        idx=idx,
                        question=sample.question,
                        model_answer=sample.model_answer,
                        expected_answer=sample.expected_answer,
                        score=None,
                        raw_score=None,
                        note=f"API error: {exc}",
                    )
                )

        if not normalized_scores:
            result = EvalResult(
                plugin_name=self.name,
                passed=False,
                errors=["All samples failed to be judged by Cerebras API"],
                sample_count=len(eval_samples),
                failed_samples=list(range(len(eval_samples))),
            )
            if getattr(self, "_save_report", False):
                save_plugin_report(self.name, rows, result)
            return result

        recommendations = self.get_recommendations(EvalResult(plugin_name=self.name, passed=True, metrics={}))

        result = aggregate_scores(
            scores=normalized_scores,
            raw_scores=raw_scores,
            failed_indices=failed_indices,
            plugin_name=self.name,
            threshold_key="min_mean_score",
            thresholds=self.thresholds,
            recommendations=recommendations,
        )
        if getattr(self, "_save_report", False):
            save_plugin_report(self.name, rows, result)
        return result

    def get_recommendations(self, result: EvalResult) -> list[str]:  # noqa: ARG002
        return [
            "Review low-scoring samples in evaluation/answers.md",
            "Consider fine-tuning with more diverse training examples",
            "Check if the model answer format matches expected format",
        ]

    def _validate_contract(self) -> None:
        model = self._param("model", _DEFAULT_MODEL)
        if not isinstance(model, str) or not model:
            raise ValueError(f"CerebrasJudgePlugin: 'model' param must be a non-empty string, got {model!r}")

        temperature = self._param("temperature", _DEFAULT_TEMPERATURE)
        if not isinstance(temperature, int | float) or not (0.0 <= float(temperature) <= 2.0):
            raise ValueError(f"CerebrasJudgePlugin: 'temperature' must be in [0.0, 2.0], got {temperature!r}")

        max_samples = self._param("max_samples", _DEFAULT_MAX_SAMPLES)
        if not isinstance(max_samples, int) or max_samples < 1:
            raise ValueError(f"CerebrasJudgePlugin: 'max_samples' must be a positive integer, got {max_samples!r}")


__all__ = [
    "CerebrasJudgePlugin",
    "CerebrasProvider",
]
