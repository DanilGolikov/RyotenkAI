"""
Integration test: live Cerebras API call.

Verifies end-to-end connectivity and correct response parsing against the
real Cerebras inference endpoint. Skipped automatically when
EVAL_CEREBRAS_API_KEY is absent from secrets.env (or env), so it never
blocks CI on machines without credentials.

Run manually:
    pytest src/tests/integration/test_cerebras_api_live.py -v -s

What is tested (real HTTP, no mocks):
- HTTP 200 response from api.cerebras.ai
- Response JSON contains valid score (1–5) and non-empty reasoning
- Score normalization to [0, 1] is correct
- CerebrasJudgePlugin.evaluate() works end-to-end with injected secrets
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_api_key() -> str | None:
    """
    Try to load EVAL_CEREBRAS_API_KEY from secrets.env or environment.
    Returns None if not found.
    """
    # 1. Environment variable (CI / exported)
    val = os.environ.get("EVAL_CEREBRAS_API_KEY", "").strip()
    if val:
        return val

    # 2. secrets.env in project root
    secrets_path = Path(__file__).parents[3] / "secrets.env"
    if secrets_path.exists():
        for line in secrets_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            if key.strip().upper() == "EVAL_CEREBRAS_API_KEY":
                return value.strip()

    return None


_API_KEY = _load_api_key()
_SKIP_REASON = (
    "EVAL_CEREBRAS_API_KEY not set in secrets.env or environment — skipping live API test"
)


# ---------------------------------------------------------------------------
# CerebrasProvider live tests
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not _API_KEY, reason=_SKIP_REASON)
class TestCerebrasProviderLive:
    """Tests against the real Cerebras API endpoint."""

    def _make_provider(self, model: str = "llama3.1-8b"):
        from src.evaluation.plugins.llm_judge.cerebras_judge import CerebrasProvider

        assert _API_KEY is not None
        return CerebrasProvider(
            api_key=_API_KEY,
            model=model,
            temperature=0.0,
            max_tokens=256,
            max_retries=2,
        )

    def test_judge_returns_valid_score(self):
        """Real API call: score must be integer 1–5."""
        provider = self._make_provider()
        response = provider.judge(
            question="What is 2 + 2?",
            expected="4",
            model_answer="4",
        )
        assert 1 <= response.score <= 5
        assert isinstance(response.reasoning, str)
        assert len(response.reasoning) > 0

    def test_judge_perfect_match_scores_high(self):
        """Identical answer should score 4 or 5."""
        provider = self._make_provider()
        answer = "QUERY GetUser() => N<User> RETURN N"
        response = provider.judge(
            question="Write a HelixQL query to get all users",
            expected=answer,
            model_answer=answer,
        )
        assert response.score >= 4, (
            f"Expected score >= 4 for perfect match, got {response.score}. "
            f"Reasoning: {response.reasoning}"
        )

    def test_judge_wrong_answer_scores_low(self):
        """Completely wrong answer should score 1 or 2."""
        provider = self._make_provider()
        response = provider.judge(
            question="Write a HelixQL query to get all users",
            expected="QUERY GetUser() => N<User> RETURN N",
            model_answer="I don't know how to write HelixQL queries.",
        )
        assert response.score <= 2, (
            f"Expected score <= 2 for wrong answer, got {response.score}. "
            f"Reasoning: {response.reasoning}"
        )

    def test_score_normalization(self):
        """Normalized score must be in [0, 1]."""
        from src.evaluation.plugins.llm_judge.cerebras_judge import _SCORE_MIN, _SCORE_RANGE

        provider = self._make_provider()
        response = provider.judge(
            question="What is the capital of France?",
            expected="Paris",
            model_answer="Paris",
        )
        normalized = (response.score - _SCORE_MIN) / _SCORE_RANGE
        assert 0.0 <= normalized <= 1.0

    def test_user_agent_header_not_blocked(self):
        """Verify no Cloudflare 403 (regression: Python-urllib was blocked)."""
        provider = self._make_provider()
        # If this raises RuntimeError with '403', the User-Agent fix regressed.
        try:
            response = provider.judge(
                question="Is the sky blue?",
                expected="Yes",
                model_answer="Yes",
            )
            assert response.score >= 1
        except RuntimeError as exc:
            if "403" in str(exc):
                pytest.fail(
                    f"Cloudflare 403 — User-Agent header fix may have regressed. Error: {exc}"
                )
            raise

    def test_invalid_model_raises_404(self):
        """Non-existent model name must raise RuntimeError with 404, not hang."""
        provider = self._make_provider(model="nonexistent-model-xyz-404")  # guaranteed invalid
        with pytest.raises(RuntimeError, match="404"):
            provider.judge(
                question="test",
                expected="test",
                model_answer="test",
            )


# ---------------------------------------------------------------------------
# CerebrasJudgePlugin end-to-end live test
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not _API_KEY, reason=_SKIP_REASON)
class TestCerebrasJudgePluginLive:
    """End-to-end: plugin with injected secrets, real samples, real API."""

    def _make_plugin(self) -> object:
        from src.evaluation.plugins.llm_judge.cerebras_judge import CerebrasJudgePlugin

        plugin = CerebrasJudgePlugin(
            params={"model": "llama3.1-8b", "max_samples": 3, "temperature": 0.0, "max_tokens": 256, "max_retries": 3},
            thresholds={"min_mean_score": 0.5},
        )
        assert _API_KEY is not None
        object.__setattr__(plugin, "_secrets", {"EVAL_CEREBRAS_API_KEY": _API_KEY})
        object.__setattr__(plugin, "_save_report", False)
        return plugin

    def _make_samples(self):
        from src.evaluation.plugins.base import EvalSample

        return [
            EvalSample(
                question="Write a HelixQL query to get all users",
                model_answer="QUERY GetUser() => N<User> RETURN N",
                expected_answer="QUERY GetUser() => N<User> RETURN N",
            ),
            EvalSample(
                question="What is the capital of France?",
                model_answer="Paris",
                expected_answer="Paris",
            ),
            EvalSample(
                question="Write a HelixQL query to get all users",
                model_answer="I don't know.",
                expected_answer="QUERY GetUser() => N<User> RETURN N",
            ),
        ]

    def test_evaluate_returns_result(self):
        """Plugin produces EvalResult with metrics after real API calls."""
        plugin = self._make_plugin()
        samples = self._make_samples()

        result = plugin.evaluate(samples)

        assert result.plugin_name == "cerebras_judge"
        assert result.sample_count == 3
        assert "mean_score" in result.metrics
        assert 0.0 <= result.metrics["mean_score"] <= 1.0
        assert "p50_score" in result.metrics

    def test_evaluate_good_samples_pass(self):
        """Two perfect matches and one wrong: mean_score should exceed 0.5."""
        from src.evaluation.plugins.base import EvalSample

        plugin = self._make_plugin()
        answer = "QUERY GetUser() => N<User> RETURN N"
        samples = [
            EvalSample(question="q1", model_answer=answer, expected_answer=answer),
            EvalSample(question="q2", model_answer=answer, expected_answer=answer),
            EvalSample(question="q3", model_answer="wrong", expected_answer=answer),
        ]

        result = plugin.evaluate(samples)

        assert result.metrics["mean_score"] > 0.5, (
            f"Expected mean_score > 0.5, got {result.metrics['mean_score']}"
        )

    def test_no_cloudflare_403_on_plugin_evaluate(self):
        """Regression: plugin.evaluate() must not produce 403 errors for any sample."""
        plugin = self._make_plugin()
        samples = self._make_samples()

        result = plugin.evaluate(samples)

        # If all samples failed due to 403, failed_samples would equal all indices
        all_failed = result.failed_samples == list(range(len(samples)))
        if all_failed and result.errors:
            pytest.fail(
                f"All samples failed — possible Cloudflare 403 regression. "
                f"Errors: {result.errors}"
            )
