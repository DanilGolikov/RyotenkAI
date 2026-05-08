"""``classify_hf_upload_error`` — turn raw stderr into a typed failure.

Coverage categories: positive, negative, boundary, invariant,
dependency-error, regression (sample stderrs from real failed runs),
logic-specific (per-status branching), combinatorial.

The classifier is pure — no IO, no async, fully deterministic — so
every branch can be exercised with a synthetic stderr string.
"""

from __future__ import annotations

import pytest

from ryotenkai_control.pipeline.stages.model_retriever.hf_errors import (
    HFUploadFailure,
    classify_hf_upload_error,
)

pytestmark = pytest.mark.unit


# Real stderr captured from a failed run (lightly trimmed). Lives at
# the top so the regression test reads well.
_REAL_401_STDERR = """\
Traceback (most recent call last):
  File "/opt/conda/lib/python3.12/site-packages/huggingface_hub/utils/_http.py", line 403, in hf_raise_for_status
    response.raise_for_status()
  File "/opt/conda/lib/python3.12/site-packages/requests/models.py", line 1024, in raise_for_status
    raise HTTPError(http_error_msg, response=self)
requests.exceptions.HTTPError: 401 Client Error: Unauthorized for url: https://huggingface.co/api/repos/create

The above exception was the direct cause of the following exception:

Traceback (most recent call last):
  File "/usr/local/bin/huggingface-cli", line 6, in <module>
    sys.exit(main())
huggingface_hub.errors.HfHubHTTPError: 401 Client Error: Unauthorized for url: https://huggingface.co/api/repos/create

Invalid username or password.
"""


# ===========================================================================
# Positive — recognised failure modes return the right kind/action
# ===========================================================================


class TestPositive:
    def test_401_unauthorized(self) -> None:
        result = classify_hf_upload_error(_REAL_401_STDERR, exit_code=1)
        assert result.kind == "auth"
        assert result.http_status == 401
        assert result.retryable is False
        assert "401" in result.short_reason
        assert "HF_TOKEN" in result.operator_action

    def test_403_forbidden(self) -> None:
        stderr = "huggingface_hub.errors.HfHubHTTPError: 403 Client Error: Forbidden"
        result = classify_hf_upload_error(stderr, exit_code=1)
        assert result.kind == "auth"
        assert result.http_status == 403
        assert result.retryable is False
        assert "write" in result.operator_action.lower() or "permission" in result.operator_action.lower()

    def test_404_not_found(self) -> None:
        stderr = "HfHubHTTPError: 404 Client Error: Not Found for url: https://huggingface.co/api/models/nonexistent/repo"
        result = classify_hf_upload_error(stderr, exit_code=1)
        assert result.kind == "not_found"
        assert result.http_status == 404
        assert result.retryable is False

    def test_429_rate_limited_retryable(self) -> None:
        stderr = "requests.exceptions.HTTPError: 429 Client Error: Too Many Requests"
        result = classify_hf_upload_error(stderr, exit_code=1)
        assert result.kind == "rate_limit"
        assert result.http_status == 429
        assert result.retryable is True

    def test_502_bad_gateway_retryable(self) -> None:
        stderr = "requests.exceptions.HTTPError: 502 Server Error: Bad Gateway"
        result = classify_hf_upload_error(stderr, exit_code=1)
        assert result.kind == "transient"
        assert result.http_status == 502
        assert result.retryable is True

    def test_503_service_unavailable_retryable(self) -> None:
        stderr = "HTTPError: 503 Server Error: Service Unavailable"
        result = classify_hf_upload_error(stderr, exit_code=1)
        assert result.kind == "transient"
        assert result.http_status == 503
        assert result.retryable is True

    def test_network_dns_failure(self) -> None:
        stderr = (
            "urllib3.exceptions.NameResolutionError: <urllib3.connection.HTTPSConnection ...>: "
            "Failed to resolve 'huggingface.co' ([Errno -3] Temporary failure in name resolution)"
        )
        result = classify_hf_upload_error(stderr, exit_code=1)
        assert result.kind == "network"
        assert result.retryable is True

    def test_network_connection_refused(self) -> None:
        stderr = "urllib3.exceptions.MaxRetryError: ConnectionRefusedError: [Errno 111]"
        result = classify_hf_upload_error(stderr, exit_code=1)
        assert result.kind == "network"
        assert result.retryable is True

    def test_quota_exceeded(self) -> None:
        stderr = "Storage quota exceeded for namespace 'foo'"
        result = classify_hf_upload_error(stderr, exit_code=1)
        assert result.kind == "quota"
        assert result.retryable is False

    def test_413_payload_too_large(self) -> None:
        stderr = "HfHubHTTPError: 413 Client Error: Payload Too Large"
        result = classify_hf_upload_error(stderr, exit_code=1)
        assert result.kind == "quota"
        assert result.http_status == 413
        assert result.retryable is False

    def test_cli_missing(self) -> None:
        stderr = "bash: huggingface-cli: command not found\n"
        result = classify_hf_upload_error(stderr, exit_code=127)
        assert result.kind == "cli_missing"
        assert result.retryable is False
        assert "huggingface_hub" in result.operator_action or "PATH" in result.operator_action


# ===========================================================================
# Negative / unrecognised → unknown but retryable (defensive default)
# ===========================================================================


class TestUnknown:
    def test_random_unrecognised_text(self) -> None:
        stderr = "something completely unrelated happened — UFO landed"
        result = classify_hf_upload_error(stderr, exit_code=1)
        assert result.kind == "unknown"
        assert result.retryable is True
        assert result.http_status is None

    def test_empty_stderr_with_nonzero_exit(self) -> None:
        """When the subprocess died silently, surface that exact fact."""
        result = classify_hf_upload_error("", exit_code=137)
        assert result.kind == "unknown"
        assert result.retryable is True
        assert "137" in result.short_reason
        assert "killed externally" in result.short_reason or "no stderr" in result.short_reason

    def test_only_whitespace_stderr(self) -> None:
        result = classify_hf_upload_error("   \n  \t\n", exit_code=1)
        assert result.kind == "unknown"
        assert result.retryable is True


# ===========================================================================
# Boundary — empty inputs, raw excerpt truncation, status edge cases
# ===========================================================================


class TestBoundary:
    def test_excerpt_truncated_to_200_chars(self) -> None:
        long_stderr = "401 Client Error: Unauthorized\n" + ("X" * 5000)
        result = classify_hf_upload_error(long_stderr, exit_code=1)
        assert len(result.raw_stderr_excerpt) <= 200

    def test_excerpt_preserves_the_signal_at_top(self) -> None:
        stderr = "401 Client Error: Unauthorized\n" + ("X" * 5000)
        result = classify_hf_upload_error(stderr, exit_code=1)
        # The leading line (the signal) is in the excerpt.
        assert "401" in result.raw_stderr_excerpt

    def test_status_400_falls_to_generic_client_error(self) -> None:
        stderr = "HfHubHTTPError: 400 Client Error: Bad Request"
        result = classify_hf_upload_error(stderr, exit_code=1)
        assert result.kind == "client_error"
        assert result.http_status == 400
        assert result.retryable is False

    def test_status_499_falls_to_generic_client_error(self) -> None:
        # Cloudflare 499 'Client Closed Request' — mapped under client_error.
        stderr = "HTTP 499 Client Error: Custom"
        result = classify_hf_upload_error(stderr, exit_code=1)
        assert result.kind == "client_error"
        assert result.http_status == 499

    def test_status_500_is_transient(self) -> None:
        stderr = "HTTP 500 Server Error: Internal"
        result = classify_hf_upload_error(stderr, exit_code=1)
        assert result.kind == "transient"
        assert result.retryable is True

    def test_status_599_is_transient(self) -> None:
        stderr = "HTTP 599 Server Error: Network connect timeout"
        result = classify_hf_upload_error(stderr, exit_code=1)
        assert result.kind == "transient"
        assert result.retryable is True


# ===========================================================================
# Invariants — frozen dataclass, well-formed log line, stable codes
# ===========================================================================


class TestInvariants:
    def test_failure_is_frozen(self) -> None:
        f = classify_hf_upload_error("404 Client Error: Not Found", exit_code=1)
        with pytest.raises((AttributeError, Exception)):
            f.kind = "auth"  # type: ignore[misc]

    def test_log_line_is_single_line(self) -> None:
        """Single-line invariant — the whole point of this refactor."""
        for stderr in (
            _REAL_401_STDERR,
            "503 Server Error",
            "MaxRetryError: connection refused",
            "garbage",
        ):
            f = classify_hf_upload_error(stderr, exit_code=1)
            log = f.to_log_line()
            assert "\n" not in log, f"newline leaked: {log!r}"
            assert "Traceback" not in log, f"traceback leaked: {log!r}"

    def test_log_line_starts_with_kind_tag(self) -> None:
        for stderr, expected_kind in (
            (_REAL_401_STDERR, "AUTH"),
            ("404 Client Error", "NOT_FOUND"),
            ("503 Server Error", "TRANSIENT"),
            ("garbage", "UNKNOWN"),
        ):
            f = classify_hf_upload_error(stderr, exit_code=1)
            assert f.to_log_line().startswith(f"[HF:{expected_kind}]")

    def test_app_error_code_is_stable(self) -> None:
        """Code shape must not drift — downstream stages match on it."""
        f = classify_hf_upload_error("401 Client Error", exit_code=1)
        assert f.to_app_error_code() == "HF_UPLOAD_AUTH"

    def test_retryable_implies_no_terminal_action(self) -> None:
        """Soft check: retryable failures should hint at retry, not give a
        final 'fix this' action that contradicts the retry signal."""
        f = classify_hf_upload_error("503 Server Error", exit_code=1)
        assert f.retryable is True
        # Acceptable phrasings that don't promise immediate user action:
        assert (
            "retry" in f.operator_action.lower()
            or "transient" in f.operator_action.lower()
        )


# ===========================================================================
# Dependency-error — never raises on weird inputs
# ===========================================================================


class TestDependencyError:
    @pytest.mark.parametrize(
        "stderr",
        [
            "",                          # empty
            None,                        # type: ignore[arg-type]
            "x",                         # one char
            "\x00\x01\x02",              # binary garbage
            "401 401 401 401",           # multiple matches
        ],
    )
    def test_classifier_never_raises(self, stderr) -> None:  # type: ignore[no-untyped-def]
        # Must not raise, must return a well-formed HFUploadFailure.
        result = classify_hf_upload_error(stderr or "", exit_code=1)
        assert isinstance(result, HFUploadFailure)
        assert result.kind in {
            "auth", "not_found", "rate_limit", "quota",
            "network", "transient", "client_error",
            "cli_missing", "unknown",
        }


# ===========================================================================
# Regression — the exact stderr that triggered this refactor
# ===========================================================================


class TestRegression:
    def test_real_world_401_traceback_classifies_clean(self) -> None:
        """The actual stderr from the user's failed run on 2026-05-07."""
        result = classify_hf_upload_error(_REAL_401_STDERR, exit_code=1)
        assert result.kind == "auth"
        assert result.http_status == 401
        assert result.retryable is False
        # The log line is what an operator sees — it must NOT contain
        # the traceback OR the multi-line 'The above exception ...' chain.
        log_line = result.to_log_line()
        assert "Traceback" not in log_line
        assert "exception" not in log_line.lower()
        assert "\n" not in log_line
        # Must contain something actionable about HF_TOKEN.
        assert "HF_TOKEN" in result.operator_action

    def test_legacy_caller_can_still_pattern_match_codes(self) -> None:
        """Stable error codes — Inference Deployer / summary report match
        on them via string equality, no regex."""
        for stderr, expected_code in (
            ("401 Client Error", "HF_UPLOAD_AUTH"),
            ("403 Client Error", "HF_UPLOAD_AUTH"),
            ("404 Client Error", "HF_UPLOAD_NOT_FOUND"),
            ("503 Server Error", "HF_UPLOAD_TRANSIENT"),
            ("MaxRetryError", "HF_UPLOAD_NETWORK"),
            ("garbage", "HF_UPLOAD_UNKNOWN"),
        ):
            f = classify_hf_upload_error(stderr, exit_code=1)
            assert f.to_app_error_code() == expected_code, (
                f"stderr={stderr!r}: expected {expected_code}, "
                f"got {f.to_app_error_code()}"
            )


# ===========================================================================
# Logic-specific — match-order matters
# ===========================================================================


class TestMatchOrder:
    def test_cli_missing_takes_precedence_over_status(self) -> None:
        """When both a CLI-missing signature AND a 401 substring appear
        (rare but possible if the wrapper script logs both), the CLI
        miss is the actual root cause — fix that first."""
        stderr = "bash: huggingface-cli: command not found\n401 Client Error from previous probe"
        result = classify_hf_upload_error(stderr, exit_code=127)
        assert result.kind == "cli_missing"

    def test_network_takes_precedence_over_unrecognised_text(self) -> None:
        stderr = "Some preamble text\nMaxRetryError: HTTPSConnectionPool(...)\nAfterword"
        result = classify_hf_upload_error(stderr, exit_code=1)
        assert result.kind == "network"

    def test_first_status_match_wins(self) -> None:
        """Real tracebacks contain the same status repeated 2-3 times.
        We pick the first one and stop — verify the dispatch is stable."""
        stderr = "401 Client Error\nMore context\n401 Client Error: Unauthorized\n"
        result = classify_hf_upload_error(stderr, exit_code=1)
        assert result.http_status == 401


# ===========================================================================
# Combinatorial — every kind has well-formed output
# ===========================================================================


class TestCombinatorial:
    @pytest.mark.parametrize(
        ("stderr_sample", "expected_kind"),
        [
            ("401 Client Error", "auth"),
            ("403 Client Error", "auth"),
            ("404 Client Error", "not_found"),
            ("413 Client Error", "quota"),
            ("429 Client Error", "rate_limit"),
            ("450 Client Error", "client_error"),
            ("503 Server Error", "transient"),
            ("MaxRetryError: ConnectionRefusedError", "network"),
            ("Storage quota exceeded", "quota"),
            ("bash: huggingface-cli: command not found", "cli_missing"),
            ("nonsense gibberish", "unknown"),
        ],
    )
    def test_kind_dispatch(self, stderr_sample: str, expected_kind: str) -> None:
        result = classify_hf_upload_error(stderr_sample, exit_code=1)
        assert result.kind == expected_kind
        # Universal: every result has a non-empty log line and stable code.
        assert result.to_log_line()
        assert result.to_app_error_code().startswith("HF_UPLOAD_")
