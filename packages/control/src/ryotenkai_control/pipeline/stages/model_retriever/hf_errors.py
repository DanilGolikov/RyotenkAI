"""HuggingFace upload error classifier.

Why this exists
---------------
The ``huggingface-cli upload`` subprocess (run over SSH on the pod) emits
multi-line Python tracebacks on stderr. The previous code path piped the
raw stderr straight into ``logger.warning(...)`` and into the final
``ModelError.message``. Result:

* ~30-line traceback per attempt × 3 retries = ~100 lines of log noise.
* The actual signal (``401 Unauthorized``, ``Repo not found``, ``Quota
  exceeded``, ``DNS lookup failed``, …) buried inside the traceback.
* Auth errors got 3 retries with 10s sleeps between them — 30s of pure
  waste because a stale token doesn't refresh between sleeps.
* All failure paths returned the same ``HF_UPLOAD_COMMAND_FAILED`` code,
  so callers (Inference Deployer, summary report) couldn't react
  differently to "fix your token" vs "Hub had a 502".

This module turns raw stderr + exit code into a structured
:class:`HFUploadFailure` carrying:

* ``kind`` — domain category (auth / not_found / rate_limit / network / …)
* ``http_status`` — extracted HTTP status when present
* ``short_reason`` — single-sentence operator-facing summary
* ``operator_action`` — what to do next (or empty when nothing helps)
* ``raw_stderr_excerpt`` — first 200 chars of original stderr for debug
* ``retryable`` — whether retrying the same call is worth waiting for

The matchers are deliberately conservative — when nothing recognised
fires, we return ``kind="unknown"`` and treat it as retryable (favour
catching transient flakes over silently skipping a real failure).
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal

# ---------------------------------------------------------------------------
# Public dataclass
# ---------------------------------------------------------------------------


HFFailureKind = Literal[
    "auth",          # 401 / 403 — token bad or wrong permissions
    "not_found",     # 404 — repo or namespace doesn't exist
    "rate_limit",    # 429 — Hub rate-limited us
    "quota",         # storage / LFS quota / payload-too-large
    "network",       # DNS / connection refused / SSL handshake / TLS
    "transient",     # 5xx server errors
    "client_error",  # other 4xx not covered above
    "cli_missing",   # huggingface-cli not on PATH
    "unknown",       # nothing matched
]


_RAW_EXCERPT_MAX_CHARS: int = 200


@dataclass(frozen=True)
class HFUploadFailure:
    """Structured representation of a failed HF upload attempt.

    Construct via :func:`classify_hf_upload_error`. The dataclass is
    frozen so callers can safely store it on retry-history lists
    without worrying about mutation.
    """

    kind: HFFailureKind
    http_status: int | None
    short_reason: str
    operator_action: str
    raw_stderr_excerpt: str
    retryable: bool

    def to_log_line(self) -> str:
        """Render a single, dense log line — no traceback, no extra newlines.

        Format:
            [HF:<KIND>] <short_reason> | action: <operator_action>
        Or, when no action is known:
            [HF:<KIND>] <short_reason>
        """
        prefix = f"[HF:{self.kind.upper()}]"
        if self.operator_action:
            return f"{prefix} {self.short_reason} | action: {self.operator_action}"
        return f"{prefix} {self.short_reason}"

    def to_app_error_code(self) -> str:
        """Stable error code for the outer ``ModelError`` / ``Err``.

        Stable across runs so callers (Inference Deployer, summary
        reporter, MLflow tags) can pattern-match without parsing
        free-form messages.
        """
        return f"HF_UPLOAD_{self.kind.upper()}"


# ---------------------------------------------------------------------------
# Pattern matchers — order matters (specific → generic)
# ---------------------------------------------------------------------------


# HTTP-status patterns. We look for "<status> Client Error" / "<status>
# Server Error" (huggingface_hub's _http.py format) and the bare
# "HTTP <status>" form for completeness. Status is extracted as int.
_HTTP_STATUS_RE = re.compile(
    r"\b(?P<status>4\d\d|5\d\d)\s+(?:Client|Server)\s+Error\b",
    re.IGNORECASE,
)
_BARE_STATUS_RE = re.compile(
    r"\bHTTP\s+(?P<status>4\d\d|5\d\d)\b",
    re.IGNORECASE,
)

# Network signatures. These are what huggingface_hub typically wraps
# under ``ConnectionError`` / ``Timeout`` from urllib3 / requests.
_NETWORK_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"\bNameResolutionError\b"),
    re.compile(r"\bgaierror\b"),
    re.compile(r"\bConnectionRefusedError\b"),
    re.compile(r"\bConnectionResetError\b"),
    re.compile(r"\bConnectTimeoutError\b"),
    re.compile(r"\bReadTimeoutError\b"),
    re.compile(r"\bSSLError\b"),
    re.compile(r"\bMaxRetryError\b"),
    re.compile(r"\bTemporary\s+failure\s+in\s+name\s+resolution\b", re.IGNORECASE),
    re.compile(r"\bConnection\s+aborted\b", re.IGNORECASE),
)

# CLI-missing signature. We probe `huggingface-cli` via SSH; if it's not
# on PATH the shell prints a cryptic "command not found".
_CLI_MISSING_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"huggingface-cli:\s+command\s+not\s+found", re.IGNORECASE),
    re.compile(r"^bash:\s+huggingface-cli", re.IGNORECASE | re.MULTILINE),
)

# Quota / storage / LFS / payload-too-large signatures.
_QUOTA_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"\bquota\s+exceeded\b", re.IGNORECASE),
    re.compile(r"\bstorage\s+limit\b", re.IGNORECASE),
    re.compile(r"\bLFS\s+quota\b", re.IGNORECASE),
    re.compile(r"\bpayload\s+too\s+large\b", re.IGNORECASE),
)


# ---------------------------------------------------------------------------
# Classifier
# ---------------------------------------------------------------------------


def classify_hf_upload_error(stderr: str, exit_code: int) -> HFUploadFailure:
    """Classify a failed huggingface-cli invocation into a structured failure.

    Conservative matching: when nothing fires we return ``kind="unknown"``
    with ``retryable=True`` so a transient flake we haven't catalogued
    yet still gets retried. False positives (treating a permanent fault
    as retryable) only cost a few extra seconds; false negatives
    (skipping retry on a real transient) are worse.

    Args:
        stderr: Raw stderr from the SSH'd ``huggingface-cli`` call.
            May be empty when the failure mode produced no text.
        exit_code: Process exit code. Currently used only as a tiebreaker
            for the "no stderr at all" case.

    Returns:
        :class:`HFUploadFailure` with all fields populated. Never raises.
    """
    excerpt = (stderr or "")[:_RAW_EXCERPT_MAX_CHARS].strip()

    # Extract HTTP status once up front so pattern matchers (e.g. the
    # quota matcher catching 'Payload Too Large' which is HTTP 413) can
    # attach the status without re-parsing.
    status: int | None = None
    _status_match = _HTTP_STATUS_RE.search(stderr or "") or _BARE_STATUS_RE.search(stderr or "")
    if _status_match:
        status = int(_status_match.group("status"))

    # ---- 1. CLI not installed --------------------------------------------
    for pat in _CLI_MISSING_PATTERNS:
        if pat.search(stderr or ""):
            return HFUploadFailure(
                kind="cli_missing",
                http_status=None,
                short_reason="huggingface-cli not found on the remote PATH",
                operator_action=(
                    "rebuild the runtime image with huggingface_hub installed, or "
                    "verify the image points at /opt/conda where the CLI lives"
                ),
                raw_stderr_excerpt=excerpt,
                retryable=False,
            )

    # ---- 2. Network / DNS / TLS ------------------------------------------
    for pat in _NETWORK_PATTERNS:
        if pat.search(stderr or ""):
            return HFUploadFailure(
                kind="network",
                http_status=None,
                short_reason="network error reaching huggingface.co (DNS / TLS / refused / reset)",
                operator_action=(
                    "transient — pipeline will retry. If it persists, check the "
                    "pod's network egress rules"
                ),
                raw_stderr_excerpt=excerpt,
                retryable=True,
            )

    # ---- 3. Quota / payload-too-large ------------------------------------
    for pat in _QUOTA_PATTERNS:
        if pat.search(stderr or ""):
            return HFUploadFailure(
                kind="quota",
                http_status=status,  # may be 413 or None depending on stderr shape
                short_reason="HF Hub rejected upload due to quota / storage limits",
                operator_action=(
                    "free up space on the target HF account or split the upload — "
                    "retry will not help"
                ),
                raw_stderr_excerpt=excerpt,
                retryable=False,
            )

    # ---- 4. HTTP status dispatch (status already extracted above) -------

    if status == 401:
        return HFUploadFailure(
            kind="auth",
            http_status=401,
            short_reason="HF auth rejected (HTTP 401 Unauthorized)",
            operator_action=(
                "invalid or missing HF token. Common cause: HF_TOKEN in the "
                "environment overrides secrets.env. Fix: `unset HF_TOKEN` "
                "before running, or refresh the token in secrets.env"
            ),
            raw_stderr_excerpt=excerpt,
            retryable=False,
        )
    if status == 403:
        return HFUploadFailure(
            kind="auth",
            http_status=403,
            short_reason="HF auth forbidden (HTTP 403)",
            operator_action=(
                "token is valid but lacks write permission on the target "
                "namespace/repo. Verify token scope (write) and that you can "
                "push to this repo from the HF web UI"
            ),
            raw_stderr_excerpt=excerpt,
            retryable=False,
        )
    if status == 404:
        return HFUploadFailure(
            kind="not_found",
            http_status=404,
            short_reason="HF Hub returned 404 (repo or namespace not found)",
            operator_action=(
                "verify integrations.huggingface.repo_id matches an existing "
                "namespace and the token can reach it"
            ),
            raw_stderr_excerpt=excerpt,
            retryable=False,
        )
    if status == 429:
        return HFUploadFailure(
            kind="rate_limit",
            http_status=429,
            short_reason="HF Hub rate-limited the request (HTTP 429)",
            operator_action=(
                "transient — pipeline will retry with backoff. Reduce upload "
                "frequency if it keeps recurring"
            ),
            raw_stderr_excerpt=excerpt,
            retryable=True,
        )
    if status == 413:
        return HFUploadFailure(
            kind="quota",
            http_status=413,
            short_reason="HF Hub rejected upload (HTTP 413 Payload Too Large)",
            operator_action="split the model into smaller files or upload via LFS",
            raw_stderr_excerpt=excerpt,
            retryable=False,
        )
    if status is not None and 500 <= status <= 599:
        return HFUploadFailure(
            kind="transient",
            http_status=status,
            short_reason=f"HF Hub server error (HTTP {status})",
            operator_action=(
                "transient on the Hub side — pipeline will retry. Check "
                "https://status.huggingface.co/ if it persists"
            ),
            raw_stderr_excerpt=excerpt,
            retryable=True,
        )
    if status is not None and 400 <= status <= 499:
        # Other 4xx — return generic client_error so callers can still
        # pattern-match on the code without us inventing fake actions.
        return HFUploadFailure(
            kind="client_error",
            http_status=status,
            short_reason=f"HF Hub client error (HTTP {status})",
            operator_action="",
            raw_stderr_excerpt=excerpt,
            retryable=False,
        )

    # ---- 5. Last resort: nothing recognised ------------------------------
    if not (stderr or "").strip() and exit_code != 0:
        return HFUploadFailure(
            kind="unknown",
            http_status=None,
            short_reason=(
                f"huggingface-cli exited {exit_code} with no stderr — "
                "subprocess may have been killed externally"
            ),
            operator_action="",
            raw_stderr_excerpt="",
            retryable=True,
        )

    return HFUploadFailure(
        kind="unknown",
        http_status=None,
        short_reason="huggingface-cli failed with an unrecognised error",
        operator_action=(
            "see raw stderr in debug log; if reproducible, file an issue with "
            "the excerpt"
        ),
        raw_stderr_excerpt=excerpt,
        retryable=True,
    )


__all__ = (
    "HFFailureKind",
    "HFUploadFailure",
    "classify_hf_upload_error",
)
