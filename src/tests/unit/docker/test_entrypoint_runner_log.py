"""
docker/training/entrypoint.sh — invariants for the runner.log redirect.

Static structural tests on the bash script. We do NOT spawn a real
container here (that's a CI-time smoke test); instead we lock in the
properties that, if violated, would break either:

* Signal forwarding (dumb-init → uvicorn): regression to ``tee`` or
  process-substitution would orphan uvicorn from dumb-init's signal
  reach and kill graceful shutdown.
* Capture window: missing ``stdbuf -oL`` would buffer stdout for
  several seconds and lose pre-import error tracebacks if Python
  dies fast.
* Fallback safety: missing writability probe would cause the
  container to fail to start when ``/workspace`` is unmounted.

If any of these tests fails, the entrypoint shell script has drifted
away from the documented architecture (see
``docs/architecture/log-collection.md``).
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

ENTRYPOINT_PATH = (
    Path(__file__).resolve().parents[4]
    / "docker" / "training" / "entrypoint.sh"
)


@pytest.fixture(scope="module")
def entrypoint_text() -> str:
    """Return the script text once per module."""
    assert ENTRYPOINT_PATH.exists(), f"entrypoint.sh not found at {ENTRYPOINT_PATH}"
    return ENTRYPOINT_PATH.read_text()


@pytest.fixture(scope="module")
def entrypoint_code(entrypoint_text: str) -> str:
    """Return entrypoint.sh with comments stripped so substring checks
    don't trip on prose that DISCUSSES forbidden patterns (e.g. the
    long header explaining why we DON'T use tee).
    """
    out_lines: list[str] = []
    for line in entrypoint_text.splitlines():
        # Strip full-line comments and trailing comments.
        stripped = line.lstrip()
        if stripped.startswith("#"):
            continue
        # Trailing inline comment removal — naive but sufficient for
        # this script which has no quoted-# in code.
        idx = line.find(" #")
        if idx >= 0:
            line = line[:idx]
        out_lines.append(line)
    return "\n".join(out_lines)


# ---------------------------------------------------------------------------
# Positive — required structural elements
# ---------------------------------------------------------------------------


def test_exec_uvicorn_uses_direct_redirect_not_tee(entrypoint_code: str) -> None:
    """The uvicorn launch line must use direct ``>>`` redirect, not tee.

    Why: tee via process substitution would put a bash subshell
    between dumb-init and uvicorn — uvicorn becomes a grandchild,
    SIGTERM stops propagating, and graceful shutdown breaks.
    """
    # Look for the exec uvicorn block.
    assert "exec stdbuf" in entrypoint_code, \
        "expected `exec stdbuf -oL -eL` before uvicorn"
    # No tee command in actual code (excluding comments).
    assert not re.search(r"\btee\b", entrypoint_code), \
        "tee command detected in code — would break dumb-init → uvicorn signal chain"
    # And the redirect must use append (>>) — not overwrite (>).
    redirect_block = entrypoint_code.split("exec stdbuf")[-1]
    assert ">> \"$RUNNER_LOG\"" in redirect_block or '>> "$RUNNER_LOG"' in redirect_block, \
        "redirect to runner log must use append (>>) so multiple boots accumulate"
    assert "2>&1" in redirect_block, \
        "stderr must be merged into stdout for capture (2>&1)"


def test_stdbuf_line_buffers_stdout_and_stderr(entrypoint_code: str) -> None:
    """stdbuf -oL forces line-buffered stdout (4KB Python default would
    miss pre-import errors if Python dies fast)."""
    assert "stdbuf -oL -eL" in entrypoint_code, \
        "stdbuf -oL -eL required for real-time line-buffered logging"


def test_runner_log_default_path_is_workspace(entrypoint_code: str) -> None:
    """Default RUNNER_LOG must point at /workspace/runner.log so that
    the Mac-side LogManager finds it at the canonical pod path."""
    assert "/workspace/runner.log" in entrypoint_code, \
        "default RUNNER_LOG path must be /workspace/runner.log"


def test_writability_probe_present(entrypoint_code: str) -> None:
    """Probe must verify the runner-log path is writable BEFORE exec.

    Without it, a read-only /workspace mount would crash the container
    immediately and we'd lose the very diagnostic the file is for.
    """
    # The probe is `: >> "$RUNNER_LOG" 2>/dev/null` followed by fallback.
    assert ': >> "$RUNNER_LOG"' in entrypoint_code or ": >> $RUNNER_LOG" in entrypoint_code, \
        "expected writability probe `: >> \"$RUNNER_LOG\"` before exec"


def test_fallback_to_tmp_when_workspace_unwritable(entrypoint_code: str) -> None:
    """When /workspace is unwritable, RUNNER_LOG must reassign to /tmp."""
    assert "/tmp/runner.log" in entrypoint_code, \
        "fallback path /tmp/runner.log not present"


# ---------------------------------------------------------------------------
# Negative / Regression — forbidden patterns
# ---------------------------------------------------------------------------


def test_no_process_substitution_redirect(entrypoint_code: str) -> None:
    """Process substitution (``> >(...)``) must never be introduced —
    it would re-break the signal chain that direct redirect preserves."""
    # Regex catches `> >(...)` even with whitespace variations.
    assert not re.search(r">\s*>\s*\(", entrypoint_code), \
        "process substitution detected — would break dumb-init signal chain"


def test_no_app_log_config_flag(entrypoint_code: str) -> None:
    """We deliberately do NOT use uvicorn --log-config:
    pre-import errors would never reach Python's logging.config and
    leak past it. Adding it back would be a silent regression."""
    assert "--log-config" not in entrypoint_code, \
        "--log-config would mask pre-import errors that runner.log captures"


def test_exec_is_last_command(entrypoint_text: str) -> None:
    """The ``exec ... uvicorn ...`` line must be the final command in
    the default-path branch — anything after it is unreachable and
    a sign of a structural mistake."""
    # Trim to default-path block (after the `if [[ $# -gt 0 ]]; then exec "$@"; fi` early-exit).
    # The actual final exec is the `exec stdbuf -oL -eL "$PY_BIN" -m uvicorn` line.
    last_exec = entrypoint_text.rfind("exec stdbuf")
    assert last_exec > 0, "the uvicorn exec line must exist"
    after = entrypoint_text[last_exec:]
    # Allow trailing whitespace / closing newline only.
    # No additional commands (lines starting with non-comment, non-whitespace).
    for line in after.splitlines()[1:]:  # skip the exec line itself
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        # Continuation lines (start with options or backslashes) are OK.
        if stripped.startswith("-") or stripped.endswith("\\") or stripped.startswith(">>"):
            continue
        if stripped.startswith('--') or "2>&1" in stripped:
            continue
        # Anything else after the exec is suspicious.
        pytest.fail(f"unexpected line after final exec: {stripped!r}")
