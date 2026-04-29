"""
docker/training/entrypoint.sh — invariants for the inert-pod design.

Static structural tests on the bash script. We do NOT spawn a real
container here (that's a CI-time smoke test); instead we lock in the
properties that, if violated, would silently re-introduce a failure
mode we've already paid for in production:

* **Sleep-infinity default.** The pod must end with ``exec sleep
  infinity`` so the Mac drives uvicorn via SSH-exec
  (``runner_launcher.py``). If a future change re-adds an
  ``exec uvicorn`` here, RunPod's CMD-override will silently
  shadow it and we go back to "/healthz never answers, no
  diagnostic available" failure mode.
* **PUBLIC_KEY block.** Provider control planes inject the Mac's
  pubkey via the env var. Removing this block breaks SSH access
  for every provider.
* **Signal-chain integrity.** dumb-init must stay PID 1 and the
  exec sleep must be a direct child. No bash subshells, no tee,
  no process substitution.

If any of these tests fails, the entrypoint shell script has
drifted away from the documented architecture (see
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
    long header explaining why we DON'T use tee / uvicorn-here).
    """
    out_lines: list[str] = []
    for line in entrypoint_text.splitlines():
        stripped = line.lstrip()
        if stripped.startswith("#"):
            continue
        idx = line.find(" #")
        if idx >= 0:
            line = line[:idx]
        out_lines.append(line)
    return "\n".join(out_lines)


# ---------------------------------------------------------------------------
# Positive — required structural elements
# ---------------------------------------------------------------------------


def test_default_path_is_sleep_infinity(entrypoint_code: str) -> None:
    """The pod's default-path branch MUST end in ``exec sleep infinity``.

    Why: the Mac (runner_launcher) will SSH-exec uvicorn after files
    are uploaded. If the entrypoint launches uvicorn itself, RunPod's
    CMD-override historically shadowed it (the pod stayed alive via
    the override's own ``sleep infinity`` but uvicorn never ran). The
    inert pod removes that ambiguity.
    """
    assert "exec sleep infinity" in entrypoint_code, \
        "default path must end in `exec sleep infinity` so the Mac drives uvicorn"


def test_no_uvicorn_exec_in_entrypoint(entrypoint_code: str) -> None:
    """Regression guard: entrypoint must NOT exec uvicorn itself.

    The Mac launches it via SSH-exec from runner_launcher. Re-adding
    an in-entrypoint launch reintroduces the RunPod CMD-override
    silent-shadow bug.
    """
    assert "uvicorn" not in entrypoint_code, \
        "entrypoint must not launch uvicorn — the Mac orchestrates it via SSH"


def test_public_key_block_present(entrypoint_code: str) -> None:
    """Provider control planes pass the Mac's SSH pubkey via the
    PUBLIC_KEY env var. The entrypoint must append it to
    /root/.ssh/authorized_keys, otherwise SSH connections fail."""
    assert "PUBLIC_KEY" in entrypoint_code, \
        "PUBLIC_KEY env-var injection block missing"
    assert "authorized_keys" in entrypoint_code, \
        "authorized_keys append missing"


def test_sshd_started_in_background(entrypoint_code: str) -> None:
    """sshd is the only thing that makes the pod reachable. The
    entrypoint must start it before the sleep."""
    assert "/usr/sbin/sshd" in entrypoint_code, \
        "sshd start command missing"


def test_custom_command_path_preserved(entrypoint_code: str) -> None:
    """``docker run image bash`` debug path must still work.

    Production providers leave CMD empty so we fall through to
    sleep-infinity, but ``docker run image bash`` for ad-hoc shells
    must still exec the user's command.
    """
    # Look for the `if [[ $# -gt 0 ]]; then exec "$@"; fi` shape.
    assert re.search(r'\$#\s*-gt\s*0', entrypoint_code), \
        "custom command path (`if [[ $# -gt 0 ]]; then exec ...`) missing"
    assert 'exec "$@"' in entrypoint_code, \
        "custom command exec missing"


# ---------------------------------------------------------------------------
# Negative / Regression — forbidden patterns
# ---------------------------------------------------------------------------


def test_no_tee_command(entrypoint_code: str) -> None:
    """tee in the code path would break dumb-init's signal chain to
    the launched process. We don't redirect uvicorn's stdout from
    the entrypoint anymore — runner_launcher does that on the Mac
    side via the SSH-exec command. Keep this guard so a future
    "let me capture stdout to a file" change doesn't re-introduce it."""
    assert not re.search(r"\btee\b", entrypoint_code), \
        "tee command detected — would break signal chain"


def test_no_process_substitution(entrypoint_code: str) -> None:
    """Process substitution (``> >(...)``) similarly orphans signal
    forwarding. Block it."""
    assert not re.search(r">\s*>\s*\(", entrypoint_code), \
        "process substitution detected"


def test_no_log_config_flag(entrypoint_code: str) -> None:
    """We deliberately do NOT pass uvicorn ``--log-config``: it only
    fires AFTER Python's logging dictConfig is applied, which means
    pre-import errors leak. Direct SSH-exec redirect from
    runner_launcher captures everything from byte 0 instead."""
    assert "--log-config" not in entrypoint_code, \
        "--log-config detected — would mask pre-import errors"


# ---------------------------------------------------------------------------
# Invariants
# ---------------------------------------------------------------------------


def test_dumb_init_signal_chain_intact(entrypoint_code: str) -> None:
    """The sleep is exec'd directly so dumb-init (PID 1) sees it as
    its immediate child. SIGTERM propagates straight through.

    A bash subshell or unset/missing exec keyword would break this.
    """
    # Find the LAST line with `sleep infinity` — must be `exec sleep infinity`.
    last_sleep = None
    for line in entrypoint_code.splitlines():
        stripped = line.strip()
        if "sleep infinity" in stripped:
            last_sleep = stripped
    assert last_sleep is not None, "no sleep infinity found"
    assert last_sleep.startswith("exec "), \
        f"final sleep must use `exec` to keep dumb-init signal chain: got {last_sleep!r}"
