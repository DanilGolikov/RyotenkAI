"""Unit tests: ``SingleNodeProvider._preempt_inference_container``.

Production hook fired by the orchestrator before training to evict any
running inference container that would otherwise hog GPU VRAM. The
method is intentionally small (~30 LoC) but has a half-dozen branches:

* No SSH client → early return (training before connect — defensive).
* SSH inspection returns "not running" → no stop, no warning.
* SSH inspection returns "running" → warning log + ``docker rm -f``
  + ``time.sleep(2)`` to let the daemon settle.
* SSH inspection raises → swallowed (best-effort).

Existing coverage in ``tests/unit/providers/single_node/test_training_health_check.py``
hits three of these (no_ssh / not_running / running_stops_it) but uses
inline ad-hoc stubs (``FakeSSH``, ``StopTracker``) instead of the
canonical ``FakeLegacySSHClient``. This file demonstrates the
post-finalization architecture:

* Canonical fake: ``tests/_fakes/legacy_ssh.py::FakeLegacySSHClient``
* Factory: ``tests/_factories/single_node_provider.py::make_single_node_provider``
* Test classes follow the
  positive/negative/boundary/invariants/dependency-errors/regressions/
  logic-specific structure mandated by the testing policy.

The new tests also fill the **branch gaps** the existing inline tests
miss — the exception-swallow path, the exact timeout/silent kwargs the
production passes, the ``ok=False`` inspection-result branch, and the
``time.sleep`` contract.

WHY this matters operationally: a regression where preemption silently
no-ops (e.g. a future refactor stops calling ``docker rm``) would leak
GPU memory and crash the training run with OOM. The mutation-testing
gate uses these branch assertions as the floor.
"""

from __future__ import annotations

import pytest

from tests._factories.single_node_provider import make_single_node_provider
from tests._fakes.legacy_ssh import FakeLegacySSHClient

# The exact docker container name the production code targets. Pinned
# here as a regression: if the production constant changes, this test
# fails loudly and forces a deliberate decision (rename here or
# preserve the contract).
_CONTAINER_NAME = "ryotenkai-inference-vllm"
_INSPECT_CMD_FRAGMENT = f"docker ps -q -f name={_CONTAINER_NAME} -f status=running"
_STOP_CMD_FRAGMENT = f"docker rm -f {_CONTAINER_NAME}"

# Production timeouts. Pinned so a regression that doubles them (and
# slows the orchestrator startup) trips the suite.
_INSPECT_TIMEOUT = 10
_STOP_TIMEOUT = 60


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _attach_ssh(provider, ssh: FakeLegacySSHClient | None) -> None:
    """Attach the SSH client manually — preempt runs BEFORE ``connect()``
    on the production lifecycle but we still want to exercise its body
    without a full ``connect()`` round-trip.
    """
    provider._ssh_client = ssh


def _disable_sleep(monkeypatch: pytest.MonkeyPatch) -> list[float]:
    """Replace ``time.sleep`` with a recorder so tests are deterministic.

    Returns a list that the test can inspect — each entry is the
    ``seconds`` arg passed to ``time.sleep``. The production code uses
    a 2-second cool-off; assertions can pin that contract.
    """
    captured: list[float] = []

    def _fake_sleep(seconds: float) -> None:
        captured.append(seconds)

    monkeypatch.setattr(
        "ryotenkai_providers.single_node.training.provider.time.sleep",
        _fake_sleep,
    )
    return captured


# ===========================================================================
# 1. Positive
# ===========================================================================


class TestPositive:
    """Container running → preemption issues docker rm + sleeps."""

    def test_running_container_is_terminated(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        ssh = FakeLegacySSHClient(host="pc")
        # Inspect returns container ID → "running"
        ssh.set_command_response(_INSPECT_CMD_FRAGMENT, success=True, stdout="abc123\n")
        provider = make_single_node_provider()
        _attach_ssh(provider, ssh)
        sleeps = _disable_sleep(monkeypatch)

        provider._preempt_inference_container()

        # exactly one stop command was issued
        stop_calls = [c for c in ssh.commands if _STOP_CMD_FRAGMENT in c]
        assert len(stop_calls) == 1
        # production sleeps exactly once at 2 seconds
        assert sleeps == [2]

    def test_running_then_idempotent_stop_command_is_safe(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """The stop command uses ``>/dev/null 2>&1 || true`` so a missing
        container does NOT raise. Pin the idempotency token."""
        ssh = FakeLegacySSHClient(host="pc")
        ssh.set_command_response(_INSPECT_CMD_FRAGMENT, success=True, stdout="abc123")
        provider = make_single_node_provider()
        _attach_ssh(provider, ssh)
        _disable_sleep(monkeypatch)

        provider._preempt_inference_container()

        stop_cmd = next(c for c in ssh.commands if _STOP_CMD_FRAGMENT in c)
        assert "|| true" in stop_cmd, (
            "stop command must be idempotent — missing `|| true` would let "
            "a missing container fail the orchestrator startup"
        )


# ===========================================================================
# 2. Negative
# ===========================================================================


class TestNegative:
    """No-op paths: ssh missing or container not running."""

    def test_no_ssh_client_returns_silently(self) -> None:
        provider = make_single_node_provider()
        _attach_ssh(provider, None)
        # Must not raise — defensive guard for "called before connect()".
        provider._preempt_inference_container()

    def test_inspect_returns_empty_stdout_means_not_running(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        ssh = FakeLegacySSHClient(host="pc")
        ssh.set_command_response(_INSPECT_CMD_FRAGMENT, success=True, stdout="")
        provider = make_single_node_provider()
        _attach_ssh(provider, ssh)
        sleeps = _disable_sleep(monkeypatch)

        provider._preempt_inference_container()

        # No stop command should have been issued
        assert all(_STOP_CMD_FRAGMENT not in c for c in ssh.commands)
        # No sleep — short-circuit before the cool-off
        assert sleeps == []

    def test_inspect_command_fails_treated_as_not_running(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Production guard ``is_running = bool(ok and stdout.strip())`` —
        when ``ok=False`` (SSH non-zero exit), preemption MUST skip the
        stop command. Otherwise a transient ``docker`` daemon hiccup
        triggers an unnecessary rm + sleep on every training start.
        """
        ssh = FakeLegacySSHClient(host="pc")
        ssh.set_command_response(_INSPECT_CMD_FRAGMENT, success=False, stdout="ignored")
        provider = make_single_node_provider()
        _attach_ssh(provider, ssh)
        sleeps = _disable_sleep(monkeypatch)

        provider._preempt_inference_container()

        assert all(_STOP_CMD_FRAGMENT not in c for c in ssh.commands)
        assert sleeps == []


# ===========================================================================
# 3. Boundary
# ===========================================================================


class TestBoundary:
    """Edge cases on the stdout-vs-running discriminator."""

    @pytest.mark.parametrize(
        "stdout_value",
        ["", " ", "\n", "\t  \n", "   \r\n  "],
        ids=["empty", "space", "lf", "mixed_ws", "crlf"],
    )
    def test_whitespace_only_stdout_treated_as_not_running(
        self,
        stdout_value: str,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Production uses ``stdout.strip()`` — every whitespace-only
        response must collapse to "not running" so we never invoke
        ``docker rm`` against a phantom container ID.
        """
        ssh = FakeLegacySSHClient(host="pc")
        ssh.set_command_response(_INSPECT_CMD_FRAGMENT, success=True, stdout=stdout_value)
        provider = make_single_node_provider()
        _attach_ssh(provider, ssh)
        _disable_sleep(monkeypatch)

        provider._preempt_inference_container()

        assert all(_STOP_CMD_FRAGMENT not in c for c in ssh.commands)

    @pytest.mark.parametrize(
        "stdout_value",
        ["a", "abc", "a\n", " a ", "0123456789ab\n"],
        ids=["single_char", "container_short", "trailing_lf", "padded", "full_id"],
    )
    def test_any_non_blank_stdout_triggers_stop(
        self,
        stdout_value: str,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        ssh = FakeLegacySSHClient(host="pc")
        ssh.set_command_response(_INSPECT_CMD_FRAGMENT, success=True, stdout=stdout_value)
        provider = make_single_node_provider()
        _attach_ssh(provider, ssh)
        _disable_sleep(monkeypatch)

        provider._preempt_inference_container()

        assert any(_STOP_CMD_FRAGMENT in c for c in ssh.commands)


# ===========================================================================
# 4. Invariants
# ===========================================================================


class TestInvariants:
    """Contract pins that hold across all running-container scenarios."""

    def test_inspect_uses_silent_true_to_avoid_log_spam(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """The inspection runs on EVERY training start. It MUST pass
        ``silent=True`` to the legacy SSHClient or production logs would
        emit one INFO line per training launch (≈ 1k lines/day on a
        busy cluster)."""
        ssh = FakeLegacySSHClient(host="pc")
        ssh.set_command_response(_INSPECT_CMD_FRAGMENT, success=True, stdout="")
        provider = make_single_node_provider()
        _attach_ssh(provider, ssh)
        _disable_sleep(monkeypatch)

        provider._preempt_inference_container()

        inspect_calls = [c for c in ssh.commands_log if _INSPECT_CMD_FRAGMENT in c.command]
        assert len(inspect_calls) == 1
        assert inspect_calls[0].silent is True

    def test_stop_uses_silent_false_for_audit_trail(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Conversely, the ``docker rm`` IS logged — it's a state change
        worth recording for postmortem."""
        ssh = FakeLegacySSHClient(host="pc")
        ssh.set_command_response(_INSPECT_CMD_FRAGMENT, success=True, stdout="abc")
        provider = make_single_node_provider()
        _attach_ssh(provider, ssh)
        _disable_sleep(monkeypatch)

        provider._preempt_inference_container()

        stop_calls = [c for c in ssh.commands_log if _STOP_CMD_FRAGMENT in c.command]
        assert len(stop_calls) == 1
        assert stop_calls[0].silent is False

    def test_inspect_timeout_is_ten_seconds(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Bounded SSH round-trip — 10s covers normal docker daemon
        round-trip without holding the training launch hostage if the
        host is unreachable."""
        ssh = FakeLegacySSHClient(host="pc")
        ssh.set_command_response(_INSPECT_CMD_FRAGMENT, success=True, stdout="")
        provider = make_single_node_provider()
        _attach_ssh(provider, ssh)
        _disable_sleep(monkeypatch)

        provider._preempt_inference_container()

        call = next(c for c in ssh.commands_log if _INSPECT_CMD_FRAGMENT in c.command)
        assert call.timeout == _INSPECT_TIMEOUT

    def test_stop_timeout_is_sixty_seconds(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """``docker rm -f`` may take a few seconds while the daemon
        sends SIGKILL; 60s is the documented ceiling. Pinned so a
        regression that drops it to 5s (causing flakes on slow
        machines) trips the suite."""
        ssh = FakeLegacySSHClient(host="pc")
        ssh.set_command_response(_INSPECT_CMD_FRAGMENT, success=True, stdout="abc")
        provider = make_single_node_provider()
        _attach_ssh(provider, ssh)
        _disable_sleep(monkeypatch)

        provider._preempt_inference_container()

        call = next(c for c in ssh.commands_log if _STOP_CMD_FRAGMENT in c.command)
        assert call.timeout == _STOP_TIMEOUT


# ===========================================================================
# 5. Dependency errors
# ===========================================================================


class TestDependencyErrors:
    """SSH transport raises during inspection: production must swallow."""

    def test_ssh_exception_during_inspect_swallowed(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Production wraps the entire inspect+stop sequence in
        ``try/except Exception`` because preemption is best-effort and
        must NEVER fail the orchestrator startup. A raised exception
        is logged at DEBUG and discarded.
        """
        ssh = FakeLegacySSHClient(host="pc")
        ssh.inject_exception(RuntimeError("SSH transport down"))
        provider = make_single_node_provider()
        _attach_ssh(provider, ssh)
        sleeps = _disable_sleep(monkeypatch)

        # Must NOT raise.
        provider._preempt_inference_container()

        # Sleep never fires when inspection raises.
        assert sleeps == []

    def test_ssh_exception_during_stop_swallowed(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Inspection succeeds → tries to stop → stop raises → still
        swallowed."""
        ssh = FakeLegacySSHClient(host="pc")
        ssh.set_command_response(_INSPECT_CMD_FRAGMENT, success=True, stdout="abc")
        # Inject ONE exception that lands on the second call (the rm).
        # The fake's counter sees the inspect first, then the stop —
        # so we set count=2 with the first call going through a canned
        # response. Workaround: use a side-effecting default.
        original_exec = ssh.exec_command
        n_calls = {"count": 0}

        def _exec_with_failure(command, background=False, timeout=30, silent=False):
            n_calls["count"] += 1
            if n_calls["count"] == 2:
                raise OSError("broken pipe")
            return original_exec(command, background=background, timeout=timeout, silent=silent)

        monkeypatch.setattr(ssh, "exec_command", _exec_with_failure)
        provider = make_single_node_provider()
        _attach_ssh(provider, ssh)
        _disable_sleep(monkeypatch)

        # Must NOT raise.
        provider._preempt_inference_container()


# ===========================================================================
# 6. Regressions
# ===========================================================================


class TestRegressions:
    """Specific bugs / contract drifts the suite must catch."""

    def test_inspect_command_shape_pinned(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """The orchestrator's pre-training preemption uses a SPECIFIC
        shape ``docker ps -q -f name=X -f status=running``. Two filters
        (name + status) are required — dropping the status filter would
        also match stopped containers and trigger pointless ``docker rm``
        on a container the user manually stopped earlier."""
        ssh = FakeLegacySSHClient(host="pc")
        ssh.set_command_response(r"docker ps -q", success=True, stdout="")
        provider = make_single_node_provider()
        _attach_ssh(provider, ssh)
        _disable_sleep(monkeypatch)

        provider._preempt_inference_container()

        inspect = ssh.commands[0]
        assert "docker ps -q" in inspect
        assert f"-f name={_CONTAINER_NAME}" in inspect
        assert "-f status=running" in inspect

    def test_container_name_constant_unchanged(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """The ``ryotenkai-inference-vllm`` container name is the contract
        between training preemption here and inference deployment in
        :mod:`single_node.inference.provider`. If one side renames
        without the other, preemption silently no-ops AND we leak GPU
        VRAM. This test pins the constant so a rename PR has to update
        both sides simultaneously."""
        ssh = FakeLegacySSHClient(host="pc")
        ssh.set_command_response(r"docker ps", success=True, stdout="")
        provider = make_single_node_provider()
        _attach_ssh(provider, ssh)
        _disable_sleep(monkeypatch)

        provider._preempt_inference_container()

        assert any(_CONTAINER_NAME in c for c in ssh.commands)


# ===========================================================================
# 7. Logic-specific
# ===========================================================================


class TestLogicSpecific:
    """Tests that exercise the boolean composition explicitly."""

    @pytest.mark.parametrize(
        ("ok", "stdout", "should_stop"),
        [
            (True, "abc", True),    # ok && non-empty       → running, stop
            (True, "", False),      # ok && empty           → not running
            (True, "  ", False),    # ok && whitespace      → not running (strip)
            (False, "abc", False),  # !ok && non-empty      → not running
            (False, "", False),     # !ok && empty          → not running
        ],
        ids=["running", "ok_empty", "ok_ws", "not_ok_with_out", "not_ok_empty"],
    )
    def test_is_running_truth_table(
        self,
        ok: bool,
        stdout: str,
        should_stop: bool,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Exhaustive truth table for ``is_running = bool(ok and stdout.strip())``."""
        ssh = FakeLegacySSHClient(host="pc")
        ssh.set_command_response(_INSPECT_CMD_FRAGMENT, success=ok, stdout=stdout)
        provider = make_single_node_provider()
        _attach_ssh(provider, ssh)
        _disable_sleep(monkeypatch)

        provider._preempt_inference_container()

        was_stopped = any(_STOP_CMD_FRAGMENT in c for c in ssh.commands)
        assert was_stopped is should_stop, (
            f"ok={ok!r} stdout={stdout!r} should_stop={should_stop} "
            f"actual_stop={was_stopped}"
        )
