"""Unit tests for src/utils/docker.py — branch coverage for key error paths."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

import src.utils.docker as docker_mod
from src.utils.docker import (
    _is_latest_tag,
    _validate_container_name,
    docker_container_exit_code,
    docker_image_exists,
    docker_is_container_running,
    docker_logs,
    docker_rm_force,
    ensure_docker_image,
)
from src.utils.result import ProviderError


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _ssh(ok: bool = True, stdout: str = "", stderr: str = "") -> MagicMock:
    """Return a minimal _ExecClient mock whose exec_command returns given values."""
    mock = MagicMock()
    mock.exec_command.return_value = (ok, stdout, stderr)
    return mock


# ---------------------------------------------------------------------------
# _validate_container_name (lines 32-40)
# ---------------------------------------------------------------------------


def test_validate_container_name_empty_string() -> None:
    res = _validate_container_name("")
    assert res.is_failure()
    assert res.unwrap_err().code == "DOCKER_INVALID_CONTAINER_NAME"


def test_validate_container_name_none_like_value() -> None:
    # The guard checks `not name` which catches falsy non-str too, but the
    # type annotation is str; pass an empty string to cover the branch.
    res = _validate_container_name("   ")  # spaces → regex mismatch
    assert res.is_failure()
    assert res.unwrap_err().code == "DOCKER_INVALID_CONTAINER_NAME"


def test_validate_container_name_unsafe_chars() -> None:
    res = _validate_container_name("bad;name")
    assert res.is_failure()
    assert "Unsafe" in res.unwrap_err().message


def test_validate_container_name_valid() -> None:
    res = _validate_container_name("my-container_01")
    assert res.is_success()


def test_validate_container_name_starts_with_digit() -> None:
    # Regex requires first char to be alphanumeric — digit is fine.
    res = _validate_container_name("1container")
    assert res.is_success()


# ---------------------------------------------------------------------------
# _is_latest_tag (lines 67-81)
# ---------------------------------------------------------------------------


def test_is_latest_tag_empty_string() -> None:
    assert _is_latest_tag("") is False


def test_is_latest_tag_none_type() -> None:
    assert _is_latest_tag(None) is False  # type: ignore[arg-type]


def test_is_latest_tag_with_digest() -> None:
    assert _is_latest_tag("ubuntu@sha256:abc123") is False


def test_is_latest_tag_implicit_latest_no_tag() -> None:
    assert _is_latest_tag("ubuntu") is True


def test_is_latest_tag_explicit_latest() -> None:
    assert _is_latest_tag("ubuntu:latest") is True


def test_is_latest_tag_specific_version() -> None:
    assert _is_latest_tag("ubuntu:22.04") is False


def test_is_latest_tag_registry_with_port_no_tag() -> None:
    # registry.example.com:5000/image — colon is port separator, not tag
    assert _is_latest_tag("registry.example.com:5000/image") is True


def test_is_latest_tag_registry_with_port_and_version() -> None:
    assert _is_latest_tag("registry.example.com:5000/image:1.2") is False


# ---------------------------------------------------------------------------
# docker_image_exists
# ---------------------------------------------------------------------------


def test_docker_image_exists_returns_true_when_ok() -> None:
    ssh = _ssh(ok=True)
    assert docker_image_exists(ssh, "myimage:latest") is True


def test_docker_image_exists_returns_false_when_fail() -> None:
    ssh = _ssh(ok=False)
    assert docker_image_exists(ssh, "myimage:latest") is False


# ---------------------------------------------------------------------------
# docker_rm_force (lines 154-170)
# ---------------------------------------------------------------------------


def test_docker_rm_force_invalid_container_name() -> None:
    ssh = _ssh()
    res = docker_rm_force(ssh, container_name="bad name")
    assert res.is_failure()
    assert res.unwrap_err().code == "DOCKER_INVALID_CONTAINER_NAME"
    ssh.exec_command.assert_not_called()


def test_docker_rm_force_command_fails_returns_err(monkeypatch: pytest.MonkeyPatch) -> None:
    """When exec_command returns ok=False, must return Err DOCKER_RM_FAILED (lines 166-167)."""
    ssh = _ssh(ok=False, stderr="no such container")
    res = docker_rm_force(ssh, container_name="mycontainer")
    assert res.is_failure()
    err = res.unwrap_err()
    assert err.code == "DOCKER_RM_FAILED"
    assert "mycontainer" in str(err.details)


def test_docker_rm_force_success() -> None:
    ssh = _ssh(ok=True)
    res = docker_rm_force(ssh, container_name="mycontainer")
    assert res.is_success()


def test_docker_rm_force_empty_stderr_uses_default_message() -> None:
    """Empty stderr → fallback message 'docker rm failed'."""
    ssh = _ssh(ok=False, stderr="")
    res = docker_rm_force(ssh, container_name="mycontainer")
    assert res.is_failure()
    assert res.unwrap_err().message == "docker rm failed"


# ---------------------------------------------------------------------------
# docker_is_container_running (line 187)
# ---------------------------------------------------------------------------


def test_docker_is_container_running_invalid_filter_with_spaces() -> None:
    ssh = _ssh()
    result = docker_is_container_running(ssh, name_filter="bad filter")
    assert result is False
    ssh.exec_command.assert_not_called()


def test_docker_is_container_running_empty_filter() -> None:
    ssh = _ssh()
    result = docker_is_container_running(ssh, name_filter="")
    assert result is False
    ssh.exec_command.assert_not_called()


def test_docker_is_container_running_returns_true_when_output() -> None:
    ssh = _ssh(ok=True, stdout="abc123\n")
    assert docker_is_container_running(ssh, name_filter="mycontainer") is True


def test_docker_is_container_running_returns_false_when_empty_stdout() -> None:
    ssh = _ssh(ok=True, stdout="")
    assert docker_is_container_running(ssh, name_filter="mycontainer") is False


# ---------------------------------------------------------------------------
# docker_logs (lines 204-220)
# ---------------------------------------------------------------------------


def test_docker_logs_invalid_container_name() -> None:
    ssh = _ssh()
    res = docker_logs(ssh, container_name="bad name")
    assert res.is_failure()
    assert res.unwrap_err().code == "DOCKER_INVALID_CONTAINER_NAME"
    ssh.exec_command.assert_not_called()


def test_docker_logs_command_fails_returns_err() -> None:
    """When exec_command returns ok=False, returns Err DOCKER_LOGS_FAILED (lines 216-217)."""
    ssh = _ssh(ok=False, stdout="", stderr="container not found")
    res = docker_logs(ssh, container_name="mycontainer")
    assert res.is_failure()
    err = res.unwrap_err()
    assert err.code == "DOCKER_LOGS_FAILED"
    assert "mycontainer" in str(err.details)


def test_docker_logs_empty_stderr_uses_default_message() -> None:
    """Empty stderr + empty stdout → fallback 'docker logs failed'."""
    ssh = _ssh(ok=False, stdout="", stderr="")
    res = docker_logs(ssh, container_name="mycontainer")
    assert res.is_failure()
    assert res.unwrap_err().message == "docker logs failed"


def test_docker_logs_success_returns_stdout() -> None:
    ssh = _ssh(ok=True, stdout="log line 1\nlog line 2\n")
    res = docker_logs(ssh, container_name="mycontainer")
    assert res.is_success()
    assert res.unwrap() == "log line 1\nlog line 2\n"


def test_docker_logs_with_tail_parameter() -> None:
    """tail parameter must be passed as --tail N in the command."""
    calls: list[str] = []
    mock_ssh = MagicMock()

    def fake_exec(cmd, **kwargs):
        calls.append(cmd)
        return (True, "line\n", "")

    mock_ssh.exec_command.side_effect = fake_exec

    res = docker_logs(mock_ssh, container_name="mycontainer", tail=50)
    assert res.is_success()
    assert calls and "--tail 50" in calls[0]


# ---------------------------------------------------------------------------
# docker_container_exit_code (lines 223-248)
# ---------------------------------------------------------------------------


def test_docker_container_exit_code_invalid_name() -> None:
    ssh = _ssh()
    res = docker_container_exit_code(ssh, container_name="bad name")
    assert res.is_failure()
    assert res.unwrap_err().code == "DOCKER_INVALID_CONTAINER_NAME"
    ssh.exec_command.assert_not_called()


def test_docker_container_exit_code_inspect_failed() -> None:
    """When exec_command fails, returns Err DOCKER_INSPECT_FAILED (lines 235-236)."""
    ssh = _ssh(ok=False, stderr="container not found")
    res = docker_container_exit_code(ssh, container_name="mycontainer")
    assert res.is_failure()
    err = res.unwrap_err()
    assert err.code == "DOCKER_INSPECT_FAILED"
    assert "mycontainer" in str(err.details)


def test_docker_container_exit_code_invalid_output() -> None:
    """Non-integer stdout → Err DOCKER_INSPECT_INVALID_OUTPUT (lines 242-243)."""
    ssh = _ssh(ok=True, stdout="not-a-number\n")
    res = docker_container_exit_code(ssh, container_name="mycontainer")
    assert res.is_failure()
    err = res.unwrap_err()
    assert err.code == "DOCKER_INSPECT_INVALID_OUTPUT"
    assert "not-a-number" in err.message


def test_docker_container_exit_code_success_zero() -> None:
    ssh = _ssh(ok=True, stdout="0\n")
    res = docker_container_exit_code(ssh, container_name="mycontainer")
    assert res.is_success()
    assert res.unwrap() == 0


def test_docker_container_exit_code_success_nonzero() -> None:
    ssh = _ssh(ok=True, stdout="137\n")
    res = docker_container_exit_code(ssh, container_name="mycontainer")
    assert res.is_success()
    assert res.unwrap() == 137


# ---------------------------------------------------------------------------
# ensure_docker_image (lines 84-151)
# ---------------------------------------------------------------------------


def test_ensure_docker_image_already_present_non_latest_skips_pull() -> None:
    """If image exists and tag != latest, skip pull and return Ok."""
    ssh = _ssh(ok=True)
    res = ensure_docker_image(ssh=ssh, image="myimage:1.0")
    assert res.is_success()
    # exec_command should only be called once (for docker_image_exists inspect)
    assert ssh.exec_command.call_count == 1


def test_ensure_docker_image_pull_failure_returns_err() -> None:
    """Pull returning ok=False → Err DOCKER_PULL_FAILED (lines 118-120)."""
    call_count = [0]

    mock_ssh = MagicMock()

    def fake_exec(cmd, **kwargs):
        call_count[0] += 1
        if "image inspect" in cmd:
            return (False, "", "")  # image not present
        if "docker pull" in cmd:
            return (False, "", "pull failed: network error")
        return (True, "", "")

    mock_ssh.exec_command.side_effect = fake_exec

    res = ensure_docker_image(ssh=mock_ssh, image="myimage:2.0")
    assert res.is_failure()
    err = res.unwrap_err()
    assert err.code == "DOCKER_PULL_FAILED"
    assert "myimage:2.0" in err.message


def test_ensure_docker_image_no_verify_after_pull_returns_ok(monkeypatch: pytest.MonkeyPatch) -> None:
    """verify_after_pull=False → Ok immediately after successful pull (lines 129-130)."""
    mock_ssh = MagicMock()

    def fake_exec(cmd, **kwargs):
        if "image inspect" in cmd:
            return (False, "", "")  # not present → trigger pull
        if "docker pull" in cmd:
            return (True, "Pulling...", "")
        return (True, "", "")

    mock_ssh.exec_command.side_effect = fake_exec

    res = ensure_docker_image(ssh=mock_ssh, image="myimage:2.0", verify_after_pull=False)
    assert res.is_success()
    # No sleep should be called since verify is skipped
    pull_calls = [c for c in mock_ssh.exec_command.call_args_list if "docker pull" in str(c)]
    assert len(pull_calls) == 1


def test_ensure_docker_image_latest_always_pulls(monkeypatch: pytest.MonkeyPatch) -> None:
    """Image with :latest tag should always trigger pull even if present locally."""
    monkeypatch.setattr(docker_mod.time, "sleep", lambda s: None)

    pull_called = [False]
    mock_ssh = MagicMock()

    def fake_exec(cmd, **kwargs):
        if "image inspect" in cmd:
            return (True, "exists", "")  # exists locally
        if "docker pull" in cmd:
            pull_called[0] = True
            return (True, "latest pulled", "")
        return (True, "", "")

    mock_ssh.exec_command.side_effect = fake_exec

    res = ensure_docker_image(ssh=mock_ssh, image="myimage:latest", verify_after_pull=False)
    assert res.is_success()
    assert pull_called[0] is True


def test_ensure_docker_image_post_pull_verify_fails_returns_err(monkeypatch: pytest.MonkeyPatch) -> None:
    """If post-pull inspect retries all fail, return Err DOCKER_IMAGE_NOT_AVAILABLE."""
    monkeypatch.setattr(docker_mod.time, "sleep", lambda s: None)

    mock_ssh = MagicMock()
    call_count = [0]

    def fake_exec(cmd, **kwargs):
        call_count[0] += 1
        if "docker pull" in cmd:
            return (True, "pulled", "")
        # All inspects fail (both pre-pull and post-pull)
        return (False, "", "")

    mock_ssh.exec_command.side_effect = fake_exec

    res = ensure_docker_image(ssh=mock_ssh, image="myimage:2.0")
    assert res.is_failure()
    assert res.unwrap_err().code == "DOCKER_IMAGE_NOT_AVAILABLE"
