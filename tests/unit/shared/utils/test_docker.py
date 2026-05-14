"""Unit tests for ``ryotenkai_shared.utils.docker`` — branch coverage for
key error paths.

Phase A2 Batch 4 (2026-05-14): the docker surface no longer returns
``Result[T, ProviderError]`` — it returns ``T`` directly and raises
:class:`ConfigInvalidError` for bad input or
:class:`ProviderUnavailableError` for transient docker / daemon /
inspect failures. Tests use ``pytest.raises``.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

import ryotenkai_shared.utils.docker as docker_mod
from ryotenkai_shared.errors import ConfigInvalidError, ProviderUnavailableError
from ryotenkai_shared.utils.docker import (
    _is_latest_tag,
    _validate_container_name,
    docker_container_exit_code,
    docker_image_exists,
    docker_is_container_running,
    docker_logs,
    docker_rm_force,
    ensure_docker_image,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _ssh(ok: bool = True, stdout: str = "", stderr: str = "") -> MagicMock:
    """Return a minimal _ExecClient mock whose exec_command returns given values."""
    mock = MagicMock()
    mock.exec_command.return_value = (ok, stdout, stderr)
    return mock


# ---------------------------------------------------------------------------
# _validate_container_name — now raises instead of returning Result
# ---------------------------------------------------------------------------


def test_validate_container_name_empty_string_raises() -> None:
    with pytest.raises(ConfigInvalidError) as exc_info:
        _validate_container_name("")
    assert exc_info.value.context.get("reason") == "DOCKER_INVALID_CONTAINER_NAME"


def test_validate_container_name_only_spaces_raises() -> None:
    # The regex rejects spaces; should raise as "unsafe".
    with pytest.raises(ConfigInvalidError) as exc_info:
        _validate_container_name("   ")
    assert exc_info.value.context.get("reason") == "DOCKER_INVALID_CONTAINER_NAME"


def test_validate_container_name_unsafe_chars_raises() -> None:
    with pytest.raises(ConfigInvalidError) as exc_info:
        _validate_container_name("bad;name")
    assert "Unsafe" in (exc_info.value.detail or "")


def test_validate_container_name_valid_returns_none() -> None:
    assert _validate_container_name("my-container_01") is None


def test_validate_container_name_starts_with_digit_returns_none() -> None:
    # Regex requires first char to be alphanumeric — digit is fine.
    assert _validate_container_name("1container") is None


# ---------------------------------------------------------------------------
# _is_latest_tag (unchanged semantics)
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
# docker_image_exists (unchanged contract)
# ---------------------------------------------------------------------------


def test_docker_image_exists_returns_true_when_ok() -> None:
    ssh = _ssh(ok=True)
    assert docker_image_exists(ssh, "myimage:latest") is True


def test_docker_image_exists_returns_false_when_fail() -> None:
    ssh = _ssh(ok=False)
    assert docker_image_exists(ssh, "myimage:latest") is False


# ---------------------------------------------------------------------------
# docker_rm_force — now raises
# ---------------------------------------------------------------------------


def test_docker_rm_force_invalid_container_name_raises() -> None:
    ssh = _ssh()
    with pytest.raises(ConfigInvalidError) as exc_info:
        docker_rm_force(ssh, container_name="bad name")
    assert exc_info.value.context.get("reason") == "DOCKER_INVALID_CONTAINER_NAME"
    ssh.exec_command.assert_not_called()


def test_docker_rm_force_command_fails_raises_provider_unavailable() -> None:
    ssh = _ssh(ok=False, stderr="no such container")
    with pytest.raises(ProviderUnavailableError) as exc_info:
        docker_rm_force(ssh, container_name="mycontainer")
    assert exc_info.value.context.get("reason") == "DOCKER_RM_FAILED"
    assert exc_info.value.context.get("container_name") == "mycontainer"


def test_docker_rm_force_success_returns_none() -> None:
    ssh = _ssh(ok=True)
    assert docker_rm_force(ssh, container_name="mycontainer") is None


def test_docker_rm_force_empty_stderr_uses_default_message() -> None:
    """Empty stderr → fallback message 'docker rm failed'."""
    ssh = _ssh(ok=False, stderr="")
    with pytest.raises(ProviderUnavailableError) as exc_info:
        docker_rm_force(ssh, container_name="mycontainer")
    assert exc_info.value.detail == "docker rm failed"


# ---------------------------------------------------------------------------
# docker_is_container_running (unchanged contract — still bool)
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
# docker_logs — now raises
# ---------------------------------------------------------------------------


def test_docker_logs_invalid_container_name_raises() -> None:
    ssh = _ssh()
    with pytest.raises(ConfigInvalidError) as exc_info:
        docker_logs(ssh, container_name="bad name")
    assert exc_info.value.context.get("reason") == "DOCKER_INVALID_CONTAINER_NAME"
    ssh.exec_command.assert_not_called()


def test_docker_logs_command_fails_raises() -> None:
    ssh = _ssh(ok=False, stdout="", stderr="container not found")
    with pytest.raises(ProviderUnavailableError) as exc_info:
        docker_logs(ssh, container_name="mycontainer")
    assert exc_info.value.context.get("reason") == "DOCKER_LOGS_FAILED"
    assert exc_info.value.context.get("container_name") == "mycontainer"


def test_docker_logs_empty_stderr_uses_default_message() -> None:
    """Empty stderr + empty stdout → fallback 'docker logs failed'."""
    ssh = _ssh(ok=False, stdout="", stderr="")
    with pytest.raises(ProviderUnavailableError) as exc_info:
        docker_logs(ssh, container_name="mycontainer")
    assert exc_info.value.detail == "docker logs failed"


def test_docker_logs_success_returns_stdout() -> None:
    ssh = _ssh(ok=True, stdout="log line 1\nlog line 2\n")
    out = docker_logs(ssh, container_name="mycontainer")
    assert out == "log line 1\nlog line 2\n"


def test_docker_logs_with_tail_parameter() -> None:
    """tail parameter must be passed as --tail N in the command."""
    calls: list[str] = []
    mock_ssh = MagicMock()

    def fake_exec(cmd, **kwargs):
        calls.append(cmd)
        return (True, "line\n", "")

    mock_ssh.exec_command.side_effect = fake_exec

    out = docker_logs(mock_ssh, container_name="mycontainer", tail=50)
    assert out == "line\n"
    assert calls and "--tail 50" in calls[0]


# ---------------------------------------------------------------------------
# docker_container_exit_code — now raises / returns int
# ---------------------------------------------------------------------------


def test_docker_container_exit_code_invalid_name_raises() -> None:
    ssh = _ssh()
    with pytest.raises(ConfigInvalidError) as exc_info:
        docker_container_exit_code(ssh, container_name="bad name")
    assert exc_info.value.context.get("reason") == "DOCKER_INVALID_CONTAINER_NAME"
    ssh.exec_command.assert_not_called()


def test_docker_container_exit_code_inspect_failed_raises() -> None:
    ssh = _ssh(ok=False, stderr="container not found")
    with pytest.raises(ProviderUnavailableError) as exc_info:
        docker_container_exit_code(ssh, container_name="mycontainer")
    assert exc_info.value.context.get("reason") == "DOCKER_INSPECT_FAILED"
    assert exc_info.value.context.get("container_name") == "mycontainer"


def test_docker_container_exit_code_invalid_output_raises() -> None:
    """Non-integer stdout → ProviderUnavailableError(DOCKER_INSPECT_INVALID_OUTPUT)."""
    ssh = _ssh(ok=True, stdout="not-a-number\n")
    with pytest.raises(ProviderUnavailableError) as exc_info:
        docker_container_exit_code(ssh, container_name="mycontainer")
    assert exc_info.value.context.get("reason") == "DOCKER_INSPECT_INVALID_OUTPUT"
    assert "not-a-number" in (exc_info.value.detail or "")


def test_docker_container_exit_code_success_zero() -> None:
    ssh = _ssh(ok=True, stdout="0\n")
    assert docker_container_exit_code(ssh, container_name="mycontainer") == 0


def test_docker_container_exit_code_success_nonzero() -> None:
    ssh = _ssh(ok=True, stdout="137\n")
    assert docker_container_exit_code(ssh, container_name="mycontainer") == 137


# ---------------------------------------------------------------------------
# ensure_docker_image — now raises
# ---------------------------------------------------------------------------


def test_ensure_docker_image_already_present_non_latest_skips_pull() -> None:
    """If image exists and tag != latest, skip pull and return None."""
    ssh = _ssh(ok=True)
    assert ensure_docker_image(ssh=ssh, image="myimage:1.0") is None
    # exec_command should only be called once (for docker_image_exists inspect)
    assert ssh.exec_command.call_count == 1


def test_ensure_docker_image_pull_failure_raises() -> None:
    """Pull returning ok=False → ProviderUnavailableError(DOCKER_PULL_FAILED)."""
    mock_ssh = MagicMock()

    def fake_exec(cmd, **kwargs):
        if "image inspect" in cmd:
            return (False, "", "")  # image not present
        if "docker pull" in cmd:
            return (False, "", "pull failed: network error")
        return (True, "", "")

    mock_ssh.exec_command.side_effect = fake_exec

    with pytest.raises(ProviderUnavailableError) as exc_info:
        ensure_docker_image(ssh=mock_ssh, image="myimage:2.0")
    assert exc_info.value.context.get("reason") == "DOCKER_PULL_FAILED"
    assert "myimage:2.0" in (exc_info.value.detail or "")


def test_ensure_docker_image_no_verify_after_pull_returns_none(monkeypatch: pytest.MonkeyPatch) -> None:
    """verify_after_pull=False → returns None immediately after pull."""
    mock_ssh = MagicMock()

    def fake_exec(cmd, **kwargs):
        if "image inspect" in cmd:
            return (False, "", "")  # not present → trigger pull
        if "docker pull" in cmd:
            return (True, "Pulling...", "")
        return (True, "", "")

    mock_ssh.exec_command.side_effect = fake_exec

    assert ensure_docker_image(ssh=mock_ssh, image="myimage:2.0", verify_after_pull=False) is None
    pull_calls = [c for c in mock_ssh.exec_command.call_args_list if "docker pull" in str(c)]
    assert len(pull_calls) == 1


def test_ensure_docker_image_latest_always_pulls(monkeypatch: pytest.MonkeyPatch) -> None:
    """Image with :latest tag should always trigger pull even if present locally."""
    monkeypatch.setattr(docker_mod.time, "sleep", lambda _s: None)

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

    assert (
        ensure_docker_image(ssh=mock_ssh, image="myimage:latest", verify_after_pull=False)
        is None
    )
    assert pull_called[0] is True


def test_ensure_docker_image_post_pull_verify_fails_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    """If post-pull inspect retries all fail, raise DOCKER_IMAGE_NOT_AVAILABLE."""
    monkeypatch.setattr(docker_mod.time, "sleep", lambda _s: None)

    mock_ssh = MagicMock()

    def fake_exec(cmd, **kwargs):
        if "docker pull" in cmd:
            return (True, "pulled", "")
        # All inspects fail (both pre-pull and post-pull)
        return (False, "", "")

    mock_ssh.exec_command.side_effect = fake_exec

    with pytest.raises(ProviderUnavailableError) as exc_info:
        ensure_docker_image(ssh=mock_ssh, image="myimage:2.0")
    assert exc_info.value.context.get("reason") == "DOCKER_IMAGE_NOT_AVAILABLE"
