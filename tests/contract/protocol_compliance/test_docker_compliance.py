"""Compliance tests for :class:`IDockerClient`.

Parametrized over ``[fake, real]``. The ``real`` variant
(:class:`LocalDockerClient`) shells out to ``docker`` via SSH —
requires ``RYOTENKAI_LIVE=1`` plus a reachable Docker host; otherwise
it ``pytest.skip``s.

Coverage:

* Protocol :func:`isinstance` check.
* image_exists / ensure_image — default register behaviour, failure
  injection, missing-image flow.
* rm_force — idempotent.
* is_container_running — returns the registered state.
* logs — returns content from the container's log buffer, supports
  ``tail``.
* container_exit_code — returns the stored exit code.

Chaos helpers live on :class:`FakeDockerClient`; real-mode skips
chaos-specific tests but still exercises the Protocol shape.
"""

from __future__ import annotations

import os
from typing import Any

import pytest

from ryotenkai_shared.infrastructure.docker import IDockerClient, LocalDockerClient
from tests._fakes.docker import FakeDockerClient

pytestmark = [
    pytest.mark.contract,
    pytest.mark.compliance,
    pytest.mark.exercises_protocol("IDockerClient"),
    pytest.mark.uses_fake("FakeDockerClient"),
]


class _StubSSH:
    """Minimal SSH stub for fake-mode tests.

    The fake doesn't touch the SSH client (records nothing through it);
    this stub keeps the Protocol typing happy without bringing in a
    real exec surface.
    """

    def exec_command(
        self,
        command: str,  # noqa: ARG002
        background: bool = False,  # noqa: ARG002
        timeout: int = 30,  # noqa: ARG002
        silent: bool = False,  # noqa: ARG002
    ) -> tuple[bool, str, str]:
        return (True, "", "")


@pytest.fixture(params=["fake", pytest.param("real", marks=pytest.mark.live)])
def docker_client(request: pytest.FixtureRequest) -> IDockerClient:
    if request.param == "real":
        if os.environ.get("RYOTENKAI_LIVE") != "1":
            pytest.skip("real IDockerClient requires RYOTENKAI_LIVE=1 and a reachable docker host")
        return LocalDockerClient()
    return FakeDockerClient()


def _as_fake(client: IDockerClient) -> FakeDockerClient:
    if not isinstance(client, FakeDockerClient):
        pytest.skip(
            "test exercises FakeDockerClient-only chaos helpers; real-mode covers Protocol shape only",
        )
    return client


@pytest.fixture()
def ssh() -> Any:
    return _StubSSH()


class TestDockerCompliance:
    def test_isinstance_protocol(self, docker_client: IDockerClient) -> None:
        assert isinstance(docker_client, IDockerClient)

    # ------------------------------------------------------------------
    # image_exists
    # ------------------------------------------------------------------

    def test_image_exists_default_true(self, docker_client: IDockerClient, ssh: Any) -> None:
        fake = _as_fake(docker_client)
        # Default: every image is "present" until explicitly marked missing.
        assert fake.image_exists(ssh, "anything:latest") is True

    def test_image_exists_after_set_missing(self, docker_client: IDockerClient, ssh: Any) -> None:
        fake = _as_fake(docker_client)
        fake.set_image_missing("foo:1")
        assert fake.image_exists(ssh, "foo:1") is False

    # ------------------------------------------------------------------
    # ensure_image
    # ------------------------------------------------------------------

    def test_ensure_image_default_succeeds(self, docker_client: IDockerClient, ssh: Any) -> None:
        fake = _as_fake(docker_client)
        result = fake.ensure_image(ssh=ssh, image="img:1")
        assert result.is_ok()
        assert fake.image_exists(ssh, "img:1") is True

    def test_ensure_image_can_fail(self, docker_client: IDockerClient, ssh: Any) -> None:
        fake = _as_fake(docker_client)
        fake.set_pull_behaviour("fail")
        result = fake.ensure_image(ssh=ssh, image="img:1")
        assert result.is_err()
        assert result.unwrap_err().code == "DOCKER_PULL_FAILED"

    def test_ensure_image_silently_missing(self, docker_client: IDockerClient, ssh: Any) -> None:
        fake = _as_fake(docker_client)
        fake.set_pull_behaviour("silently_missing")
        result = fake.ensure_image(ssh=ssh, image="img:1")
        assert result.is_err()
        assert result.unwrap_err().code == "DOCKER_IMAGE_NOT_AVAILABLE"

    # ------------------------------------------------------------------
    # rm_force
    # ------------------------------------------------------------------

    def test_rm_force_unknown_container_is_ok(
        self, docker_client: IDockerClient, ssh: Any,
    ) -> None:
        fake = _as_fake(docker_client)
        result = fake.rm_force(ssh, container_name="never-existed")
        assert result.is_ok()

    def test_rm_force_marks_container_removed(
        self, docker_client: IDockerClient, ssh: Any,
    ) -> None:
        fake = _as_fake(docker_client)
        fake.register_container("c-1", state="running")
        result = fake.rm_force(ssh, container_name="c-1")
        assert result.is_ok()
        assert fake.is_container_running(ssh, name_filter="c-1") is False

    def test_rm_force_failure_injection(
        self, docker_client: IDockerClient, ssh: Any,
    ) -> None:
        fake = _as_fake(docker_client)
        fake.fail_next_n_calls("rm_force", 1)
        result = fake.rm_force(ssh, container_name="c-1")
        assert result.is_err()
        # Second call recovers.
        result = fake.rm_force(ssh, container_name="c-1")
        assert result.is_ok()

    # ------------------------------------------------------------------
    # is_container_running
    # ------------------------------------------------------------------

    def test_is_container_running_unknown_is_false(
        self, docker_client: IDockerClient, ssh: Any,
    ) -> None:
        fake = _as_fake(docker_client)
        assert fake.is_container_running(ssh, name_filter="unknown") is False

    def test_is_container_running_reflects_state(
        self, docker_client: IDockerClient, ssh: Any,
    ) -> None:
        fake = _as_fake(docker_client)
        fake.register_container("c-1", state="running")
        assert fake.is_container_running(ssh, name_filter="c-1") is True
        fake.set_container_state("c-1", "exited")
        assert fake.is_container_running(ssh, name_filter="c-1") is False

    # ------------------------------------------------------------------
    # logs
    # ------------------------------------------------------------------

    def test_logs_empty_for_unknown_container(
        self, docker_client: IDockerClient, ssh: Any,
    ) -> None:
        fake = _as_fake(docker_client)
        result = fake.logs(ssh, container_name="never-existed")
        assert result.is_ok()
        assert result.unwrap() == ""

    def test_logs_returns_appended_content(
        self, docker_client: IDockerClient, ssh: Any,
    ) -> None:
        fake = _as_fake(docker_client)
        fake.register_container("c-1")
        fake.append_logs("c-1", "line-1")
        fake.append_logs("c-1", "line-2")
        result = fake.logs(ssh, container_name="c-1")
        assert result.is_ok()
        assert "line-1" in result.unwrap()
        assert "line-2" in result.unwrap()

    def test_logs_tail_returns_last_n(
        self, docker_client: IDockerClient, ssh: Any,
    ) -> None:
        fake = _as_fake(docker_client)
        for i in range(5):
            fake.append_logs("c-1", f"line-{i}")
        result = fake.logs(ssh, container_name="c-1", tail=2)
        assert result.is_ok()
        assert result.unwrap() == "line-3\nline-4"

    def test_logs_failure_injection(
        self, docker_client: IDockerClient, ssh: Any,
    ) -> None:
        fake = _as_fake(docker_client)
        fake.fail_next_n_calls("logs", 1)
        result = fake.logs(ssh, container_name="c-1")
        assert result.is_err()

    # ------------------------------------------------------------------
    # container_exit_code
    # ------------------------------------------------------------------

    def test_exit_code_unknown_returns_zero(
        self, docker_client: IDockerClient, ssh: Any,
    ) -> None:
        fake = _as_fake(docker_client)
        result = fake.container_exit_code(ssh, container_name="unknown")
        assert result.is_ok()
        assert result.unwrap() == 0

    def test_exit_code_reflects_set_value(
        self, docker_client: IDockerClient, ssh: Any,
    ) -> None:
        fake = _as_fake(docker_client)
        fake.set_exit_code("c-1", 137)
        result = fake.container_exit_code(ssh, container_name="c-1")
        assert result.is_ok()
        assert result.unwrap() == 137

    def test_exit_code_failure_injection(
        self, docker_client: IDockerClient, ssh: Any,
    ) -> None:
        fake = _as_fake(docker_client)
        fake.fail_next_n_calls("container_exit_code", 1)
        result = fake.container_exit_code(ssh, container_name="c-1")
        assert result.is_err()

    # ------------------------------------------------------------------
    # Call log — assertion surface
    # ------------------------------------------------------------------

    def test_call_log_records_method_and_kwargs(
        self, docker_client: IDockerClient, ssh: Any,
    ) -> None:
        fake = _as_fake(docker_client)
        fake.logs(ssh, container_name="c-1", tail=5)
        fake.rm_force(ssh, container_name="c-1")
        calls = fake.calls
        assert ("logs", {"container_name": "c-1", "tail": 5, "timeout_seconds": 30}) in calls
        assert ("rm_force", {"container_name": "c-1", "timeout_seconds": 60}) in calls
