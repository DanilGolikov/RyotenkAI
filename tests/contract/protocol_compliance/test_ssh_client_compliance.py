"""Compliance tests for :class:`ISSHClient`.

Fake-only by default. ``real`` is parametrized but ``pytest.skip``s
until a thin :class:`SSHClient` adapter lands.
"""

from __future__ import annotations

import os
from typing import Any

import pytest

from ryotenkai_shared.infrastructure.ssh import (
    ISSHClient,
    SSHConnectionError,
    SSHTransferError,
)
from tests._fakes.ssh import FakeSSHClient

pytestmark = [
    pytest.mark.contract,
    pytest.mark.compliance,
    pytest.mark.exercises_protocol("ISSHClient"),
    pytest.mark.uses_fake("FakeSSHClient"),
    pytest.mark.asyncio,
]


@pytest.fixture(params=["fake", pytest.param("real", marks=pytest.mark.live)])
def ssh_client(request: pytest.FixtureRequest, manual_clock: Any) -> ISSHClient:
    if request.param == "real":
        if os.environ.get("RYOTENKAI_LIVE") != "1":
            pytest.skip("real ISSHClient requires RYOTENKAI_LIVE=1")
        pytest.skip("real ISSHClient not yet wired into compliance suite")
    return FakeSSHClient(clock=manual_clock)


def _as_fake(client: ISSHClient) -> FakeSSHClient:
    assert isinstance(client, FakeSSHClient)
    return client


class TestSSHClientCompliance:
    async def test_isinstance_protocol(self, ssh_client: ISSHClient) -> None:
        assert isinstance(ssh_client, ISSHClient)

    async def test_host_property_visible(self, ssh_client: ISSHClient) -> None:
        assert isinstance(ssh_client.host, str)
        assert ssh_client.host

    async def test_exec_default_success(self, ssh_client: ISSHClient) -> None:
        result = await ssh_client.exec("echo hello")
        assert result.success
        assert result.exit_code == 0

    async def test_exec_canned_response(self, ssh_client: ISSHClient) -> None:
        fake = _as_fake(ssh_client)
        fake.set_command_response(
            r"^cat /workspace/marker$",
            exit_code=0, stdout="alive\n",
        )
        result = await ssh_client.exec("cat /workspace/marker")
        assert result.stdout == "alive\n"

    async def test_upload_then_file_exists(self, ssh_client: ISSHClient) -> None:
        await ssh_client.upload_file("local.txt", "/remote/path/file.txt")
        assert await ssh_client.file_exists("/remote/path/file.txt") is True
        assert await ssh_client.file_exists("/remote/missing") is False

    async def test_download_unknown_file_raises(self, ssh_client: ISSHClient) -> None:
        with pytest.raises(SSHTransferError):
            await ssh_client.download_file("/remote/missing", "local.txt")

    async def test_close_blocks_further_calls(self, ssh_client: ISSHClient) -> None:
        await ssh_client.close()
        with pytest.raises(SSHConnectionError):
            await ssh_client.exec("anything")

    # -- chaos surface --------------------------------------------------

    async def test_chaos_connect_timeout_blocks_everything(
        self, ssh_client: ISSHClient,
    ) -> None:
        fake = _as_fake(ssh_client)
        fake.inject_connect_timeout(True)
        with pytest.raises(SSHConnectionError):
            await ssh_client.exec("anything")
        fake.inject_connect_timeout(False)
        # Recovery — call succeeds.
        result = await ssh_client.exec("anything")
        assert result.success

    async def test_chaos_command_failure_returns_nonzero(
        self, ssh_client: ISSHClient,
    ) -> None:
        fake = _as_fake(ssh_client)
        fake.inject_command_failure(count=2)
        first = await ssh_client.exec("anything")
        assert first.exit_code != 0
        second = await ssh_client.exec("anything")
        assert second.exit_code != 0
        third = await ssh_client.exec("anything")
        assert third.exit_code == 0

    async def test_chaos_transfer_failure(self, ssh_client: ISSHClient) -> None:
        fake = _as_fake(ssh_client)
        fake.inject_transfer_failure(count=1)
        with pytest.raises(SSHTransferError):
            await ssh_client.upload_file("local.txt", "/remote/p")
        # Recovery — second upload succeeds.
        await ssh_client.upload_file("local.txt", "/remote/p")

    async def test_chaos_disconnect_after_n(self, ssh_client: ISSHClient) -> None:
        fake = _as_fake(ssh_client)
        fake.inject_disconnect_after_n_commands(2)
        # Three execs: first two succeed, third fails (disconnected).
        await ssh_client.exec("a")
        await ssh_client.exec("b")
        with pytest.raises(SSHConnectionError):
            await ssh_client.exec("c")

    async def test_snapshot_is_json_serializable(self, ssh_client: ISSHClient) -> None:
        import json
        fake = _as_fake(ssh_client)
        await ssh_client.upload_file("l", "/r")
        snap = fake.snapshot()
        json.dumps(snap)
        assert "/r" in snap["fs_keys"]


__all__: list[str] = []
