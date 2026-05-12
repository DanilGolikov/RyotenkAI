"""Scenario 7 — ``stale_ssh_control_master``.

The SSH ControlMaster goes stale after N commands; the next exec
raises :class:`SSHConnectionError`. A real client would reconnect
transparently; we model that here by re-opening the fake SSH client
and re-issuing the failed command — and assert it succeeds.
"""

from __future__ import annotations

from datetime import timedelta

from ryotenkai_shared.infrastructure.ssh import SSHConnectionError
from tests._fakes.ssh import FakeSSHClient
from tests._harness.chaos import ScenarioContext, register_scenario
from tests._harness.clock import ManualClock
from tests.chaos.scenarios._base import ScenarioBase


@register_scenario
class StaleSshControlMaster(ScenarioBase):
    name = "stale_ssh_control_master"
    tags = ["network", "ssh"]
    recovery_window = timedelta(seconds=30)

    async def precondition(self, ctx: ScenarioContext) -> None:
        clock = ManualClock()
        client = FakeSSHClient(host="pod-1", clock=clock)
        ctx.extras["ssh"] = client

    async def inject(self, ctx: ScenarioContext) -> None:
        ssh: FakeSSHClient = ctx.extras["ssh"]
        ssh.inject_disconnect_after_n_commands(2)
        ctx.debug_recorder.record("inject", "disconnect_after_2")

    async def steady_state(self, ctx: ScenarioContext) -> None:
        ssh: FakeSSHClient = ctx.extras["ssh"]
        # First two execs succeed.
        for i in range(2):
            r = await ssh.exec(f"echo {i}")
            if r.exit_code != 0:
                raise AssertionError(f"pre-disconnect exec {i} failed: {r!r}")
        # Third should hit the disconnect.
        try:
            await ssh.exec("echo failure_expected")
        except SSHConnectionError:
            pass
        else:
            raise AssertionError("expected SSHConnectionError on stale master")

        # Reconnect = build a new fake; idempotent retry succeeds.
        ssh2 = FakeSSHClient(host="pod-1")
        ctx.extras["ssh"] = ssh2
        r = await ssh2.exec("echo recovered")
        if r.exit_code != 0:
            raise AssertionError("post-reconnect exec did not succeed")
        ctx.debug_recorder.record("steady_state", "reconnect_transparent")

    async def cleanup(self, ctx: ScenarioContext) -> None:
        ctx.extras.clear()
        await super().cleanup(ctx)


__all__: list[str] = []
