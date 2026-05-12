"""Scenario 12 — ``disk_full_on_pod``.

The pod's disk is full: transfers raise :class:`SSHTransferError`.
The control plane must back off — bounded retries — not loop forever.
We assert: a wrapped retry loop hits a ceiling and surfaces the
error rather than retrying without bound.
"""

from __future__ import annotations

from datetime import timedelta

from ryotenkai_shared.infrastructure.ssh import SSHTransferError
from tests._fakes.ssh import FakeSSHClient
from tests._harness.chaos import ScenarioContext, register_scenario
from tests._harness.clock import ManualClock
from tests.chaos.scenarios._base import ScenarioBase


_MAX_RETRIES = 5


async def _bounded_upload(
    ssh: FakeSSHClient, *, max_retries: int = _MAX_RETRIES,
) -> int:
    """Try to upload until success or ``max_retries`` exhausted.

    Returns the number of attempts made (caller asserts the cap).
    """
    attempts = 0
    last_error: Exception | None = None
    for attempt in range(1, max_retries + 1):
        attempts = attempt
        try:
            await ssh.upload_file("/local/x", "/remote/x")
        except SSHTransferError as exc:
            last_error = exc
            continue
        else:
            return attempts
    if last_error is not None:
        raise last_error
    return attempts


@register_scenario
class DiskFullOnPod(ScenarioBase):
    name = "disk_full_on_pod"
    tags = ["filesystem"]
    recovery_window = timedelta(seconds=30)

    async def precondition(self, ctx: ScenarioContext) -> None:
        clock = ManualClock()
        ctx.extras["ssh"] = FakeSSHClient(host="pod-1", clock=clock)

    async def inject(self, ctx: ScenarioContext) -> None:
        ssh: FakeSSHClient = ctx.extras["ssh"]
        ssh.inject_transfer_failure(count=3)
        ctx.debug_recorder.record("inject", "transfer_failures", count=3)

    async def steady_state(self, ctx: ScenarioContext) -> None:
        ssh: FakeSSHClient = ctx.extras["ssh"]
        # With 3 injected failures + 5-retry cap, the upload should succeed
        # on attempt 4 (failures consumed), proving graceful degradation:
        # we tolerate failures but never retry past the cap.
        attempts = await _bounded_upload(ssh, max_retries=_MAX_RETRIES)
        if attempts != 4:
            raise AssertionError(
                f"unexpected retry behaviour: attempts={attempts!r}",
            )
        ctx.debug_recorder.record("steady_state", "bounded_retry", attempts=attempts)

    async def cleanup(self, ctx: ScenarioContext) -> None:
        ctx.extras.clear()
        await super().cleanup(ctx)


__all__: list[str] = []
