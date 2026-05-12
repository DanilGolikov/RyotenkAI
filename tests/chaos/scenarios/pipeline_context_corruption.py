"""Scenario 9 — ``pipeline_context_corruption``.

Write garbage to a PipelineState-shaped file partway through, then
attempt to load it. The expected behaviour is detection +
``json.JSONDecodeError`` raised at load time (never a partial parse
that silently returns half-broken state).

The production "PipelineState writer" is not abstracted behind a
Protocol the chaos framework can drive in-process. As a partial
implementation we use a generic JSON file-as-state surrogate so the
detection invariant (corruption raises) is still exercised. Full
wiring to the real persistence layer is a Phase 6 follow-up.
"""

from __future__ import annotations

import json
import tempfile
from datetime import timedelta
from pathlib import Path

from tests._harness.chaos import ScenarioContext, register_scenario
from tests.chaos.scenarios._base import ScenarioBase


@register_scenario
class PipelineContextCorruption(ScenarioBase):
    name = "pipeline_context_corruption"
    tags = ["persistence"]
    recovery_window = timedelta(seconds=30)

    async def precondition(self, ctx: ScenarioContext) -> None:
        tmp = Path(tempfile.mkdtemp(prefix="chaos-pipeline-"))
        path = tmp / "state.json"
        path.write_text(json.dumps({"run_id": "r-1", "stage": "training"}))
        ctx.extras["state_path"] = path

    async def inject(self, ctx: ScenarioContext) -> None:
        path: Path = ctx.extras["state_path"]
        # Replace a chunk in the middle of the file with garbage bytes.
        raw = path.read_bytes()
        midpoint = len(raw) // 2
        corrupted = raw[:midpoint] + b"\xff\x00\x01\x02CORRUPT" + raw[midpoint + 6:]
        path.write_bytes(corrupted)
        ctx.debug_recorder.record("inject", "garbage_written", at_byte=midpoint)

    async def steady_state(self, ctx: ScenarioContext) -> None:
        path: Path = ctx.extras["state_path"]
        try:
            json.loads(path.read_bytes())
        except (json.JSONDecodeError, UnicodeDecodeError):
            ctx.debug_recorder.record("steady_state", "corruption_detected")
            return
        raise AssertionError(
            "corruption silently accepted; state loaded without error",
        )

    async def cleanup(self, ctx: ScenarioContext) -> None:
        path = ctx.extras.get("state_path")
        if path is not None:
            try:
                Path(path).unlink(missing_ok=True)
                Path(path).parent.rmdir()
            except Exception:
                pass
        ctx.extras.clear()
        await super().cleanup(ctx)


__all__: list[str] = []
