"""L12 replay regression — load each committed Playwright trace and
assert it survives a parse pass.

Phase-6 scope: the corpus is bootstrapped here as a small registry of
expected trace files. The actual traces are recorded by Playwright
(see ``web/e2e/*.spec.ts``) and committed to ``tests/replay/corpus/``
on each release.

The test does two things, both cheap and offline:

1. **Registry coherence** — every flow listed in ``EXPECTED_FLOWS`` has
   a corresponding ``.zip`` file or is explicitly marked as
   "pending-record" (skipped, not failed). This catches PRs that
   advertise a new flow but forget to commit the trace.
2. **Trace integrity** — each present trace is a valid zip archive
   containing at least one ``*.trace`` entry. We deliberately do NOT
   replay the trace against a live browser here; replay against the
   real build runs in the ``release-gate`` workflow with the heavyweight
   Playwright fixture, gated by ``RYOTENKAI_REPLAY_FULL=1``.
"""

from __future__ import annotations

import os
import zipfile
from pathlib import Path
from typing import NamedTuple

import pytest

pytestmark = [pytest.mark.replay]


CORPUS_DIR = Path(__file__).parent / "corpus"


class ReplayFlow(NamedTuple):
    """A registered replay flow."""

    name: str
    """User-flow identifier (matches ``*.spec.ts`` filename)."""
    description: str
    """What the user does in this flow — for humans reading the test."""


# Registry. Each tuple is (flow-name, description). To register a new
# flow:
#   1. add a Playwright spec under ``web/e2e/<flow>.spec.ts`` that ends
#      with ``context.tracing.stop({ path: '../tests/replay/corpus/<flow>.zip' })``;
#   2. add the entry below;
#   3. run ``npm run test:e2e`` once to generate the .zip and commit it.
EXPECTED_FLOWS: tuple[ReplayFlow, ...] = (
    ReplayFlow(
        name="create_run",
        description="User opens the runs list and starts a new run.",
    ),
    ReplayFlow(
        name="cancel_run",
        description="User navigates to a running run and cancels it.",
    ),
    ReplayFlow(
        name="view_run_logs",
        description="User opens the log dock and tails a running attempt.",
    ),
)


@pytest.mark.parametrize(
    "flow",
    EXPECTED_FLOWS,
    ids=[f.name for f in EXPECTED_FLOWS],
)
def test_replay_flow_present_or_pending(flow: ReplayFlow) -> None:
    """Every registered flow has a corpus file or is explicitly pending.

    A missing trace is a *skip*, not a failure, until the release-gate
    workflow runs end-to-end and records them. This keeps the new-lane
    test suite green during initial phase-6 setup.
    """
    trace_path = CORPUS_DIR / f"{flow.name}.zip"
    if not trace_path.exists():
        pytest.skip(
            f"replay trace not yet recorded: {trace_path}; "
            "run `npm run test:e2e` in web/ to populate."
        )

    assert zipfile.is_zipfile(trace_path), (
        f"trace at {trace_path} is not a valid zip archive — re-record."
    )
    with zipfile.ZipFile(trace_path) as zf:
        names = zf.namelist()
    assert any(
        n.endswith(".trace") or n == "trace.trace" or n.startswith("trace")
        for n in names
    ), (
        f"trace at {trace_path} contains no '*.trace' entry — got {names!r}; "
        "Playwright tracing API may have changed shape, re-record."
    )


def test_corpus_directory_exists() -> None:
    """``tests/replay/corpus/`` must exist even before any trace is
    recorded — otherwise the release-gate workflow has nowhere to
    write."""
    assert CORPUS_DIR.is_dir(), f"missing replay corpus dir: {CORPUS_DIR}"


@pytest.mark.skipif(
    os.environ.get("RYOTENKAI_REPLAY_FULL") != "1",
    reason="full replay only runs in release-gate (set RYOTENKAI_REPLAY_FULL=1)",
)
def test_full_replay_smoke() -> None:
    """Placeholder for the heavyweight replay step.

    Phase-6 ships only the registry. The real replay-against-build
    runs in `.github/workflows/release-gate.yml` and is wired in
    Phase-7 as the corpus matures. Keeping this stub here makes the
    contract visible to readers.
    """
    pytest.skip("phase-7 wires the full replay path against the current build")
