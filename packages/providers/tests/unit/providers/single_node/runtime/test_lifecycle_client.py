"""Phase 14.B — :class:`NoOpPodLifecycleClient` contract.

Pure-stdlib unit tests for the single-node no-op impl. 7-category
coverage applied even though the impl is trivial — invariant tests
catch a future refactor that accidentally pulls a transport
dependency into this path.
"""

from __future__ import annotations

import inspect
from pathlib import Path

import pytest

from ryotenkai_shared.constants import PROVIDER_SINGLE_NODE
from ryotenkai_providers.single_node.runtime.lifecycle_client import (
    NoOpPodLifecycleClient,
)
from ryotenkai_shared.infrastructure.lifecycle import PodTerminalOutcome
from ryotenkai_shared.infrastructure.lifecycle import (
    IPodLifecycleClient,
    LifecycleActionResult,
)


pytestmark = pytest.mark.asyncio


# ---------------------------------------------------------------------------
# 1. Positive — every method returns the SKIPPED sentinel
# ---------------------------------------------------------------------------


class TestPositive:
    async def test_terminate_returns_skipped(self) -> None:
        client = NoOpPodLifecycleClient()
        result = await client.terminate(resource_id="any-id")
        assert isinstance(result, LifecycleActionResult)
        assert result.outcome == PodTerminalOutcome.SKIPPED
        assert result.attempts_made == 0
        assert result.last_error is None

    async def test_pause_returns_skipped(self) -> None:
        client = NoOpPodLifecycleClient()
        result = await client.pause(resource_id="any-id")
        assert result.outcome == PodTerminalOutcome.SKIPPED
        assert result.attempts_made == 0

    async def test_resume_returns_skipped(self) -> None:
        client = NoOpPodLifecycleClient()
        result = await client.resume(resource_id="any-id")
        assert result.outcome == PodTerminalOutcome.SKIPPED
        assert result.attempts_made == 0

    async def test_provider_name_is_single_node(self) -> None:
        client = NoOpPodLifecycleClient()
        assert client.provider_name == PROVIDER_SINGLE_NODE


# ---------------------------------------------------------------------------
# 2. Negative — N/A (NoOp can't fail)
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# 3. Boundary — empty + arbitrary resource_id values accepted
# ---------------------------------------------------------------------------


class TestBoundary:
    async def test_empty_resource_id_accepted(self) -> None:
        client = NoOpPodLifecycleClient()
        result = await client.terminate(resource_id="")
        assert result.outcome == PodTerminalOutcome.SKIPPED

    async def test_arbitrary_resource_id_ignored(self) -> None:
        # Pin: resource_id is part of the Protocol but ignored by
        # single-node. The same SKIPPED sentinel comes back regardless
        # of input.
        client = NoOpPodLifecycleClient()
        r1 = await client.terminate(resource_id="x")
        r2 = await client.terminate(resource_id="y" * 100)
        assert r1 is r2  # shared sentinel, not just equal

    async def test_skipped_sentinel_is_frozen(self) -> None:
        client = NoOpPodLifecycleClient()
        result = await client.terminate(resource_id="x")
        with pytest.raises(Exception):  # FrozenInstanceError
            result.outcome = "terminated"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# 4. Invariants — Protocol conformance + transport-isolation
# ---------------------------------------------------------------------------


class TestInvariants:
    @pytest.mark.asyncio(loop_scope=None)
    async def test_conforms_to_ipodlifecycleclient(self) -> None:
        client = NoOpPodLifecycleClient()
        assert isinstance(client, IPodLifecycleClient)

    @pytest.mark.asyncio(loop_scope=None)
    async def test_module_does_not_import_httpx(self) -> None:
        # Pin: the single-node lifecycle client must NOT pull in any
        # HTTP transport. A regression here would silently fatten the
        # single-node deployment image with httpx and its deps.
        module_path = Path(
            inspect.getsourcefile(NoOpPodLifecycleClient)  # type: ignore[arg-type]
        )
        source = module_path.read_text(encoding="utf-8")
        assert "import httpx" not in source
        assert "from httpx" not in source

    @pytest.mark.asyncio(loop_scope=None)
    async def test_module_does_not_import_runpod(self) -> None:
        # Pin: no cross-provider knowledge in the no-op impl.
        module_path = Path(
            inspect.getsourcefile(NoOpPodLifecycleClient)  # type: ignore[arg-type]
        )
        source = module_path.read_text(encoding="utf-8")
        assert "providers.runpod" not in source
        assert "from runpod" not in source


# ---------------------------------------------------------------------------
# 5. Dependency errors — N/A (no transport)
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# 6. Regressions — pre-14.B SKIPPED string preserved verbatim
# ---------------------------------------------------------------------------


class TestRegressions:
    async def test_skipped_outcome_string_value_unchanged(self) -> None:
        # Pin: the wire-string is ``"skipped"`` (lowercase). Operator
        # dashboards parse this verbatim — Phase 9.A / 11.B convention.
        client = NoOpPodLifecycleClient()
        result = await client.terminate(resource_id="x")
        assert result.outcome == "skipped"


# ---------------------------------------------------------------------------
# 7. Logic-specific — last_error is always None (no failure modes)
# ---------------------------------------------------------------------------


class TestLogicSpecific:
    async def test_last_error_is_none_for_all_three_methods(self) -> None:
        client = NoOpPodLifecycleClient()
        for coro in (
            client.terminate(resource_id="x"),
            client.pause(resource_id="x"),
            client.resume(resource_id="x"),
        ):
            result = await coro
            assert result.last_error is None
            assert result.raw_response_excerpt is None
