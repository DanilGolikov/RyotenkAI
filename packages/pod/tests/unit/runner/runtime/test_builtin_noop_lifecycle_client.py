"""Pod-side built-in no-op lifecycle client contract.

Replaces the deleted ``test_lifecycle_client.py`` in providers/single_node/
— the no-op now lives in the pod package
(:class:`ryotenkai_pod.runner.runtime.provider_registry._BuiltinNoOpLifecycleClient`)
because it's used by every provider with
``capabilities.supports_lifecycle_actions=false`` rather than being a
single_node-specific class.

7-category coverage applied to a trivial impl on purpose: the invariants
(Protocol conformance + transport-isolation) catch a future refactor
that accidentally pulls a transport dependency into this path.
"""

from __future__ import annotations

import inspect
from pathlib import Path

import pytest

from ryotenkai_pod.runner.runtime.provider_registry import (
    _BuiltinNoOpLifecycleClient,
    _SKIPPED_RESULT,
)
from ryotenkai_shared.infrastructure.lifecycle import (
    IPodLifecycleClient,
    LifecycleActionResult,
    PodTerminalOutcome,
)

pytestmark = pytest.mark.asyncio


# ---------------------------------------------------------------------------
# 1. Positive — every method returns the shared SKIPPED sentinel
# ---------------------------------------------------------------------------


class TestPositive:
    async def test_terminate_returns_skipped(self) -> None:
        client = _BuiltinNoOpLifecycleClient()
        result = await client.terminate(resource_id="any-id")
        assert isinstance(result, LifecycleActionResult)
        assert result.outcome == PodTerminalOutcome.SKIPPED
        assert result.attempts_made == 0
        assert result.last_error is None

    async def test_pause_returns_skipped(self) -> None:
        client = _BuiltinNoOpLifecycleClient()
        result = await client.pause(resource_id="any-id")
        assert result.outcome == PodTerminalOutcome.SKIPPED
        assert result.attempts_made == 0

    async def test_resume_returns_skipped(self) -> None:
        client = _BuiltinNoOpLifecycleClient()
        result = await client.resume(resource_id="any-id")
        assert result.outcome == PodTerminalOutcome.SKIPPED
        assert result.attempts_made == 0


# ---------------------------------------------------------------------------
# 2. Negative — N/A (NoOp can't fail)
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# 3. Boundary — empty + arbitrary resource_id values accepted
# ---------------------------------------------------------------------------


class TestBoundary:
    async def test_empty_resource_id_accepted(self) -> None:
        client = _BuiltinNoOpLifecycleClient()
        result = await client.terminate(resource_id="")
        assert result.outcome == PodTerminalOutcome.SKIPPED

    async def test_arbitrary_resource_id_returns_shared_sentinel(self) -> None:
        # Pin: resource_id is part of the Protocol but ignored by the
        # no-op. The same SKIPPED sentinel comes back regardless of
        # input — verified through ``is`` (object identity), not just
        # equality, since the impl reuses one module-level instance.
        client = _BuiltinNoOpLifecycleClient()
        r1 = await client.terminate(resource_id="x")
        r2 = await client.terminate(resource_id="y" * 100)
        assert r1 is r2
        assert r1 is _SKIPPED_RESULT

    async def test_skipped_sentinel_is_frozen(self) -> None:
        client = _BuiltinNoOpLifecycleClient()
        result = await client.terminate(resource_id="x")
        with pytest.raises(Exception):  # FrozenInstanceError or AttributeError
            result.outcome = "terminated"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# 4. Invariants — Protocol conformance + transport-isolation
# ---------------------------------------------------------------------------


class TestInvariants:
    async def test_conforms_to_ipodlifecycleclient(self) -> None:
        client = _BuiltinNoOpLifecycleClient()
        assert isinstance(client, IPodLifecycleClient)

    async def test_provider_name_reads_runtime_env(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Pin: provider_name is the env-set value, not a hardcoded
        # constant. Replaces the legacy ``return PROVIDER_SINGLE_NODE``
        # — the no-op is shared by every cap-gated-off provider, so
        # the right name is whatever the env declared at lifespan boot.
        monkeypatch.setenv("RYOTENKAI_RUNTIME_PROVIDER", "single_node")
        client = _BuiltinNoOpLifecycleClient()
        assert client.provider_name == "single_node"
        monkeypatch.setenv("RYOTENKAI_RUNTIME_PROVIDER", "future_local_provider")
        assert client.provider_name == "future_local_provider"

    async def test_module_does_not_import_httpx(self) -> None:
        # Pin: the no-op MUST NOT pull in any HTTP transport.
        module_path = Path(
            inspect.getsourcefile(_BuiltinNoOpLifecycleClient)  # type: ignore[arg-type]
        )
        source = module_path.read_text(encoding="utf-8")
        assert "import httpx" not in source
        assert "from httpx" not in source

    async def test_module_does_not_import_runpod_provider(self) -> None:
        # Pin: pod-side noop has no provider-specific knowledge.
        module_path = Path(
            inspect.getsourcefile(_BuiltinNoOpLifecycleClient)  # type: ignore[arg-type]
        )
        source = module_path.read_text(encoding="utf-8")
        assert "ryotenkai_providers.runpod" not in source


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
        # Replaced the deleted single_node noop's identical assertion.
        client = _BuiltinNoOpLifecycleClient()
        result = await client.terminate(resource_id="x")
        assert result.outcome == "skipped"


# ---------------------------------------------------------------------------
# 7. Logic-specific — last_error is always None (no failure modes)
# ---------------------------------------------------------------------------


class TestLogicSpecific:
    async def test_last_error_is_none_for_all_three_methods(self) -> None:
        client = _BuiltinNoOpLifecycleClient()
        for coro in (
            client.terminate(resource_id="x"),
            client.pause(resource_id="x"),
            client.resume(resource_id="x"),
        ):
            result = await coro
            assert result.last_error is None
            assert result.raw_response_excerpt is None
