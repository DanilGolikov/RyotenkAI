"""Phase 5 — typed event emission for ``RunPodProvider``.

Sister of ``test_provider_events.py`` for the single_node provider:
pins the ``ryotenkai.control.gpu.ssh_provisioned`` contract for the
runpod variant. Bypasses the heavy GraphQL-client construction in
``__init__`` via :meth:`RunPodProvider.from_resume_metadata`, which
the resume flow already uses for the same reason — we don't need a
real RunPod API client to exercise emit wiring.
"""

from __future__ import annotations

import logging
import pytest

from ryotenkai_providers.runpod.training.provider import RunPodProvider
from tests._fakes.event_emitter import FakeEventEmitter


pytestmark = pytest.mark.unit


def _make_provider() -> RunPodProvider:
    """Build a minimal :class:`RunPodProvider` that supports
    :meth:`set_emitter` / :meth:`_emit_ssh_provisioned` without
    hitting RunPod's API."""
    provider = RunPodProvider.from_resume_metadata(api_key="dummy")
    # The from_resume_metadata factory mirrors __init__ but doesn't
    # initialise the Phase-5 emitter slots. Mirror the manual setup
    # the unit-under-test depends on so the helper has the expected
    # attributes.
    provider._emitter = None
    provider._emitter_missing_warned = False
    return provider


class TestPositive:
    def test_emit_ssh_provisioned_publishes_envelope(self) -> None:
        provider = _make_provider()
        emitter = FakeEventEmitter()
        provider.set_emitter(emitter)
        provider._emit_ssh_provisioned(run_id="run-x", host="10.0.0.5")
        assert len(emitter.emitted) == 1
        ev = emitter.emitted[0]
        assert ev.kind == "ryotenkai.control.gpu.ssh_provisioned"
        assert ev.run_id == "run-x"
        assert ev.payload.host == "10.0.0.5"

    def test_emit_severity_is_info(self) -> None:
        provider = _make_provider()
        emitter = FakeEventEmitter()
        provider.set_emitter(emitter)
        provider._emit_ssh_provisioned(run_id="r", host="h")
        assert emitter.emitted[0].severity == "info"


class TestNegative:
    def test_no_emitter_warns_once(self, caplog: pytest.LogCaptureFixture) -> None:
        provider = _make_provider()
        with caplog.at_level(logging.WARNING, logger="ryotenkai"):
            provider._emit_ssh_provisioned(run_id="r", host="h")
            provider._emit_ssh_provisioned(run_id="r", host="h")
        warnings = [
            r.message for r in caplog.records
            if "ssh_provisioned" in r.message or "no emitter" in r.message.lower()
        ]
        assert len(warnings) == 1


class TestInvariants:
    def test_set_emitter_replaces_previous(self) -> None:
        provider = _make_provider()
        first = FakeEventEmitter()
        second = FakeEventEmitter()
        provider.set_emitter(first)
        provider.set_emitter(second)
        provider._emit_ssh_provisioned(run_id="r", host="h")
        assert first.emitted == []
        assert len(second.emitted) == 1
