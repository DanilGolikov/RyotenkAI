"""Phase 5 — typed event emission for ``SingleNodeProvider``.

Pins the contract for the ``ryotenkai.control.gpu.ssh_provisioned``
envelope: emitted exactly once after a successful SSH handshake,
carrying the configured host + a fingerprint placeholder. Pre-Phase 5
the provider published only via :data:`logger.info` and was invisible
to the unified event system.

These tests exercise the small :meth:`SingleNodeProvider._emit_ssh_provisioned`
helper directly so we don't need to drag the full ``connect()`` SSH
choreography (subprocess + ssh_client) into the assertion. Connect
itself is covered by integration tests; here we pin the typed-event
behaviour that Phase 6 will rely on when reports start consuming it.
"""

from __future__ import annotations

import logging
import pytest

from tests._factories.single_node_provider import make_single_node_provider
from tests._fakes.event_emitter import FakeEventEmitter


pytestmark = pytest.mark.unit


class TestPositive:
    """Happy path: emitter wired, emit fires."""

    def test_emit_ssh_provisioned_alias_mode(self) -> None:
        provider = make_single_node_provider(alias="pc")
        emitter = FakeEventEmitter()
        provider.set_emitter(emitter)
        provider._emit_ssh_provisioned(run_id="run-1")
        assert len(emitter.emitted) == 1
        ev = emitter.emitted[0]
        assert ev.kind == "ryotenkai.control.gpu.ssh_provisioned"
        assert ev.run_id == "run-1"
        # Alias mode → no explicit key path is forwarded; fingerprint
        # placeholder stays empty per the helper docstring.
        assert ev.payload.key_fingerprint == ""

    def test_emit_ssh_provisioned_explicit_mode(self) -> None:
        provider = make_single_node_provider(host="1.2.3.4", user="u")
        emitter = FakeEventEmitter()
        provider.set_emitter(emitter)
        provider._emit_ssh_provisioned(run_id="r")
        ev = emitter.emitted[0]
        assert ev.payload.host  # explicit host surfaces


class TestNegative:
    """No emitter wired → warning logged once + no emit."""

    def test_no_emitter_logs_once(self, caplog: pytest.LogCaptureFixture) -> None:
        provider = make_single_node_provider()
        with caplog.at_level(logging.WARNING, logger="ryotenkai"):
            provider._emit_ssh_provisioned(run_id="r")
            provider._emit_ssh_provisioned(run_id="r")
        warning_lines = [
            r.message for r in caplog.records
            if "ssh_provisioned" in r.message or "no emitter" in r.message.lower()
        ]
        # Helper logs exactly once even on repeated calls — the
        # second invocation must be silent.
        assert len(warning_lines) == 1


class TestInvariants:
    def test_set_emitter_replaces_previous(self) -> None:
        provider = make_single_node_provider()
        first = FakeEventEmitter()
        second = FakeEventEmitter()
        provider.set_emitter(first)
        provider.set_emitter(second)
        provider._emit_ssh_provisioned(run_id="r")
        assert first.emitted == []
        assert len(second.emitted) == 1

    def test_emit_severity_is_info(self) -> None:
        provider = make_single_node_provider()
        emitter = FakeEventEmitter()
        provider.set_emitter(emitter)
        provider._emit_ssh_provisioned(run_id="r")
        assert emitter.emitted[0].severity == "info"
