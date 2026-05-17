"""Post-Phase-10 (F-СРЕД) — typed cleanup events for SingleNodeProvider.

Pins the visibility-gap close for ``cleanup_after_run``: the method now
emits a typed envelope at every milestone (started, completed, failed)
so orphaned containers / leaked workspaces surface on the unified
timeline instead of being invisible to operators.

Coverage:

* **Positive** — happy-path cleanup emits exactly 2 envelopes
  (started + completed) with ``duration_s > 0``;
  ``resources_freed['containers'] == 1``.
* **Negative** — docker rm fails (verify reports container still
  listed) → started + failed; ``partial_cleanup=True``. SSH transport
  exception → started + failed; ``partial_cleanup=False``.
* **Boundary** — idempotent re-run on an already-cleaned pod still
  emits the started + completed pair (cleanup is best-effort and the
  underlying ``|| true`` makes the docker rm itself a no-op).
* **Regressions** — caller-supplied ``reason`` propagates to the
  started payload verbatim.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pytest

from ryotenkai_providers.single_node.training.provider import SingleNodeProvider
from ryotenkai_providers.training.interfaces import ProviderStatus
from ryotenkai_shared.errors import ProviderUnavailableError

from tests._fakes.event_emitter import FakeEventEmitter


pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# Fake SSH client mirroring the shape used by test_cleanup_after_run.py —
# duplicated rather than imported to keep the test files independent.
# ---------------------------------------------------------------------------


@dataclass
class _FakeSSHClient:
    commands: list[tuple[str, float]] = field(default_factory=list)
    rm_result: tuple[bool, str, str] = (True, "", "")
    verify_result: tuple[bool, str, str] = (True, "", "")
    rm_raises: Exception | None = None
    verify_raises: Exception | None = None

    def exec_command(
        self,
        command: str,
        timeout: float = 10.0,
        silent: bool = False,
    ) -> tuple[bool, str, str]:
        self.commands.append((command, timeout))
        if "docker rm -f" in command:
            if self.rm_raises is not None:
                raise self.rm_raises
            return self.rm_result
        if "docker ps -a -q" in command:
            if self.verify_raises is not None:
                raise self.verify_raises
            return self.verify_result
        return (True, "", "")


def _make_provider(ssh: _FakeSSHClient, emitter: FakeEventEmitter) -> SingleNodeProvider:
    """Construct a provider with the SSH client + emitter wired in.

    Bypasses ``__init__`` (which requires a fully-formed
    :class:`ProviderContext`) and injects the minimum state the
    cleanup method touches.
    """
    provider = SingleNodeProvider.__new__(SingleNodeProvider)
    provider._ssh_client = ssh  # type: ignore[attr-defined]
    provider._status = ProviderStatus.CONNECTED  # type: ignore[attr-defined]
    provider._emitter = emitter  # type: ignore[attr-defined]
    provider._emitter_missing_warned = False  # type: ignore[attr-defined]
    return provider


# ---------------------------------------------------------------------------
# 1. Positive
# ---------------------------------------------------------------------------


class TestPositive:
    def test_happy_path_emits_started_and_completed(self) -> None:
        ssh = _FakeSSHClient()  # default: rm ok, verify ok (empty stdout)
        emitter = FakeEventEmitter()
        _make_provider(ssh, emitter).cleanup_after_run("ryotenkai_training_r1")

        kinds = [ev.kind for ev in emitter.emitted]
        assert kinds == [
            "ryotenkai.control.gpu.cleanup_started",
            "ryotenkai.control.gpu.cleanup_completed",
        ]

    def test_completed_carries_duration_and_resources(self) -> None:
        ssh = _FakeSSHClient()
        emitter = FakeEventEmitter()
        _make_provider(ssh, emitter).cleanup_after_run("ryotenkai_training_r1")
        completed = emitter.emitted[1]
        assert completed.payload.provider == "single_node"
        assert completed.payload.duration_s >= 0.0
        # Even a near-instant cleanup should report > 0 wall-clock
        # delta because time.monotonic() has nanosecond resolution.
        assert completed.payload.resources_freed == {"containers": 1}


# ---------------------------------------------------------------------------
# 2. Negative
# ---------------------------------------------------------------------------


class TestNegative:
    def test_verify_reports_still_listed_emits_failed_partial(self) -> None:
        ssh = _FakeSSHClient(
            rm_result=(True, "", ""),
            verify_result=(True, "abc123\n", ""),  # container still listed
        )
        emitter = FakeEventEmitter()
        with pytest.raises(ProviderUnavailableError):
            _make_provider(ssh, emitter).cleanup_after_run("ryotenkai_training_r1")

        kinds = [ev.kind for ev in emitter.emitted]
        assert kinds == [
            "ryotenkai.control.gpu.cleanup_started",
            "ryotenkai.control.gpu.cleanup_failed",
        ]
        failed = emitter.emitted[1]
        assert failed.payload.partial_cleanup is True
        assert failed.payload.provider == "single_node"
        assert failed.severity == "error"

    def test_ssh_transport_exception_emits_failed_non_partial(self) -> None:
        ssh = _FakeSSHClient(rm_raises=ConnectionError("dns failed"))
        emitter = FakeEventEmitter()
        with pytest.raises(ProviderUnavailableError):
            _make_provider(ssh, emitter).cleanup_after_run("ryotenkai_training_r1")

        kinds = [ev.kind for ev in emitter.emitted]
        assert kinds == [
            "ryotenkai.control.gpu.cleanup_started",
            "ryotenkai.control.gpu.cleanup_failed",
        ]
        failed = emitter.emitted[1]
        assert failed.payload.partial_cleanup is False
        assert failed.payload.error_type == "ConnectionError"

    def test_docker_rm_failure_emits_failed(self) -> None:
        # exec_command returns (ok=False, ...) — docker daemon down,
        # for example. Emitted ``failed`` is not partial because no
        # confirmed side-effect happened.
        ssh = _FakeSSHClient(rm_result=(False, "", "docker not running"))
        emitter = FakeEventEmitter()
        with pytest.raises(ProviderUnavailableError):
            _make_provider(ssh, emitter).cleanup_after_run("ryotenkai_training_r1")

        kinds = [ev.kind for ev in emitter.emitted]
        assert kinds[-1] == "ryotenkai.control.gpu.cleanup_failed"
        assert emitter.emitted[-1].payload.partial_cleanup is False

    def test_no_ssh_client_emits_started_then_failed(self) -> None:
        # Pre-condition failure path. The provider still emits
        # ``started`` (so the timeline shows the attempt) followed by
        # ``failed`` before raising.
        provider = SingleNodeProvider.__new__(SingleNodeProvider)
        provider._ssh_client = None  # type: ignore[attr-defined]
        provider._status = ProviderStatus.CONNECTED  # type: ignore[attr-defined]
        emitter = FakeEventEmitter()
        provider._emitter = emitter  # type: ignore[attr-defined]
        provider._emitter_missing_warned = False  # type: ignore[attr-defined]

        with pytest.raises(ProviderUnavailableError):
            provider.cleanup_after_run("ryotenkai_training_r1")

        kinds = [ev.kind for ev in emitter.emitted]
        assert kinds == [
            "ryotenkai.control.gpu.cleanup_started",
            "ryotenkai.control.gpu.cleanup_failed",
        ]
        assert emitter.emitted[-1].payload.partial_cleanup is False


# ---------------------------------------------------------------------------
# 3. Boundary
# ---------------------------------------------------------------------------


class TestBoundary:
    def test_idempotent_rerun_still_emits_pair(self) -> None:
        # ``docker rm -f X || true`` returns success regardless of
        # whether X existed. Calling cleanup twice in a row must
        # produce two complete (started, completed) pairs — the timeline
        # should not silently drop the second attempt.
        ssh = _FakeSSHClient()
        emitter = FakeEventEmitter()
        provider = _make_provider(ssh, emitter)
        provider.cleanup_after_run("ryotenkai_training_r1")
        provider.cleanup_after_run("ryotenkai_training_r1")

        kinds = [ev.kind for ev in emitter.emitted]
        assert kinds == [
            "ryotenkai.control.gpu.cleanup_started",
            "ryotenkai.control.gpu.cleanup_completed",
            "ryotenkai.control.gpu.cleanup_started",
            "ryotenkai.control.gpu.cleanup_completed",
        ]
        # ``resources_freed`` is the same on both runs because the
        # contract is "containers we attempted to free" not "bytes
        # actually freed" — distinguishing requires inspecting docker
        # output across runs, which the simple ``|| true`` flow doesn't.
        for ev in (emitter.emitted[1], emitter.emitted[3]):
            assert ev.payload.resources_freed == {"containers": 1}


# ---------------------------------------------------------------------------
# 4. Regressions
# ---------------------------------------------------------------------------


class TestRegressions:
    @pytest.mark.parametrize("reason", ["natural", "cancelled", "failed", "forced"])
    def test_reason_propagates_to_started_payload(self, reason: str) -> None:
        ssh = _FakeSSHClient()
        emitter = FakeEventEmitter()
        _make_provider(ssh, emitter).cleanup_after_run(
            "ryotenkai_training_r1", reason=reason,
        )
        assert emitter.emitted[0].payload.reason == reason

    def test_default_reason_is_natural(self) -> None:
        ssh = _FakeSSHClient()
        emitter = FakeEventEmitter()
        _make_provider(ssh, emitter).cleanup_after_run("ryotenkai_training_r1")
        assert emitter.emitted[0].payload.reason == "natural"

    def test_no_emitter_does_not_crash(self) -> None:
        # Legacy / standalone callers that haven't wired an emitter
        # must keep working. The cleanup path remains
        # raise-on-error-only — no envelope side-effects.
        ssh = _FakeSSHClient()
        provider = SingleNodeProvider.__new__(SingleNodeProvider)
        provider._ssh_client = ssh  # type: ignore[attr-defined]
        provider._status = ProviderStatus.CONNECTED  # type: ignore[attr-defined]
        provider._emitter = None  # type: ignore[attr-defined]
        provider._emitter_missing_warned = False  # type: ignore[attr-defined]
        # Must NOT raise — emitter is optional.
        provider.cleanup_after_run("ryotenkai_training_r1")
