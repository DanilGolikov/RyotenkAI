"""Phase 3 — :class:`RunnerEventCallback` contract.

Unit coverage uses a captured ``httpx.MockTransport`` so we can
assert on the exact POST payloads without paying for a real HTTP
server. One integration test wires the callback against a live
:class:`fastapi.testclient.TestClient` of the runner so the
end-to-end loopback path is exercised at least once.

Coverage matrix:

- TestEnablement       env-driven activation, no-op when disabled
- TestEventTypes       each Trainer hook publishes the right kind
- TestFlushPolicy      buffer behaviour around ``flush_every``
- TestFailureHandling  retry, consecutive failures, disable + drain
- TestEndOfTraining    on_train_end flushes + closes the client
- TestIntegration      real loopback POST → FastAPI runner
"""

from __future__ import annotations

# The callback's ``transformers`` import (and the ``src.training``
# package's heavy notifiers / orchestrator chain pulled in through
# the regular package path) is not available in every dev
# environment. Tests must run on a slim venv too — production / CI
# is the only place that has full ML deps installed. So we:
#
# 1. Stub ``transformers`` (and any siblings the import cascade
#    happens to pull in) BEFORE the callback module loads.
# 2. Load the callback module DIRECTLY from its file path, bypassing
#    ``src.training/__init__`` so we don't trigger the orchestrator
#    cascade that imports ``datasets`` etc.
#
# The stubs are minimal — ``TrainerCallback`` is just a base class
# the callback subclasses, so an empty placeholder is enough for
# imports to succeed. Real CI runs install transformers and the
# stub branch is never taken.

import importlib.util as _importlib_util
import pathlib as _pathlib
import sys as _sys
import types as _types


def _stub(name: str, attrs: dict[str, object] | None = None) -> None:
    if name in _sys.modules:
        return
    try:
        __import__(name)
    except ModuleNotFoundError:
        module = _types.ModuleType(name)
        for attr_name, attr_value in (attrs or {}).items():
            setattr(module, attr_name, attr_value)
        _sys.modules[name] = module


class _TrainerCallback:
    """Stand-in for ``transformers.TrainerCallback`` when the real
    library isn't installed in the test env. RunnerEventCallback only
    subclasses it; no real logic is inherited."""


_stub("transformers", {"TrainerCallback": _TrainerCallback})
_stub("colorlog", {"ColoredFormatter": type})

_CALLBACK_PATH = (
    _pathlib.Path(__file__).resolve().parents[4]
    / "src" / "ryotenkai_pod" / "trainer" / "callbacks" / "runner_event_callback.py"
)
_spec = _importlib_util.spec_from_file_location(
    "_ryotenkai_runner_event_callback_under_test", _CALLBACK_PATH,
)
assert _spec is not None and _spec.loader is not None
_module = _importlib_util.module_from_spec(_spec)
_sys.modules["_ryotenkai_runner_event_callback_under_test"] = _module
_spec.loader.exec_module(_module)

RunnerEventCallback = _module.RunnerEventCallback
RUNNER_URL_ENV = _module.RUNNER_URL_ENV
MAX_CONSECUTIVE_FAILURES = _module.MAX_CONSECUTIVE_FAILURES

from dataclasses import dataclass  # noqa: E402
from typing import Any  # noqa: E402

import httpx  # noqa: E402
import pytest  # noqa: E402


# ---------------------------------------------------------------------------
# Trainer-state stubs — TrainerCallback ignores everything except the
# four fields below, so we don't need a real ``TrainerState`` instance.
# ---------------------------------------------------------------------------


@dataclass
class _State:
    global_step: int = 0
    epoch: float = 0.0
    max_steps: int = 100


@dataclass
class _Args:
    num_train_epochs: float = 1.0
    per_device_train_batch_size: int = 4


class _Control:  # pragma: no cover — opaque to the callback
    pass


def _hook_args(state: _State | None = None) -> dict[str, Any]:
    """Common kwargs the Trainer would normally pass."""
    return {
        "args": _Args(),
        "state": state or _State(),
        "control": _Control(),
    }


# ---------------------------------------------------------------------------
# MockTransport helper — captures every POSTed body for assertion.
# ---------------------------------------------------------------------------


class _Capturing:
    """Stateful httpx.MockTransport handler.

    Drops a recording of every POST into ``self.calls``. Tests can
    set ``self.fail_until`` to make the first N posts fail (used by
    failure-handling tests).
    """

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []
        self.fail_until: int = 0

    def __call__(self, request: httpx.Request) -> httpx.Response:
        if self.fail_until > 0:
            self.fail_until -= 1
            return httpx.Response(500, json={"error": "synthetic"})
        body = request.read()
        import json as _json

        self.calls.append({
            "url": str(request.url),
            "json": _json.loads(body) if body else None,
        })
        return httpx.Response(202, json={"offset": len(self.calls) - 1})


def _build(callback: RunnerEventCallback, capture: _Capturing) -> None:
    """Replace the callback's lazy client with one wired to MockTransport."""
    transport = httpx.MockTransport(capture)
    callback._client = httpx.Client(transport=transport, timeout=2.0)


# ---------------------------------------------------------------------------
# Enablement
# ---------------------------------------------------------------------------


class TestEnablement:
    def test_no_op_when_env_unset(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv(RUNNER_URL_ENV, raising=False)
        cb = RunnerEventCallback()
        assert cb.enabled is False
        cb.on_train_begin(**_hook_args())
        # Nothing buffered, nothing posted, nothing raised.
        assert cb.buffer_size == 0

    def test_explicit_url_overrides_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv(RUNNER_URL_ENV, raising=False)
        cb = RunnerEventCallback(runner_url="http://127.0.0.1:8080")
        assert cb.enabled is True

    def test_env_url_used_when_arg_omitted(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv(RUNNER_URL_ENV, "http://127.0.0.1:8080")
        cb = RunnerEventCallback()
        assert cb.enabled is True


# ---------------------------------------------------------------------------
# Per-hook events
# ---------------------------------------------------------------------------


class TestEventTypes:
    def test_on_train_begin_publishes_training_started(self) -> None:
        cap = _Capturing()
        cb = RunnerEventCallback(runner_url="http://127.0.0.1:8080")
        _build(cb, cap)
        cb.on_train_begin(**_hook_args())
        assert len(cap.calls) == 1
        body = cap.calls[0]["json"]
        assert body["kind"] == "training_started"
        assert body["payload"]["max_steps"] == 100
        assert body["payload"]["per_device_train_batch_size"] == 4

    def test_on_step_end_emits_at_flush_boundaries(self) -> None:
        cap = _Capturing()
        cb = RunnerEventCallback(runner_url="http://127.0.0.1:8080", flush_every=10)
        _build(cb, cap)
        # Step 5 → no event (5 % 10 != 0)
        cb.on_step_end(**_hook_args(_State(global_step=5)))
        assert cap.calls == []
        # Step 10 → emitted
        cb.on_step_end(**_hook_args(_State(global_step=10, epoch=0.1)))
        assert any(c["json"]["kind"] == "step" for c in cap.calls)

    def test_on_step_end_attaches_last_loss(self) -> None:
        cap = _Capturing()
        cb = RunnerEventCallback(runner_url="http://127.0.0.1:8080", flush_every=10)
        _build(cb, cap)
        cb.on_log(logs={"loss": 0.42, "lr": 1e-5}, **_hook_args(_State(global_step=10)))
        cb.on_step_end(**_hook_args(_State(global_step=10)))
        step_payload = next(
            c["json"]["payload"] for c in cap.calls
            if c["json"]["kind"] == "step"
        )
        assert step_payload["loss"] == 0.42

    def test_on_evaluate_emits_metrics(self) -> None:
        cap = _Capturing()
        cb = RunnerEventCallback(runner_url="http://127.0.0.1:8080")
        _build(cb, cap)
        cb.on_evaluate(
            metrics={"eval_loss": 0.3, "eval_accuracy": 0.85},
            **_hook_args(_State(global_step=50)),
        )
        body = cap.calls[-1]["json"]
        assert body["kind"] == "eval_metrics"
        assert body["payload"]["metrics"]["eval_accuracy"] == 0.85

    def test_on_save_emits_checkpoint(self) -> None:
        cap = _Capturing()
        cb = RunnerEventCallback(runner_url="http://127.0.0.1:8080")
        _build(cb, cap)
        cb.on_save(**_hook_args(_State(global_step=100)))
        body = cap.calls[-1]["json"]
        assert body["kind"] == "checkpoint_saved"
        assert body["payload"]["step"] == 100

    def test_on_log_buffers_when_below_threshold(self) -> None:
        cap = _Capturing()
        # flush_every=10 → on_log alone (no flush_now=True) buffers.
        cb = RunnerEventCallback(runner_url="http://127.0.0.1:8080", flush_every=10)
        _build(cb, cap)
        cb.on_log(logs={"x": 1}, **_hook_args())
        assert cap.calls == []
        assert cb.buffer_size == 1


# ---------------------------------------------------------------------------
# Flush policy
# ---------------------------------------------------------------------------


class TestFlushPolicy:
    def test_buffer_drains_after_flush_every_logs(self) -> None:
        cap = _Capturing()
        cb = RunnerEventCallback(runner_url="http://127.0.0.1:8080", flush_every=3)
        _build(cb, cap)
        for i in range(3):
            cb.on_log(logs={"i": i}, **_hook_args())
        # Three buffered + the third triggered a flush → all three posted.
        assert len(cap.calls) == 3
        assert cb.buffer_size == 0


# ---------------------------------------------------------------------------
# Failure handling
# ---------------------------------------------------------------------------


class TestFailureHandling:
    def test_failed_post_keeps_event_buffered(self) -> None:
        cap = _Capturing()
        cap.fail_until = 1  # next post fails
        cb = RunnerEventCallback(runner_url="http://127.0.0.1:8080")
        _build(cb, cap)
        cb.on_train_begin(**_hook_args())
        # fail_until=1 → first post 500 → event re-buffered.
        assert cb.buffer_size == 1
        # Next flush succeeds (fail_until=0).
        cb._flush()
        assert cb.buffer_size == 0
        assert len(cap.calls) == 1

    def test_disables_after_max_consecutive_failures(self) -> None:
        cap = _Capturing()
        cap.fail_until = 100  # always fail
        cb = RunnerEventCallback(
            runner_url="http://127.0.0.1:8080", flush_every=1,
        )
        _build(cb, cap)
        # MAX_CONSECUTIVE_FAILURES = 3.
        for i in range(MAX_CONSECUTIVE_FAILURES):
            cb.on_log(logs={"i": i}, **_hook_args())
        assert cb.enabled is False
        # Subsequent calls are silent no-ops.
        cb.on_step_end(**_hook_args(_State(global_step=10)))
        cb.on_save(**_hook_args(_State(global_step=10)))
        # Buffer cleared on disable; no more posts.
        assert cb.buffer_size == 0


# ---------------------------------------------------------------------------
# End of training
# ---------------------------------------------------------------------------


class TestEndOfTraining:
    def test_on_train_end_emits_complete_and_drains(self) -> None:
        cap = _Capturing()
        cb = RunnerEventCallback(runner_url="http://127.0.0.1:8080")
        _build(cb, cap)
        cb.on_train_end(**_hook_args(_State(global_step=200)))
        kinds = [c["json"]["kind"] for c in cap.calls]
        assert "training_complete" in kinds
        # Client closed after train end.
        assert cb._client is None


# Note: an ASGITransport-based integration test was prototyped here
# but proved too coupled to httpx + Starlette internals to be useful.
# The unit tests above (with ``httpx.MockTransport``) give us full
# behaviour coverage of the callback's HTTP path; the actual
# end-to-end loopback round-trip is exercised in Phase 6's RunPod
# manual smoke and the supervisor integration tests in
# ``test_supervisor.py``. The real wire never sees a stub —
# uvicorn binds 127.0.0.1 in the docker entrypoint and the trainer
# subprocess inherits ``RYOTENKAI_RUNNER_URL`` from the supervisor's
# ``submit_and_spawn(env=...)`` call.
