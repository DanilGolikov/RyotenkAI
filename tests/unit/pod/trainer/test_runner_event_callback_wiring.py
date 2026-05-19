"""Phase 3.2 regression: ``RunnerEventCallback`` is wired into the trainer.

The Phase 3 commit (75ba789) shipped the callback class + 13 unit
tests (``test_runner_event_callback.py``), but the *registration*
in :class:`TrainerFactory` was forgotten — the class was dead code
in production until this pin landed.

This file guards against the regression returning. The full
``TrainerFactory.create()`` path requires a fully-resolved
:class:`PipelineConfig` + heavy ML deps, so we inspect the factory
module's source to assert the wire is present. It's a coarse-
grained pin, but it catches the actual failure mode (someone
removes the env-driven append block) cheaply and without paying
for the full trainer setup.

The companion :file:`test_runner_event_callback.py` covers the
callback's *behaviour* with ``httpx.MockTransport``; this file
covers the *factory-level integration* — that the production
factory adds the callback to the trainer's callback list when
``RYOTENKAI_RUNNER_URL`` is set in the trainer's env.
"""

from __future__ import annotations

import inspect

import pytest

# Importing the factory triggers ML-stack imports that aren't always
# available in the dev venv (no datasets package). Skip the whole
# module gracefully if the import fails — the wire-presence test then
# falls back to a static read of the source file.
factory_mod: object | None
try:
    from ryotenkai_pod.trainer.trainers import factory as factory_mod  # type: ignore[no-redef]
except Exception:
    factory_mod = None


def _factory_source() -> str:
    """Return the factory module source whether or not import succeeded."""
    if factory_mod is not None:
        return inspect.getsource(factory_mod)
    # Fallback: read the file directly. Same regex target either way.
    from pathlib import Path

    # Batch 6b: anchor on worktree root (tests/unit/pod/trainer/.. = 4 up)
    # to reach packages/pod/src/ryotenkai_pod/trainer/trainers/factory.py.
    # In legacy (packages/pod/tests/unit/trainer/) parents[4] = packages —
    # which was already broken; the fallback was dead-code because the
    # ``from ryotenkai_pod.trainer.trainers import factory`` import on
    # line 36 always succeeds in the full dev venv.
    here = Path(__file__).resolve().parents[4]
    target = here / "packages" / "pod" / "src" / "ryotenkai_pod" / "trainer" / "trainers" / "factory.py"
    return target.read_text(encoding="utf-8")


class TestFactoryWiring:
    """Pin the source-level invariants the wire depends on."""

    def test_factory_imports_runner_event_callback(self) -> None:
        src = _factory_source()
        assert "RunnerEventCallback" in src, (
            "TrainerFactory must import RunnerEventCallback "
            "(Phase 3.2 wiring)."
        )
        assert "RUNNER_URL_ENV" in src, (
            "TrainerFactory must reference RUNNER_URL_ENV — the "
            "env-driven activation gate."
        )

    def test_factory_appends_callback_when_env_set(self) -> None:
        # The append call is what binds the callback into the
        # trainer's callback list. A future refactor that splits the
        # factory should preserve this line (or move it to wherever
        # callbacks are aggregated; the test then fails LOUDLY and the
        # owner updates the pin).
        #
        # Phase 9.C — the factory now binds the constructor return value
        # to a local name (``runner_event_callback``) so the cancellation
        # callback can reuse its publish channel. Accept either:
        #   * ``callbacks.append(RunnerEventCallback())`` (pre-9.C)
        #   * ``runner_event_callback = RunnerEventCallback()``
        #     followed by ``callbacks.append(runner_event_callback)``
        src = _factory_source()
        legacy = "callbacks.append(RunnerEventCallback())" in src
        named = (
            "runner_event_callback = RunnerEventCallback()" in src
            and "callbacks.append(runner_event_callback)" in src
        )
        assert legacy or named, (
            "TrainerFactory must append RunnerEventCallback when "
            "RYOTENKAI_RUNNER_URL is set in the trainer env. Either form: "
            "callbacks.append(RunnerEventCallback()) OR "
            "runner_event_callback = RunnerEventCallback() + "
            "callbacks.append(runner_event_callback)."
        )

    def test_wire_is_env_gated_not_unconditional(self) -> None:
        """Defensive: the append must be inside an env check.

        An unconditional append would attach the callback even in
        local-mode runs (no runner attached) — the callback's own
        no-op short-circuit would handle it, but the registration
        contract documents env-driven activation, and unit tests of
        local trainings shouldn't see it in the callback list.
        """
        src = _factory_source()
        # Look for the env check in the same paragraph as the append.
        # ``str.find`` lets us assert ordering without a regex.
        env_check = src.find("os.environ.get(RUNNER_URL_ENV)")
        # ``import os as _os`` aliases os in the factory; tolerate either.
        if env_check < 0:
            env_check = src.find("environ.get(RUNNER_URL_ENV)")
        # Either pre-9.C inline append or 9.C named-binding append.
        append_call = src.find("callbacks.append(RunnerEventCallback())")
        if append_call < 0:
            append_call = src.find("callbacks.append(runner_event_callback)")
        assert env_check >= 0, "missing env-driven activation gate"
        assert append_call >= 0, "missing append call"
        # The append must come AFTER the env check (within ~400 chars
        # tolerates a comment block between them).
        assert append_call > env_check, (
            "callback append must be inside the env check, not before it"
        )
        assert (append_call - env_check) < 400, (
            "env check and append are too far apart — refactor likely "
            "broke the conditional"
        )


class TestTerminalCallbackWiring:
    """Post-cleanup regression: a single :class:`TerminalCallback`
    parametric on ``reason`` handles both the cancellation
    (``reason="cancel"``) and natural-completion (``reason="complete"``)
    paths. The cancel-reason callback must be inserted at index 0 of
    the callback list inside the same env-gated block."""

    def test_factory_imports_terminal_callback(self) -> None:
        src = _factory_source()
        assert "TerminalCallback" in src, (
            "TrainerFactory must import TerminalCallback "
            "(post-cleanup unified terminal-state wiring)."
        )

    def test_factory_inserts_terminal_callback_at_index_zero(self) -> None:
        """Insert at idx 0, not append -- must run BEFORE HF Trainer's
        auto-registered MLflow callback. Order is the contract: HF
        MLflow callback owns ``end_run()`` on ``on_train_end``; our
        callback flips ``control.should_save+should_training_stop``
        on ``on_step_end`` so HF observes it before it decides whether
        to keep stepping."""
        src = _factory_source()
        assert "callbacks.insert(" in src, (
            "TrainerFactory must insert TerminalCallback at index 0 "
            "(BEFORE HF MLflow callback)."
        )
        insert_idx = src.find("callbacks.insert(")
        assert insert_idx >= 0
        snippet = src[insert_idx:insert_idx + 250]
        assert "callbacks.insert(\n                0," in snippet or \
               "callbacks.insert(0," in snippet, (
            "callbacks.insert(...) must use index 0 as the first arg; "
            f"snippet was: {snippet!r}"
        )
        assert "TerminalCallback(" in snippet, (
            "insert(0, ...) must construct TerminalCallback"
        )
        assert 'reason="cancel"' in snippet, (
            "first TerminalCallback must use reason=\"cancel\""
        )

    def test_terminal_wire_is_env_gated(self) -> None:
        src = _factory_source()
        env_check = src.find("environ.get(RUNNER_URL_ENV)")
        terminal_insert = src.find("callbacks.insert(")
        assert env_check >= 0, "missing env-driven activation gate"
        assert terminal_insert >= 0, "missing TerminalCallback insert"
        assert terminal_insert > env_check, (
            "TerminalCallback insert must be inside the env check"
        )
        assert (terminal_insert - env_check) < 4000, (
            "env check and TerminalCallback insert are too far apart"
        )

    def test_terminal_wire_passes_event_publisher(self) -> None:
        """Factory must wire ``event_publisher=`` so the callback can
        emit ``cancellation_finalized`` / ``completion_finalized`` via
        the same runner publish channel as RunnerEventCallback."""
        src = _factory_source()
        assert "event_publisher" in src, (
            "TrainerFactory must pass event_publisher=... to TerminalCallback"
        )
        assert "flush_now=True" in src, (
            "Terminal event publisher must call _publish with flush_now=True"
        )

    def test_terminal_wire_after_runner_event_callback(self) -> None:
        src = _factory_source()
        runner_append = src.find("callbacks.append(RunnerEventCallback())")
        if runner_append < 0:
            runner_append = src.find("callbacks.append(runner_event_callback)")
        terminal_insert = src.find("callbacks.insert(")
        assert runner_append >= 0, "missing RunnerEventCallback append"
        assert runner_append < terminal_insert, (
            "RunnerEventCallback.append must come BEFORE the TerminalCallback inserts"
        )

    def test_factory_inserts_both_cancel_and_complete_terminal_callbacks(self) -> None:
        """Two TerminalCallback inserts -- one with reason="cancel" at
        index 0 and one with reason="complete" at index 1."""
        src = _factory_source()
        assert 'reason="cancel"' in src
        assert 'reason="complete"' in src


@pytest.mark.skipif(
    factory_mod is None,
    reason="factory module failed to import in this environment",
)
class TestFactoryRuntime:
    """Runtime-level regressions that don't need a full ``create()`` call."""

    def test_runner_event_callback_class_constructible_no_op(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Importing + constructing the callback must succeed in the
        no-op (env-unset) path. The factory imports it lazily —
        verify the lazy import target works without env."""
        monkeypatch.delenv("RYOTENKAI_RUNNER_URL", raising=False)
        from ryotenkai_pod.trainer.callbacks.runner_event_callback import (
            RUNNER_URL_ENV,
            RunnerEventCallback,
        )

        cb = RunnerEventCallback()
        # Disabled when env is unset — callback is a TrainerCallback
        # subclass and its ``_publish`` short-circuits.
        assert RUNNER_URL_ENV == "RYOTENKAI_RUNNER_URL"
        assert cb is not None
