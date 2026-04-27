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
    from src.training.trainers import factory as factory_mod  # type: ignore[no-redef]
except Exception:  # noqa: BLE001 — env-specific import issues
    factory_mod = None


def _factory_source() -> str:
    """Return the factory module source whether or not import succeeded."""
    if factory_mod is not None:
        return inspect.getsource(factory_mod)
    # Fallback: read the file directly. Same regex target either way.
    from pathlib import Path

    here = Path(__file__).resolve().parents[4]
    target = here / "src" / "training" / "trainers" / "factory.py"
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
        # The exact append call is what binds the callback into the
        # trainer's callback list. A future refactor that splits the
        # factory should preserve this line (or move it to wherever
        # callbacks are aggregated; the test then fails LOUDLY and the
        # owner updates the pin).
        src = _factory_source()
        assert "callbacks.append(RunnerEventCallback())" in src, (
            "TrainerFactory must append RunnerEventCallback() when "
            "RYOTENKAI_RUNNER_URL is set in the trainer env."
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
        append_call = src.find("callbacks.append(RunnerEventCallback())")
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


class TestPhase9ACancellationCallbackWiring:
    """Phase 9.A regression: CancellationCallback must be inserted at
    index 0 of the callback list inside the same env-gated block."""

    def test_factory_imports_cancellation_callback(self) -> None:
        src = _factory_source()
        assert "CancellationCallback" in src, (
            "TrainerFactory must import CancellationCallback "
            "(Phase 9.A wiring)."
        )

    def test_factory_inserts_cancellation_callback_at_index_zero(self) -> None:
        """Insert at idx 0, not append — must run BEFORE HF Trainer's
        auto-registered MLflow callback. Order is the contract: HF
        MLflow callback owns ``end_run()`` on ``on_train_end``; our
        callback flips ``control.should_save+should_training_stop``
        on ``on_step_end`` so HF observes it before it decides whether
        to keep stepping."""
        src = _factory_source()
        # Match either the 9.A signature (no kwargs) or the 9.B
        # signature (passes mlflow_manager). The pin is "insert(0,
        # CancellationCallback(...))" with no positional args before
        # the kwarg. ``CancellationCallback(`` immediately after
        # ``insert(0,`` and ``)`` closing within ~120 chars is enough.
        assert "callbacks.insert(" in src, (
            "TrainerFactory must insert CancellationCallback at index 0 "
            "(BEFORE HF MLflow callback). Phase 9.1.E ordering decision."
        )
        # Find the actual insert call and confirm the index is 0.
        insert_idx = src.find("callbacks.insert(")
        assert insert_idx >= 0
        # Read ahead and ensure the first arg is exactly ``0``.
        snippet = src[insert_idx:insert_idx + 120]
        assert "callbacks.insert(\n                0," in snippet or \
               "callbacks.insert(0," in snippet, (
            "callbacks.insert(...) must use index 0 as the first arg; "
            f"snippet was: {snippet!r}"
        )
        assert "CancellationCallback(" in snippet, (
            "insert(0, ...) must construct CancellationCallback"
        )

    def test_cancellation_wire_is_env_gated(self) -> None:
        """Same env gate as RunnerEventCallback — CancellationCallback
        only runs inside the in-pod runner where stop signals are
        meaningful. Local-mode trainings (no runner attached) skip
        both callbacks."""
        src = _factory_source()
        env_check = src.find("environ.get(RUNNER_URL_ENV)")
        cancel_insert = src.find("callbacks.insert(")
        assert env_check >= 0, "missing env-driven activation gate"
        assert cancel_insert >= 0, "missing CancellationCallback insert"
        # Insert MUST come AFTER the env check (inside the conditional).
        assert cancel_insert > env_check, (
            "CancellationCallback insert must be inside the env check"
        )
        # Wider window than the RunnerEventCallback test because Phase 9.A
        # added ~30 lines of explanatory comments between the env check
        # and the cancel-insert. ~2500 chars still tolerates a few more
        # comment blocks before becoming a real problem.
        assert (cancel_insert - env_check) < 2500, (
            "env check and CancellationCallback insert are too far "
            "apart — likely a refactor broke the conditional grouping"
        )

    def test_cancellation_wire_after_runner_event_callback(self) -> None:
        """Both callbacks live in the same env-gated block. The order
        in source (Runner first, then Cancellation) is intentional:

        - ``RunnerEventCallback.append`` happens at the existing
          end-of-list position so its event hooks fire AFTER HF
          completes a step.
        - ``CancellationCallback`` is inserted at index 0 so its
          on_step_end hook fires BEFORE the HF MLflow callback
          (which auto-registers at the end of the list).

        If a future refactor swaps the order in source, the
        runtime semantic changes. Pin both spots."""
        src = _factory_source()
        runner_append = src.find("callbacks.append(RunnerEventCallback())")
        cancel_insert = src.find("callbacks.insert(")
        assert runner_append < cancel_insert, (
            "RunnerEventCallback.append must come BEFORE "
            "callbacks.insert(0, CancellationCallback(...)) in source order"
        )


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
        from src.training.callbacks.runner_event_callback import (
            RUNNER_URL_ENV,
            RunnerEventCallback,
        )

        cb = RunnerEventCallback()
        # Disabled when env is unset — callback is a TrainerCallback
        # subclass and its ``_publish`` short-circuits.
        assert RUNNER_URL_ENV == "RYOTENKAI_RUNNER_URL"
        assert cb is not None
