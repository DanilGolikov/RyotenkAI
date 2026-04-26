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
