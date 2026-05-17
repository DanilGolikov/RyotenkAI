"""Post-Phase-10 (A-СРЕД) — per-plugin typed events for the validator runner.

Pins the visibility-gap close around
``ryotenkai_control.pipeline.stages.dataset_validator.plugin_runner.PluginRunner``:
each plugin invocation now publishes a typed envelope trio (started,
completed | failed) on the unified timeline, mirroring the evaluation
runner's per-plugin emission.

Coverage:

* **Positive** — 3 passing plugins emit 6 envelopes
  (3 ``plugin_started`` + 3 ``plugin_completed``); per-plugin ordering
  preserved.
* **Negative** — a plugin that throws emits ``plugin_started`` then
  ``plugin_failed`` carrying ``error_type`` / ``message`` /
  truncated traceback excerpt.
* **Invariants** — per-plugin ordering: every ``plugin_started`` is
  immediately followed by its ``plugin_completed | plugin_failed``
  match; no orphaned started events. plugin_name surfaces in payload.
* **Dependency errors** — emitter is ``None`` → runner stays silent
  but still executes plugin logic (legacy / standalone fixtures).
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from ryotenkai_control.data.validation.base import ValidationResult
from ryotenkai_control.pipeline.stages.dataset_validator.plugin_runner import PluginRunner
from ryotenkai_shared.errors import DatasetValidationFailedError

from tests._fakes.event_emitter import FakeEventEmitter


pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# Plugin doubles — mirror the shape exercised by the existing
# test_plugin_runner.py, kept inline so the test files stay independent.
# ---------------------------------------------------------------------------


class _OkPlugin:
    """A plugin that returns ``passed=True`` with sample-count metric."""

    def __init__(self, name: str = "ok") -> None:
        self.name = name
        self.version = "1.0.0"
        self.params: dict = {}
        self.thresholds: dict = {}

    def get_description(self) -> str:
        return f"description-of-{self.name}"

    def validate(self, dataset):  # noqa: ARG002 — dataset unused
        return ValidationResult(
            plugin_name=self.name,
            passed=True,
            params={},
            thresholds={},
            metrics={"sample_count": 100.0},
            warnings=[],
            errors=[],
            execution_time_ms=1.0,
        )

    def get_recommendations(self, result):  # noqa: ARG002
        return []


class _CrashPlugin:
    """A plugin whose ``validate`` raises an exception."""

    def __init__(self, exc_factory) -> None:
        self.name = "crashy"
        self.version = "0.1"
        self.params: dict = {}
        self.thresholds: dict = {}
        self._exc_factory = exc_factory

    def get_description(self) -> str:
        return "crashes for testing"

    def validate(self, dataset):  # noqa: ARG002
        raise self._exc_factory()

    def get_recommendations(self, result):  # noqa: ARG002
        return []


def _plugin_tuple(plugin, plugin_id: str | None = None):
    """Build a ``PluginTuple`` shape matching ``PluginLoader`` output."""
    return (plugin_id or plugin.name, plugin.name, plugin, {"train"})


def _dataset_config_no_critical():
    """Return a dataset-config double whose validations.critical_failures == 0
    (i.e. fail-soft mode — the runner finishes the loop regardless)."""
    return SimpleNamespace(validations=SimpleNamespace(critical_failures=0))


# ---------------------------------------------------------------------------
# 1. Positive
# ---------------------------------------------------------------------------


class TestPositive:
    def test_three_passing_plugins_emit_six_envelopes(self) -> None:
        emitter = FakeEventEmitter()
        runner = PluginRunner(emitter=emitter, run_id="r1")
        plugins = [
            _plugin_tuple(_OkPlugin("p1")),
            _plugin_tuple(_OkPlugin("p2")),
            _plugin_tuple(_OkPlugin("p3")),
        ]
        runner.run(
            dataset_name="ds",
            dataset_path="/tmp/d",
            dataset=object(),  # plugin ignores it
            dataset_config=_dataset_config_no_critical(),
            plugins=plugins,
            split_name="train",
        )

        kinds = [ev.kind for ev in emitter.emitted]
        assert kinds == [
            "ryotenkai.control.dataset.validation_plugin_started",
            "ryotenkai.control.dataset.validation_plugin_completed",
            "ryotenkai.control.dataset.validation_plugin_started",
            "ryotenkai.control.dataset.validation_plugin_completed",
            "ryotenkai.control.dataset.validation_plugin_started",
            "ryotenkai.control.dataset.validation_plugin_completed",
        ]
        # plugin_name surfaces on every envelope.
        plugin_names = [ev.payload.plugin_name for ev in emitter.emitted]
        assert plugin_names == ["p1", "p1", "p2", "p2", "p3", "p3"]

    def test_completed_carries_duration_and_dataset_path(self) -> None:
        emitter = FakeEventEmitter()
        runner = PluginRunner(emitter=emitter, run_id="r1")
        runner.run(
            dataset_name="ds",
            dataset_path="/tmp/d/train.jsonl",
            dataset=object(),
            dataset_config=_dataset_config_no_critical(),
            plugins=[_plugin_tuple(_OkPlugin("p1"))],
            split_name="train",
        )
        started, completed = emitter.emitted
        assert started.payload.dataset_path == "/tmp/d/train.jsonl"
        assert started.payload.plugin_version == "1.0.0"
        assert completed.payload.duration_s >= 0.0


# ---------------------------------------------------------------------------
# 2. Negative
# ---------------------------------------------------------------------------


class TestNegative:
    def test_plugin_crash_emits_started_then_failed(self) -> None:
        emitter = FakeEventEmitter()
        runner = PluginRunner(emitter=emitter, run_id="r1")
        plugins = [_plugin_tuple(_CrashPlugin(lambda: RuntimeError("boom")))]

        # Crashes count as a failed plugin. With critical_failures=0
        # the runner finishes the loop and raises ``DatasetValidationFailedError``
        # at the end (failures collected into the build-result branch).
        with pytest.raises(DatasetValidationFailedError):
            runner.run(
                dataset_name="ds",
                dataset_path="/tmp/d",
                dataset=object(),
                dataset_config=_dataset_config_no_critical(),
                plugins=plugins,
                split_name="train",
            )

        kinds = [ev.kind for ev in emitter.emitted]
        assert kinds == [
            "ryotenkai.control.dataset.validation_plugin_started",
            "ryotenkai.control.dataset.validation_plugin_failed",
        ]
        failed_ev = emitter.emitted[1]
        assert failed_ev.payload.error_type == "RuntimeError"
        assert "boom" in failed_ev.payload.message
        # Traceback excerpt is bounded and non-empty.
        assert failed_ev.payload.traceback_excerpt
        assert len(failed_ev.payload.traceback_excerpt) <= 2048

    def test_traceback_excerpt_is_truncated_for_huge_traceback(self) -> None:
        # Deeply nested call → long traceback → must be clipped.
        emitter = FakeEventEmitter()
        runner = PluginRunner(emitter=emitter, run_id="r1")

        def _make_deep_exc() -> Exception:
            def recurse(depth: int) -> None:
                if depth <= 0:
                    raise ValueError("X" * 5000)
                recurse(depth - 1)

            try:
                recurse(40)
            except Exception as e:
                return e
            return ValueError("unreachable")

        plugins = [_plugin_tuple(_CrashPlugin(_make_deep_exc))]
        with pytest.raises(DatasetValidationFailedError):
            runner.run(
                dataset_name="ds",
                dataset_path="/tmp/d",
                dataset=object(),
                dataset_config=_dataset_config_no_critical(),
                plugins=plugins,
                split_name="train",
            )
        failed_ev = emitter.emitted[-1]
        assert len(failed_ev.payload.traceback_excerpt) <= 2048
        # The truncation marker is at the tail when clipping happened.
        if len(failed_ev.payload.traceback_excerpt) == 2048:
            assert failed_ev.payload.traceback_excerpt.endswith("…[truncated]")


# ---------------------------------------------------------------------------
# 3. Invariants
# ---------------------------------------------------------------------------


class TestInvariants:
    def test_every_started_is_followed_by_a_terminal_envelope(self) -> None:
        # Mix of pass + crash plugins. The terminal envelope is either
        # ``plugin_completed`` (validate returned) or ``plugin_failed``
        # (validate threw). Every plugin contributes exactly two
        # envelopes — no orphaned ``plugin_started``.
        emitter = FakeEventEmitter()
        runner = PluginRunner(emitter=emitter, run_id="r1")
        plugins = [
            _plugin_tuple(_OkPlugin("p1")),
            _plugin_tuple(_CrashPlugin(lambda: ValueError("nope"))),
            _plugin_tuple(_OkPlugin("p3")),
        ]
        with pytest.raises(DatasetValidationFailedError):
            runner.run(
                dataset_name="ds",
                dataset_path="/tmp/d",
                dataset=object(),
                dataset_config=_dataset_config_no_critical(),
                plugins=plugins,
                split_name="train",
            )

        # Walk pairs and assert per-plugin ordering.
        kinds = [ev.kind for ev in emitter.emitted]
        # Expect: started, completed (p1), started, failed (crashy),
        #         started, completed (p3) — six envelopes total.
        assert len(kinds) == 6
        # Pairwise sanity: positions 0/2/4 must be ``plugin_started``;
        # positions 1/3/5 must be ``plugin_completed`` or
        # ``plugin_failed``.
        for i in (0, 2, 4):
            assert kinds[i] == "ryotenkai.control.dataset.validation_plugin_started"
        for i in (1, 3, 5):
            assert kinds[i] in {
                "ryotenkai.control.dataset.validation_plugin_completed",
                "ryotenkai.control.dataset.validation_plugin_failed",
            }
        # Per-plugin name pairs match.
        names = [ev.payload.plugin_name for ev in emitter.emitted]
        assert names == ["p1", "p1", "crashy", "crashy", "p3", "p3"]

    def test_run_id_is_forwarded(self) -> None:
        emitter = FakeEventEmitter()
        runner = PluginRunner(emitter=emitter, run_id="my-run-42")
        runner.run(
            dataset_name="ds",
            dataset_path="/tmp/d",
            dataset=object(),
            dataset_config=_dataset_config_no_critical(),
            plugins=[_plugin_tuple(_OkPlugin("p1"))],
            split_name="train",
        )
        for ev in emitter.emitted:
            assert ev.run_id == "my-run-42"


# ---------------------------------------------------------------------------
# 4. Dependency errors
# ---------------------------------------------------------------------------


class TestDependencyErrors:
    def test_no_emitter_silently_skips_emission(self) -> None:
        # Legacy fixture: PluginRunner() with no emitter wired.
        # ``run`` must still execute the plugins and return the
        # success dict — only emission is skipped.
        runner = PluginRunner()  # emitter omitted -> None
        result = runner.run(
            dataset_name="ds",
            dataset_path="/tmp/d",
            dataset=object(),
            dataset_config=_dataset_config_no_critical(),
            plugins=[_plugin_tuple(_OkPlugin("p1"))],
            split_name="train",
        )
        # Sanity: plugin executed (metrics flowed through).
        assert any(k.endswith(".p1.sample_count") for k in result)
