"""``make_monitor_with_log_manager`` — factory for :class:`TrainingMonitor`.

Post Phase-B / Phase 6.3b the :class:`TrainingMonitor` constructor only
takes ``(config, secrets, *, emitter)``; the rich state (``_provider``,
``_client``, ``_log_manager`` etc.) is populated inside :meth:`execute`
from the pipeline context that :class:`TrainingLauncher` left behind.
Phase 4 (event-system unification, 2026-05-16) — the legacy
``callbacks`` keyword was removed in favour of an optional
:class:`IEventEmitter`.

Unit tests that exercise helper methods (``_build_log_manager_from_context``,
``_handle_trainer_exited``, ``_recover_pod_if_needed``, …) need a monitor
instance whose ``_provider``/``_client``/``_log_manager`` already have
useful values without going through a full ``execute()`` round-trip.
The pre-Phase-B greenfield tests papered over this with an inline
``_make_monitor`` helper that bypassed ``__init__`` via ``__new__``;
that helper drifted (missing ``_provider``/``_client``) when the
class grew new fields.

This factory replaces that helper. It calls the *real* :meth:`__init__`
(stub :class:`PipelineConfig` is enough — the constructor doesn't read
nested fields) and then applies caller-provided overrides on top, so
production attribute layout is mirrored 1:1.

Pattern follows :mod:`tests._fakes.provider_context` and
:mod:`tests._fakes.dataset_source`: a thin factory returning a real
instance, not a ``MagicMock(spec=...)``.

Usage::

    from tests._fakes.training_monitor import make_monitor_with_log_manager

    monitor = make_monitor_with_log_manager(
        _client=fake_job_client,
        _provider=fake_provider,
        _resource_id="pod-1",
    )
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any


def _load_training_monitor_module() -> Any:
    """Import the monitor module directly to avoid dragging the heavy
    ``ryotenkai_control.pipeline.stages`` package init.

    Mirrors the loader in
    ``tests/unit/control/pipeline/test_training_monitor_v2.py``.
    Subsequent calls re-use the cached module.
    """
    cached = sys.modules.get("ryotenkai_monitor_test")
    if cached is not None:
        return cached
    here = Path(__file__).resolve()
    repo_root = here.parents[2]
    src_path = (
        repo_root
        / "packages"
        / "control"
        / "src"
        / "ryotenkai_control"
        / "pipeline"
        / "stages"
        / "training_monitor.py"
    )
    spec = importlib.util.spec_from_file_location(
        "ryotenkai_monitor_test", str(src_path),
    )
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules["ryotenkai_monitor_test"] = mod
    spec.loader.exec_module(mod)
    return mod


def _stub_pipeline_config() -> Any:
    """Minimal stand-in for :class:`PipelineConfig`.

    :class:`PipelineStage.__init__` only stores ``self.config = config``;
    :class:`TrainingMonitor.__init__` doesn't dereference any nested
    field. A :class:`SimpleNamespace` with an empty ``.training``
    namespace is enough for tests that don't touch config-driven paths.
    """
    return SimpleNamespace(training=SimpleNamespace())


def make_monitor_with_log_manager(**overrides: Any) -> Any:
    """Build a real :class:`TrainingMonitor` for unit tests.

    The factory:

    1. Calls the production :meth:`TrainingMonitor.__init__` with a
       stub config (no secrets, no emitter unless overridden) so the
       attribute layout matches production exactly.
    2. Lets callers override any attribute via ``**overrides`` — typical
       overrides are ``_client`` (a fake :class:`JobClient`),
       ``_provider`` (an :class:`IRecoveryProbeProvider` impl),
       ``_log_manager`` / ``_runner_log_manager``
       (:class:`LogFetcher` fakes), ``_resource_id``, ``_secrets``,
       and ``_emitter``.

    The returned instance is a real ``TrainingMonitor`` so production
    code paths (``_dispatch_event``, ``_handle_trainer_exited``,
    ``_recover_pod_if_needed``, …) execute against it untouched.

    Special-cased kwargs:

    * ``config`` — replaces the default stub :class:`PipelineConfig`.
    * ``secrets`` — passed through to ``__init__``.
    * ``emitter`` — passed through to ``__init__`` as the optional
      :class:`IEventEmitter` (Phase 4 replacement for ``callbacks``).

    Any other key is set on the instance after construction via
    ``setattr``. Unknown attribute names are accepted: production reads
    attributes by name, so a typo would manifest as a test failure
    rather than a factory error.
    """
    mod = _load_training_monitor_module()
    monitor_cls = mod.TrainingMonitor

    config = overrides.pop("config", None) or _stub_pipeline_config()
    secrets = overrides.pop("secrets", None)
    emitter = overrides.pop("emitter", None)

    monitor = monitor_cls(config=config, secrets=secrets, emitter=emitter)

    for key, value in overrides.items():
        setattr(monitor, key, value)
    return monitor


__all__ = ["make_monitor_with_log_manager"]
