"""Single-owner lifecycle for MLflow runs (Phase M2 redesign).

This subpackage owns the write-side of the MLflow integration on the
control plane. It replaces the four parallel close paths and three
parallel preflight probes of the legacy
:mod:`ryotenkai_control.pipeline.mlflow_attempt` package with one
narrow, idempotent surface:

* :class:`PreflightConnectivityCheck` -- fail-fast probe BEFORE any
  run is opened (mitigates the legacy "preflight opens a probe run
  to test connectivity" anti-pattern).
* :class:`ParentRunOpener` -- opens the root run and nested attempt
  runs, stamping all required ``ryotenkai.lineage.*`` /
  ``ryotenkai.lifecycle.*`` / ``ryotenkai.attempt.*`` tags exactly
  once.
* :class:`MlflowFinalizer` -- idempotent close path: tag check ->
  journal upload -> lifecycle tags -> ``set_terminated``. Never
  raises.
* :class:`RunLifecycleCoord` -- atexit/SIGTERM/SIGINT mutex'd owner
  of finalization; the ONLY caller of :meth:`MlflowFinalizer.finalize`
  for the long-running pipeline.

See ``docs/plans/vectorized-fluttering-mist.md`` (Phase M2) for design.
"""

from __future__ import annotations

from ryotenkai_control.pipeline.mlflow.lifecycle.coord import RunLifecycleCoord
from ryotenkai_control.pipeline.mlflow.lifecycle.finalizer import MlflowFinalizer
from ryotenkai_control.pipeline.mlflow.lifecycle.opener import ParentRunOpener
from ryotenkai_control.pipeline.mlflow.lifecycle.preflight import (
    PreflightConnectivityCheck,
)

__all__ = [
    "MlflowFinalizer",
    "ParentRunOpener",
    "PreflightConnectivityCheck",
    "RunLifecycleCoord",
]
