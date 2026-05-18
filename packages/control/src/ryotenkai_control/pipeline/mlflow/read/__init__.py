"""Read-path components for control-plane MLflow interactions.

This package contains the new Phase M3.A read-path implementations:

* :class:`~.client.MlflowReadClient` — concrete
  :class:`~ryotenkai_shared.infrastructure.mlflow.protocols.IRunQuery`
  implementation. Replaces ad-hoc ``MlflowClient()`` constructions at
  the six legacy callsites (rewired in M3.B).
* :class:`~.tree_walker.RunTreeWalker` — single BFS walker over the
  run hierarchy. Subsumes the three independent traversals scattered
  across the existing reports/deletion/summary_reporter code.

Both components are additive in M3.A — the legacy callsites continue
to use ``MlflowClient()`` directly until M3.B re-wires them.
"""

from __future__ import annotations

from ryotenkai_control.pipeline.mlflow.read.client import MlflowReadClient
from ryotenkai_control.pipeline.mlflow.read.tree_walker import (
    RunNode,
    RunTreeWalker,
)

__all__ = [
    "MlflowReadClient",
    "RunNode",
    "RunTreeWalker",
]
