"""Immutable MLflow run handle (value object).

Carried across the system after a run is opened or adopted. Replaces
loose ``Any``-typed run references and the ``mlflow.entities.Run``
leakage from the wide ``IMLflowManager`` Protocol.

Frozen dataclass — safe to share across threads and to embed in
event payloads. Hashable by ``run_id``.

The ``status`` field is a :class:`~.protocols.RunStatus` value at
the moment of construction; it does NOT live-update from the
server. Use :class:`~.protocols.IRunQuery.get_run` to refresh.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ryotenkai_shared.infrastructure.mlflow.protocols import RunStatus


@dataclass(frozen=True, slots=True)
class RunHandle:
    """Immutable handle to an MLflow run.

    Fields:
        run_id:         MLflow-assigned UUID.
        experiment_id:  Parent experiment id.
        parent_run_id:  ``None`` for top-level runs; set for nested
                        children. Useful for journal context tags
                        and lineage queries.
        tracking_uri:   Resolved URI of the server the run was
                        created against. Frozen at open-time so
                        readers (deletion, finalization) know where
                        to look even if process-wide URI changes
                        later.
        status:         :class:`~.protocols.RunStatus` at construction.
                        Treat as a snapshot, NOT live state.
    """

    run_id: str
    experiment_id: str
    parent_run_id: str | None
    tracking_uri: str
    status: RunStatus

    def __hash__(self) -> int:
        return hash(self.run_id)


__all__ = ["RunHandle"]
