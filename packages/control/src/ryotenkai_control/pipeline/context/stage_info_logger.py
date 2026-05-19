"""Post-stage MLflow info logging (legacy shim).

Historical behaviour: after each stage completed, the orchestrator called
:meth:`StageInfoLogger.log` which emitted stage-specific MLflow events /
params / metrics via the wide ``IMLflowManager``.

After the wide-manager retirement, every call site received
``mlflow_manager=None`` so the per-stage handlers were dead. The class is
kept as a no-op shim because the bootstrap still wires an instance into
:class:`StageExecutionLoop`'s ctor (``stage_info_logger=``) — keeping the
shim avoids cascading constructor changes across the bootstrap and the
loop's ``__slots__``.

Stage-specific MLflow logging is now performed via narrow protocols
inside each stage (e.g. ``MlflowTransport`` for params / tags / metrics).
"""

from __future__ import annotations

from typing import Any


class StageInfoLogger:
    """No-op shim retained for bootstrap-wiring compatibility.

    All ``log(...)`` invocations are silent — the legacy wide-manager
    surface is retired and per-stage logging now happens via narrow
    protocols inside individual stages.
    """

    def log(
        self,
        *,
        context: dict[str, Any],
        stage_name: str,
    ) -> None:
        """No-op; retained so existing bootstrap call sites do not break."""
        # Intentionally empty — see module docstring.
        del context, stage_name


__all__ = ["StageInfoLogger"]
