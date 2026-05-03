"""Stage execution loop and related primitives.

The execution package owns "what happens after the launch is prepared":
the for-loop that drives every stage, per-stage bookkeeping (artifact
collectors, MLflow events, context propagation), outcome handlers
(failure/success/interrupt), and the four exception classes that can
terminate a pipeline mid-run.

Components here never own PipelineState — all state mutations go through
the injected :class:`AttemptController`. They do own *execution* concerns:
timing, logging, artifact flushing, and the decision of whether a stage
is eligible to run.
"""

from ryotenkai_control.pipeline.execution.restart_inspector import RestartPointsInspector
from ryotenkai_control.pipeline.execution.stage_execution_loop import StageExecutionLoop
from ryotenkai_control.pipeline.execution.stage_planner import StagePlanner, is_inference_runtime_healthy
from ryotenkai_control.pipeline.execution.stage_registry import StageRegistry

__all__ = [
    "RestartPointsInspector",
    "StageExecutionLoop",
    "StagePlanner",
    "StageRegistry",
    "is_inference_runtime_healthy",
]
