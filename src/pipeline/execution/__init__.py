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

from src.pipeline.execution.stage_execution_loop import StageExecutionLoop

__all__ = ["StageExecutionLoop"]
