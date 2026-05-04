"""Canonical log filenames — single source of truth.

This module owns the **string literals** for every log file the pipeline
produces. Both pod-side (:mod:`src.utils.pod_layout`) and Mac-side
(:mod:`src.utils.logs_layout`) layouts import their filenames from
here, so a rename happens in exactly one place and the LogFetcher HTTP read
mapping (1:1 by filename) cannot drift.

Why a separate module: ``pod_layout`` and ``logs_layout`` describe
*structure* (which directories exist, where things live), while this
module describes *identity* (what individual files are called). The two
concerns evolve at different rates — a new artefact needs both a
filename here AND placement in one of the layouts, but a structural
refactor (e.g. ``state/`` → ``runtime/state/``) only changes layout.

Audit invariant: every literal log filename in the codebase MUST come
from a constant defined here. Hardcoded ``"runner.log"`` strings in
business logic are a bug — replace them with the constant.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Pod-side filenames (written on the remote training pod)
# ---------------------------------------------------------------------------

#: Pod-side trainer subprocess stdout/stderr ground-truth.
#: Single writer = the runner's :class:`Supervisor` pump. Captures
#: regular Python tracebacks AND native faulthandler dumps (the
#: trainer activates ``faulthandler.enable()`` which writes to stderr,
#: which the pump tees into this file).
#: Pulled by ``LogFetcher`` (HTTP) to ``<attempt>/logs/<this name>`` on Mac.
TRAINER_STDIO_LOG = "trainer.stdio.log"

#: Pod-side uvicorn / FastAPI runner stdout. Captured by the
#: Mac-orchestrated launch command's ``nohup ... >> runner.log 2>&1``
#: redirect — survives pre-import errors (ImportError, SyntaxError)
#: that fire BEFORE the runner ever spawns the trainer.
RUNNER_LOG = "runner.log"

# ---------------------------------------------------------------------------
# Mac-side filenames (written by the Mac orchestrator)
# ---------------------------------------------------------------------------

#: Aggregated stream of every stage's logger output for one attempt.
#: Lives at ``<attempt>/logs/<this name>``. The orchestrator's logging
#: setup writes here directly; not pulled from a remote host.
PIPELINE_LOG = "pipeline.log"

#: Suffix for per-stage log files (``<slug>.log``). Stage slug is
#: derived from the stage name via :func:`src.utils.logs_layout._slugify`.
STAGE_LOG_SUFFIX = ".log"

# ---------------------------------------------------------------------------
# FSM persistence (runner-internal state)
# ---------------------------------------------------------------------------

#: Latest atomic snapshot of the JobLifecycleFSM state (pod-side).
#: Located at ``<workspace>/state/<this name>``.
FSM_STATE_JSON = "job.json"

#: Append-only audit log of every FSM transition (pod-side).
#: Located at ``<workspace>/state/<this name>``.
FSM_STATE_JSONL = "job.jsonl"

# ---------------------------------------------------------------------------
# Pipeline config (uploaded to pod)
# ---------------------------------------------------------------------------

#: Pipeline config YAML uploaded by FileUploader to
#: ``<workspace>/config/<this name>`` on the pod.
PIPELINE_CONFIG_FILE = "pipeline_config.yaml"


__all__ = [
    "FSM_STATE_JSON",
    "FSM_STATE_JSONL",
    "PIPELINE_CONFIG_FILE",
    "PIPELINE_LOG",
    "RUNNER_LOG",
    "STAGE_LOG_SUFFIX",
    "TRAINER_STDIO_LOG",
]
