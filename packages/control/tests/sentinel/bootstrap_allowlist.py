"""Allowlist of Mac-side modules permitted to issue SSH ``exec_command``.

Single source of truth — kept in its own module (not inline in the
sentinel test) so changes require an explicit PR review against
this file with a written justification (RP22 mitigation —
allowlist staleness).

Each entry MUST carry a ``# why`` comment explaining the
bootstrap-or-out-of-scope rationale. The reviewer's job is to
push back when an SSH addition isn't a real bootstrap or pod→Mac
data-pull concern.

Companion docs: :doc:`docs/architecture/SSH_SURFACE.md`.
"""

from __future__ import annotations

# A. Mac→pod bootstrap (before /healthz returns 200).
BOOTSTRAP_MODULES: frozenset[str] = frozenset({
    # nohup uvicorn launch — the runner cannot start itself.
    "ryotenkai_control.pipeline.stages.managers.deployment.runner_launcher",
    # rsync orchestration + post-sync verify hook (the verify hook
    # itself migrates to HTTP via PR-3.3 → PR-2.5 endpoint).
    "ryotenkai_control.pipeline.stages.managers.deployment.code_syncer",
    # uv pip install for plugin deps. Per plan Q-NEW-1 kept as
    # bootstrap-extended — one-shot, no streaming requirement.
    "ryotenkai_control.pipeline.stages.managers.deployment.dependency_installer",
    # tar-pipe + SCP — TRANSITIONAL. PR-3.3 migrates the call
    # sites to ``JobClient.upload_file`` (PR-2.4 endpoint) and this
    # entry should be removed.
    "ryotenkai_control.pipeline.stages.managers.deployment.file_uploader",
    # Pre-bootstrap "Test connection" probe — runs from the web UI
    # before any pipeline starts. Cannot use JobClient because the
    # tunnel doesn't exist yet.
    "ryotenkai_control.api.services.connection_test",
})

# B. Pod→Mac data pull (out of scope per plan §10 — separate boundary).
DATA_PULL_MODULES: frozenset[str] = frozenset({
    # Adapter download — SCP-stream of trained model weights from
    # the pod's /workspace/output/. May get its own HTTP plan later.
    "ryotenkai_control.pipeline.stages.model_retriever.hf_uploader",
    # Metrics buffer JSONL pull. Eligible for migration to a
    # generalised /api/v1/files/{name} endpoint in a future PR.
    "ryotenkai_control.pipeline.stages.model_retriever.metrics_buffer_retriever",
})

ALLOWLIST: frozenset[str] = BOOTSTRAP_MODULES | DATA_PULL_MODULES


__all__ = [
    "ALLOWLIST",
    "BOOTSTRAP_MODULES",
    "DATA_PULL_MODULES",
]
