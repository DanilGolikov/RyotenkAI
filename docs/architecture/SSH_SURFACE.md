# SSH Surface — Mac↔pod boundary

**Status:** Active enforcement (Phase 3 of `2026-05-04-transport-unification-http-runtime-v2.md`).

This document is the **canonical allowlist** of SSH calls the
Mac orchestrator may issue against a pod. Anything not on this
list is a bug — and AST sentinel test
[test_no_runtime_ssh_exec_command](../../packages/control/tests/sentinel/test_no_runtime_ssh_exec_command.py)
will block CI on any new violation.

The plan's contract: *after the runner's `/healthz` returns 200,
the Mac MUST NOT issue any further SSH `exec_command` for
runtime concerns*. Diagnostics, log tails, file uploads, import
checks — all flow through HTTP+WS via `JobClient`.

## Allowlist (current state, post Phase 2)

### A. Bootstrap (Mac→pod, before runner is up)

These calls run **once** at the start of a pipeline run, before
uvicorn binds 8080. After `/healthz` is green they are not
re-issued.

| File | Function | Purpose |
|---|---|---|
| [runner_launcher.py](../../packages/control/src/ryotenkai_control/pipeline/stages/managers/deployment/runner_launcher.py) | `_launch_uvicorn` | `nohup python -m uvicorn ryotenkai_pod.runner.main:app …` |
| [code_syncer.py](../../packages/control/src/ryotenkai_control/pipeline/stages/managers/deployment/code_syncer.py) | rsync wrapper | `rsync packages/ → /workspace/` (incremental, <50 ms typical) |
| [dependency_installer.py](../../packages/control/src/ryotenkai_control/pipeline/stages/managers/deployment/dependency_installer.py) | `run_pip_install` / `run_uv_install` | `uv pip install <plugin-deps>` (per plan Q-NEW-1: kept as bootstrap-extended; one-shot, no streaming requirement) |
| [file_uploader.py](../../packages/control/src/ryotenkai_control/pipeline/stages/managers/deployment/file_uploader.py) | tar-pipe + SCP | **Transitional** — will migrate to `POST /api/v1/files/upload` in PR-3.3. Listed here for clarity but earmarked for removal. |
| [api/services/connection_test.py](../../packages/control/src/ryotenkai_control/api/services/connection_test.py) | connection probe | One-shot pre-bootstrap connectivity test triggered from the web UI's "Test connection" button. |

### B. Pod→Mac data pull (out of scope — separate boundary)

Plan §10 deliberately excluded pod→Mac large file pulls from
this transport unification. They remain SSH for now and may get
their own follow-up plan when scope permits.

| File | Function | Purpose |
|---|---|---|
| [hf_uploader.py](../../packages/control/src/ryotenkai_control/pipeline/stages/model_retriever/hf_uploader.py) | adapter download | SCP-stream the trained adapter weights from pod's `/workspace/output/` to `runs/<id>/...` |
| [metrics_buffer_retriever.py](../../packages/control/src/ryotenkai_control/pipeline/stages/model_retriever/metrics_buffer_retriever.py) | metrics buffer pull | `cat <buffer>` reads the pod-side training metrics JSONL |

### C. Provider boundary (Mac→cloud-API, not Mac→pod)

The RunPod GraphQL API is HTTP, but it lives behind the **provider
abstraction** (`ryotenkai_providers.runpod.runtime.lifecycle_client`)
and is conceptually a different boundary — Mac talks to RunPod's
control plane, not to a specific pod's runner.

| Concern | Endpoint | File |
|---|---|---|
| Pod lifecycle | `https://api.runpod.io/...` | `packages/providers/.../runpod/runtime/lifecycle_client.py` |

Document only — these calls are **NOT** in scope for the SSH
sentinel since they are HTTPS to a third-party endpoint.

## Forbidden

Anything else. Specifically:

* `ssh.exec_command("dmesg …")` — use `JobClient.get_diagnostics()`.
* `ssh.exec_command("nvidia-smi …")` — use
  `JobClient.get_resources()` for snapshot or
  `JobClient.get_diagnostics()` for full GPU table.
* `ssh.exec_command("stat -c %s …")` / `tail -c …` — use
  `JobClient.get_log_size()` / `JobClient.read_log()`.
* `ssh.exec_command("python … runtime_check.py …")` — use
  `JobClient.check_imports()`.
* `ssh.exec_command("cat /workspace/…")` for non-log files
  (config, dataset, plugin) — use `JobClient.read_log()` for
  log-shaped files; for arbitrary binary, prefer the future
  `GET /api/v1/files/{name}` (out of scope here).

## How to add a new SSH call (rare)

1. Justify in writing: why HTTP cannot serve this call. Bootstrap
   is the only legitimate reason today.
2. Open a PR adding the call to one of the allowlisted files in
   section A above. Adding a new file in section A requires a
   discussion in the PR description plus a corresponding entry
   in `bootstrap_allowlist.py`.
3. The AST sentinel test will pass — but reviewers should ask
   *why* the call can't be HTTP.

## Enforcement layers

1. **importlinter contract** (PR-3.2) — forbids importing
   `ryotenkai_shared.utils.ssh_client` outside the allowlisted
   modules.
2. **AST sentinel test** (this PR) — walks every Python file
   under `packages/control/...` and fails if it finds a
   `.exec_command(` call outside the allowlist.
3. **PR review** — humans catching what static analysis can't:
   the underlying intent.

Defence in depth — any one layer slipping should not let a
runtime SSH call sneak in.
