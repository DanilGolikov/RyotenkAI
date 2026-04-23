# ADR updates pending (Repowise tool was unavailable at implementation time)

Run these through `update_decision_records` once the Repowise MCP server responds again (tool returned `Unknown tool: update_decision_records`, likely a post-pull sync issue).

## 1. CREATE new active ADR

```
action: create
title: TUI removed — web is the canonical interactive UI
status: active
context: |
  The TUI (src/tui/, ~1.2k LOC + 11 tests ~2.1k LOC) was a sibling client to
  the pipeline state store alongside CLI and the FastAPI backend. With the
  web UI now primary, the TUI became dead weight: every pipeline-layer
  refactor cost double work (TUI + web), and the `textual` dependency
  pulled in extra wheel weight for no user-facing benefit.
decision: |
  Hard-remove the module: deleted src/tui/ and src/tests/unit/tui/; removed
  the `ryotenkai tui` Typer command from src/main.py; dropped the three
  tui-specific tests from src/tests/unit/test_main_cli.py; removed
  `textual[syntax]>=8.1.0,<9.0.0` from pyproject.toml; cleaned all
  doc mentions (README.md TUI section + 3 screenshots, CONTRIBUTING.md,
  run.sh --tui flag and help text, docs/web-ui.md CLI/TUI coexistence
  section). Historic hash-named log file `tui_launch.log` on disk is kept
  as-is to preserve backward compatibility with existing runs.
rationale: |
  Web is now the primary UI per user directive. Keeping a deprecated
  TUI alongside the web backend doubled maintenance cost on every
  pipeline-layer refactor (ConfigBuilder, PipelineStateStore, launch
  semantics) and forced the `textual` dependency on every install.
alternatives:
  - "Soft-disable (keep src/tui/, drop CLI entry): rejected — leaves ~3k LOC of dead code + textual dep for no benefit, and invites drift."
  - "Rename tui_launch.log to launch.log: rejected — breaking for existing runs on disk; symbolic string only, not worth the migration."
consequences:
  - "~3.3k LOC + 3 screenshots + 1 python dep gone; developer install is lighter and wheel/lockfile shrinks."
  - "Single interactive UI to maintain — all UX work converges on the React app."
  - "`tui_launch.log` filename on disk is historical; a follow-up rename to `launch.log` is possible when there is a state/migration mechanism."
affected_files:
  - "src/main.py"
  - "src/tests/unit/test_main_cli.py"
  - "pyproject.toml"
  - "README.md"
  - "CONTRIBUTING.md"
  - "run.sh"
  - "docs/web-ui.md"
  - "src/api/__init__.py"
  - "src/api/main.py"
  - "src/api/services/run_service.py"
  - "src/pipeline/deletion.py"
  - "src/pipeline/live_logs.py"
  - "src/pipeline/presentation.py"
  - "src/pipeline/state/queries.py"
  - "src/pipeline/state/cache.py"
  - "src/pipeline/launch/runtime.py"
  - "src/tests/unit/pipeline/state/test_cache.py"
tags: ["architecture", "ui", "deprecation", "cleanup", "web"]
```

## 2. CREATE new active ADR

```
action: create
title: run.lock lifecycle stays inside PipelineStateStore (sibling-invariant hardening)
status: active
context: |
  Invariant #1 (from src/pipeline/state/run_lock_guard.py) says only
  src/pipeline/state/store.py owns run.lock lifecycle: acquire_run_lock()
  creates, PipelineRunLock.release() / RunLockGuard.__exit__ removes. Audit
  surfaced that src/api/services/launch_service.py:interrupt() was doing
  `(run_dir / "run.lock").unlink()` directly for stale-lock cleanup, bypassing
  the store, hardcoding the filename, and without a pid content-match.
decision: |
  Added two helpers to src/pipeline/state/store.py:
    - read_lock_pid(lock_path: Path) -> int | None
    - remove_stale_lock(lock_path: Path, *, expected_pid: int) -> bool
  remove_stale_lock re-reads the pid= line right before unlink and skips
  the removal if another process has taken ownership. Both are re-exported
  from src.pipeline.state and the legacy read_lock_pid in
  src/pipeline/launch/runtime.py is now a thin adapter delegating to the
  canonical store function. launch_service.interrupt() uses
  PipelineStateStore(run_dir).lock_path + remove_stale_lock instead of
  hardcoded unlink. Also migrated src/main.py:_resolve_config to
  PipelineStateStore(run_dir).load() (removes duplicate json.loads and
  picks up schema_version validation for free).
rationale: |
  Single source of truth for the lock-file lifecycle prevents drift if the
  lock filename or payload format ever changes (we already write
  `pid=<n>\nstarted_at=<iso>\n`, so the reader should live beside the
  writer). Content-matched unlink closes a theoretical race where a PID
  check on a stale lock is followed by a legitimate acquire from another
  process; the content-match protects the new owner.
alternatives:
  - "Simple unlink without content-match: rejected — keeps race window open when pid is reused by another run that just started."
  - "Move read_lock_pid out of pipeline/launch entirely: deferred — kept as thin adapter to avoid breaking external callers of src.pipeline.launch."
consequences:
  - "API interrupt path is now race-safe: another process's lock cannot be accidentally removed."
  - "pipeline_state.json consumption is unified — all readers go through PipelineStateStore.load() (with schema_version validation) rather than ad-hoc json.loads."
  - "read_lock_pid in launch/runtime.py has one extra function-call hop; negligible cost."
affected_files:
  - "src/pipeline/state/store.py"
  - "src/pipeline/state/__init__.py"
  - "src/pipeline/launch/runtime.py"
  - "src/api/services/launch_service.py"
  - "src/main.py"
tags: ["architecture", "sibling-client", "state", "lock", "api", "invariant"]
```

## 3. UPDATE existing active ADR

Find the existing "Sibling Client Architecture for Pipeline State" via
`update_decision_records(action="list", filter_tag="architecture")`, then:

```
action: update
decision_id: <found>
consequences:
  - "The backend imports src.pipeline directly rather than wrapping the CLI through subprocesses for read paths."
  - "CLI and Web UI can coexist and operate on the same state without feature flags."
  - "TUI was removed 2026-04 (see separate ADR); CLI + Web UI are the two sibling clients going forward."
  - "Read path uses a process-local mtime-keyed cache (src/pipeline/state/cache.py) and HTTP ETag/304 for efficiency; writers stay on PipelineStateStore."
  - "Lock-file (run.lock) lifecycle stays inside PipelineStateStore (see sibling-invariant ADR)."
```
