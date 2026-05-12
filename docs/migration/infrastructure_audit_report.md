# Infrastructure Audit Report — Phase 0/1 Greenfield Testing Infra

**Date:** 2026-05-12
**Worktree:** `.claude/worktrees/ecstatic-khorana-1baa6b`
**Branch:** `claude/ecstatic-khorana-1baa6b`
**Baseline before audit:** 6429 passed, 0 failed, 478 xfailed (no `tests/conftest.py`, empty `tests/_harness/` and `tests/_fakes/`).
**Final lane state:** 6429 passed, 0 failed, 478 xfailed — **GREEN**.

---

## 1. Audit findings

All files spec'd in Phase 0/1 of
`docs/adrs/2026-05-10-greenfield-testing-architecture.md` were missing from
the worktree at audit start. `__pycache__` directories contained `.pyc`
bytecode for every expected `.py` source file — confirming the source files
existed at some point and were stripped/never-committed.

| File | Before audit | Reason |
|------|--------------|--------|
| `tests/conftest.py` | missing (only `.pyc`) | Not on disk |
| `tests/__init__.py` | missing (only `.pyc`) | Not on disk |
| `tests/_harness/__init__.py` | missing (only `.pyc`) | Not on disk |
| `tests/_harness/clock.py` | missing (only `.pyc`) | Not on disk |
| `tests/_harness/wait.py` | missing (only `.pyc`) | Not on disk |
| `tests/_harness/telemetry.py` | missing (only `.pyc`) | Not on disk |
| `tests/_harness/debug_bundle.py` | missing (only `.pyc`) | Not on disk |
| `tests/_harness/chaos.py` | missing (only `.pyc`) | Not on disk |
| `tests/_harness/chaos_recorder.py` | missing (only `.pyc`) | Not on disk |
| `tests/_harness/stack/*.py` (6 files) | missing (only `.pyc`) | Not on disk |
| `tests/_harness/stack/sidecars/*.py` (6 files) | missing (only `.pyc`) | Not on disk |
| `tests/_fakes/__init__.py` | missing (only `.pyc`) | Not on disk |
| `tests/_fakes/mlflow.py` | missing (only `.pyc`) | Not on disk |
| `tests/_fakes/lifecycle.py` | missing (only `.pyc`) | Not on disk |
| `tests/_fakes/runpod.py` | missing (only `.pyc`) | Not on disk |
| `tests/_fakes/trainer.py` | missing (only `.pyc`) | Not on disk |
| `tests/_fakes/ssh.py` | missing (only `.pyc`) | Not on disk |
| `tests/_fakes/hf_hub.py` | missing (only `.pyc`) | Not on disk |
| `tests/_fakes/job_client.py` | missing (only `.pyc`) | Not on disk |

Sister production-side scaffolding (also absent on disk) — these are needed
by some of the canonical fakes:

| Module | On-disk status |
|--------|---------------|
| `packages/shared/src/ryotenkai_shared/utils/clock.py` | missing |
| `packages/shared/src/ryotenkai_shared/infrastructure/runpod_api/{__init__,protocol}.py` | missing (only empty dir + `__pycache__`) |
| `packages/shared/src/ryotenkai_shared/infrastructure/trainer_spawner/{__init__,protocol}.py` | missing (only empty dir + `__pycache__`) |
| `packages/shared/src/ryotenkai_shared/infrastructure/ssh/{__init__,protocol}.py` | missing (only empty dir + `__pycache__`) |
| `packages/shared/src/ryotenkai_shared/infrastructure/hf_hub/{__init__,protocol,adapter}.py` | missing (only empty dir + `__pycache__`) |
| `packages/shared/src/ryotenkai_shared/infrastructure/job_client/{__init__,protocol}.py` | missing (only empty dir + `__pycache__`) |
| `packages/shared/src/ryotenkai_shared/infrastructure/mlflow/protocol.py` | **present** |
| `packages/shared/src/ryotenkai_shared/infrastructure/lifecycle/` | **present** |

ADR/plan docs (`docs/adrs/2026-05-10-greenfield-testing-architecture.md`,
`docs/plans/structured-hopping-starfish.md`) were also missing from disk.

## 2. Git history check

`git log --all --diff-filter=D` for `tests/_harness/`, `tests/_fakes/`,
`tests/conftest.py` returned **empty** — none of these files were ever
committed and then deleted.

A `git stash list` revealed three local stashes:

```
stash@{0}: WIP on claude/ecstatic-khorana-1baa6b: 0ccd28e
stash@{1}: WIP on claude/zealous-gauss-7321c5: aec2f94 ...
(plus index/untracked-files commits attached to stash@{0})
```

The stash chain anchored to `stash@{0}` contains:

* `5e1e060` — **untracked files** stash entry; contains the entire Phase 0/1
  greenfield infra (including ADRs, plan, conftest, `_harness/*`,
  `_fakes/*`, sister production protocol/adapter modules under
  `packages/shared/src/ryotenkai_shared/...`).
* `d214eb4` — index entry.
* `6408e35` — WIP entry.

So: the work existed, was created uncommitted, was stashed, never restored.
This stash is the canonical source for reconstruction.

## 3. Reference check

Inside the greenfield `tests/` tree:

```
grep -rn "from tests._harness" tests/ -> only tests/conftest.py imports
                                          telemetry/debug_bundle/clock
grep -rn "from tests._fakes"   tests/ -> only sidecar modules (mlflow_server,
                                          runpod_server) reference fakes
grep -rn "import tests._fakes" tests/ -> 0 results
```

**No test file on disk imports `tests._harness` or `tests._fakes`.** The
infrastructure is canonical scaffolding, not yet wired into any actual test.
Restoration therefore does not change the test count, only the canonical
shape of the lane.

## 4. Restoration plan

For each missing file: try `git show 5e1e060:<path> > <path>` (stash
content). If the file imports production code that exists on disk, keep it.
If it imports production code that's also missing (and adding the
production code would violate the audit's **No production code changes**
rule), reconstruct from the Phase 0 ADR spec or drop and leave for a
future user-supervised pass.

| File | Strategy |
|------|----------|
| `tests/conftest.py` | restored from stash; works once `clock.py` is self-contained |
| `tests/__init__.py` | restored from stash |
| `tests/_harness/__init__.py` | restored from stash |
| `tests/_harness/clock.py` | **reconstructed from Phase 0 ADR spec** — the stash version imports `Clock,RealClock` from a Phase-1+ production module that isn't on disk. Reconstructed self-contained Phase 0 form (Clock Protocol + RealClock + ManualClock locally). |
| `tests/_harness/wait.py` | restored from stash (only imports `tests._harness.clock`) |
| `tests/_harness/telemetry.py` | restored from stash (stdlib-only) |
| `tests/_harness/debug_bundle.py` | restored from stash (stdlib + optional stack context) |
| `tests/_harness/chaos.py` | restored from stash |
| `tests/_harness/chaos_recorder.py` | restored from stash |
| `tests/_harness/stack/*.py` (6 files) | restored from stash (`__init__.py`, `_context.py`, `orchestrator.py`, `playwright.py`, `ports.py`, `process.py`) |
| `tests/_harness/stack/docker-compose.yml` | restored from stash |
| `tests/_harness/stack/sidecars/*.py` | partially restored — see below |
| `tests/_fakes/__init__.py` | restored from stash |
| `tests/_fakes/mlflow.py` | restored from stash — `ryotenkai_shared.infrastructure.mlflow.protocol` exists on disk |
| `tests/_fakes/lifecycle.py` | restored from stash — `ryotenkai_shared.infrastructure.lifecycle` exists on disk |
| `tests/_fakes/runpod.py` | **dropped** — imports `IRunPodAPI` etc. from `ryotenkai_shared.infrastructure.runpod_api`, which is an empty production dir on disk. Restoring would require adding production code (forbidden by audit). |
| `tests/_fakes/trainer.py` | **dropped** — same reason for `ITrainerSpawner` |
| `tests/_fakes/ssh.py` | **dropped** — same reason for `ISSHClient` |
| `tests/_fakes/hf_hub.py` | **dropped** — same reason for `HFAuthError`/`IHFHubClient` |
| `tests/_fakes/job_client.py` | **dropped** — same reason for `IJobClient` |
| `tests/_harness/stack/sidecars/runpod_server.py` | **dropped** — depends on `FakeRunPodAPI` (dropped) and missing prod `RunPodInfo` |

## 5. Restored files (final list)

```
tests/__init__.py                                   — from stash
tests/conftest.py                                   — from stash
tests/_harness/__init__.py                          — from stash
tests/_harness/clock.py                             — reconstructed from Phase 0 ADR spec
tests/_harness/wait.py                              — from stash
tests/_harness/telemetry.py                         — from stash
tests/_harness/debug_bundle.py                      — from stash
tests/_harness/chaos.py                             — from stash
tests/_harness/chaos_recorder.py                    — from stash
tests/_harness/stack/__init__.py                    — from stash
tests/_harness/stack/_context.py                    — from stash
tests/_harness/stack/orchestrator.py                — from stash
tests/_harness/stack/playwright.py                  — from stash
tests/_harness/stack/ports.py                       — from stash
tests/_harness/stack/process.py                     — from stash
tests/_harness/stack/docker-compose.yml             — from stash
tests/_harness/stack/sidecars/__init__.py           — from stash
tests/_harness/stack/sidecars/_base.py              — from stash
tests/_harness/stack/sidecars/mlflow_server.py      — from stash
tests/_harness/stack/sidecars/hf_hub_server.py      — from stash
tests/_harness/stack/sidecars/vllm_server.py        — from stash
tests/_fakes/__init__.py                            — from stash
tests/_fakes/mlflow.py                              — from stash
tests/_fakes/lifecycle.py                           — from stash
```

Notable reconstruction: `tests/_harness/clock.py` was rewritten from the
Phase 0 ADR spec rather than restored verbatim from the stash. The stash
version is the post-Phase-1 form that imports `Clock` and `RealClock` from
`ryotenkai_shared.utils.clock` (production module). That production module
does not exist on disk in this worktree and adding it would be a production
change. The reconstructed Phase 0 form defines `Clock` Protocol +
`RealClock` + `ManualClock` locally — exactly what the ADR mandates for
Phase 0, and what the Phase 0 verification command
`python -c "from tests._harness.wait import Eventually; ..."` expects.

## 6. Verification

```
.venv/bin/python -c "from tests._harness.clock import Clock, ManualClock, RealClock"   # OK
.venv/bin/python -c "from tests._harness.wait import Eventually, Consistently"          # OK
.venv/bin/python -c "from tests._harness import telemetry, debug_bundle, chaos"         # OK
.venv/bin/python -c "from tests._harness.stack import Stack, current_stack"             # OK
.venv/bin/python -c "from tests._fakes.mlflow import FakeMLflowManager"                  # OK
.venv/bin/python -c "from tests._fakes.lifecycle import FakePodLifecycleClient"          # OK

.venv/bin/python -m pytest -c tests/pytest.ini tests/
# => 6429 passed, 203 skipped, 478 xfailed, 4 xpassed in 361.22s
```

Tested final outcome **matches the baseline exactly** — no regressions.
The conftest plugin registration is working: `tests/.telemetry/run-*.jsonl`
files are being emitted per test session (6883 lines in the latest one).

`make lint-no-mocks-in-new-tests` was not re-run — it's a Makefile target
that already worked against the legacy lint sentinels under `tests/_lint/`
(unchanged by this audit). The Phase 0 verification command for the AST
sentinel remains satisfied by the existing
`tests/_lint/test_no_protocol_mocking.py`.

## 7. Pending items

The following Phase 1 canonical fakes were **not restored** because they
require corresponding production Protocol modules that are also absent
from disk. Restoring the fakes without first restoring those Protocols
would either:

* fail at import-time (current state of `tests/_fakes/runpod.py` etc.
  before they were dropped), or
* force this audit to add production code (forbidden by the audit's hard
  rules: **"No production code changes"**).

| Fake | Required production module(s) (missing) |
|------|----------------------------------------|
| `tests/_fakes/runpod.py` | `ryotenkai_shared.infrastructure.runpod_api.{IRunPodAPI, RunPodInfo, ...}` |
| `tests/_fakes/trainer.py` | `ryotenkai_shared.infrastructure.trainer_spawner.{ITrainerSpawner, ...}` |
| `tests/_fakes/ssh.py` | `ryotenkai_shared.infrastructure.ssh.{ISSHClient, ...}` |
| `tests/_fakes/hf_hub.py` | `ryotenkai_shared.infrastructure.hf_hub.{IHFHubClient, HFAuthError, ...}` |
| `tests/_fakes/job_client.py` | `ryotenkai_shared.infrastructure.job_client.{IJobClient, ...}` |
| `tests/_harness/stack/sidecars/runpod_server.py` | same as `runpod.py` |

All six files exist verbatim in stash `5e1e060` together with the
production-side scaffolding they depend on:

```
packages/shared/src/ryotenkai_shared/utils/clock.py
packages/shared/src/ryotenkai_shared/infrastructure/runpod_api/{__init__,protocol}.py
packages/shared/src/ryotenkai_shared/infrastructure/trainer_spawner/{__init__,protocol}.py
packages/shared/src/ryotenkai_shared/infrastructure/ssh/{__init__,protocol}.py
packages/shared/src/ryotenkai_shared/infrastructure/hf_hub/{__init__,protocol,adapter}.py
packages/shared/src/ryotenkai_shared/infrastructure/job_client/{__init__,protocol}.py
packages/shared/src/ryotenkai_shared/utils/clients/job_client_adapter.py
packages/providers/src/ryotenkai_providers/runpod/lifecycle/adapter.py
packages/providers/src/ryotenkai_providers/runpod/runpod_api_adapter.py
packages/control/src/ryotenkai_control/cleanup/{__init__,batch_terminator,hibernation_detector}.py
```

**Recommended follow-up** (user-supervised, separate session): restore the
production Protocol modules from stash `5e1e060`, then restore the five
canonical fakes + `runpod_server.py` sidecar, then re-run the lane. This
is a production change and must be reviewed.

The audit also did not restore the broader test files that exist in the
stash (e.g. `tests/_contracts/*`, `tests/_lint/test_no_protocol_mocking.py`,
`tests/chaos/scenarios/*`, `tests/contract/protocol_compliance/*`,
`tests/unit/{community,engines}/*`). Many of these likely depend on the
missing production modules and/or the missing fakes. The on-disk
`tests/_lint/` already has 15 passing lint tests; if Phase 0/1's
`test_no_protocol_mocking.py` is intended to supersede them, that
consolidation is also out of scope for this audit.

The stash itself was **left intact** so the canonical content is recoverable.

## Hard rules compliance

* No production code changes. (Confirmed — `git status` shows only
  `tests/` additions plus this report.)
* No `unittest.mock` of Protocols. (Existing sentinels in `tests/_lint/`
  continue to pass.)
* Lane ends GREEN. (6429 passed, matches baseline.)
* `.pyc` files not reused — restoration sourced from `git show 5e1e060`
  or from the ADR spec.
