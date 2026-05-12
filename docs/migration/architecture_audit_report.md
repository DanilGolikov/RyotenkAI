# Architecture Integrity Audit — Post Test Migration

**Date:** 2026-05-12
**Worktree:** `.claude/worktrees/ecstatic-khorana-1baa6b`
**Scope:** verify the "structured-hopping-starfish" architectural promises
after the 7-batch test migration.
**Auditor stance:** read-only verification; no fixes attempted.

## Executive summary

The migration *layout* claim — 6 packages moved, 6429 passing, 0 failures —
holds. The migration *architectural* claim does not. Three of the plan's
core promises are absent or stubbed:

1. **Canonical Fakes per Protocol** — 7 Fake classes exist on disk, but
   5 of 7 do not import (the Protocol modules they target are empty
   stub directories containing only stale `__pycache__` bytecode).
2. **Compliance harness** — `tests/contract/protocol_compliance/`
   contains only `__pycache__`. There are 0 compliance tests.
3. **Hermetic stack tests** — `tests/stack/` contains only `__pycache__`.
   There are 0 L6 stack tests. The sidecar harness infrastructure
   under `tests/_harness/stack/` exists but has no consumers.

Plus auxiliary integrity:

- The Protocol-mocking sentinel referenced in the audit instructions
  (`tests/_lint/test_no_protocol_mocking.py`) **does not exist**.
- importlinter rules pass (1 documented baseline breakage; no new
  violations from migration).
- Production code (`packages/*/src/`) is unmodified in the working
  tree — migration is genuinely additive there.
- L0-L12 layer cake is name-only: only L1 (unit), L3 (contract),
  L4 (integration), L5 (e2e) are populated. L2 (component), L6 (stack),
  L7 (property), L8 (golden), L9 (chaos), L10 (load), L11 (replay) are
  empty directories.

Overall integrity score: **4 / 10**. The packagization succeeded;
the test architecture promises did not land.

---

## Note on mandatory references

The audit instructions cite two documents that do not exist in the
worktree:

- `docs/plans/structured-hopping-starfish.md` — not found anywhere
  under `docs/plans/`.
- `tests/_lint/test_no_protocol_mocking.py` — not found anywhere in
  `tests/_lint/`.

Conclusions about "the plan's expected Protocol list" are inferred from
the audit prompt itself, the actual production Protocol definitions,
the canonical Fakes on disk (which document their intended targets),
and the surviving migration logs under `docs/migration/`.

---

## D1 — Protocol inventory

Production-code `Protocol` classes (subset that follow the `I*Protocol`
or `class .*\(Protocol\)` convention):

| Protocol | Location | runtime_checkable? | Concrete impl |
|---|---|---|---|
| `IMLflowManager` | `packages/shared/src/ryotenkai_shared/infrastructure/mlflow/protocol.py:32` | **NO** | `ryotenkai_pod.trainer.managers.mlflow_manager.MLflowManager` |
| `IMLflowGateway` | `packages/shared/src/ryotenkai_shared/infrastructure/mlflow/gateway.py:31` | NO | `MLflowGateway`, `NullMLflowGateway` (same file) |
| `IMLflowPrimitives` | `packages/pod/src/ryotenkai_pod/trainer/mlflow/primitives.py:14` | NO | (pod-internal, mlflow primitives wrapper) |
| `IPodLifecycleClient` | `packages/shared/src/ryotenkai_shared/infrastructure/lifecycle/protocol.py:107` | YES | `RunPodPodLifecycleClient`, `NoOpPodLifecycleClient` |
| `IInferenceEngine` | `packages/engines/src/ryotenkai_engines/interfaces.py:307` | YES | engines plugins (vllm, …) |
| `IRecoveryProbeProvider` | `packages/providers/src/ryotenkai_providers/training/interfaces.py:374` | YES | provider impls |
| `ICapacityErrorClassifier` | `packages/providers/src/ryotenkai_providers/training/interfaces.py:414` | YES | provider impls |
| `IGPUProvider` | `packages/providers/src/ryotenkai_providers/training/interfaces.py:445` | YES | RunPod / single_node |
| `ITerminalActionProvider` | `packages/providers/src/ryotenkai_providers/training/interfaces.py:729` | YES | RunPod / single_node |
| `InferenceEventLogger` | `packages/providers/src/ryotenkai_providers/inference/interfaces.py:32` | YES | provider impls |
| `IInferenceProvider` | `packages/providers/src/ryotenkai_providers/inference/interfaces.py:116` | YES | provider impls |
| `IModelInference` | `packages/control/src/ryotenkai_control/evaluation/model_client/interfaces.py:14` | YES | model-client impls |
| `IReportBlockPlugin` | `packages/control/src/ryotenkai_control/reports/plugins/interfaces.py:89` | YES | report plugin impls |
| `IExperimentDataProvider` | `packages/control/src/ryotenkai_control/reports/domain/interfaces.py:13` | NO | report domain impls |
| `IMetricAnalyzer` | `packages/control/src/ryotenkai_control/reports/domain/interfaces.py:24` | NO | report domain impls |
| `IPercentileCalculator` | `packages/control/src/ryotenkai_control/reports/domain/interfaces.py:32` | NO | report domain impls |
| `IMemoryManager` | `packages/pod/src/ryotenkai_pod/trainer/container.py:54` | YES | trainer mem manager |
| `IStrategyFactory` | `packages/pod/src/ryotenkai_pod/trainer/container.py:104` | YES | trainer strategy factory |
| `IDatasetLoader` | `packages/pod/src/ryotenkai_pod/trainer/container.py:140` | YES | trainer dataset loader |
| `ITrainerFactory` | `packages/pod/src/ryotenkai_pod/trainer/container.py:172` | YES | trainer factory |

### Expected vs found

The audit prompt lists 8 expected Protocols. Mapping:

| Expected | Found? | Notes |
|---|---|---|
| `IMLflowManager` | Yes | Exists but **not runtime_checkable** (testing limitation). |
| `IPodLifecycleClient` | Yes | runtime_checkable, fully wired. |
| `IRunPodAPI` | **NO** | `packages/shared/src/ryotenkai_shared/infrastructure/runpod_api/` is an empty stub directory (only `__pycache__`). |
| `ITrainerSpawner` | **NO** | `infrastructure/trainer_spawner/` empty stub. |
| `ISSHClient` | **NO** | `infrastructure/ssh/` empty stub. |
| `IHFHubClient` | **NO** | `infrastructure/hf_hub/` empty stub. |
| `IJobClient` | **NO** | `infrastructure/job_client/` empty stub. |
| `Clock` | Partial | `tests/_harness/clock.py` defines a `Clock` Protocol + `RealClock` for tests only. No production `Clock` Protocol. |

### Stub directories — proof

```
packages/shared/src/ryotenkai_shared/infrastructure/
├── runpod_api/        ← only __pycache__
├── ssh/               ← only __pycache__
├── hf_hub/            ← only __pycache__
├── job_client/        ← only __pycache__
├── trainer_spawner/   ← only __pycache__
├── mlflow/            ← real (protocol.py + gateway.py + env/uri helpers)
└── lifecycle/         ← real (protocol.py + outcomes.py + availability.py)
```

The `__pycache__` retains bytecode from a prior commit (verified by
`ImportError: cannot import name 'IRunPodAPI' from
'ryotenkai_shared.infrastructure.runpod_api' (unknown location)` —
import path resolves to the directory but the source `.py` is gone).
Either the source was deleted in a refactor and never replaced, or
those Protocols were *planned but never written*.

---

## D2 — Fake inventory

| Protocol target | Fake | File | snapshot()? | chaos surface? | Importable? | Protocol satisfied? |
|---|---|---|---|---|---|---|
| `IMLflowManager` | `FakeMLflowManager` | `tests/_fakes/mlflow.py:91` | yes | yes (`fail_next_n_calls`, `transient_*`) | yes | can't isinstance-check (Protocol not `@runtime_checkable`) |
| `IPodLifecycleClient` | `FakePodLifecycleClient` | `tests/_fakes/lifecycle.py:71` | yes | yes | yes | **YES, isinstance OK** |
| `IRunPodAPI` | `FakeRunPodAPI` | `tests/_fakes/runpod.py:51` | yes | yes (`rate_limit`, `transient`, `partial`) | **NO** | target Protocol missing |
| `ISSHClient` | `FakeSSHClient` | `tests/_fakes/ssh.py:55` | yes | yes (`connect_timeout`, `command_failures`, `disconnect_after`) | **NO** | target Protocol missing |
| `IHFHubClient` | `FakeHFHubClient` | `tests/_fakes/hf_hub.py:63` | yes | yes (`rate_limited`, `transient`, `auth_failure`, `corrupted_download`) | **NO** | target Protocol missing |
| `IJobClient` | `FakeJobClient` | `tests/_fakes/job_client.py:57` | yes | yes (`rate_limited`, `timeout`, `not_found`, `network_partition`) | **NO** | target Protocol missing |
| `ITrainerSpawner` | `FakeTrainerSpawner` | `tests/_fakes/trainer.py:61` | yes | yes (`oom`, `callback_failures`, `slow_start`, `signal_ignored`) | **NO** | target Protocol missing |

### Findings

- All 7 canonical Fakes follow the `Fake<ProtocolMinusI>` naming.
- All 7 implement `snapshot()` and a `reset_chaos()` method, plus at
  least one named chaos surface.
- 5 of 7 are **not importable** because the target Protocol module
  is an empty stub directory.
- 1 of 7 (`FakeMLflowManager`) is importable but cannot be
  `isinstance`-checked against its Protocol.
- 1 of 7 (`FakePodLifecycleClient`) is fully wired end-to-end.
- The entire `tests/_fakes/` tree is **untracked in git** (working
  tree only, never committed).
- Only `tests/_harness/stack/sidecars/{runpod_server,mlflow_server}.py`
  consume the canonical Fakes. **No test under `tests/{unit,integration,
  contract,e2e}/` imports from `tests._fakes`** (verified by
  `grep -rln "from tests._fakes" tests/`).

Net: the Fakes infrastructure exists *as a design artifact* but is
not wired to any production test path or to its own Protocols.

---

## D3 — Compliance harness

```
tests/contract/protocol_compliance/
└── __pycache__/   ← only directory entry
```

**0 compliance test files exist.** The directory is a husk.
For every Protocol in D1, there is **no parametrized
`[fake_factory, real_factory]` test**. There is no `RYOTENKAI_LIVE`
env-gated path. The `tests/contract/engines/test_engine_protocol_parity.py`
exists but tests engine *plugin parity* (signatures across multiple
engines), not Protocol↔Fake equivalence.

---

## D4 — Sentinel coverage

The audit prompt instructs to read `tests/_lint/test_no_protocol_mocking.py`
and verify its `_PROTOCOLS` list.

```
$ ls tests/_lint/
bootstrap_allowlist.py
test_control_no_pod_imports.py
test_discriminator_uniformity.py
test_no_control_imports.py
test_no_downstream_imports.py
test_no_io_in_engine_prepare.py
test_no_pod_imports.py
test_no_provider_imports.py
test_no_runtime_ssh_exec_command.py
test_runner_api_dto_location.py
test_shared_is_leaf.py
```

**`test_no_protocol_mocking.py` does not exist.** The synthetic-violation
test described in the audit prompt cannot be run — there is no scanner
to feed.

For reference, the surviving lint sentinels enforce package boundaries
(no_pod_imports, no_control_imports, etc.) and surface-area invariants
(no_runtime_ssh_exec_command, no_io_in_engine_prepare,
discriminator_uniformity, runner_api_dto_location, shared_is_leaf).
Collected: 15 sentinel tests.

Independent search for `unittest.mock` usage in tests: **173 test files
import `unittest.mock` / `MagicMock` / `AsyncMock`**. Without a
Protocol-mocking sentinel, the claim "no mocks-of-Protocols anywhere"
is unverifiable; spot-checks find `MagicMock(spec=I…)` usages
(e.g. `tests/unit/control/pipeline/execution/test_stage_registry.py`)
that the missing sentinel would presumably flag.

---

## D5 — importlinter rules

```
$ .venv/bin/lint-imports
Analyzed 552 files, 1463 dependencies.

shared has no internal deps (must be leaf)                  KEPT
community depends only on shared                            KEPT
pod depends only on shared and community                    KEPT
providers depend only on shared (no community/pod/control)  KEPT
control must not import pod (Mac vs in-pod boundary)        BROKEN
trainer subpackage must not import runner subpackage        KEPT
runner subpackage must not import trainer subpackage        KEPT
runtime-only modules must not import SSHClient              KEPT
engines is a leaf (depends only on shared)                  KEPT
generic code must not import concrete engine modules        KEPT

Contracts: 9 kept, 1 broken.
```

Broken contract: `control → pod` (3 imports). These match the
**documented baseline** identified in the migration plan:

- `ryotenkai_control.data.validation.standalone → ryotenkai_pod.trainer.strategies.factory`
- `ryotenkai_control.pipeline.mlflow_attempt.manager → ryotenkai_pod.trainer.managers.mlflow_manager`
- `ryotenkai_control.pipeline.stages.dataset_validator.stage → ryotenkai_pod.trainer.data_loaders.factory`

The test migration did **not** introduce new boundary violations.

---

## D6 — Test layer distribution

Collection counts (`pytest --collect-only -q`):

| Layer (directory) | Collected | % of 7114 | Target (audit prompt) |
|---|---:|---:|---|
| `tests/unit/` (L0 + L1) | 6811 | **95.7%** | ~70% |
| `tests/contract/` (L3) | 75 | 1.1% | part of 15% |
| `tests/integration/` (L4) | 157 | 2.2% | part of 15% |
| `tests/e2e/` (L5) | 56 | 0.8% | part of 10% |
| `tests/_lint/` (sentinels) | 15 | 0.2% | — |
| `tests/component/` (L2) | **0** | 0% | part of 70% |
| `tests/stack/` (L6) | **0** | 0% | part of 10% |
| `tests/property/` (L7) | **0** | 0% | part of 5% |
| `tests/golden/` (L8) | **0** | 0% | part of 5% |
| `tests/chaos/` (L9) | **0** | 0% | part of 5% |
| `tests/load/` (L10) | **0** | 0% | part of 5% |
| `tests/replay/` (L11) | **0** | 0% | part of 5% |
| **Total** | **7114** | 100% | |

Severe top-heaviness:

- 95.7% in unit layer vs ~70% target.
- 0% in 7 of the 11 expected layers.
- The pyramid is "unit + a sliver" rather than the planned layer cake.

The unit layer probably contains a mix of L0 (pure logic) and L1
(component-with-fakes) tests; the migration did not split them.

---

## D7 — Hermetic stack integrity

```
$ ls tests/stack/
__pycache__   web

$ ls tests/stack/web/
__pycache__

$ .venv/bin/python -m pytest -c tests/pytest.ini tests/stack/ --collect-only
collected 0 items
no tests collected in 0.00s
```

The stack-test directory contains no `.py` files at top level — only
stale bytecode from prior runs in `__pycache__/`
(`test_runpod_via_sidecar`, `test_smoke_control_plane`, `test_smoke_boot`,
`test_smoke_web`). The audit-prompt command
`pytest tests/stack/test_smoke_boot.py` can not be executed because
that file does not exist on disk.

The *harness* (`tests/_harness/stack/`) is intact:

- `orchestrator.py`, `ports.py`, `process.py`, `playwright.py`, `_context.py`
- `sidecars/{mlflow_server,runpod_server,vllm_server,hf_hub_server,_base}.py`
- `docker-compose.yml`

So the sidecar infrastructure is fully built, the Fakes that back the
sidecars are fully built — and there are no tests to use either.

L6 layer status: **infrastructure-only, zero coverage.**

---

## D8 — xfail hotspots

Total greenfield xfails: **478** (from batch_7c log).
Strict-True : Strict-False ratio: **184 : 12** (verified by grep).

Top 20 files by xfail count:

| Count | File |
|---:|---|
| 15 | `tests/unit/control/pipeline/test_stages_deployer.py` |
| 13 | `tests/unit/control/pipeline/inference/test_single_node_config_v3.py` |
|  9 | `tests/unit/control/pipeline/test_training_monitor_v2.py` |
|  8 | `tests/unit/control/test_pipeline_orchestrator_missing_lines.py` |
|  8 | `tests/unit/control/pipeline/launch/test_restart_options.py` |
|  7 | `tests/unit/control/pipeline/test_stages_model_retriever.py` |
|  7 | `tests/unit/control/pipeline/test_orchestrator_cleanup_hardening.py` |
|  7 | `tests/unit/control/pipeline/stages/test_inference_deployer.py` |
|  7 | `tests/unit/control/pipeline/stages/managers/deployment/test_code_syncer.py` |
|  7 | `tests/unit/control/pipeline/providers/runpod/test_api_client.py` |
|  7 | `tests/unit/community/test_phase_complete_coverage.py` |
|  6 | `tests/unit/control/test_architectural_guardrails.py` |
|  4 | `tests/unit/control/pipeline/inference/test_runpod_pods_provider.py` |
|  4 | `tests/unit/control/cli/test_commands_smoke.py` |
|  4 | `tests/integration/control/api/test_plugins.py` |
|  3 | `tests/unit/pod/trainer/test_trainer_factory_reward_routing.py` (23 tests under class marker) |
|  3 | `tests/unit/pod/trainer/test_run_training_observability.py` (5 tests under module marker) |
|  3 | `tests/unit/pod/test_run_training.py` (4 tests under class marker) |
|  3 | `tests/unit/control/pipeline/test_stages_model_evaluator.py` |
|  2 | `tests/unit/shared/utils/test_config_provider_validation.py` |

Files at the top of this list are concentrated in pipeline / stage /
deployment code where the production surface has drifted faster than
the tests. Per `xfail_debt.md`, most are genuine "test pins API that
no longer exists" — they are dead candidates, not bugs.

Single-file hotspot signal: any file with > 10 xfails is a
high-value DEAD candidate (likely cheaper to rewrite or delete than
to maintain xfail markers). Three files meet that threshold.

---

## D9 — Production code drift

```
$ git diff HEAD --name-only -- 'packages/*/src/'
(empty)

$ git diff --cached HEAD --name-only -- 'packages/*/src/'
(empty)
```

**Zero production files modified in the working tree.** The two paths
called out in the initial `gitStatus` header
(`packages/shared/.../mlflow/protocol.py`,
`packages/providers/.../runpod/training/constants.py`) are already
committed at HEAD (verified via `git log -- <file>`) — they are part
of the merged Phase B baseline and PR-3 (runpod constants), not
working-tree drift.

The migration is **strictly additive on production code**, as
documented.

---

## Overall integrity score: 4 / 10

| Dimension | Score | Reason |
|---|---:|---|
| Packagization layout | 9/10 | 5 packages, importlinter green except baseline. |
| Production-code drift | 10/10 | None. Migration is additive. |
| Protocol surface | 3/10 | 2 of 8 expected Protocols exist; 5 are stub directories. |
| Fake surface | 4/10 | All 7 Fakes designed; 5 are non-importable; 0 consumers. |
| Compliance harness | 0/10 | Empty directory. |
| Sentinel coverage | 4/10 | Existing 11 sentinels work; the architectural anti-mock sentinel does not exist. |
| Layer cake (L0-L12) | 2/10 | 4 of 12 layers populated; pyramid is unit-heavy by ~25%. |
| Hermetic stack | 2/10 | Infrastructure built, 0 tests. |
| xfail discipline | 6/10 | 184 strict-True vs 12 strict-False is healthy; 3 files > 10 xfails are real debt. |
| Greenfield lane health | 9/10 | 6429 pass, 0 fail, 478 xfailed — the visible green is real. |

The migration **delivered the move**. It did **not** deliver the
architectural scaffolding the plan promised would back the move.
The greenfield lane is green because Fakes aren't wired in and
compliance tests don't exist, not because invariants hold.

---

## Critical gaps — top 5 in priority order

1. **Restore the 5 missing Protocol modules**
   (`runpod_api`, `ssh`, `hf_hub`, `job_client`, `trainer_spawner`).
   Each has a Fake on disk that imports specific names (`IRunPodAPI`,
   `ISSHClient`, `HFAuthError`, `IJobClient`, `ITrainerSpawner`).
   Cross-reference Fake `__init__.py` import lists against the Fakes'
   public API to reconstruct the Protocols. **Blocks every downstream
   gap below.**

2. **Add `@runtime_checkable` to `IMLflowManager`**.
   The Fake exists; the Protocol does not opt in. Single one-line
   change unblocks `isinstance`-based compliance checks.

3. **Create the compliance harness**.
   `tests/contract/protocol_compliance/test_<protocol>_compliance.py`
   parametrized over `[fake_factory, real_factory]` with
   `RYOTENKAI_LIVE=1` gating real impl. At minimum: 7 files, one per
   canonical Fake. Without this, nothing enforces fake==real.

4. **Write the Protocol-mocking sentinel**.
   `tests/_lint/test_no_protocol_mocking.py` with a hard-coded
   `_PROTOCOLS` list scanning `tests/` for `MagicMock(spec=I…)` /
   `@patch("…IProto…")` / `Mock(spec=I…)`. Without it, the
   "canonical Fakes only" rule is unenforced and 173 mock-using test
   files cannot be audited.

5. **Populate or remove the empty layer directories**.
   `tests/{component,stack,property,golden,chaos,load,replay}/` are
   misleading as-is — they suggest coverage that does not exist.
   Either start migrating relevant tests in (e.g. property-based
   tests already in unit/, chaos scenarios that exist as
   `tests/chaos/scenarios/`) or document them as future-work
   placeholders.

---

## Recommendations — cleanup batches (suggested order)

**Batch A: unblock Fakes (1-2 days)**
1. Reconstruct the 5 missing Protocol modules from Fake import sites
   and the surviving production code that already implements those
   contracts (`RunPodAPIClient`, `SSHClient`, paramiko wrappers,
   etc.). Production impls already exist; this batch creates the
   Protocol declarations the Fakes import.
2. Add `@runtime_checkable` to `IMLflowManager`.
3. Verify all 7 Fakes import cleanly and pass `isinstance` checks.

**Batch B: compliance harness (2-3 days)**
4. Write `tests/contract/protocol_compliance/_helpers.py` defining a
   `protocol_compliance` parametrize decorator.
5. Write 7 compliance tests; gate real-impl path on
   `RYOTENKAI_LIVE=1` env.
6. Confirm Fake == real by running both paths in CI for at least one
   Protocol (start with `IPodLifecycleClient`, since that's the only
   end-to-end-wired one today).

**Batch C: sentinel restore (0.5 day)**
7. Write `tests/_lint/test_no_protocol_mocking.py` with a hard-coded
   `_PROTOCOLS` list (use D1's table) scanning `tests/` for spec=I…
   patterns. Generate the initial allowlist by running it; commit
   the allowlist; the test fails on any new addition.

**Batch D: xfail garbage-collect (1 day)**
8. Triage the 3 files with > 10 xfails (`test_stages_deployer.py`,
   `test_single_node_config_v3.py`, `test_training_monitor_v2.py`).
   Per `xfail_debt.md` these are stale pins, not bugs — either
   rewrite each test against current production API or delete.
9. Same for the 3 module/class-level group xfails (23 + 5 + 4 tests
   in trainer-factory / observability / FileHandler clusters).

**Batch E: hermetic stack (variable, 3-5 days)**
10. Write 1 boot-smoke stack test
    (`tests/stack/test_smoke_boot.py`) using the existing
    `tests/_harness/stack/orchestrator.py` to spin up all sidecars
    and verify they answer health endpoints.
11. Write 1 end-to-end stack test that exercises a real control-plane
    flow through the sidecars — this validates that Fakes match real
    sidecar wire format. Without this, the sidecar Fakes can drift
    silently.

**Batch F: layer cleanup (0.5 day)**
12. Delete empty directories or add `README.md` placeholders
    documenting them as L2/L6/L7/L8/L9/L10/L11 stubs with target
    coverage. The current state silently suggests broader coverage
    than exists.
