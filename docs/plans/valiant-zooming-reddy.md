# Test Infrastructure Finalization — Final Plan

> Author: Claude (architecture-lead role) | Date: 2026-05-12
> Detailed risk analysis + alternatives: [docs/plans/test-infrastructure-finalization.md](test-infrastructure-finalization.md)

## Context

After 6 phases of mock-elimination + greenfield migration + agent-driven mutation testing, the test infrastructure has a solid **architectural skeleton** but several open hatches prevent it from being declared "production-ready":

- 2 collection errors in `tests/chaos/` (duplicate scenario registration) and `tests/load/` (missing `RunLoader` exports)
- 3 pre-existing lane failures (`test_garbage_event_content` HF `KeyError: 'Version'`, plus 2 parity tests)
- 88 strict-True xfailed tests (DRIFT bucket: ProviderContext / DatasetSource / Pydantic / training_monitor cluster drift)
- Mutation testing `mode: advisory` with baseline only for 3/5 hotspots
- `mlflow_relay.py` threshold 0.55 (< spec floor 0.65)
- Quarterly mock-policy review documented but not scheduled
- No "test file exists" sentinel — files can land without ANY test

**Goal**: every regression vector gets a permanent gate, the lane is green from clean clone, feature work can resume without future tech-debt blowback. **Decisions confirmed (2026-05-12)**: Full sweep xfails / Auto-flip mutation / Collectable+chaos-nightly / Test-file-exists sentinel.

---

## Phases (in execution order)

### F1 — Structural cleanup (must-do; no compromises)

**F1.1 Idempotent chaos scenario registration**
- File: [tests/_harness/chaos.py:139](tests/_harness/chaos.py:139) `register_scenario` raises `RuntimeError` on duplicate name.
- Change: if `new_class is registered_class` (same identity), silently no-op; else raise as before.
- This handles `tests/chaos/scenarios/<name>.py` paired with `test_<name>.py` both decorating the same class.
- Verify: `pytest tests/chaos/test_catalog.py --collect-only` clean.

**F1.2 Restore `RunLoader` / `RunLoaderConfig` exports**
- File: [tests/load/runloader/__init__.py](tests/load/runloader/__init__.py)
- Current exports: `RunLoaderScenario`, `SLOResult`, `SLOSpec`, `run_scenario`.
- Missing per the smoke test: `RunLoader`, `RunLoaderConfig`.
- Action: investigate framework.py to determine if these are renamed classes; if missing, add a thin wrapper (`@dataclass class RunLoaderConfig`, `class RunLoader` that calls `run_scenario`). Time-box: 2 hours; fallback `collect_ignore = ["tests/load/test_smoke_load.py"]` in conftest with a `# TODO(load-framework)` comment.

**F1.3 Disposition `test_garbage_event_content` HF Version KeyError**
- Root cause investigation order: (1) `community/libs/helixql/*.toml` parse error (line 1 column 7) — fix TOML if straightforward; (2) `huggingface_hub` version compatibility — `pytest.skipif` if env-only.
- Time-box: 1 hour. Fallback: skipif with clear reason linking to a tracking issue.

**F1.4 Other pre-existing failures**
- `test_list_terminate_via_real_http` — confirm `RYOTENKAI_LIVE=1` gating works; if test runs unconditionally, add gate.
- `test_parser_always_returns_apiexception` (api/parity test) — investigate, fix or `skipif`.
- Acceptance: `pytest tests/unit tests/integration tests/e2e` reports `0 failed`.

**F1.5 Clean-collection sentinel**
- New: `tests/_lint/test_clean_collection.py` — runs `pytest --collect-only` on each top-level test tree; asserts zero `ERROR collecting`.

**F1.6 `.github/workflows/chaos-nightly.yml`**
- Cron `0 4 * * *` (1h after mutation-nightly).
- Runs `pytest tests/chaos/ -m chaos -v`.
- Advisory only (workflow doesn't gate merges); blocking after 7 days observation.

**STOP-AND-REPORT checkpoint after F1**: lane green, collection clean, chaos workflow added.

---

### F2 — Full sweep xfail triage (5-7 days)

**F2.0 Bucketing pass** → produces `docs/migration/xfail_full_sweep.md` with 88 tests classified:
| Bucket | Definition | Action |
|---|---|---|
| RESOLVABLE-NOW | test-only refactor using existing fakes/factories | F2.1 convert |
| RESOLVABLE-WITH-HELPER | needs new fake/factory | F2.2 build helper, F2.3 convert |
| DEAD | production code removed; test references nothing real | F2.4 delete |
| HARD-DRIFT | substantial test rewrite needed; production behavior intact | F2.5 document with `xfail-debt:<id>` |

**HARD CHECKPOINT after F2.0**: if RESOLVABLE-NOW + RESOLVABLE-WITH-HELPER > 80 tests, stop and re-negotiate scope.

**F2.1 Convert RESOLVABLE-NOW** (est. 30-40 tests)
- Use existing helpers: `tests/_fakes/provider_context.py::make_provider_context`, `tests/_fakes/dataset_source.py::make_dataset_local|hf|with_kind`.
- Per-cluster commits.

**F2.2 Build new helpers** (est. 2 hours each):
- `tests/_fakes/single_node_config.py::make_single_node_training_config(**overrides)` — typed pydantic factory.
- `tests/_fakes/training_monitor.py::make_monitor_with_log_manager(**overrides)` — wires `_provider`, `_client`, log manager.
- Helpers MUST mirror Phase B production API surface; verify via `inspect.signature`.

**F2.3 Convert RESOLVABLE-WITH-HELPER** (est. 20-30 tests).

**F2.4 Delete DEAD tests** (est. 10-15 tests)
- Per-test runtime verification: `hasattr(module, attr)` returns `False` before deletion.
- One commit per file; deletion list logged in `xfail_full_sweep.md`.

**F2.5 Document HARD-DRIFT remainder** (est. < 10 tests)
- Update each xfail reason to contain `xfail-debt:<unique-id>`.
- Add matching row to `docs/migration/xfail_debt.md` with owner + trigger (e.g., "rewrite when feature X touches monitor postmortem path").

**F2.6 Sentinel `test_xfail_debt_completeness`**
- AST-walks all `@pytest.mark.xfail(strict=True, ...)` decorators.
- Asserts each reason contains `xfail-debt:<id>` AND that id exists in `xfail_debt.md`.
- Escape hatch: `xfail-debt:ad-hoc-<timestamp>` token auto-creates a placeholder row (one-time emergency unblock).

**Acceptance**: `pytest tests/_lint/test_xfail_debt_completeness.py` green; < 10 strict-True xfails remain.

---

### F3 — Mutation testing graduation

**F3.1 Pin cosmic-ray in `uv.lock`**
- `uv add --dev cosmic-ray>=8.4.6,<9.0.0`; verify lock entry with hash.

**F3.2 Run full nightly baseline in CI**
- After F1 merges, trigger `.github/workflows/mutation-nightly.yml` manually (`workflow_dispatch`).
- Commit produced `scripts/mutation/ratchet_baseline.json` with all 5 hotspots.

**F3.3 Address `mlflow_relay.py` surviving mutations** (per Phase 6 report mutations 3, 4, 6)
- New tests in `tests/unit/pod/runner/test_mlflow_relay.py`:
  1. Per-kind dispatch: publish one event per kind (`mlflow_metric`, `mlflow_param`, `mlflow_tag`); assert correct client method fires.
  2. Cooldown boundary: `now - opened_at == active_cooldown` exactly; expect `closed = False`.
  3. Breaker defaults: `MLflowRelayCircuitBreaker()` (no kwargs) → `failure_threshold == 3`, `initial_cooldown_s == 1.0`, etc.
- Re-run mutation testing on `mlflow_relay.py`; raise threshold to 0.65 in `.mutation-hotspots.yml`.

**F3.4 Auto-flip script `scripts/mutation/maybe_flip_to_blocking.py`**
- Reads last 4 nightly reports + last 14 days PR-mutation runs via `gh run list`.
- Conditions: all nightlies green AND zero false-positive PR runs.
- If satisfied: patches `.mutation-hotspots.yml` mode → `blocking`, runs `gh pr create` with summary.
- Else: prints status (no side effects).

**F3.5 `.github/workflows/mutation-flip-check.yml`**
- Weekly cron, runs the script; auto-PR on success.

**F3.6 Mutation gate uses `git merge-base` baseline for feature branches**
- Modify `scripts/mutation/check_ratchet.py`: when running on a PR, the baseline is the kill rate at `merge-base origin/main HEAD`, not current `main`.
- This prevents new branches inheriting old debt being blocked.

---

### F4 — Process gates (permanent enforcement)

**F4.1 Verify allowlist 365-day decay**
- Manually add a stale entry to `tests/_lint/_mock_allowlist.py` with `renewed_date` 366 days ago.
- Run sentinel; expect failure with message "stale entry".
- Revert. Document the verification in `docs/testing/mock_policy.md`.

**F4.2 Quarterly mock-policy review — GitHub Action**
- `.github/workflows/quarterly-review.yml` (cron `0 9 1 */3 *`).
- Step 1: `python scripts/mock_inventory.py` produces fresh inventory.
- Step 2: diff against committed `docs/migration/mock_inventory.{csv,md}` + `_mock_allowlist.py`.
- Step 3: `gh issue create` with the delta as body.

**F4.3 Subagent self-validation gate — `.claude/CLAUDE.md`**
- Add section "Agent testing workflow":
  - When subagent diff touches `packages/*/src/**/*.py`: MUST run `bash scripts/mutation/validate_agent_output.sh main` before declaring done.
  - When adding new `@pytest.mark.xfail(strict=True, ...)`: MUST add matching row to `docs/migration/xfail_debt.md` in same PR.
- Include exact diff in execution; show user before write.

**F4.4 Verify sentinel dynamic Protocol discovery**
- Add `tests/_lint/test_protocol_discovery_invariants.py`:
  - Discovers all `Protocol`-decorated classes in `packages/*/src/`.
  - Asserts count > 30 (currently 38; allows growth but catches regression if discovery breaks).

**F4.5 New-xfail sentinel** — already specified in F2.6.

**F4.6 "Test file exists" sentinel `tests/_lint/test_every_module_has_tests.py`**
- Walks `packages/*/src/**/*.py` (skip `__init__.py`).
- For each module, greps for `from <module>` or `import <module>` in `tests/**/*.py`.
- Allowlist: `tests/_lint/no_test_required.yaml` — modules legitimately without tests (pure dataclass / type-stub / constants) listed with reason.
- Fail if module unreferenced AND not allowlisted.

---

### F5 — Documentation + Dashboard

**F5.1 `docs/testing/README.md`** — canonical index linking mock_policy, mutation_testing, xfail_debt, onboarding.

**F5.2 `scripts/testing/status.py`** — emits `docs/testing/STATUS.md`:
- Tests collected / passing / xfailed / skipped (with timestamp).
- Mock count (from `scripts/mock_inventory.py`).
- xfail count + age distribution.
- Latest mutation kill rates per hotspot (from `scripts/mutation/reports/`).
- Allowlist entries + age.
- Every section includes `source-data-as-of: <timestamp>`.
- Sentinel `test_status_freshness.py` asserts STATUS.md emit time < 14 days old.

**F5.3 `Makefile` targets**:
```
test-quick:        .venv/bin/python -m pytest tests/unit -q
test-full:         .venv/bin/python -m pytest tests/unit tests/integration tests/e2e
test-mutation:     bash scripts/mutation/run_on_diff.sh
test-status:       .venv/bin/python scripts/testing/status.py
```

---

### F6 — Validation + Merge

**F6.1** Fresh clone smoke: `git clone`, `uv sync --all-extras`, `make test-full` → green.
**F6.2** Trigger draft PR touching `event_bus.py`; verify `.github/workflows/mutation-pr.yml` produces kill rate comment.
**F6.3** Sentinel verification matrix: each new lint runs in isolation + as part of full lane.
**F6.4** Commit + fast-forward to `RESEACRH`.
**F6.5** Trigger one nightly run; verify baselines for remaining hotspots are populated.

---

## Test variations covered

| Variation | Where in plan |
|---|---|
| Positive + negative | F1 each fix, F2.1-F2.4 conversions, F3.3 mlflow tests |
| Boundary | F3.3 mutation 4 (exactly-on-cooldown), existing eventually/clock harness |
| Invariants | F4.4 protocol discovery sentinel, existing sentinel suite |
| Dependency errors | F1.3 (HF version), F1.4 (LIVE-gated), `pytest.importorskip` patterns |
| Regressions | F1.5 collection sentinel, F2.6 xfail sentinel, F3.5 mutation ratchet |
| Logic-specific | F3.3 per-kind dispatch / breaker defaults |
| Combinatorial | F1.6 chaos catalog (after F1.1 unblocks) |

---

## Acceptance criteria (with exact verification commands)

| # | Criterion | Command | Expected |
|---|-----------|---------|----------|
| AC1 | Zero collection errors | `pytest tests/ --collect-only -q 2>&1 \| tail -3` | `N tests collected`, no `ERROR` |
| AC2 | Main lane green | `pytest tests/unit tests/integration tests/e2e -q` | `N passed, M xfailed, 0 failed` |
| AC3 | xfail debt sentinel green | `pytest tests/_lint/test_xfail_debt_completeness.py` | passed |
| AC4 | Mutation baseline complete | `jq '.files \| length' scripts/mutation/ratchet_baseline.json` | `5` |
| AC5 | mlflow_relay threshold raised | `grep mlflow_relay .mutation-hotspots.yml` | `min_kill_rate: 0.65` |
| AC6 | `mode: blocking` flip script exists | `ls scripts/mutation/maybe_flip_to_blocking.py` | present |
| AC7 | Quarterly review workflow exists | `ls .github/workflows/quarterly-review.yml` | present |
| AC8 | Chaos nightly exists | `ls .github/workflows/chaos-nightly.yml` | present |
| AC9 | Allowlist decay verified | manual: stale entry → sentinel fails | documented in mock_policy.md |
| AC10 | Protocol discovery invariant | `pytest tests/_lint/test_protocol_discovery_invariants.py` | passed |
| AC11 | "test file exists" sentinel | `pytest tests/_lint/test_every_module_has_tests.py` | passed |
| AC12 | docs/testing/README.md exists + links | `grep -l mock_policy docs/testing/README.md` | match |
| AC13 | Status dashboard works | `python scripts/testing/status.py && head -1 docs/testing/STATUS.md` | Markdown header |
| AC14 | Makefile targets work | `make test-quick && make test-status` | both green |
| AC15 | cosmic-ray pinned | `grep cosmic-ray uv.lock` | entry with hash |
| AC16 | CLAUDE.md agent gate updated | `grep validate_agent_output .claude/CLAUDE.md` | match |
| AC17 | Lane green after merge | `git log RESEACRH..HEAD --oneline` | empty |

---

## Critical files to be modified

**Read-and-edit**:
- `tests/_harness/chaos.py` — F1.1 idempotent register
- `tests/load/runloader/__init__.py` — F1.2 add exports (or `framework.py`)
- `community/libs/helixql/*.toml` — F1.3 fix if root cause
- `.mutation-hotspots.yml` — F3.3 raise mlflow threshold (later: F3.4 mode flip)
- `tests/_lint/_mock_allowlist.py` — F4.1 (temporary stale entry then revert)
- `.claude/CLAUDE.md` — F4.3 agent workflow section
- `docs/migration/xfail_debt.md` — F2.5 HARD-DRIFT rows
- `Makefile` — F5.3 targets
- `pyproject.toml` / `uv.lock` — F3.1 cosmic-ray pin

**Created**:
- `tests/_lint/test_clean_collection.py` (F1.5)
- `tests/_lint/test_xfail_debt_completeness.py` (F2.6)
- `tests/_lint/test_protocol_discovery_invariants.py` (F4.4)
- `tests/_lint/test_every_module_has_tests.py` (F4.6)
- `tests/_lint/test_status_freshness.py` (F5.2)
- `tests/_lint/no_test_required.yaml` (F4.6 allowlist)
- `tests/_fakes/single_node_config.py` (F2.2)
- `tests/_fakes/training_monitor.py` (F2.2)
- `docs/migration/xfail_full_sweep.md` (F2.0)
- `docs/testing/README.md` (F5.1)
- `docs/testing/STATUS.md` (F5.2 auto-generated)
- `scripts/mutation/maybe_flip_to_blocking.py` (F3.4)
- `scripts/testing/status.py` (F5.2)
- `.github/workflows/mutation-flip-check.yml` (F3.5)
- `.github/workflows/chaos-nightly.yml` (F1.6)
- `.github/workflows/quarterly-review.yml` (F4.2)

**Functions/utilities reused**:
- [tests/_fakes/provider_context.py](tests/_fakes/provider_context.py) `make_provider_context()`, `attach_manifest_capabilities()` — F2.1
- [tests/_fakes/dataset_source.py](tests/_fakes/dataset_source.py) `make_dataset_local/hf/with_kind()` — F2.1
- [tests/_lint/test_no_protocol_mocking.py](tests/_lint/test_no_protocol_mocking.py) `_collect_protocols()` — F4.4
- [scripts/mock_inventory.py](scripts/mock_inventory.py) — F4.2 quarterly review
- [scripts/mutation/orchestrate.py](scripts/mutation/orchestrate.py), [scripts/mutation/check_ratchet.py](scripts/mutation/check_ratchet.py) — F3
- [scripts/mutation/validate_agent_output.sh](scripts/mutation/validate_agent_output.sh) — F4.3

---

## Verification (end-to-end test of the test infrastructure)

After all 6 phases:

1. **From clean clone**: `git clone … && cd … && uv sync --all-extras && make test-full` → expect green.
2. **Mutation gate end-to-end**: open draft PR touching `event_bus.py`; check that PR comment shows kill rate; intentionally introduce a tautological test (`assert True`) and verify mutation testing catches the gap.
3. **xfail debt sentinel end-to-end**: add an `@pytest.mark.xfail(strict=True, reason="test")` without `xfail-debt:` token; run sentinel; expect failure.
4. **Allowlist decay end-to-end**: temporarily push `renewed_date` 366 days back; sentinel fails; revert.
5. **Status dashboard end-to-end**: `make test-status`; read STATUS.md; verify timestamps + numbers match reality.
6. **chaos-nightly dry run**: trigger via `workflow_dispatch`; expect green or expected chaos-detected failures.
7. **Mutation auto-flip dry run**: `python scripts/mutation/maybe_flip_to_blocking.py --dry-run`; verify it correctly identifies "not yet stable enough" until 4 nightlies pass.

---

## Schedule estimate

| Phase | Estimate | Parallelizable with |
|---|---|---|
| F1 (structural) | 1-2 days | — |
| F2 (full sweep xfails) | 5-7 days | F4-F5 partially |
| F3 (mutation graduation) | 1 day work + nights for nightly baselines | F4 |
| F4 (process gates) | 1-2 days | F5 |
| F5 (docs + dashboard) | 1 day | F4 |
| F6 (validation + merge) | 0.5 days | — |
| **Total** | **9-13 days work; ~2-3 calendar weeks** | |

**Stop-and-report checkpoints**: after F1, after F2.0 bucketing, after F2 complete, after F3, after F6.
