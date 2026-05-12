# Test Infrastructure Finalization Plan

> Status: **draft — awaiting user approval**
> Author: Claude (architecture-lead role)
> Date: 2026-05-12
> Goal: deliver a **production-ready, regression-proof** test infrastructure
>       so feature work can proceed without future tech-debt blowback.

## Definition of "ready" (architectural)

`Ready` is NOT "zero xfails / 100% coverage / blocking mutation gates from day 1."
`Ready` is: **every regression vector has a permanent gate that catches it BEFORE merge**.

The 6 regression vectors:

1. **Collection regressions** — pytest must collect every test tree without errors.
2. **Lane health** — main lane is green; xfails are deliberate and documented.
3. **Mock regression** — sentinel + allowlist + decay block new `unittest.mock` of Protocols.
4. **Quality regression** — mutation testing catches tautological tests on hotspots; ratchet prevents kill-rate drift.
5. **Test-debt regression** — every new xfail goes through `xfail_debt.md` review; quarterly automated audit.
6. **Onboarding regression** — `make test` works from a clean clone with no manual steps.

Once each vector has its gate, feature work proceeds and the gates do the maintenance.

---

## Open hatches (inventory — 2026-05-12)

### Tier-1 — STRUCTURAL (must close before declaring "ready")

| # | Item | Impact | Files |
|---|------|--------|-------|
| S1 | `tests/chaos/test_catalog.py` collection error: duplicate `ChaosScenario` name | every chaos run errors at collection | `tests/chaos/scenarios/*.py` paired with `test_*.py` — both register |
| S2 | `tests/load/test_smoke_load.py` ImportError: `RunLoader`/`RunLoaderConfig` not exported | load lane non-collectable | `tests/load/runloader/__init__.py` re-exports |
| S3 | `test_garbage_event_content` fails on `huggingface_hub.hf_api` `KeyError: 'Version'` | env-dependent flaky failure | `community/libs/helixql` TOML parse error + HF version |
| S4 | Other 2-3 pre-existing failures need disposition (test_list_terminate_via_real_http, test_parser_always_returns_apiexception) | lane not green | env / parity tests |

### Tier-2 — DEBT-TRACKED (resolve OR formally document)

| # | Item | Disposition path |
|---|------|------------------|
| D1 | 88 strict-True xfails (DRIFT bucket from audit) | Triage per cluster — RESOLVE small clusters, DOCUMENT large ones with owner+SLA |
| D2 | mutation `mode: advisory` until threshold stabilizes | flip after S1-S4 + complete baseline + 1 nightly green |
| D3 | Mutation baseline incomplete (3/5 hotspots) | run `scripts/mutation/run_full.sh` locally, commit baselines for remaining 2 |
| D4 | `mlflow_relay.py` threshold 0.55 < spec floor 0.65 | add tests for surviving mutations 3/6/8 (per-kind dispatch, breaker boundary, breaker defaults) |

### Tier-3 — PROCESS GATES (must be PERMANENT, not aspirational)

| # | Item | Permanence mechanism |
|---|------|---------------------|
| P1 | Allowlist 365-day decay enforcement | sentinel test in CI (already exists — verify it fires) |
| P2 | Quarterly mock-policy review | `scheduled-tasks` cron — NOT documentation-only |
| P3 | Subagent self-validation gate | reference in `.claude/CLAUDE.md` agent workflow |
| P4 | Sentinel dynamic Protocol discovery | runtime test that discovers all 38 Protocols (already exists — verify behavior) |
| P5 | New-xfail blocker | sentinel that fails if a strict-True xfail is added without an `xfail_debt.md` entry |

### Tier-4 — DOCUMENTATION + DASHBOARD

| # | Item | Deliverable |
|---|------|-------------|
| DOC1 | Canonical `docs/testing/README.md` index | links mock_policy, mutation_testing, xfail_debt, onboarding |
| DOC2 | "Test infrastructure status" auto-generated section | `scripts/testing/status.py` produces a Markdown summary |
| DOC3 | Onboarding doc: `make test` from clean clone | `make test-quick` (lane), `make test-full` (slow), `make test-mutation` |

### Tier-5 — DEFERRED (NOT blockers for "ready", but tracked)

| # | Item | Why not now |
|---|------|------------|
| F1 | Coverage `fail_under` baseline (Helixir memory goal) | Mutation testing is the upgrade. Add only if mutation testing proves insufficient. |
| F2 | PyTation hybrid mutation operators (ICSE 2026) | Research-stage tool; cosmic-ray remains canonical. Re-evaluate Q3 2026. |
| F3 | Mutation testing for `api/main.py` (no unit suite) | Write unit tests first; covered by feature work. |

---

## Workplan (in execution order)

### Phase F1 — Structural cleanup (must-do, no compromises)

**F1.1 Fix chaos duplicate scenario registration**
- Root cause: each scenario directory has `scenario.py` (definition) + `test_scenario.py` (pytest wrapper). Both `@register_scenario`, name clashes.
- Decision: **delete the duplicate**. Either the definition or the test file owns the registration; the other doesn't register. Industry pattern: one source of truth.
- Action: keep `test_*.py` (because pytest discovers them), delete the bare `<name>.py` siblings — OR keep the bare definitions, have tests import them without re-registering. Choose based on which has the more-complete logic.

**F1.2 Fix `tests/load/runloader` exports**
- Root cause: `__init__.py` re-exports `RunLoaderScenario`, `SLOResult`, `SLOSpec`, `run_scenario`. The smoke test imports `RunLoader` and `RunLoaderConfig` — names that don't exist.
- Decision: ADD the classes (probably a thin wrapper over `run_scenario`) — they are the documented public API per the smoke test's contract.
- Validation: smoke test collects + passes; no other tests broken.

**F1.3 `test_garbage_event_content` HF version issue**
- Root cause: `huggingface_hub.hf_api` raises `KeyError: 'Version'` due to version drift; community/libs/helixql has a broken TOML.
- Disposition options:
  1. Pin `huggingface_hub` version that doesn't crash.
  2. Fix the helixql TOML.
  3. Skip the test on env where HF imports fail (with explicit skipif).
- Decision: investigate root cause; if env-only, add `pytest.skipif` with reason; if helixql TOML, fix it.

**F1.4 Other pre-existing failures**
- `test_list_terminate_via_real_http`: integration test that needs real HTTP — likely behind `RYOTENKAI_LIVE=1`. Verify gating is correct.
- `test_parity.py::test_every_documented_get_endpoint_responds`: API parity. Verify scope is documented.

**F1.5 Verify clean collection across ALL test trees**
- `pytest tests/ --collect-only` must return zero errors.
- Add a CI step (or sentinel test) that asserts this — prevents collection regression.

### Phase F2 — DEBT triage (resolve small, document large)

**F2.1 Cluster the 88 strict-True xfails**

Top clusters (from `xfail_audit_report.md`):
- ~82: provider `__init__` signature drift (ProviderContext)
- ~22: `_extract_datasets_for_readme` typed source drift
- ~16: `SingleNodeTrainingConfig` pydantic drift
- ~12: training_monitor postmortem/log-manager drift
- ~6 small clusters

**F2.2 Resolve small clusters (~30 tests)**
- ProviderContext drift: extend `tests/_fakes/provider_context.py::make_provider_context()` if needed; convert 1 cluster as POC; if fast, do all.
- DatasetSource drift: extend `tests/_fakes/dataset_source.py` factories; convert.

**F2.3 Document large clusters with explicit OWNERS + SLAs**
- For each large cluster that can't be resolved now: add a row to `xfail_debt.md` with:
  - Owner (the feature owner of the area)
  - Trigger ("will be resolved when feature X is touched")
  - Last-touched date (for decay enforcement)

**F2.4 Add `xfail_debt.md` enforcement sentinel**
- New lint: every `@pytest.mark.xfail(strict=True, ...)` must appear in `xfail_debt.md` (by test ID or pattern).
- Reason text in code must mention an `xfail-debt:<id>` reference.
- Mechanism: parametrized test that loads xfail_debt.md, AST-extracts all xfails, asserts 1:1.

### Phase F3 — Mutation testing graduation

**F3.1 Complete baseline locally**
- Run `bash scripts/mutation/run_full.sh` on local machine (35-min budget per file, ~3 hours total).
- Commit `scripts/mutation/ratchet_baseline.json` with all 5 hotspot baselines.

**F3.2 Validate PEP 604 noise reduction empirically**
- For each new baseline file, confirm `mutations_skipped` / `mutations_total` < 30%.
- If a file has > 30% skipped, investigate (likely real bitwise-or in production).

**F3.3 Address `mlflow_relay.py` surviving mutations**
- Add 3 targeted tests:
  1. Per-kind dispatch test (mutation 3): publish one event per kind, assert correct client method fires.
  2. Cooldown boundary test (mutation 4): test exactly-on-boundary returns `False`.
  3. Breaker defaults test (mutation 6): test `MLflowRelayCircuitBreaker()` with no kwargs produces documented defaults.
- Re-run mutation testing on `mlflow_relay.py`; raise threshold to 0.65 in `.mutation-hotspots.yml`.

**F3.4 Flip `mode: blocking` after first green nightly**
- Trigger: 1 nightly run on `RESEACRH` with all 5 hotspots green + ratchet check passes.
- Manual flip: edit `.mutation-hotspots.yml`. Commit. PR description explains the flip.

### Phase F4 — Process gates

**F4.1 Verify allowlist decay actually fires**
- Manually add an entry with a 366-day-old `renewed_date`, run sentinel, expect failure.
- Revert.
- Document this verification in `mock_policy.md`.

**F4.2 Schedule quarterly mock-policy review as DURABLE cron**
- Use `mcp__scheduled-tasks__create_scheduled_task` with `cronExpression: "0 9 1 */3 *"` (first of every 3rd month, 9 AM).
- Task: read latest `mock_inventory.csv`, compare against `_mock_allowlist.py`, produce a Markdown delta.
- The task creates a new GitHub issue or just leaves a Markdown report (your preference).

**F4.3 Subagent `.claude/CLAUDE.md` update**
- Add an "Agent testing workflow" section:
  - Before declaring "done" on production code changes: run `bash scripts/mutation/validate_agent_output.sh`.
  - If new xfail added: must add row to `xfail_debt.md` in same PR.
  - If new Protocol introduced: sentinel auto-detects (no manual step).

**F4.4 Verify sentinel dynamic Protocol discovery**
- Add a new dummy Protocol in a test fixture; sentinel should pick it up automatically.
- Document the assertion.

**F4.5 New-xfail sentinel**
- Add `tests/_lint/test_xfail_debt_completeness.py` that:
  - Walks all `@pytest.mark.xfail(strict=True, ...)` decorators via AST.
  - Loads `docs/migration/xfail_debt.md`.
  - Asserts each marker has a reason-text mentioning an entry that exists in xfail_debt.md (by file path or test ID).

### Phase F5 — Documentation + Dashboard

**F5.1 `docs/testing/README.md` — canonical index**
- Top-level: "How testing works in this repo" — 1-page summary linking everything.
- Sections: layers (L0-L12), policies (mock, mutation), debt (xfail), onboarding.

**F5.2 `scripts/testing/status.py` — auto-generated status dashboard**
- Emits Markdown summary:
  - Tests collected
  - Tests passing / xfailed / skipped
  - Mock count (from `scripts/mock_inventory.py`)
  - xfail count + age distribution
  - Latest mutation testing kill rates (from `scripts/mutation/reports/`)
  - Allowlist entries + age
- Output: `docs/testing/STATUS.md` (regenerated on demand; sentinel test verifies it's < 7 days stale).

**F5.3 `Makefile` targets**
- `make test-quick`: unit lane only, ~90s.
- `make test-full`: all test trees, ~6 min.
- `make test-mutation`: incremental mutation on diff, ~25 min.
- `make test-status`: regenerate STATUS.md.

### Phase F6 — Validation + Cleanup

**F6.1 Full lane green from clean clone**
- Fresh worktree, `uv sync --all-extras`, `make test-full`. Expect: green.

**F6.2 Mutation testing PR-time gate**
- Open a draft PR touching `event_bus.py`. Verify `.github/workflows/mutation-pr.yml` runs, reports kill rate, leaves PR comment.

**F6.3 Sentinel verification matrix**
- Each sentinel test runs in isolation + as part of the full lane. Document.

**F6.4 Commit + merge to RESEARCH**

---

## Test types covered (per requirements)

Required variations + mapping to this plan:

| Variation | Coverage in plan |
|---|---|
| **Positive + negative** | F1.1-F1.4 (each fix has a "does the right thing fail / pass" check) |
| **Boundary** | F3.3 mutation 4 (exactly-on-cooldown boundary); existing eventually/clock harness |
| **Invariants** | Existing sentinel suite (Protocol discovery, no-mock policy, import-linter) |
| **Dependency errors** | F1.3 (HF version), F1.4 (LIVE-gated tests); each must fail loudly OR skip with reason |
| **Regressions** | F4.5 (new-xfail sentinel), F1.5 (collection regression sentinel), ratchet check |
| **Logic-specific** | F3.3 mlflow_relay surviving mutations; per-kind dispatch |
| **Combinatorial** | Existing chaos catalog + load runloader (after F1.1, F1.2 unblock them) |

---

## Risks + Decisions (3-iteration deep-think)

| # | Risk | Decision | Application |
|---|------|----------|-------------|
| R1.1 | Deleting bare `scenario.py` may break `_discover_scenarios()` coupling | Make `@register_scenario` IDEMPOTENT on identical class — accept duplicates if same class | F1.1 strategy = decorator fix, NOT file deletion |
| R1.2 | `RunLoader` design is real work, not a 5-min fix | DEFER. Mark `tests/load/` as collect-ignore in conftest; move to Tier-5 deferred | F1.2 reworded |
| R1.3 | HF version pin propagates instability | Try helixql TOML fix first; fallback `pytest.skipif` if env-dependent. **Time-box 1 hour** | F1.3 reworded |
| R1.4 | 88 DRIFT xfails triage is days of work | **Hard cap**: fix 2 smallest clusters (~30 tests); document rest with owner+SLA | F2.2/F2.3 reworded |
| R1.5 | Mutation baseline takes ~3hrs wall clock | Run in BACKGROUND (`Bash run_in_background`); do F4–F5 in parallel | Phase ordering note |
| R1.6 | Quarterly cron via scheduled-tasks unreliable | Use `.github/workflows/quarterly-review.yml` cron instead | F4.2 method changed |
| R1.7 | xfail-debt sentinel parser brittle | Simple contract: reason text contains `xfail-debt:<id>` token | F4.5 spec |
| R2.1 | Mutation baseline env-coupled to local machine | Baseline from CI nightly ONLY; commit after one stable run | F3.1 reworded |
| R2.2 | `mode: blocking` flip is fragile | Hold advisory MIN 4 weeks + 4 consecutive green nightlies; no auto-flip | F3.4 wording |
| R2.3 | Makefile conflicts with `uv run` | Every target uses `.venv/bin/python -m pytest` | F5.3 spec |
| R2.4 | STATUS.md staleness → false reassurance | Emit data-source timestamps; sentinel asserts max age 14 days | F5.2 contract |
| R2.5 | New-xfail sentinel blocks emergency flakes | Scope strict-True only; escape hatch `xfail-debt: ad-hoc-<timestamp>` | F4.5 spec |
| R2.6 | Mutation gate blocks feature work on PRE-EXISTING low kill rates | Use `git merge-base origin/main HEAD` as baseline for new branches | F3 spec |
| R2.7 | Acceptance criteria not measurable | Each criterion gets exact command + expected output | Acceptance section rewritten |
| R3.1 | cosmic-ray version drift | Pin via `uv add --dev cosmic-ray>=8.4.6,<9.0.0` → enters `uv.lock` | F3.1 |
| R3.2 | Should we add `fail_under` coverage gate? | **OPEN — ASK USER (Q1 below)** | TBD |
| R3.3 | chaos/load wired into CI nightly? | **OPEN — ASK USER (Q2 below)** | TBD |
| R3.4 | Subagent gate slows agent productivity | Trigger only when subagent diff touches `packages/*/src/**/*.py` | F4.3 spec |
| R3.5 | CLAUDE.md mutation without user consent | Include exact diff in plan; user approves before write | F4.3 spec |
| R3.6 | Cron creation timing | Create at end (F6.4), not at plan approval | F6.4 |
| R3.7 | helixql/HF rabbit-hole risk | Time-box 1 hour, fallback `skipif` | F1.3 |
| R3.8 | Plan size → drift mid-execution | Each phase has STOP-AND-REPORT checkpoint | F1-F6 all |

---

## Decisions confirmed (2026-05-12, user-approved)

| Q | Decision | Impact on plan |
|---|----------|----------------|
| Q1 xfail scope | **Full sweep** | Phase F2 expands: triage + convert ALL 88 strict-True xfails to either passing OR DEAD-delete OR documented hard-DRIFT. Estimated 5-7 days work. |
| Q2 mutation flip | Auto: 4 green nightlies + 2 weeks no FP | New script `scripts/mutation/maybe_flip_to_blocking.py` checks last 4 nightlies + 2-week PR runs; auto-flips `.mutation-hotspots.yml` and opens PR |
| Q3 chaos/load | Collectable + chaos-nightly | F1.1 fixes idempotent registration; NEW: `.github/workflows/chaos-nightly.yml` runs catalog daily; tests/load/ collectable-only |
| Q4 coverage gate | Not needed + "test file exists" sentinel | New: `tests/_lint/test_every_module_has_tests.py` asserts every `packages/*/src/.../*.py` (non-`__init__`) has at least one `tests/**/test_*.py` referencing it |

## Plan additions from these decisions

### F2 expanded — Full sweep xfail triage

**F2.0 Bucketing pass**
- Read every strict-True xfail; classify into:
  - `RESOLVABLE-NOW` (test-only refactor against existing helpers/factories) — convert in F2.1
  - `RESOLVABLE-WITH-HELPER` (needs a new tests/_fakes/* factory or fixture) — write helper in F2.2, convert in F2.3
  - `DEAD` (production code removed; test references nothing real) — delete in F2.4
  - `HARD-DRIFT` (significant test rewrite, but production behavior is real) — document with owner+trigger in F2.5
- Output: `docs/migration/xfail_full_sweep.md` listing every test with its bucket.

**F2.1 Convert RESOLVABLE-NOW (estimated 30-40 tests)**
- DatasetSource cluster (~22 tests in `test_stages_model_retriever.py`)
- Provider __init__ cluster — extend `tests/_fakes/provider_context.py::make_provider_context()`; sweep all 82 sites.

**F2.2 Build helpers for RESOLVABLE-WITH-HELPER (~20-30 tests)**
- `tests/_fakes/single_node_config.py::make_single_node_training_config(**overrides)` — typed pydantic factory.
- `tests/_fakes/training_monitor.py::make_monitor_with_log_manager()` — wires `_provider`, `_client`, etc.

**F2.3 Convert RESOLVABLE-WITH-HELPER (~20-30 tests)**

**F2.4 Delete DEAD (~10-15 tests)**
- Verify each via runtime introspection (`hasattr` check) before deletion.

**F2.5 Document HARD-DRIFT remainder (~5-10 tests)**
- Each gets an `xfail-debt:<id>` token in reason; entry in `xfail_debt.md` with owner+trigger.

**Acceptance**: < 10 strict-True xfails remain; all referenced in `xfail_debt.md`; sentinel `test_xfail_debt_completeness` green.

### F3.4 expanded — auto-flip script

**F3.4.1** `scripts/mutation/maybe_flip_to_blocking.py`:
- Read last 4 nightly reports from GitHub artifacts (`gh run list --workflow=mutation-nightly.yml`)
- Read last 14 days of PR mutation runs (`gh run list --workflow=mutation-pr.yml --created '>=2 weeks ago'`)
- IF all nightlies green AND zero false-positive PR runs → patch `.mutation-hotspots.yml` `mode: advisory` → `mode: blocking`
- Open auto-PR via `gh pr create` with summary of stable runs
- ELSE: print "advisory stays for now; needs N more green nightlies"

**F3.4.2** Schedule via `.github/workflows/mutation-flip-check.yml` weekly cron.

### F4.6 — "test file exists" sentinel

`tests/_lint/test_every_module_has_tests.py`:
- Walk `packages/*/src/**/*.py` (skip `__init__.py`)
- For each module, search for `from <module>` or `import <module>` in any `tests/**/*.py`
- Allow override via `.test-coverage-allowlist.yaml` (modules that genuinely don't need a test, e.g., pure type stubs)
- Fail if any module unreferenced AND not allowlisted.

### F1.6 — chaos-nightly workflow

`.github/workflows/chaos-nightly.yml`:
- cron `0 4 * * *` (after mutation-nightly at 03:00 UTC)
- `pytest tests/chaos/ -m chaos -v`
- Upload `tests/.debug_bundles/` on failure
- Advisory only initially (workflow does NOT block any merge); blocking after first 7 days observation.

## Risks added (4th iteration — Full-sweep specific)

| # | Risk | Mitigation |
|---|------|------------|
| R4.1 | F2 takes 7-10 days, not 5-7 — scope blowout | Hard checkpoint after F2.0 (bucketing): if RESOLVABLE-NOW + RESOLVABLE-WITH-HELPER > 80 tests, STOP and re-negotiate scope |
| R4.2 | Converting xfails may break the lane (a previously-xfailed test passes but its surrounding test code is stale) | Per-file commits with re-run; revert on lane red |
| R4.3 | DEAD test deletion may delete tests the user wants to keep as documentation | Verify each DEAD candidate via `hasattr` runtime check; output to `xfail_full_sweep.md` for review before deletion |
| R4.4 | Building 2 new `tests/_fakes/*` helpers may take longer than expected (helpers need to mirror evolving production APIs) | Time-box each helper to 2 hours; if blocked, downgrade its tests to HARD-DRIFT (document, don't convert) |
| R4.5 | Auto-flip script reads from GitHub API; depends on `gh` CLI availability + token | Document in CLAUDE.md the prereqs; script gracefully exits with message if `gh` missing |
| R4.6 | chaos-nightly needs sidecar stack (fake-runpod, fake-mlflow, fake-vllm) — may not be CI-friendly | First version uses Docker Compose; if too heavy, mark as `manual-only` and revisit |
| R4.7 | "test file exists" sentinel false-positives on modules that legitimately have no tests (config dataclasses, type stubs, __init__) | Allowlist `.test-coverage-allowlist.yaml` with reason per entry; sentinel reads it |

---

## Acceptance criteria (with exact verification commands)

| # | Criterion | Verification command | Expected |
|---|-----------|---------------------|----------|
| AC1 | Zero collection errors | `.venv/bin/python -m pytest tests/ --collect-only -q 2>&1 \| tail -3` | last line: `N tests collected in M.MMs`, no `ERROR collecting` |
| AC2 | Main lane green | `.venv/bin/python -m pytest tests/unit tests/integration tests/e2e --no-header -q` | `N passed, M xfailed, 0 failed` |
| AC3 | xfails all strict-True + documented | `pytest tests/_lint/test_xfail_debt_completeness.py` | green |
| AC4 | Mutation baseline complete | `jq '.files \| keys \| length' scripts/mutation/ratchet_baseline.json` | `5` |
| AC5 | mlflow_relay threshold raised | `grep mlflow_relay .mutation-hotspots.yml` shows `min_kill_rate: 0.65` AND latest nightly report passes | green |
| AC6 | `.mutation-hotspots.yml` mode decision | either `mode: blocking` after 4 green nightlies OR explanatory comment for staying advisory | comment present |
| AC7 | Mock policy quarterly review scheduled | `cat .github/workflows/quarterly-review.yml` | file exists + cron set |
| AC8 | Allowlist decay enforcement verified | Manual: temporarily add stale entry, run sentinel, expect failure | failure produced |
| AC9 | Sentinel dynamic Protocol discovery | `.venv/bin/python -m pytest tests/_lint/test_no_protocol_mocking.py -v 2>&1 \| grep 'discovered'` | `38` protocols (or current count, > 30) |
| AC10 | `docs/testing/README.md` exists + links all policies | `ls docs/testing/README.md && grep -l mock_policy docs/testing/README.md` | both succeed |
| AC11 | Status dashboard works | `.venv/bin/python scripts/testing/status.py > /tmp/status.md && head -1 /tmp/status.md` | Markdown header line |
| AC12 | Makefile targets work | `make test-quick` (smoke), `make test-status` | green outputs |
| AC13 | cosmic-ray pinned in uv.lock | `grep cosmic-ray uv.lock` | entry present with hash |
| AC14 | CLAUDE.md agent gate updated | `grep validate_agent_output .claude/CLAUDE.md` | match found |
| AC15 | Final commit + merge | `git log RESEACRH..HEAD --oneline` empty after merge | empty |

