# Testing in RyotenkAI

> Canonical index for everything testing-related. Read this first when
> you're not sure where to look.

## TL;DR

The repository's tests are governed by **three layered gates**:

1. **Lane health** — `pytest tests/unit tests/integration tests/e2e`
   must report `0 failed`. xfails are allowed but each must carry an
   `xfail-debt:<id>` token documented in
   [docs/migration/xfail_debt.md](../migration/xfail_debt.md).

2. **Mock discipline** — no mocking of Protocol types in greenfield
   `tests/`. Each `unittest.mock` use is either deleted, replaced with a
   canonical fake from `tests/_fakes/`, or pinned in
   `tests/_lint/_mock_allowlist.py` with a renewable date. Stale
   entries (renewed > 365 days ago) automatically decay and fail CI.
   See [mock_policy.md](mock_policy.md).

3. **Quality (mutation testing)** — hotspot files declared in
   [`.mutation-hotspots.yml`](../../.mutation-hotspots.yml) are
   protected by per-PR mutation testing (advisory → blocking after
   stability bar). The nightly job enforces a 5-pp kill-rate ratchet.
   See [mutation_testing.md](mutation_testing.md).

Everything else here is implementation detail of those three gates.

---

## Quick reference

| What | Where | Enforced by |
|---|---|---|
| Test layout & categories | [tests/README.md](../../tests/README.md) | layout convention |
| Mock policy + allowlist | [mock_policy.md](mock_policy.md) | `tests/_lint/test_no_protocol_mocking.py` |
| Mutation testing policy | [mutation_testing.md](mutation_testing.md) | `.github/workflows/mutation-*.yml` |
| xfail debt ledger | [../migration/xfail_debt.md](../migration/xfail_debt.md) | `tests/_lint/test_xfail_debt_completeness.py` |
| Test-coverage debt allowlist | [../../tests/_lint/no_test_required.yaml](../../tests/_lint/no_test_required.yaml) | `tests/_lint/test_every_module_has_tests.py` |
| Status dashboard (auto-gen) | [STATUS.md](STATUS.md) | `tests/_lint/test_status_freshness.py` |

## Sentinel suite (`tests/_lint/`)

Each sentinel guards a different invariant:

| Sentinel | Guards |
|---|---|
| `test_no_protocol_mocking.py` | Protocols are not mocked; concrete `MagicMock(spec=...)` allowlisted |
| `test_xfail_debt_completeness.py` | Every strict-True xfail has an `xfail-debt:<id>` token + ledger row |
| `test_allowlist_decay_verified.py` | The 365-day decay mechanism itself fires when fed a stale entry |
| `test_protocol_discovery_invariants.py` | Dynamic Protocol discovery sees ≥ 30 protocols + named anchors |
| `test_every_module_has_tests.py` | Every `packages/*/src/**/*.py` is referenced by ≥ 1 test (or allowlisted) |
| `test_clean_collection.py` | `pytest --collect-only` succeeds on every top-level test tree |
| `test_status_freshness.py` | `docs/testing/STATUS.md` was regenerated within 14 days |

## Onboarding workflow

```bash
# 1. Clone + sync deps
uv sync --all-extras

# 2. Quick lane (~90s)
make test-quick

# 3. Full lane (~6 min)
make test-full

# 4. Sentinel only
.venv/bin/python -m pytest tests/_lint

# 5. Mutation testing on your diff (incremental)
make test-mutation

# 6. Refresh the status dashboard
make test-status
```

## Agent testing workflow

The `.claude/CLAUDE.md` section "Agent testing workflow (mandatory)"
describes the gates a subagent must satisfy before declaring "done"
when its diff touches `packages/*/src/`. Summary:

1. Run `bash scripts/mutation/validate_agent_output.sh` — must pass.
   (Default base ref is the integration branch — currently `RESEACRH`;
   override via `MUTATION_BASE_REF=<ref>` or pass as `$1`.)
2. Any new strict-True xfail MUST carry an `xfail-debt:<id>` token AND
   a matching `xfail_debt.md` row.
3. No Protocol mocking. Sentinel will catch you.
4. New production modules must have a test reference OR an explicit
   `no_test_required.yaml` entry.

## Test layers (12-layer cake)

| L | Name | Where | Purpose |
|---|---|---|---|
| L0 | Static analysis | sentinels in `tests/_lint/` | Catch structural regressions cheaply |
| L1 | Unit | `tests/unit/` | One class, all collaborators replaced with fakes |
| L2 | Component | `tests/unit/.../*_component.py` (marker) | One production class + canonical fakes |
| L3 | Contract | `tests/contract/` | Protocol compliance fake/real parametrize |
| L4 | Integration | `tests/integration/` | Multiple production classes wired together |
| L5 | Stack | `tests/stack/` | Hermetic sidecar stack (HTTP) |
| L6 | E2E | `tests/e2e/` | Full pipeline from CLI invocation |
| L7 | Golden | `tests/golden/` | Snapshot of artefacts (reports, configs) |
| L8 | Property | `tests/_harness/property.py` consumers | Hypothesis-driven invariants |
| L9 | Chaos | `tests/chaos/` | Failure injection + recovery assertions |
| L10 | Load | `tests/load/` | SLO-bound performance scenarios |
| L11 | Mutation | `scripts/mutation/` | External validation of suite quality |
| L12 | Replay regression | `tests/replay/` (planned) | Real-prod artefact replays |
