# Mock Policy (Greenfield Tests)

> Status: **enforced** by `tests/_lint/test_no_protocol_mocking.py`
> Last updated: 2026-05-12 (Phase 5 of the mock-elimination plan)

This document is the canonical reference for when `unittest.mock` may
appear in `tests/` and how the allowlist process works.

## TL;DR

1. **Default: do not mock.** Prefer real construction with fakes from
   `tests/_fakes/`, factories from `tests/_factories/`, or
   `SimpleNamespace` data carriers.
2. **Protocol mocking is banned.** `MagicMock(spec=IFoo)` /
   `@patch("…IFoo")` is rejected by the sentinel
   (`test_no_mocking_of_protocols_in_tests`).
3. **AsyncMock / `MagicMock(spec=ConcreteClass)` need an allowlist
   entry.** Either eliminate the mock or add a row to
   `tests/_lint/_mock_allowlist.py` with a justification.
4. **External-library `@patch` is excused by pattern.** Patching
   `torch.cuda.*`, `time.*`, `mlflow.*`, `huggingface_hub.*`,
   `subprocess.*`, `datasets.*`, `peft.*`, `concurrent.futures.*`, or
   `httpx.*` passes the sentinel automatically.
5. **Allowlist entries expire after 365 days** unless their `renewed=`
   field is bumped during the yearly review.

## Why we eliminate mocks

Phases 1–4 of [mock-elimination-architecture.md](../plans/mock-elimination-architecture.md)
established three problems with the historical mock-heavy style:

* **False positives during refactors.** A test that does
  `@patch("…JobClient")` + `assert_called_with(X)` tests *interaction*,
  not behaviour. Renaming a method (`send → dispatch`) breaks the test
  while production still works.
* **Cargo culture.** 153 files with mocks meant new contributors
  copied the nearest pattern; the mock universe self-replicated.
* **Pyramid imbalance.** 93% of tests were unit tests with frozen
  low-level details, leaving integration thin.

Phases 2A and 3B removed the bulk: `MagicMock()` data carriers became
`SimpleNamespace`; `MagicMock(spec=ConcreteClass)` became real factory
construction or canonical fakes.

## When mocks ARE legitimate

Phase 3A surfaced ~73 `AsyncMock` usages that **should stay**. These
fall into four families:

* **Interaction tests on async proxies.** The CLI / API routers forward
  args from one layer to another; the contract IS the call shape
  (`await_args.kwargs.get("grace_seconds") == 15.0`). A real fake
  cannot substitute -- the assertion is on the call, not the result.
* **Lifecycle observers.** Tests that verify `client.aclose()` /
  `tunnel.close()` is awaited exactly once at the right moment.
  Counters on `await_count` cannot be replicated cheaply by a fake.
* **Retry-loop sleep accounting.** A retry helper that calls
  `sleep(2.5)` exactly five times -- pinning the schedule by
  `assert_awaited_with(2.5)` / `await_count == 5` requires a mock.
* **Module-level free async functions.** Patching a free async
  function (`patch("module.foo", new=AsyncMock(...))`) is the canonical
  pattern; there is no class to fake.

All other `AsyncMock` usages have been refactored away in Phase 3A.

The Phase 3B audit also kept exactly one `MagicMock(spec=…)`: the
`threading.Timer` case in `tests/unit/shared/utils/test_cancellation.py`.
A real `Timer` would spawn a thread and inject flakiness for zero
coverage gain.

## The allowlist

The allowlist lives in
[`tests/_lint/_mock_allowlist.py`](../../tests/_lint/_mock_allowlist.py).
Each entry pins either a single `(path, line)` (preferred) or covers a
glob path with a pattern key (used for external-lib `@patch` only).

```python
AllowlistEntry(
    path="tests/unit/control/api/routers/test_jobs_router.py",
    line=319,
    pattern="AsyncMock_interaction",
    reason="Verifies runner.request_stop proxy forwards grace_seconds=15.0 -- behaviour IS the call.",
    added="2026-05-12",
    renewed="2026-05-12",
)
```

Required fields:

* `path` -- pinned file path or glob pattern (for `line=0`).
* `line` -- exact line number for the mock call site (`0` for pattern entries).
* `pattern` -- short identifier (see "Patterns vocabulary" in the
  module docstring).
* `reason` -- one-sentence explanation of WHY the mock is legitimate.
* `added` -- ISO date when the entry first appeared.
* `renewed` -- ISO date of the most recent review.

### Adding an entry

Adding to the list is a smell. Before opening a PR with a new entry,
walk through this decision tree:

1. **Can the test be rewritten with a fake / factory / SimpleNamespace?**
   If yes -- do that. No allowlist entry needed.
2. **Is the production class hard to construct?** Build a fake under
   `tests/_fakes/` instead.
3. **Is the test asserting on the call itself (interaction contract)?**
   Then the mock is load-bearing. Add an allowlist entry with a
   one-sentence reason explaining what the assertion pins.

A new allowlist entry should always come with a comment near the
allowlist row in code review, e.g. *"This is a CLI proxy interaction
test -- assert_awaited_with grace_seconds is the contract."*

### Removing an entry

Removing entries is encouraged. When you refactor a test to drop its
mock, also drop the corresponding allowlist row in the same PR. The
sentinel will catch you if you remove the row without removing the
mock -- the test fails until both sides match.

### Renewing entries

Every allowlist entry has a `renewed` ISO date. After 365 days
without renewal, the sentinel test
`test_allowlist_entries_renewed_within_365_days` fails -- forcing
either re-blessing or elimination.

To renew: bump the `renewed` date in code review, with a one-line
note in the PR description stating "still legitimate because …".

## Patterns vocabulary

Pinned entries use one of these `pattern` values:

| Pattern | Meaning |
|---|---|
| `AsyncMock_interaction` | per-method async stub; test asserts on `await_args` / `await_count` |
| `AsyncMock_lifecycle_observer` | `aclose()` / `close()` lifecycle hook fires exactly once |
| `AsyncMock_retry_sleep` | retry-loop sleep schedule pinned by `await_count` / `assert_awaited_with` |
| `AsyncMock_module_free_async` | patches a module-level free async function (no Protocol involved) |
| `AsyncMock_factory_default` | `AsyncMock()` as default in a test-factory parameter signature |
| `MagicMock_spec_threading_Timer` | one-off cancellation-test observer for `threading.Timer.cancel()` |

Pattern entries (line=0) use one of these:

| Pattern | Prefix matched |
|---|---|
| `patch_torch_cuda` | `torch.cuda.*` |
| `patch_time` | `time.sleep`, `time.time`, `time.monotonic`, `time.perf_counter` |
| `patch_mlflow_external` | `mlflow.set_tracking_uri`, `mlflow.genai.*`, etc. |
| `patch_huggingface_hub` | `huggingface_hub.*` |
| `patch_subprocess` | `subprocess.run`, `subprocess.Popen`, `subprocess.check_*` |
| `patch_datasets` | `datasets.load_dataset`, `datasets.get_dataset_*` |
| `patch_peft` | `peft.*` |
| `patch_concurrent_futures` | `concurrent.futures.*` |
| `patch_httpx_external` | `httpx.*` |

The prefix-to-pattern mapping lives in `_mock_allowlist.EXTERNAL_PATCH_PREFIXES`.
Adding a new external library requires both a pattern entry and a
prefix-tuple update.

## Good examples

### Real factory + real construction

```python
from tests._factories.run_data import make_run_data
rd = make_run_data(metrics={"train_loss": 0.5})
# rd is a real mlflow.entities.RunData
```

### SimpleNamespace data carrier

```python
from types import SimpleNamespace
config = SimpleNamespace(
    get_primary_dataset=lambda: SimpleNamespace(kind="local"),
    training=SimpleNamespace(strategies=()),
)
```

### Canonical fake

```python
from tests._fakes.mlflow import FakeMLflowManager
mlflow = FakeMLflowManager()
mlflow.setup()
# ... exercise SUT ...
assert mlflow.log_dict_calls == [...]
```

## Bad examples

### Protocol mock (banned)

```python
# Sentinel rejects this:
client = MagicMock(spec=IJobClient)
client.get_status.return_value = {"state": "running"}
```

**Fix:** use `tests/_fakes/job_client.py::FakeJobClient`.

### Bare MagicMock as data carrier

```python
# Tolerated for now (Phase 2A still in flight); will be banned later.
obj = MagicMock()
obj.foo = 1
obj.bar = 2
```

**Fix:** use `SimpleNamespace(foo=1, bar=2)` (matches xUnit pattern
"Dummy Object").

### Patching an internal private function

```python
# Tolerated for now (Phase 4 still in flight); will be banned later.
@patch("ryotenkai_control.pipeline.foo._private_helper")
def test_…(_): ...
```

**Fix:** refactor production for dependency injection (additive
constructor param with default), then test with a real fake.

## How the sentinel runs

```bash
make test-mock-policy        # the canonical entry point
# or directly:
.venv/bin/python -m pytest -c tests/pytest.ini tests/_lint/test_no_protocol_mocking.py -v
```

The sentinel runs in every PR via the pre-merge lane and on `main`
post-merge. The two failure modes are:

* `test_no_unallowlisted_mocks` -- you added a mock without an
  allowlist row.
* `test_allowlist_entries_renewed_within_365_days` -- the yearly
  renewal slipped.

Both surface the exact file and line.

## History

* **2026-05-12 (Phase 5):** allowlist + sentinel extension.
* **2026-05-12 (Phase 3B):** `MagicMock(spec=ConcreteClass)` eliminated
  bar one threading.Timer KEEP.
* **2026-05-12 (Phase 3A):** `AsyncMock` audit; 9 conversions, 73 KEEPS.
* **Earlier 2026-05:** Phase 2A converted ~270 `MagicMock()` data
  carriers to `SimpleNamespace`; Phase 2B replaced
  `patch.dict(os.environ)` with `monkeypatch.setenv`.

See [`docs/migration/phase_3a_log.md`](../migration/phase_3a_log.md),
[`docs/migration/phase_3b_log.md`](../migration/phase_3b_log.md), and
[`docs/migration/mock_inventory.md`](../migration/mock_inventory.md)
for the full audit trail.

---

## Quarterly Allowlist Review (Task 3 mechanism)

**Cadence:** every 90 days (or when sentinel flags stale entries).

**Trigger:**
- Automatic: `test_allowlist_entries_renewed_within_365_days` sentinel fires
  in CI when ANY entry's `renewed=` field is >365 days old → PR blocked
- Manual: maintainer schedules quarterly review

**Process** (~1 hour/quarter):
1. Run `pytest tests/_lint/test_no_protocol_mocking.py -v` — confirms current state
2. For each entry approaching 365 days:
   - Re-read the test
   - Verify the `reason=` is still accurate (production behavior unchanged)
   - Either:
     - **Renew**: bump `renewed=` to today's date in `_mock_allowlist.py`
     - **Remove**: delete the entry (the test no longer needs the mock OR the test was removed)
     - **Refactor**: convert the mock to a Fake/factory; remove entry
3. Re-run sentinel; commit changes

**Verification:**
```bash
.venv/bin/python -m pytest -c tests/pytest.ini tests/_lint/test_no_protocol_mocking.py::test_allowlist_entries_renewed_within_365_days -v
```

This is the **only** mechanism preventing allowlist sprawl. Without quarterly
review, entries accumulate and the lint becomes cargo. Lifecycle of an entry:
`added → renewed (1×/year) → removed`. Maximum allowlist size grows linearly
with TRUE legitimate use, not with technical debt.

---

## Natural Erosion Policy (Task 4 mechanism)

**Principle:** every PR that touches a file in `packages/<pkg>/src/` MUST
also clean up the corresponding test file's mocks if any of these apply:

| When you touch `packages/X/src/foo.py` | Required action in same PR |
|---|---|
| Add/change/remove a method | Migrate any `patch.object(*, "method_name")` in `tests/.../test_foo.py` |
| Add/change a constructor param | Migrate `MagicMock(spec=Foo)` to factory or Fake |
| Refactor a function signature | Audit test mocks for that function path |
| Touch an internal helper (`_private`) | Refactor test to use real instance (not patch.object) |

**Why this works** (proven in industry — k8s, Argo, Grafana):
- Mocks evaporate organically as code evolves
- No "big-bang migration" risk
- New code follows new patterns (sentinel-enforced)
- Old code dies a natural death

**Enforcement:** code review checklist. NOT automated (would be too noisy).
A reviewer who sees an unchanged mock-heavy test alongside a touched
production file should request the migration in the same PR.

**Tracking:** quarterly metric — `grep -rc "unittest.mock" tests/ | wc -l`.
Trend should be monotonically decreasing. If it stalls > 2 quarters,
investigate (likely a hotspot that needs a dedicated refactor PR).

---

## Tooling reference

- Sentinel: [tests/_lint/test_no_protocol_mocking.py](../../tests/_lint/test_no_protocol_mocking.py)
- Allowlist: [tests/_lint/_mock_allowlist.py](../../tests/_lint/_mock_allowlist.py)
- Mock inventory script: [scripts/mock_inventory.py](../../scripts/mock_inventory.py)
- Codemods: [scripts/codemods/](../../scripts/codemods/) (4 libcst transforms)
- Compliance harness: [tests/contract/protocol_compliance/](../../tests/contract/protocol_compliance/)

## History

- 2026-05-12 — Phase 5 landed; allowlist sentinel + 75 entries + 365-day decay
- 2026-05-12 — Phase 4A docker DI; Phase 4B mlflow_manager kwarg
- 2026-05-12 — Quarterly review + Natural Erosion policies documented (this doc)
