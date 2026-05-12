# Phase 3A log — AsyncMock conversions

Phase 3A targeted the `AsyncMock` population identified in the
[mock inventory](mock_inventory.md): ~82 invocations across 9
production test files (Phase 2 audit had said "~93" because of double-counting
imports + lines).

## Categorization

Built `scripts/audit/asyncmock_usage.py` — an AST sweep that for each
`AsyncMock(...)` call emits:

* `file:line`
* enclosing variable / keyword / positional-arg context
* shape (`bare` / `return_value` / `side_effect` / `spec`)
* whether the surrounding test uses interaction-style asserts
  (`assert_awaited*`, `await_count`, `await_args`, `call_args*`)
* guessed replacement (FakeJobClient / FakeMLflowManager / KEEP / noop)

```bash
.venv/bin/python -m scripts.audit.asyncmock_usage > /tmp/asyncmock_audit.csv
```

### Pre-Phase-3A landscape (82 invocations)

| Class | Count | Action |
|---|---|---|
| KEEP (per-method async stub, e.g. `client.get_status = AsyncMock(return_value={...})`) | 35 | KEEP |
| KEEP (interaction test — `assert_awaited*` / `await_count` / `await_args` on it) | 22 | KEEP |
| KEEP (`patch(..., new=AsyncMock(...))` for module-level async free function) | 8 | KEEP |
| KEEP (per-method async stub with `side_effect=...`) | 7 | KEEP |
| Replace with `async def _noop(...): pass` (bare placeholder coroutine callable) | 10 | CONVERT |

The 10 "Replace with noop" hits broke down as:

* 9 in `tests/unit/pod/runner/test_mlflow_relay.py` — bare `AsyncMock()`
  passed positionally to `MLflowRelay(...)` as the `forward_fn`
  argument; the tests *never* invoke it (they exercise the relay's
  queue / circuit-breaker invariants), so the mock surface adds zero
  value and merely defeats the sentinel.
* 1 in `tests/unit/pod/runner/test_pod_terminator_retry.py` line 60 —
  the `_make_terminator` factory's `sleep=sleep or AsyncMock()`
  fallback. Every test in the file passes an explicit `AsyncMock`
  (line 84, 112, ...) so the fallback path is dead code. Left alone
  to keep the type hint (`sleep: AsyncMock | None`) honest — a future
  Phase 4 cleanup can swap to a real `async def` placeholder.

### Why no Fake substitution

None of the remaining `AsyncMock` usages corresponds to a whole Protocol
(`IJobClient` / `IPodLifecycleClient` / `IMLflowManager`). All are:

1. **Per-method async stubs on a `SimpleNamespace` façade** (Phase 2A had
   already converted the surrounding façades from `MagicMock()` to
   `SimpleNamespace`; the AsyncMock literal pins one method's return
   value):

   ```python
   runner = SimpleNamespace(
       get_status=AsyncMock(return_value={"state": "running"}),
   )
   ```

   Substituting `FakeJobClient` here would *over-fake* — the runner-side
   proxy under test only consumes `get_status`, doesn't care about the
   full IJobClient state machine, and verifying the wire shape (the
   point of this test) needs the per-call return-value, not a state
   machine. Per Phase 3A's "**Per-file judgment**" rule, KEEP.

2. **Interaction tests on `client.method` (`assert_awaited_once_with`,
   `await_args.kwargs`).** Replacing the AsyncMock with a Fake breaks
   the test's contract: it's *asserting on the call*, not the result.

3. **Module-level free async functions** (`patch("pkg.module.func",
   new=AsyncMock(return_value=...))`). No Fake exists for a free
   function; `patch.new=AsyncMock` *is* the canonical pattern here.

This matches the project plan's "**~50 ~~Async client mocks → existing
Fakes~~**" estimate being optimistic — Phase 2A's MagicMock-→
SimpleNamespace codemod already absorbed most of the "façade" work.
What remains is the **callable surface** of those façades.

## Conversions applied

### `tests/unit/pod/runner/test_mlflow_relay.py` (9 hits → 0)

Replaced 9 `AsyncMock()` positional-arg placeholders with a single
module-level `async def _noop_forward(_event)` coroutine.

```python
# BEFORE
relay = MLflowRelay(AsyncMock(), worker_idle_poll_s=0.01)

# AFTER
async def _noop_forward(_event: dict[str, Any]) -> None:
    """No-op forward_fn placeholder for tests that don't exercise the forwarder."""

relay = MLflowRelay(_noop_forward, worker_idle_poll_s=0.01)
```

Removed the now-unused `from unittest.mock import AsyncMock` import.

Tests: **40 passed, 3 skipped** — unchanged from before the conversion.

## AsyncMock count delta

| Metric | Before | After |
|---|---|---|
| `AsyncMock(...)` call invocations across `tests/unit/` | 82 | 73 |
| Files with at least one `AsyncMock` call | 9 | 9 (mlflow_relay still has the `__doc__` reference but 0 actual calls) |
| `from unittest.mock import AsyncMock` imports | 9 | 8 |

Conversion ratio: **9 / 82 ≈ 11%**. This is consistent with the
"AsyncMock conversions are HIGHER variance than Phase 2A" note from
the plan — the bulk of remaining cases are either legitimate
interaction-tests or per-method stubs on `SimpleNamespace` façades
that were already Phase 2A-blessed.

## Lane status

```
6825 passed, 0 failed, 291 skipped, 88 xfailed, 7 xpassed in 369.78s
```

Identical to pre-Phase-3A baseline. No tests broken, no regressions.

## Files KEPT with WHY

| File | Pattern | Why kept |
|---|---|---|
| `tests/unit/control/api/routers/test_jobs_router.py` | 13 × `AsyncMock(return_value=...)` / `side_effect` on `SimpleNamespace(get_status=...)`, `subscribe_events=...`, `request_stop=...` | Per-method stubs; tests use `await_args.kwargs` to verify the proxy forwards `grace=15` as `grace_seconds=15.0` etc. Interaction-test contract. |
| `tests/unit/control/cli/test_job_command.py` | 12 × same pattern, with `assert_awaited_once`, `await_args.kwargs.get("grace_seconds")` | Same — CLI proxy interaction contract. |
| `tests/unit/control/pipeline/test_training_monitor_v2.py` | 11 × `client.aclose = AsyncMock(return_value=None)` + `client.get_status = AsyncMock(...)` followed by `client.aclose.assert_awaited_once` / `assert_not_awaited` | Interaction-test on `cleanup()` lifecycle: SUT must NOT call `aclose` before `cleanup()`. AsyncMock is the cheapest way to record the await. |
| `tests/unit/pod/runner/test_pod_terminator_retry.py` | 9 × `sleep = AsyncMock()` followed by `sleep.assert_not_awaited()` / `sleep.assert_awaited_with(2.5)` / `sleep.await_count == 5` | Pure interaction: pin that the retry loop sleeps exactly N times with the right tick. |
| `tests/unit/pod/runner/api/test_resources.py` | 8 × `patch("...default_health_snapshot", new=AsyncMock(return_value=snap))` | Patching a module-level free async function. No Protocol involved; no Fake exists or would help. |
| `tests/unit/control/pipeline/stages/managers/deployment/test_training_launcher_v2.py` | 11 × `SimpleNamespace(health_check=AsyncMock(...), submit_job=AsyncMock(...), aclose=AsyncMock(...))` | The whole test class is `@pytest.mark.xfail(strict=True)` post-packagization; touching it would either un-xfail a known-broken assertion or re-revive a maintenance liability. Skip per "lane must end GREEN" rule. |
| `tests/unit/control/pipeline/stages/managers/deployment/test_training_launcher_runner.py` | 4 × same `SimpleNamespace`-with-method-stubs pattern, partly in xfail tests | Mixed; the non-xfail uses still rely on `assert_called_once` on a synchronous launch_runner mock, so the `AsyncMock` is needed just to satisfy the constructor surface. KEEP. |
| `tests/unit/control/pipeline/stages/managers/deployment/test_file_uploader.py` | 1 × `SimpleNamespace(upload_file=AsyncMock(return_value=MagicMock(bytes_written=42)))` with `await_count` / `await_args.args[0]` interaction | Interaction-test; `fake_client.upload_file.await_args.args[0]` pins the upload target enum. |

The one `AsyncMock` left in `test_mlflow_relay.py` is a docstring
reference (cosmetic), not a call. The import statement `from
unittest.mock import AsyncMock` was deleted.

## Open issues for Phase 3B / 4

* **`test_training_launcher_v2.py` / `test_training_launcher_runner.py`
  xfail debt.** Most `AsyncMock` chains there are inside `@xfail`
  test classes that have drifted post-packagization. Phase 3B should
  either un-xfail (fix the SUT/test wiring) or delete those test
  classes — once the xfail is resolved, the `AsyncMock` chains can be
  swapped for `FakeJobClient` + a fake SSH tunnel.

* **`test_pod_terminator_retry.py` factory default.** The `sleep =
  sleep or AsyncMock()` fallback is dead code (every caller passes
  `sleep` explicitly). A trivial Phase 4 cleanup could replace it
  with `async def _noop_sleep(_: float) -> None: pass` and drop the
  type hint reference to `AsyncMock`. Leave as-is for now to avoid
  churning a passing file.

* **No new Fakes were needed.** The hypothesis that "~50 AsyncMock
  usages map to existing Fakes" turned out to be optimistic — Phase
  2A's MagicMock→SimpleNamespace codemod already eliminated the
  "whole-Protocol mock" pattern; what remains is finer-grained
  per-method stubbing and interaction-testing of async APIs, which is
  the *correct* use of AsyncMock per the test-doubles taxonomy.

* **Sentinel allowlist (Phase 5).** The 73 remaining `AsyncMock`
  invocations should be reviewed once the bulk Phase 4 refactors land.
  If they are still load-bearing interaction tests, they can be
  added to `tests/_lint/_mock_allowlist.py` (per the Phase 5 plan in
  `docs/plans/mock-elimination-architecture.md`) with a one-line
  WHY-comment each.

## Verification commands

```bash
cd /Users/daniil/MyProjects/RyotenkAI/.claude/worktrees/ecstatic-khorana-1baa6b

# Categorization
.venv/bin/python -m scripts.audit.asyncmock_usage > /tmp/asyncmock_audit.csv
tail -n +2 /tmp/asyncmock_audit.csv | cut -d, -f7- | sort | uniq -c | sort -rn

# Per-file invocation count
grep -rho "AsyncMock(" tests/unit/ | wc -l   # 73 after Phase 3A

# Lane green
.venv/bin/python -m pytest -c tests/pytest.ini tests/ 2>&1 | tail -3
```
