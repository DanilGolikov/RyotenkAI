# Phase 2A Log — `MagicMock()` → `SimpleNamespace` codemod

> Phase 2A of the mock-elimination plan
> ([docs/plans/mock-elimination-architecture.md](../plans/mock-elimination-architecture.md)):
> mechanically convert bare `MagicMock()` data carriers to
> `types.SimpleNamespace`.  This is the highest-leverage, lowest-risk
> mechanical conversion in the elimination plan.

## TL;DR

| Metric | Before | After | Delta |
|---|---:|---:|---:|
| `MagicMock` occurrences in `tests/` | 2884 | 2636 | **−248** |
| Test files containing `MagicMock` | 150 | 146 | −4 |
| Files modified by codemod (kept) | — | 52 | +52 |
| Files reverted by auto-verifier | — | 24 | (revert preserved Mock semantics) |
| Lane status | 6823 passed | 6823 passed | unchanged |
| Codemod meta-tests | — | 18 passed | new |

The lane stayed GREEN throughout.  No production code was touched.

## Codemod design

### File layout

```
scripts/
├── __init__.py
└── codemods/
    ├── __init__.py
    ├── magicmock_to_simplenamespace.py   # libcst codemod
    ├── apply_with_revert.py              # per-file apply+verify wrapper
    ├── test_cases/                       # before/after fixtures
    │   ├── simple_data_carrier/
    │   ├── multi_attr_data_carrier/
    │   ├── data_carrier_with_callable_attrs/
    │   ├── delete_unused_magicmock_import/
    │   ├── reassignment_after_use/
    │   ├── post_use_attribute_mutation/
    │   ├── skip_callable/
    │   ├── skip_with_spec/
    │   ├── skip_with_return_value/
    │   ├── skip_with_assert_called/
    │   ├── skip_kwargs_construct/
    │   ├── skip_chained_attr/
    │   ├── skip_monkeypatch_setattr_target/
    │   ├── skip_interleaved_statement/
    │   ├── nested_attr_assignment/
    │   └── inline_argument/
    └── tests/
        └── test_magicmock_to_simplenamespace.py
```

### Detection rules — what gets CONVERTED

`var = MagicMock()` is convertible iff **all** of the following hold:

1. The constructor is bare — no positional args; the only allowed
   kwargs are plain data ones (NOT `spec=`, `spec_set=`, `wraps=`,
   `return_value=`, `side_effect=`, `name=`, `configure_mock=`).
2. The variable is never read inside an assignment target like
   `var.return_value = …`, `var.assert_called_with(…)`, or
   `var.foo.return_value = …`.
3. The variable is never used as a *nested* attribute-assignment
   target (`var.foo.bar = …`) — SimpleNamespace cannot auto-create
   intermediate namespaces.
4. The variable is never passed as the **value** argument to
   `setattr(...)`, `monkeypatch.setattr(...)`, `monkeypatch.setitem`,
   `patch.object(...)`, `patch.dict(...)`, or `patch.multiple(...)`.
   Any such alias hands the value to the system under test, which may
   use Mock-only semantics we cannot see.
5. The variable is never called directly (`var(...)`).

When converted, attribute assignments `var.attr = X` that appear in a
**physically contiguous** block immediately after `var = MagicMock()`
(no interleaved statements) are folded into the constructor as
kwargs:

```python
m = MagicMock()      ──►   m = SimpleNamespace(foo=1, bar=2)
m.foo = 1
m.bar = 2
```

Mutations after the first non-attribute-assignment statement are left
as plain attribute writes — `SimpleNamespace` supports them.

### Why the contiguity rule matters

This bug was caught during expansion to `tests/unit/control/`:

```python
client = MagicMock()
client.get_status = AsyncMock(return_value={"state": "running"})

async def _stream(_job_id, **_kwargs):
    yield ...

client.subscribe_events = _stream   # <-- _stream defined above THIS line only
```

Without the contiguity rule, the codemod folded `subscribe_events=_stream`
into the constructor at the top — generating `client = SimpleNamespace(
get_status=…, subscribe_events=_stream)` *before* `_stream` was
defined.  `UnboundLocalError` at runtime.  Fixed by closing the
absorption window on any non-`var.attr = X` interleaving.

### Import handling

* If any conversion happens in the module, `from types import
  SimpleNamespace` is inserted (after `from __future__` imports if
  any, else after the module docstring, else at the top — with PEP 8
  blank-line conventions).
* If `MagicMock` is no longer used anywhere outside imports, the
  `MagicMock` name is dropped from the `from unittest.mock import …`
  alias list.  If the import list becomes empty, the line is removed.

### Auto-verify wrapper

`scripts/codemods/apply_with_revert.py` runs the codemod *per-file*
and immediately runs that file's pytest tests.  If they fail, it
reverts the file to its pre-codemod state.  This pattern guarantees
the lane stays green even when the codemod's local analysis missed a
cross-file semantic dependency (typically: production code accessing
attributes the test never explicitly set, relying on `MagicMock`'s
auto-creation).

## Test cases — 18 meta-tests

| Scenario | Behaviour |
|---|---|
| `simple_data_carrier` | `m = MagicMock(); m.value = 42` → `m = SimpleNamespace(value=42)` |
| `multi_attr_data_carrier` | multiple attribute assigns folded into kwargs |
| `data_carrier_with_callable_attrs` | the dominant `runner = MagicMock(); runner.X = AsyncMock(...)` pattern |
| `delete_unused_magicmock_import` | drops `MagicMock` from imports when no longer used |
| `reassignment_after_use` | post-use attribute write left as plain mutation |
| `post_use_attribute_mutation` | attribute read does NOT disqualify; later attribute write kept as mutation |
| `skip_callable` | `m()` invocation → keep MagicMock |
| `skip_with_spec` | `MagicMock(spec=X)` → keep |
| `skip_with_return_value` | `m.return_value = …` → keep |
| `skip_with_assert_called` | `m.assert_called_with(…)` → keep |
| `skip_kwargs_construct` | `MagicMock(return_value=…)` → keep |
| `skip_chained_attr` | `m.foo.return_value = …` → keep |
| `skip_monkeypatch_setattr_target` | passing to `monkeypatch.setattr` → keep |
| `skip_interleaved_statement` | non-attr-assign between init and `var.X = Y` closes absorption |
| `nested_attr_assignment` | `m.foo.bar = X` → keep (SimpleNamespace can't auto-create `foo`) |
| `inline_argument` | inline `MagicMock()` (no assignment) left untouched |
| `idempotent` | running twice yields the same result |
| `preserves_unrelated_code` | files without `MagicMock` are unchanged |

## Application results

Applied via `apply_with_revert.py` (per-file dry-run + verify + revert).

| Scope | Files scanned | Files kept | Files reverted | Conversions kept |
|---|---:|---:|---:|---:|
| `tests/unit/control/api/` (pilot) | 8 | 2 | 0 | 32 |
| `tests/unit/control/` (full) | 136 | 30 | 14 | 163 |
| `tests/unit/{community,engines,providers,shared}/` | 86 | 3 | 1 | 12 |
| `tests/unit/pod/` | 99 | 16 | 9 | 61 |
| `tests/{component,contract,integration,property}/` | 46 | 3 | 0 | 5 |
| **Total** | **375** | **54** | **24** | **273** |

(The 273 *conversions kept* number counts the codemod's
`changed_count` per file — i.e. converted `MagicMock()` data-carrier
sites.  Each conversion is at least one MagicMock instantiation
removed; some sites also collapse multiple attribute-assign lines
into one `SimpleNamespace(...)` constructor.  Reduction in raw
`MagicMock` mentions: 2884 → 2636 = **−248**.)

## Reverted files (Phase 2D / 4 candidates)

These files have `MagicMock(...)` data carriers that *look*
convertible to local analysis but cause downstream failures, almost
always because production code under test calls an attribute on the
mock that the test never explicitly set (relying on MagicMock's
auto-attribute behaviour).  Each is a candidate for explicit refactor
in Phase 2D (real pydantic instance via factory) or Phase 3 (DI
refactor):

* `tests/unit/control/evaluation/test_system_prompt_loader.py`
* `tests/unit/control/pipeline/execution/test_stage_execution_loop.py`
* `tests/unit/control/pipeline/execution/test_stage_planner.py`
* `tests/unit/control/pipeline/execution/test_stage_planner_comprehensive.py`
* `tests/unit/control/pipeline/launch/test_launch_preparator.py`
* `tests/unit/control/pipeline/mlflow_attempt/test_manager_comprehensive.py`
* `tests/unit/control/pipeline/reporting/test_summary_reporter.py`
* `tests/unit/control/pipeline/test_gpu_deployer_runner_log.py`
* `tests/unit/control/pipeline/test_model_retriever_metrics_replay.py`
* `tests/unit/control/pipeline/test_stages_model_retriever.py`
* `tests/unit/control/pipeline/validators/test_runtime_validators.py`
* `tests/unit/control/test_mlflow_events.py`
* `tests/unit/control/test_pipeline_orchestrator.py`
* `tests/unit/control/test_pipeline_orchestrator_missing_lines.py`
* `tests/unit/pod/test_chain_runner.py`
* `tests/unit/pod/test_dataset_loaders.py`
* `tests/unit/pod/test_phase_executor.py`
* `tests/unit/pod/trainer/managers/test_model_saver_manager.py`
* `tests/unit/pod/trainer/mlflow/test_run_analytics.py`
* `tests/unit/pod/trainer/orchestrator/test_adapter_cache_phase_executor.py`
* `tests/unit/pod/trainer/orchestrator/test_resume_manager.py`
* `tests/unit/pod/trainer/test_strategies.py`
* `tests/unit/pod/trainer/test_trainer_builder.py`
* `tests/unit/shared/utils/test_cancellation.py`

## Edge cases observed

1. **Auto-attribute reliance.** The single biggest source of reverts.
   Tests do `cfg = MagicMock(); cfg.foo = 1` and the production code
   calls `cfg.bar()` — works on MagicMock (auto-creates `bar`), fails
   on SimpleNamespace.  Cannot be detected by file-local analysis.
   Mitigated by `apply_with_revert.py`.
2. **Aliasing via `monkeypatch.setattr`.** When a candidate becomes a
   production binding (`monkeypatch.setattr(real_obj, "attr", fake)`),
   downstream code may use Mock-only semantics through the alias.
   Detected and skipped explicitly.
3. **Aliasing via `facade.attr = fake`.** Less common pattern; the
   `test_config_service_reward_strategy.py` fixture builds a façade
   whose attribute is the fake.  In that file, the outer façade
   converts (it's only read via attributes the test sets), the inner
   `fake_plugins` is preserved because it's passed to
   `monkeypatch.setattr`.  Works correctly.
4. **Forward-referenced RHS.** A nested function defined between
   `var = MagicMock()` and `var.x = nested_func` cannot be folded
   without changing semantics.  Detected via the contiguity check.
5. **`MagicMock(return_value=X)`.** Treated as a callable definition,
   not a data carrier.  Skipped.
6. **`MagicMock(spec=X)`.** Out of Phase 2A scope (Phase 2D).
   Skipped.

## Open issues for Phase 2B-E

* **Phase 2B** — `patch.dict("os.environ", ...)` → `monkeypatch.setenv(...)`.
  ~50 sites.  Codemod TBD.
* **Phase 2C** — `@patch("time.monotonic", side_effect=[...])` →
  `ManualClock`.  Requires SUT to accept a Clock; typically already
  does or trivial to add.
* **Phase 2D** — `MagicMock(spec=PydanticModel)` → real model via
  `tests/_factories/`.  Most of the *reverted* control tests above
  fit this category — convert their `cfg`/`config` mocks to real
  pydantic instances built by factories.
* **Phase 2E** — `@patch("...time.*")` / `@patch("...random.*")` →
  seeded fixtures.
* **Phase 3A** — `AsyncMock` for HTTP/SSH/MLflow clients → canonical
  fakes under `tests/_fakes/`.  Many of the reverted *pod trainer*
  tests fit this category.
* **Phase 5** — sentinel: extend `tests/_lint/test_no_protocol_mocking.py`
  with `MagicMock_bare` ban once the bulk is converted.

## Verification commands

```bash
cd /Users/daniil/MyProjects/RyotenkAI/.claude/worktrees/ecstatic-khorana-1baa6b

# Codemod meta-tests (always pass)
.venv/bin/python -m pytest scripts/codemods/tests/ -q

# Re-apply with per-file verify+revert (idempotent — no-op now)
.venv/bin/python -m scripts.codemods.apply_with_revert tests/unit/

# Full lane (must pass 6823 / 0 failed)
.venv/bin/python -m pytest -c tests/pytest.ini tests/ 2>&1 | tail -3

# MagicMock occurrence delta
grep -rc MagicMock tests/ | awk -F: '{s += $2} END {print "MagicMock:", s}'
```
