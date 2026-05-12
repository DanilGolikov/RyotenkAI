# Phase 3B — `MagicMock(spec=ConcreteClass)` Elimination

Status: **COMPLETE**. Lane green: **6829 passed, 0 failed**, 88 xfailed
(was 6825 before; +4 from new factory unit tests).

## Summary

| Metric | Before | After |
|---|---:|---:|
| Concrete-class `MagicMock(spec=X)` invocations | 22 | 1 (KEPT) |
| `MagicMock(spec=[...])` list-spec (acceptable) | 5 | 5 |
| New factory modules | 0 | 2 (`run_data`, `__init__`) |
| New factory tests | 0 | 4 |
| `FakeMLflowManager` surface additions | — | `log_dict` + observation API |

All concrete-class spec= mocks eliminated except `threading.Timer`,
which is documented in-place with a WHY-comment per Phase 3B rules.

## Per-class conversions

### 3B.1 — `MagicMock(spec=MLflowManager)` ×7 → `FakeMLflowManager`

Files touched:
* `tests/unit/control/pipeline/test_stage_artifact_collector.py`

Strategy: extended the canonical
[`tests/_fakes/mlflow.py:FakeMLflowManager`](../tests/_fakes/mlflow.py)
with a `log_dict(dictionary, artifact_file, *, run_id)` method plus a
small observation API (`log_dict_calls`, `set_log_dict_error`,
`set_log_dict_return`). Tests now construct a real fake, call
`setup()` to activate it (or skip `setup()` to leave it inactive), and
assert on captured `log_dict_calls` directly instead of `Mock` call
introspection.

The fake's `log_dict` lives outside the `IMLflowManager` Protocol
because the Protocol only catalogues the *pipeline-facing* surface;
`log_dict` is exposed by the concrete trainer-side `MLflowManager` and
relied on by `save_stage_artifact`. The docstring on
`FakeMLflowManager.log_dict` explains this asymmetry.

Tests pass (71 / 71).

### 3B.2 — `MagicMock(spec=PipelineConfig)` ×4 → `SimpleNamespace`

Files touched:
* `tests/unit/control/pipeline/test_stages_validator.py` (3)
* `tests/unit/control/pipeline/stages/dataset_validator/test_plugin_loader.py` (1)

Strategy: `DatasetValidator` and `PluginLoader` only read
`get_primary_dataset()` and `training.strategies` on their config
argument. A real `PipelineConfig` requires a full `model` /
`providers` / `datasets` registry; building one for these tests would
swamp the test in unrelated setup. The replacement is a
`SimpleNamespace` with callable attributes — duck-typed test double,
no `MagicMock` magic.

NOT a factory: a `make_pipeline_config` factory would still produce
the same "test double for a config-shaped read surface", just with a
heavier baseline. The tests already document via the factory
docstring why a `SimpleNamespace` is preferred.

Tests pass (16 / 16).

### 3B.3 — `MagicMock(spec=PipelineStage)` ×5 → `_FakeStage` test double

Files touched:
* `tests/e2e/control/test_stages_integration.py`

Strategy: the original fixtures invoked `stage.run(config, secrets,
context)` which is **not** the real `PipelineStage.run(context)`
signature. The `spec=PipelineStage` claim was load-bearing-by-name
only. Introduced a small in-test `_FakeStage` class with:
* `stage_name: str`
* `run(config, secrets, context) -> Result`
* `set_run_return(value)` to flip side-effect from happy to failure
  for error-path tests
* `call_count` for assertion compatibility

The fake's docstring explains why we keep the non-canonical
`run(config, secrets, context)` shape rather than refactor to the
real ABC — the E2E intentionally tests sequence-of-callables
orchestration, not the ABC contract.

Tests pass (13 / 13).

### 3B.4 — `MagicMock(spec=RunData)` ×2 → `make_run_data` factory

Files touched:
* `tests/unit/control/reports/test_report_v2.py`
* NEW: `tests/_factories/run_data.py`
* NEW: `tests/_factories/test_run_data.py` (4 unit tests)
* NEW: `tests/_factories/__init__.py` (package docstring)

Strategy: `mlflow.entities.RunData` is a pure value object with
read-only `metrics`/`params`/`tags` properties that only accept lists
of `Metric`/`Param`/`RunTag` entities. The factory takes plain dicts
and converts under the hood — tests get ergonomic kwargs, the
returned object is the real `RunData`.

Tests pass (11 / 11 report_v2 + 4 / 4 factory).

### 3B.5 — `MagicMock(spec=DatasetSourceLocal/HF)` ×2 → already handled

The two usages live inside
[`tests/_fakes/dataset_source.py`](../tests/_fakes/dataset_source.py)
itself — they are the canonical factory's internal fallback for
inputs that fail Pydantic validation (int, empty, etc.). This was
already the approved Phase 2 strategy; no change required.

### 3B.6 — Misc one-offs

| Class | Count | Conversion | File |
|---|---:|---|---|
| `IterableDataset` | 1 | Minimal real subclass of `datasets.IterableDataset` so `isinstance` works without spawning streaming I/O | `tests/unit/control/pipeline/stages/dataset_validator/test_split_loader.py` |
| `Request` (fastapi) | 1 | Real Starlette `Request` from ASGI scope | `tests/unit/pod/runner/api/test_errors.py` |
| `threading.Timer` | 1 | **KEPT** with WHY-comment | `tests/unit/shared/utils/test_cancellation.py` |
| `EndpointInfo` | 0 | Not actually present in code-level usages | — |

#### KEPT entry — `threading.Timer`

```python
# WHY MagicMock(spec=threading.Timer): we need to observe
# ``timer.cancel()`` was invoked exactly once during reset.
# A real ``threading.Timer`` cannot be substituted here without
# actually starting a background thread (its constructor expects
# a callable + non-zero interval and starting/cancelling
# introduces flaky timing into a pure-logic test). The mock
# captures the side-effect contract without spawning a thread.
timer = MagicMock(spec=threading.Timer)
```

A real Timer would require a callable + non-zero interval; the test
only verifies that the reset path calls `cancel()` on whatever
timer is armed. Substituting a real Timer here would introduce
thread-spawn / timing flakiness for zero coverage gain.

## Factories created

### `tests/_factories/run_data.py`

```python
from tests._factories.run_data import make_run_data

rd = make_run_data(
    metrics={"train_loss": 0.5},
    params={"learning_rate": 0.001},
    tags={"mlflow.runName": "exp"},
)
# rd is a real mlflow.entities.RunData
```

Defaults to empty `metrics`/`params`/`tags`; coerces params to
strings (mlflow's convention). Covered by 4 unit tests in
`tests/_factories/test_run_data.py`.

## Spec= count delta

| Category | Before | After |
|---|---:|---:|
| Concrete-class spec= (judgment work) | 22 | 1 |
| List-spec `MagicMock(spec=[...])` (acceptable, attribute allowlist) | 5 | 5 |
| Synthetic strings in lint sentinel | 4 | 4 |
| Internal fallback inside `tests/_fakes/dataset_source.py` | 2 | 2 |

The remaining 1 concrete-class spec mock is `threading.Timer`, kept
with a WHY-comment.

## Lane status

```
6829 passed, 291 skipped, 88 xfailed, 7 xpassed
```

vs Phase 3A baseline (6825 passed). The +4 are the new factory unit
tests (`tests/_factories/test_run_data.py`). 0 failures.

## Open issues / Phase 4 hand-off

* `tests/e2e/control/test_stages_integration.py` retains a non-
  canonical `stage.run(config, secrets, context)` signature in its
  `_FakeStage`. The real `PipelineStage.run(context)` takes only
  `context`. Phase 4 should consider either (a) refactoring the E2E
  to drive real stages with a `(config, secrets)` carried *inside*
  the context, or (b) deleting these tests in favour of the
  integration suite that exercises real stages. The fake's docstring
  flags this for future cleanup.

* `FakeMLflowManager.log_dict` is an extension *beyond* the
  `IMLflowManager` Protocol. If the pipeline ever moves
  `save_stage_artifact` to call something narrower than the concrete
  `MLflowManager` (e.g. an `IArtifactSink` Protocol), this addition
  becomes the canonical sink-fake — it already has call tracking and
  programmable failure injection.

* No production code was modified in Phase 3B (per the hard-rules).
  The earlier "extract a class to make it constructible" escape
  hatch was not needed.

## Reproduce

```bash
cd /Users/daniil/MyProjects/RyotenkAI/.claude/worktrees/ecstatic-khorana-1baa6b
test -L .venv || ln -sf /Users/daniil/MyProjects/RyotenkAI/.venv .venv

# Concrete-class spec= count
grep -rE "^[^#]*= MagicMock\(spec=[A-Z]" tests/ --include="*.py" \
  | grep -v "_fakes/\|_factories/\|test_no_protocol_mocking"
# Expected: 1 line (threading.Timer, with WHY-comment)

# Factory tests
.venv/bin/python -m pytest -c tests/pytest.ini tests/_factories/ -v

# Lane
.venv/bin/python -m pytest -c tests/pytest.ini tests/ -q
# Expected: 6829 passed, 0 failed
```
