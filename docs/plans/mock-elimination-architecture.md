# Mock Elimination — Architectural Plan

> Долгосрочный, архитектурно правильный план по устранению `unittest.mock` из greenfield-тестов. **Не "прям щас"**, а правильное решение на перспективу.

---

## Context

Greenfield-миграция тестов завершена (commit `b5e6740` на RESEACRH). Лента зелёная: 6823 passed, 0 failed, 88 xfailed. Но **2334 `unittest.mock` вызова в 153 файлах** — это унаследованный pattern из легаси, который попал в greenfield через lift-and-shift.

Mock-инвентарь (от regex-аудита, AST-уточнение в работе):
- **1620** bare `MagicMock()` — data carriers (самая большая группа)
- **420** `patch.object` / `patch.dict`
- **159** `@patch(...)` decorators
- **93** `AsyncMock`
- **40** `MagicMock(spec=ConcreteClass)`
- **2** `create_autospec`
- **0** `mock_open`

Top-3 файла по mock-heavy:
- `tests/unit/control/test_pipeline_orchestrator.py` — 382
- `tests/unit/pod/test_phase_executor.py` — 363
- `tests/unit/control/test_mlflow_events.py` — 172

## Why mocks are the residual problem

(Resume архитектурного анализа предыдущих фаз)

**Mock-heavy тесты дают ложный сигнал:**
- `@patch("...JobClient")` + `assert_called_with(X)` тестирует **взаимодействие**, не **поведение**
- При рефакторинге `JobClient.send` → `JobClient.dispatch` тест падает, хотя система работает
- Это **false positive**, который тратит часы на investigation

**Cargo culture:**
- 153 файла с mock'ами → молодой разработчик копирует ближайший паттерн → mock-вселенная самовоспроизводится

**Pyramid imbalance:**
- 92.7% unit (target 70%) — много unit-тестов с mock'ами замораживают low-level детали

**Подтверждение из community 2024-2026 ресерча:**
- HN consensus (June 2024): "engineers spent time on mocks easily outstripped time a good build engineer would have spent creating a fast reusable test framework"
- xUnit Patterns (Meszaros): fakes — отдельная категория, не "вариант mock"
- Martin Fowler "Mocks Aren't Stubs": mockist tests куда более brittle при рефакторинге чем classicist (real + stub) tests
- Real Python: autospec лечит signature drift, но не behavior drift

## Goal

**Уменьшить с 153 файлов и 2334 mock-вызовов до <20 файлов и <200 вызовов** — оставив только **legitimate interaction tests** с явным WHY-комментарием.

Не "удалить все mocks", а **eliminate UNACCEPTABLE patterns**. Acceptable mocks (interaction-tests на external APIs, поведение = вызов) остаются с документацией.

---

## Architecture vision (target state)

После elimination структура `tests/` выглядит так:

```
tests/
├── _fakes/                          # Канонические Fakes per Protocol/heavy class
│   ├── mlflow.py, lifecycle.py, runpod.py, ... (уже есть, 9 штук)
│   └── (новые при необходимости)
├── _factories/                      # NEW — pure-function builders for values
│   ├── pipeline_config.py           # make_pipeline_config(...)
│   ├── dataset_source.py            # уже есть, перенести сюда
│   ├── pod_info.py                  # make_pod_info(...)
│   ├── run_dto.py                   # make_run_dto(...)
│   └── ...
├── _helpers/                        # NEW — common test utilities
│   ├── caplog_wrappers.py           # типизированные ассерты для caplog
│   ├── env_isolation.py             # extensions of monkeypatch.setenv
│   └── time_travel.py               # ManualClock fixtures
├── _harness/                        # Eventually, Clock, telemetry, debug, chaos, stack
├── _lint/                           # sentinels — extended in Phase 5
│   ├── test_no_protocol_mocking.py  # уже есть
│   └── test_no_bare_magicmock.py    # NEW — Phase 5
└── unit/ component/ contract/ ...   # сами тесты
```

**Тесты используют:**
- **Real pydantic instances** (через `_factories/`)
- **SimpleNamespace** для ad-hoc data carriers (NOT bare MagicMock)
- **Canonical Fakes** для stateful collaborators
- **pytest `caplog` / `monkeypatch.setenv` / `tmp_path`** для среды (НЕ unittest.mock)
- **Real classes constructed with Fake deps** для DI'd компонентов

**Acceptable осталось:**
- `caplog` (pytest-native)
- `monkeypatch.setenv` (pytest-native)
- `monkeypatch.setattr` (pytest-native, для env-level — НЕ для production functions)
- Rare interaction-test spies на external APIs (subprocess.run, requests.get) с **WHY-комментарием**

---

## Mock taxonomy — детальная категоризация

### Категория A: MECHANICAL (high-volume bulk conversion)

| Pattern | Count | Conversion | Risk |
|---|---|---|---|
| `MagicMock()` bare data carrier | ~1620 | `SimpleNamespace(...)` | Low |
| `patch.dict("os.environ", {...})` | ~50 | `monkeypatch.setenv(...)` | Very low (pytest-native) |
| `@patch("time.monotonic"/"time.time")` | ~10 | `ManualClock` injection | Low |
| `MagicMock(spec=PydanticModel)` | ~30 | real `Model(...)` via factory | Low |
| `@patch("ryotenkai_*.IProtocol*")` | ~0 (banned) | already done | N/A |

**Subtotal: ~1710 mechanical conversions.**

### Категория B: FAKE-DRIVEN (medium-volume, fakes-based)

| Pattern | Count | Conversion | Risk |
|---|---|---|---|
| `MagicMock(spec=StatefulConcreteClass)` | ~10 | real construct + Fakes for deps | Medium |
| `AsyncMock()` for HTTP/SSH/MLflow clients | ~50 | existing canonical Fakes | Medium |
| Heavy `patch.object(instance, "method")` chains | ~100 | DI refactor (production additive) | Medium |

**Subtotal: ~160 fake-driven conversions.**

### Категория C: REFACTOR-OR-DELETE (judgment per-file)

| Pattern | Count | Action | Risk |
|---|---|---|---|
| `@patch("module._private_function")` | ~80 | Test smell — refactor or DELETE | Medium |
| `patch.object(...)` on internal attrs | ~100 | Same — refactor or DELETE | Medium |
| `@patch("module.SOME_CONSTANT")` | ~30 | Parametrized fixture | Low |

**Subtotal: ~210 judgment conversions. Some become DELETEs.**

### Категория D: ACCEPTABLE (keep with WHY-comment)

| Pattern | Count | Action |
|---|---|---|
| `@patch("subprocess.run")` for system calls | ~20 | Keep, add WHY-comment |
| `@patch("logging.*")` for behavior-is-call tests | ~10 | Migrate to `caplog` if simple, else keep |
| `monkeypatch.*` (already pytest, not unittest.mock) | N/A | Not in inventory |
| Rare external-API spies | ~30 | Keep with WHY-comment |

**Subtotal: ~60 acceptable. Documented in allowlist.**

### Категория E: BANNED (already enforced)

| Pattern | Count | Mechanism |
|---|---|---|
| `MagicMock(spec=IProtocol)` | 0 | `test_no_protocol_mocking.py` sentinel |
| `@patch("...IProtocol")` | 0 | same sentinel |
| `create_autospec(IProtocol)` | 0 | same sentinel |

**Subtotal: 0 (sentinel-enforced from Day 1).**

### Totals

- Mechanical: ~1710 (73%)
- Fake-driven: ~160 (7%)
- Refactor-or-delete: ~210 (9%)
- Acceptable: ~60 (3%)
- Banned: 0 (8% of original 2334 were banned-style; now 0)
- **Estimated final state:** ~60 acceptable mocks in <20 files. **Reduction: 97%.**

---

## Phased approach (7 phases)

### Phase 1 — INVENTORY (1 batch, ~1 day)

**Deliverable:**
- `scripts/mock_inventory.py` — AST-based scanner
- `docs/migration/mock_inventory.csv` — per-pattern row
- `docs/migration/mock_inventory.md` — categorized summary

**Subagent in progress** (background) — see `a9dc7595fe4e5fd95`.

**Exit criterion:** every mock invocation in `tests/` categorized as MECHANICAL/JUDGMENT/REFACTOR/ACCEPTABLE.

### Phase 2 — MECHANICAL conversions (5 sub-batches, ~5 days)

In leverage order (count × inverse_complexity):

**Phase 2A — `MagicMock()` → `SimpleNamespace` (~1620 conversions)**

Build a codemod (libcst-based, NOT regex):
1. Find `MagicMock()` constructed bare (no spec=)
2. Identify what's done with it next 5 lines
3. If pure data carrier (`obj.foo = 1; obj.bar = 2`) → emit `SimpleNamespace(foo=1, bar=2)`
4. If used as method-return on a parent mock → keep (interaction test)

Manual review: a small sample of 10% to validate codemod.

**Phase 2B — `patch.dict("os.environ", ...)` → `monkeypatch.setenv` (~50)**

Codemod:
1. Find `@patch.dict(os.environ, {...}, clear=False)` or `with patch.dict(os.environ, ...)`
2. Convert to `monkeypatch.setenv(...)` fixture usage
3. Pytest auto-cleans on test end (no `clear()` reset needed)

**Phase 2C — Time patches → `ManualClock` (~10)**

Codemod:
1. Find `@patch("time.monotonic", side_effect=[...])` or similar
2. Inject `ManualClock` into the SUT (if SUT accepts Clock — it should, per `_harness/clock.py`)
3. Use `clock.advance(seconds)` in test

If SUT doesn't accept Clock → additive production change (constructor param with default `RealClock()`).

**Phase 2D — `MagicMock(spec=PydanticModel)` → real instances (~30)**

Build `tests/_factories/` per pydantic model. For each `MagicMock(spec=X)`:
1. Identify pydantic model X
2. Build factory `make_X(**overrides) -> X` with sensible defaults
3. Replace `MagicMock(spec=X)` with `make_X(...)`

**Phase 2E — `@patch("...time.")` and `@patch("...random.")` → seeded fixtures (~20)**

Codemod or manual: replace with `ManualClock` (time) or `random.Random(seed=...)` injection.

**Exit criterion Phase 2:** ~1730 conversions done. Lane stays green. xfail count unchanged (or improves).

### Phase 3 — FAKE-DRIVEN conversions (3 sub-batches, ~3 days)

**Phase 3A — Async client mocks → existing Fakes (~50)**

For each `AsyncMock(...)` used as HTTP/SSH/MLflow client:
1. Identify the client class
2. Use existing canonical Fake (FakeJobClient, FakeSSHClient, FakeMLflowManager, FakeRunPodAPI)
3. Inject via constructor

**Phase 3B — `MagicMock(spec=StatefulClass)` → real construct (~10)**

For each `MagicMock(spec=PipelineOrchestrator)` etc.:
1. Check if real class is cheap to construct with Fake deps
2. If yes → real construct (preferred)
3. If no → build new Fake under `_fakes/`

**Phase 3C — Heavy `patch.object` chains → DI refactor (~50)**

For SUTs with 5+ patch.object on internal attrs:
1. Identify the deps being patched
2. Additive production: add constructor params with defaults
3. Tests pass deps explicitly instead of patching

**Exit criterion Phase 3:** All Protocol/client mocks gone. Real classes constructed with Fakes.

### Phase 4 — REFACTOR-OR-DELETE (5 sub-batches, ~5 days)

For each `@patch("module._private_function")` or `patch.object(instance, "_internal_method")`:

**Decision tree:**
1. Is `_private_function` testable through the public API? → DELETE the mock, test through public API
2. Is it really an external dependency in disguise? → Extract Protocol, add to canonical Fakes
3. Is the test redundant with compliance/property/component tests? → DELETE
4. Otherwise → ACCEPT with WHY-comment + add to allowlist

Per-file judgment. Batch by package (control pipeline, pod runner, etc.).

**Exit criterion Phase 4:** No `@patch` of internal functions/methods.

### Phase 5 — SENTINEL EXTENSION (1 batch, ~1 day)

Extend `tests/_lint/test_no_protocol_mocking.py` (or add new sentinel `test_no_bare_magicmock.py`):

```python
_BANNED_PATTERNS = {
    "MagicMock_bare": "Use SimpleNamespace or canonical Fake instead",
    "MagicMock_spec_pydantic": "Use real pydantic instance via tests/_factories/",
    "patch_internal_function": "Refactor production for DI, then construct with Fakes",
    "patch_dict_os_environ": "Use monkeypatch.setenv(...) (pytest-native)",
}
```

Allowlist file `tests/_lint/_mock_allowlist.py` for documented interaction tests:

```python
ALLOWED_MOCKS = {
    "tests/unit/some_file.py::test_subprocess_call": "Behavior IS the subprocess.run invocation; cannot test otherwise",
}
```

Sentinel ensures allowlist shrinks monotonically (entries can be removed but not added without review).

### Phase 6 — MUTATION TESTING (1 batch, ~2 days)

**Tool choice: Cosmic Ray** (not mutmut).

Rationale (per 2024-2026 industry comparisons + ICSE 2026 PyTation paper):
- Cosmic Ray is most mature for production; widest operator set; parallel execution; ~460K recent downloads
- mutmut had incomplete results in academic comparison
- mutatest + MutPy unmaintained (last commits 2-6 years ago)
- PyTation (newest 2026 research) finds DIFFERENT mutations than Cosmic Ray — could complement, but Cosmic Ray is safer first choice

```bash
pip install cosmic-ray
cosmic-ray init config.yml session.sqlite
cosmic-ray exec session.sqlite
cr-html session.sqlite > report.html
```

For each mutation that survives (no test catches it):
- If a test SHOULD catch it but doesn't → fix the test
- If no test could reasonably catch it → low-signal test indication; investigate which tests cover that line

Identify tests with 0 mutation-kill rate → DEAD candidates → DELETE.

**Exit criterion Phase 6:** mutation kill rate >70% on top-10 high-traffic production modules.

### Phase 7 — CONTINUOUS (ongoing)

- Sentinel in PR-blocking CI lane
- Allowlist shrinks monotonically (audit quarterly)
- Test-quality dashboard (mutation kill rate per package, mock count per file)
- New test PRs: must follow new patterns or document why not

**Exit criterion Phase 7:** N/A (continuous).

---

## Timeline

| Phase | Duration | Cumulative |
|---|---|---|
| 1 — Inventory | 1 day | 1 day |
| 2 — Mechanical | 5 days | 6 days |
| 3 — Fake-driven | 3 days | 9 days |
| 4 — Refactor-or-delete | 5 days | 14 days |
| 5 — Sentinel | 1 day | 15 days |
| 6 — Mutation testing | 2 days | 17 days |
| 7 — Continuous | ∞ | ongoing |

**~3 weeks focused effort.** Lane stays green throughout.

---

## Tooling

### Codemod choice — `libcst` (NOT regex)

Why libcst:
- Preserves comments, whitespace, formatting
- AST-aware (won't mess up multi-line)
- Used by Black, isort, Instagram's monorepo refactors

Install: `pip install libcst`.

Codemods live in `scripts/codemods/`:
- `magicmock_to_simplenamespace.py`
- `patchdict_environ_to_monkeypatch.py`
- `time_patch_to_manualclock.py`
- `magicmock_spec_to_factory.py`

Each codemod:
1. Takes a file path
2. Does AST transformation
3. Writes back the transformed source
4. Has a `--dry-run` flag that prints diff without writing
5. Has unit tests (codemod tests itself before being applied broadly)

### Codemod testing — libcst `assertCodemod` pattern

Per libcst official docs and SeatGeek's "Refactoring Python with LibCST" pattern (industry-validated):

Each codemod has tests in `scripts/codemods/tests/`:
- `test_cases/<scenario>/before.py` — original code
- `test_cases/<scenario>/after.py` — expected refactored code
- Test uses `CodemodTest.assertCodemod(before, after, **kwargs)` from `libcst.codemod`

This is **meta-testing**: tests of the test-conversion tool. Critical.

**TDD workflow** (SeatGeek-validated):
1. Write `before.py` + `after.py` for a new scenario
2. Run test → fails (codemod doesn't handle it)
3. Extend codemod to make test pass
4. Repeat for next scenario

**Rollout** (also SeatGeek-validated):
- Phase 2A pilot: run codemod on 20-200 lines of test code, review by hand
- Once confidence: roll out to full scope
- Refusal-to-handle: codemod skips file with `SkipFile` exception + emits a `# TODO(codemod-skipped): <reason>` comment; manual conversion later

---

## Per-test variation requirements (per user's testing checklist)

For each new factory/Fake/helper introduced, must cover:

1. **Позитивный** — happy path returns expected value
2. **Негативный** — invalid inputs raise expected errors
3. **Граничный** — empty inputs, max sizes, edge values
4. **Инвариант** — round-trip serialization, idempotency
5. **Ошибки зависимостей** — what happens if injected Fake fails
6. **Регрессия** — captures bug that was found
7. **Логика-specific** — specific business rules
8. **Комбинаторный** — Hypothesis-driven where applicable

---

## Risks + 3 iterations of analysis

### Iteration 1 — surface risks

**R1: Codemod produces broken Python in edge cases.**
- Mitigation: dry-run + diff review on 10% sample before full run; codemod's own tests cover edge patterns.

**R2: Lane goes red mid-batch.**
- Mitigation: per-file commit + verify after each. Revert single file if breaks; never break the lane.

**R3: Some `MagicMock()` is actually used as a method spy (not data carrier).**
- Mitigation: codemod detects context (assignment vs argument) and skips spy cases.

**R4: `SimpleNamespace` doesn't support all `MagicMock` magic methods (e.g., `__call__`).**
- Mitigation: codemod detects callable usage; falls back to keep `MagicMock` for callables.

**R5: Real pydantic instance requires fields the test doesn't care about.**
- Mitigation: factories with sensible defaults; tests pass overrides only for relevant fields.

**R6: Production class refactor for DI (Phase 3C) requires non-additive changes.**
- Mitigation: prefer additive (default param value); if non-additive needed, halt + flag.

### Iteration 2 — deeper risks

**R7: Mock elimination reveals dormant bugs (tests start failing).**
- Honest assessment: This is GOOD news, but disruptive. Schedule fixing in Phase 4-5.
- Mitigation: if >5 dormant bugs per batch, slow down and triage.

**R8: Some tests rely on `MagicMock` returning truthy for `__bool__` — replacing with `None` data carrier breaks them.**
- Mitigation: codemod detects bool-usage; emits richer SimpleNamespace.

**R9: `tests/_factories/` grows large.**
- Mitigation: per-package subdir. `_factories/control/`, `_factories/pod/`, etc.

**R10: Mutation testing is slow (mutmut runs minutes per module).**
- Mitigation: parallel + scope-limited (top-10 modules only); nightly lane.

**R11: Pytest `monkeypatch.setattr` blurs line with unittest.mock.**
- Decision: `monkeypatch.setattr` ONLY for env-level (env vars, sys.modules). NOT for production functions. Document in sentinel.

**R12: Subagent reliability — past agents over-reported success.**
- Mitigation: per-batch verification by me (count check + spot-read 5 files). Same protocol as migration audits.

### Iteration 3 — architectural risks

**R13: After elimination, codebase is harder to test temporarily (until factories built).**
- Mitigation: Phase 2D builds factories BEFORE deleting MagicMock(spec=X). Always factory-first.

**R14: Allowlist (Phase 5) becomes dumping ground for "too hard to fix".**
- Mitigation: quarterly review; allowlist entries decay after 6 months unless renewed.

**R15: Sentinel false-positives block legitimate work.**
- Mitigation: allowlist for documented cases; escape hatch with required reviewer.

**R16: Loss of testing flexibility — some legitimate interaction tests harder to write.**
- Mitigation: keep MagicMock available for explicit interaction tests; just sentinel-flag bare `MagicMock()` as data carrier.

**R17: Fake explosion (Phase 3B creates many Fakes).**
- Mitigation: Fake only for stateful classes with real production analog. Stateless concrete classes → SimpleNamespace or real.

**R18: Mutation testing flags too many "DEAD" tests that are actually defensive.**
- Mitigation: human review of mutation results; not auto-delete. Mutation score < threshold = investigate, not delete.

**R19: Refactoring production for DI (Phase 3C) could destabilize legacy code paths.**
- Mitigation: ONLY additive (constructor param with default). Never break existing call sites.

**R20: Codemod misses runtime-only patterns (e.g., `Mock` aliased on import).**
- Mitigation: codemod uses scope-aware analysis (libcst MetadataWrapper); test on the actual codebase.

### Open questions (need answer before execution)

**Q1: Should we also eliminate `monkeypatch.setattr` for production functions (not env)?**
- **Deep think 3 iter:**
  - Iter 1: `monkeypatch.setattr(module, "func", replacement)` is pytest-native, not unittest.mock. But same anti-pattern (patching production module attr).
  - Iter 2: Sentinel for `monkeypatch.setattr` is harder — function patches are sometimes legitimate (e.g., patching CWD for tests).
  - Iter 3: Decision: scope sentinel to **unittest.mock only** for now. Address `monkeypatch.setattr` as separate effort if it remains a problem post-Phase 5.
- **Answer:** OUT OF SCOPE. Focus on unittest.mock first.

**Q2: How to handle tests that have BOTH legitimate and illegitimate mocks?**
- **Deep think 3 iter:**
  - Iter 1: Convert illegitimate, keep legitimate with WHY-comment.
  - Iter 2: Per-mock decision: codemod runs on each call site, leaves others alone.
  - Iter 3: Decision: yes, codemod does per-mock conversion; tests can have N kept + M converted mocks.
- **Answer:** Per-mock conversion (granular).

**Q3: What if a test SHOULD use a Fake but no canonical Fake exists for the class?**
- **Deep think 3 iter:**
  - Iter 1: Build new Fake in Phase 3B.
  - Iter 2: Cost-benefit: if only 1-2 tests need it, real construct is cheaper.
  - Iter 3: Rule: Fake required if (a) class has state machine AND (b) ≥5 tests need it. Else real-construct or SimpleNamespace.
- **Answer:** Rule formalized.

**Q4: How to verify codemod correctness at scale?**
- **Deep think 3 iter:**
  - Iter 1: Run full test suite after each batch.
  - Iter 2: Also: AST-diff before/after to ensure no logic change beyond mock conversion.
  - Iter 3: Plus: spot-check 10% of converted files by hand.
- **Answer:** Triple gate — test suite + AST diff + spot-check.

**Q5: What if codemod conversions reveal that an "elimination" actually corrupts coverage?**
- **Deep think 3 iter:**
  - Iter 1: Coverage diff per batch: must not decrease significantly.
  - Iter 2: If coverage drops, investigate the affected lines. Either: (a) the test was DEAD (Welcome, deletion), (b) the conversion lost a meaningful assertion (fix the conversion).
  - Iter 3: Rule: coverage drop >2% per batch → halt, investigate.
- **Answer:** Coverage gate per batch.

---

## Best practices validation

(To be filled in after audit subagent completes inventory + I cross-check against industry refs)

Sources to validate against:
1. **Martin Fowler "Mocks Aren't Stubs"** (2007, still canonical)
2. **xUnit Test Patterns by Meszaros** (test doubles taxonomy)
3. **"Fake It Don't Mock It" by Shai Yallin** (2022)
4. **HN discussion June 2024** (fakes vs mocks, real-DB vs mocks)
5. **Real Python autospec guidance**
6. **Google Testing Blog** (testing on the toilet — fake-driven testing)

Cross-check:
- Phase 2A (SimpleNamespace) matches xUnit pattern "Dummy Object"
- Phase 2D (real pydantic) matches Fowler "classicist" school
- Phase 3 (Fakes) matches Yallin's "Fake, Don't Mock"
- Phase 6 (mutation testing) matches industry standard (PIT for Java, mutmut for Python)

---

## Critical files

### Will be created
- `scripts/mock_inventory.py` (Phase 1) — AST scanner
- `scripts/codemods/{magicmock_to_simplenamespace,patchdict_to_monkeypatch,time_to_manualclock,spec_to_factory}.py` (Phase 2)
- `scripts/codemods/tests/` — codemod meta-tests
- `tests/_factories/` — pydantic-model builders (Phase 2D)
- `tests/_helpers/` — caplog wrappers, env isolation (Phase 4)
- `tests/_lint/test_no_bare_magicmock.py` (Phase 5) — new sentinel
- `tests/_lint/_mock_allowlist.py` (Phase 5) — documented exceptions
- `docs/migration/mock_inventory.{csv,md}` (Phase 1)
- `docs/migration/mock_elimination_log_*.md` per batch

### Will be modified (additive)
- `tests/_lint/test_no_protocol_mocking.py` — extend with new bans
- Production code: ONLY additive constructor params with defaults (Phase 3C)
- `pyproject.toml` — add `libcst` to dev deps; add `cosmic-ray` (Phase 6)
- `Makefile` — add `lint-no-bare-mocks`, `run-codemods` targets
- `.github/workflows/presubmit-blocking.yml` — add new sentinels
- `.github/workflows/nightly.yml` — add mutation testing job

### Will be modified (test files only — 153 files)
- All files using `unittest.mock` — converted per batch

---

## Verification per phase

### Phase 1 (Inventory)
```bash
.venv/bin/python scripts/mock_inventory.py > docs/migration/mock_inventory.csv
wc -l docs/migration/mock_inventory.csv  # expect ~2400 lines
.venv/bin/python -m pytest -c tests/pytest.ini tests/ 2>&1 | tail -3  # 0 failed
```

### Phase 2 (Mechanical, per sub-batch)
```bash
# Run codemod on N files
python scripts/codemods/magicmock_to_simplenamespace.py tests/unit/<scope> --apply

# Verify lane stays green
.venv/bin/python -m pytest -c tests/pytest.ini tests/ 2>&1 | tail -3

# Coverage diff
pytest --cov=packages --cov-report=term-missing tests/ | tee /tmp/coverage_after.txt
diff /tmp/coverage_before.txt /tmp/coverage_after.txt | head

# Mock count delta
grep -c "MagicMock\|@patch\|AsyncMock" tests/<scope>/  # expect decrease
```

### Phase 3 (Fake-driven, per sub-batch)
- Same as Phase 2 + isinstance checks for new Fakes

### Phase 4 (Refactor-or-delete)
- Coverage diff per file (don't lose meaningful coverage)
- Mutation score check on affected production files

### Phase 5 (Sentinel)
- Synthetic violation test: introduce bare `MagicMock()` in temp file, sentinel fails
- Allowlist tests: documented exemptions are accepted

### Phase 6 (Mutation testing)
- mutmut on top-10 modules → kill rate ≥70%

### Phase 7 (Continuous)
- PR-blocking sentinel
- Weekly burn-down dashboard

---

## Success metrics

| Metric | Start | Target |
|---|---|---|
| Files using `unittest.mock` | 153 | **<20** |
| Total `unittest.mock` invocations | ~2334 | **<200** |
| `MagicMock()` bare | ~1620 | **0** |
| `MagicMock(spec=Pydantic)` | ~30 | **0** |
| `@patch("...IProtocol")` | 0 | **0** (maintained) |
| `tests/_factories/` files | 0 | ~10 (one per major pydantic model) |
| Mutation kill rate (top-10 mods) | unknown | **>70%** |
| Lane green throughout | ✅ | ✅ (maintained) |
| Production code refactoring (additive only) | 11 files | ~20 files (Phase 3C additions) |

---

## Anti-goals (what we WON'T do)

- **NOT** rewrite tests that already use canonical Fakes
- **NOT** delete tests without coverage diff or mutation analysis
- **NOT** force-convert legitimate interaction tests (subprocess.run spies, etc.)
- **NOT** introduce new mocking frameworks (pytest-mock, etc.)
- **NOT** break the green lane
- **NOT** require non-additive production changes
- **NOT** rush — 3 weeks of careful work beats 1 week of broken tests

---

## Next actions

1. ✅ Audit subagent running in background (Phase 1)
2. ⏳ Wait for inventory → cross-check categorization
3. Launch Phase 2A subagent (highest leverage, lowest risk)
4. Iterate batches 2A → 2E → 3A → 3B → 3C → 4 → 5 → 6
5. Each batch: subagent does the work, I verify (count + spot-read), proceed
6. Final report at end of Phase 6
