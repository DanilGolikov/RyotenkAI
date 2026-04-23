# Починка sibling-invariant для run.lock + удаление TUI

## Контекст

Активная ADR «Sibling Client Architecture for Pipeline State» декларирует: `PipelineStateStore` (src/pipeline/state/store.py) — единственный канонический writer для `pipeline_state.json` и `run.lock`. Канонический lifecycle lock — через `acquire_run_lock()` и `PipelineRunLock.release()` / `RunLockGuard`. Invariant #1 из [run_lock_guard.py](../../src/pipeline/state/run_lock_guard.py): «every acquired run.lock must be released through the guarded path».

Дипсинк выявил две категории проблем:

**1. Sibling-инвариант нарушен в нескольких точках:**

- `src/api/services/launch_service.py:97` — API при `interrupt()` напрямую делает `(run_dir / "run.lock").unlink()` для cleanup stale-lock. Это обходит PipelineStateStore, хардкодит имя файла, и не делает content-match (читает pid, проверяет alive, а между этими шагами другой процесс может захватить lock заново — теоретическая race).
- `src/main.py:335` — CLI-утилита `_load_config_for_run()` читает `pipeline_state.json` напрямую через `json.loads(state_file.read_text())` вместо `PipelineStateStore(run_dir).load()`. Дублирует десериализационную логику и не проверяет schema_version.

**2. TUI — устаревшая функциональность:**

Web — основной UI. TUI (`src/tui/`, ~1177 LOC + 11 тестов на ~2141 LOC) не нужен. Единственный внешний потребитель — CLI-команда `ryotenkai tui` в `src/main.py:1270-1332`. Зависимость `textual[syntax]>=8.1.0,<9.0.0` используется только TUI. Упоминания в README.md (TUI-секция + 3 скриншота), CONTRIBUTING.md, run.sh, docs/web-ui.md.

## Изменения

### Часть A: Sibling-invariant fix

**A1. Новая функция в [src/pipeline/state/store.py](../../src/pipeline/state/store.py):**

Добавить рядом с `acquire_run_lock()`:

```python
def read_lock_pid(lock_path: Path) -> int | None:
    """Read pid= line from a run.lock file. None if missing/corrupt."""
    try:
        content = lock_path.read_text(encoding="utf-8")
    except OSError:
        return None
    for line in content.splitlines():
        line = line.strip()
        if line.startswith("pid="):
            with contextlib.suppress(ValueError):
                return int(line.split("=", 1)[1])
    return None


def remove_stale_lock(lock_path: Path, *, expected_pid: int) -> bool:
    """Best-effort unlink of a run.lock whose current pid matches expected_pid.
    Returns True if unlinked, False if pid drifted (another process took the lock) or file gone.
    Race-safe: re-reads pid from the file right before unlink; skips if it changed.
    """
    current = read_lock_pid(lock_path)
    if current != expected_pid:
        return False
    with contextlib.suppress(FileNotFoundError):
        lock_path.unlink()
    return True
```

Канонический `read_lock_pid` переезжает из `src/pipeline/launch/runtime.py:174-185` в `store.py` (лучшее место: read/write lock-файла в одном модуле). Старое место превращается в re-export (`from src.pipeline.state.store import read_lock_pid`), чтобы не ломать API `src.pipeline.launch`.

**A2. Обновить API-сервис:**

В [src/api/services/launch_service.py](../../src/api/services/launch_service.py) строки 90-102:

```python
def interrupt(run_dir: Path) -> InterruptResponse:
    store = PipelineStateStore(run_dir)
    pid = read_lock_pid(store.lock_path)
    if pid is None:
        return InterruptResponse(interrupted=False, pid=None, reason="no_lock_file")
    if not is_process_alive(pid):
        remove_stale_lock(store.lock_path, expected_pid=pid)
        return InterruptResponse(interrupted=False, pid=pid, reason="process_not_found")
    ok = interrupt_launch_process(pid)
    if not ok:
        return InterruptResponse(interrupted=False, pid=pid, reason="signal_failed")
    return InterruptResponse(interrupted=True, pid=pid, reason=None)
```

Замена: `(run_dir / "run.lock").unlink()` → `remove_stale_lock(store.lock_path, expected_pid=pid)`. Импорты: `PipelineStateStore`, `read_lock_pid`, `remove_stale_lock` из `src.pipeline.state.store` (или через `src.pipeline.launch` для read_lock_pid, т.к. он там reexport).

**A3. CLI-утилита `_load_config_for_run` в [src/main.py](../../src/main.py):330-354:**

Заменить ручное чтение `pipeline_state.json` на `PipelineStateStore(run_dir).load()`:

```python
if run_dir is not None:
    try:
        state = PipelineStateStore(run_dir.expanduser().resolve()).load()
    except PipelineStateLoadError as exc:
        raise ValueError(f"Cannot load state for {run_dir}: {exc}") from exc
    config_path_str = state.config_path
    if not config_path_str:
        raise ValueError(
            f"Run '{run_dir.name}' was created before config tracking was added.\n"
            f"Pass the config explicitly:\n"
            f"  ./run.sh /path/to/config.yaml {run_dir}"
        )
    resolved = Path(config_path_str)
    if not resolved.exists():
        raise ValueError(
            f"Config from state no longer exists: {resolved}\nPass --config explicitly to override."
        )
    return resolved
```

### Часть B: Hard-remove TUI

**B1. Удалить каталоги целиком:**
- `src/tui/` (весь)
- `src/tests/unit/tui/` (весь)

**B2. Удалить CLI-команду `tui` из [src/main.py](../../src/main.py):1270-1332** (Typer `@app.command(name="tui")` функция `ryotenkai_tui` + её хелперы; ~63 строки).

Проверить и удалить относящиеся к ней тесты в `src/tests/unit/test_main_cli.py` (grep по `tui` / `ryotenkai_tui`).

**B3. Удалить зависимость из [pyproject.toml](../../pyproject.toml):**

Убрать строку `"textual[syntax]>=8.1.0,<9.0.0",` (строка 28).

**B4. Документация:**

- [README.md](../../README.md) — удалить TUI-секцию «### Interactive TUI», все упоминания `ryotenkai tui`, ссылки на `tui_runs_list.png`, `tui_run_detail.png`, `tui_eval_answers.png` + сами файлы скриншотов, строку `├── tui/             # Terminal UI (Textual)` из дерева модулей.
- [CONTRIBUTING.md](../../CONTRIBUTING.md) — удалить строку `- 'src/tui/' — Terminal UI (Textual)`.
- [run.sh](../../run.sh) — удалить ~6 строк документации TUI-команды и примеров.
- [docs/web-ui.md](../../docs/web-ui.md) — переписать строки 5, 60, 94-97: вместо «CLI/TUI/Web UI coexist» → «CLI/Web UI coexist»; удалить параллели с TUI как активным клиентом.

### Часть C: Обновление ADR

**C1.** Добавить **active** ADR: `TUI removed — web is the canonical interactive UI` (context + decision + rationale + consequences), с affected_files = удалённые пути.

**C2.** Обновить **существующую active ADR** `Sibling Client Architecture for Pipeline State` (id поиск через `update_decision_records(action="list", filter_tag="architecture")`): в `decision` убрать упоминание TUI; в `consequences` добавить «TUI removed 2026-04; CLI + Web UI are the two sibling clients going forward». Добавить `src/pipeline/state/store.py` (со ссылкой на `remove_stale_lock`) в affected_files.

## Критические файлы

**Изменяются:**
- [src/pipeline/state/store.py](../../src/pipeline/state/store.py) — добавить `read_lock_pid`, `remove_stale_lock`
- [src/pipeline/launch/runtime.py](../../src/pipeline/launch/runtime.py) — `read_lock_pid` становится re-export
- [src/api/services/launch_service.py](../../src/api/services/launch_service.py) — использовать новые функции
- [src/main.py](../../src/main.py) — заменить ручной load на `PipelineStateStore.load()`, удалить `tui`-команду
- [pyproject.toml](../../pyproject.toml) — убрать `textual[syntax]`
- [README.md](../../README.md), [CONTRIBUTING.md](../../CONTRIBUTING.md), [run.sh](../../run.sh), [docs/web-ui.md](../../docs/web-ui.md)

**Удаляются целиком:**
- `src/tui/`
- `src/tests/unit/tui/`

**Только для сверки (не менять):**
- [src/pipeline/state/run_lock_guard.py](../../src/pipeline/state/run_lock_guard.py) — Invariant #1 reference
- [src/api/services/delete_service.py](../../src/api/services/delete_service.py), [src/api/services/run_service.py](../../src/api/services/run_service.py) — используют `read_lock_pid` только для чтения, safe

**Для follow-up (вне этого плана):**
- `scripts/batch_smoke.py:151` — читает `pipeline_state.json` напрямую; оставить на будущее, это вспомогательный скрипт.
- `src/pipeline/state/queries.py`, `src/pipeline/run_queries.py` — `.exists()` на state-файле при скане run-директорий; read-only, приемлемо.

## Verification

1. **Тесты:** `pytest` — всё должно проходить. Особенно:
   - `src/tests/integration/api/test_launch_and_interrupt.py` — покрывает interrupt() со stale-lock (строка 77, 106-112).
   - `src/tests/e2e/api/test_full_launch_cycle.py` — e2e flow, строка 173 проверяет что run.lock удаляется после "stale" pid=999999.

2. **Lint/Type:** `ruff check .` + `ruff format .` + `mypy .` — без ошибок.

3. **Grep-проверки после удаления TUI:**
   ```
   grep -R "src\.tui\|from src\.tui\|import src\.tui" src/ tests/ scripts/ docs/ web/ README.md CONTRIBUTING.md run.sh
   ```
   Ожидание: 0 матчей (кроме plan-файлов в docs/plans/).
   ```
   grep -R "textual" src/ scripts/
   ```
   Ожидание: 0 матчей.

4. **CLI:** после изменений запустить `python -m src.main --help` — команды `tui` быть не должно, остальные команды присутствуют.

5. **Integration smoke:** поднять web-backend (`python -m src.main serve ...`), убедиться что `/api/v1/runs/<id>/interrupt` нормально работает и корректно удаляет stale lock (можно подделать через `(run_dir / "run.lock").write_text("pid=999999\n...")`).

## Обязательный пост-шаг

После завершения правок: `update_decision_records` — создание ADR «TUI removed» и обновление существующего «Sibling Client Architecture» (см. §C).
