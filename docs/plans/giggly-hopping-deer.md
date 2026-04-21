# Plan: fix stage-context leak across threading boundaries

> Follow-up to iterations 1–2 (per-stage logging with `LogLayout` + root-handler). This iteration addresses a runtime-discovered correctness bug.

## Контекст

В боевом запуске `run_20260421_161711_1dwzw` обнаружилось: в `dataset_validator.log` есть **11 строк**, а в `pipeline.log` за тот же стейдж — **полный лог плагинов** валидации (avg_length, empty_ratio, diversity, gold_syntax — несколько десятков записей). Записи теряются по маршруту в per-stage файл.

**Корень**: `_current_stage: ContextVar[str | None]` в [src/utils/logger.py](../../src/utils/logger.py) устанавливается в `stage_logging_context` в main thread. По PEP 567 `ContextVar` **не копируется** в `threading.Thread` / `ThreadPoolExecutor` workers автоматически — только в `asyncio.Task`. Worker видит `_current_stage.get() == None`, `_StageContextFilter` отклоняет запись → она не попадает в stage-handler. В `pipeline.log` запись есть, потому что root-handler у него без stage-фильтра.

## Карта утечки (explore-отчёт)

| Файл:строка | Тип | Приоритет | Что теряется |
|---|---|---|---|
| [src/pipeline/stages/dataset_validator.py:294](../../src/pipeline/stages/dataset_validator.py:294) | `ThreadPoolExecutor` для плагинов валидации | **HIGH** | ~50 записей плагинов на стейдж |
| [src/training/orchestrator/phase_executor/adapter_cache.py:60](../../src/training/orchestrator/phase_executor/adapter_cache.py:60) | `ThreadPoolExecutor` timeout-wrapper для HF-upload | MEDIUM | HF hub upload/download |
| [src/infrastructure/mlflow/gateway.py:149](../../src/infrastructure/mlflow/gateway.py:149) | `ThreadPoolExecutor` для MLflow prompt loading | MEDIUM | MLflow prompt-load events |
| [src/training/orchestrator/shutdown_handler.py:176-177](../../src/training/orchestrator/shutdown_handler.py:176) | signal handlers | LOW | Shutdown-callbacks |

**Вне скоупа**: subprocess remote execution (deployment_manager/ssh_client) — это процесс на GPU-сервере, ContextVar там физически нет. Решается через [log_manager.py](../../src/pipeline/stages/managers/log_manager.py) — SSH-tail стягивает `training.log` с пода в `logs/training.log` (уже работает).

## Решение: гибрид

### A. Global fallback в `src/utils/logger.py` (страховка от всех будущих ThreadPool'ов)

Module-level `_active_stage: str | None` — валиден под инвариантом «оркестратор выполняет один стейдж за раз».

```python
_active_stage: str | None = None  # fallback when ContextVar doesn't propagate

class _StageContextFilter(logging.Filter):
    def filter(self, record):
        ctx = _current_stage.get()
        if ctx is None:
            ctx = _active_stage  # threadpool/signal-handler fallback
        return ctx == self._stage_name

@contextmanager
def stage_logging_context(stage_name, layout):
    ...
    global _active_stage
    prev_active = _active_stage
    _active_stage = stage_name
    token = _current_stage.set(stage_name)
    try:
        yield stage_log_path
    finally:
        _current_stage.reset(token)
        _active_stage = prev_active  # restore nested/outer stage correctly
        ...
```

Почему global безопасен:
- `CANONICAL_STAGE_ORDER` — последовательное выполнение (один активный стейдж за раз).
- `ThreadPoolExecutor` в workers блокирует main-thread до `executor.shutdown(wait=True)` (это гарантия `with`-блока) → гонка между stage A завершилось / stage B началось / воркер A ещё пишет — **невозможна** в текущей архитектуре.
- GIL гарантирует atomic чтение/запись module-level строки.
- `prev_active`-паттерн зеркалит token-паттерн ContextVar → nested stages корректно восстанавливают внешний.

### B. `copy_context()` точечно в 3 ThreadPool'ах (canonical pattern)

```python
from contextvars import copy_context
ctx = copy_context()
future = executor.submit(ctx.run, fn, *args)
```

**Зачем, если есть global fallback?**
- Canonical — следует PEP 567.
- Готово к будущему параллельному пайплайну (если когда-то появится — global fallback станет некорректным, а copy_context продолжит работать).
- `copy_context` захватывает ВСЕ ContextVars, не только наш `_current_stage` — например, MLflow active run / OpenTelemetry trace ID тоже прокинутся автоматически.

## Критичные файлы

Правим:
- [src/utils/logger.py](../../src/utils/logger.py) — `_active_stage` module-level + `_StageContextFilter.filter` fallback + `stage_logging_context` set/restore
- [src/pipeline/stages/dataset_validator.py:294](../../src/pipeline/stages/dataset_validator.py:294) — `copy_context` обёртка `executor.submit`
- [src/training/orchestrator/phase_executor/adapter_cache.py:60](../../src/training/orchestrator/phase_executor/adapter_cache.py:60) — `copy_context` обёртка
- [src/infrastructure/mlflow/gateway.py:149](../../src/infrastructure/mlflow/gateway.py:149) — `copy_context` обёртка

Новые тесты:
- [src/tests/unit/utils/test_logger.py](../../src/tests/unit/utils/test_logger.py) — **5 тестов** под threading/nested/global-fallback (см. ниже)

Не трогаем:
- `shutdown_handler.py` — global fallback покроет signal handler автоматически; copy_context здесь неприменим (не submission).
- `deployment_manager.py`, `ssh_client.py` — remote execution, другой класс проблемы (уже решён через LogManager).

## Тесты (все in-process, без оркестратора)

1. **`test_stage_filter_captures_threadpool_record_via_global_fallback`** — ThreadPoolExecutor.submit без copy_context → запись появляется в stage.log.
2. **`test_stage_filter_captures_threadpool_record_via_copy_context`** — то же, но с `copy_context().run` → запись появляется (проверяет canonical path).
3. **`test_active_stage_reset_on_exit`** — после выхода из stage context `_active_stage is None`; запись из отдельно запущенного после thread НЕ попадает в stage.log.
4. **`test_nested_stage_restores_outer_active_stage`** — nested `with stage_logging_context("outer"): with stage_logging_context("inner"):` → после inner `_active_stage == "outer"`.
5. **`test_active_stage_survives_exception_in_stage`** — raise внутри stage, `finally` восстанавливает предыдущее состояние.

Все тесты используют module-level `monkeypatch.setattr(logger_module, "_active_stage", None)` в fixture для изоляции.

## Verification

1. `pytest src/tests/unit/utils/test_logger.py -xvs` — новые и существующие тесты.
2. `pytest src/tests/unit/pipeline src/tests/unit/utils src/tests/unit/api src/tests/integration/api src/tests/integration/test_phase_executor_restart_flow.py src/tests/integration/test_strategy_orchestrator_restart_flow.py -q` — должно остаться 1641+ passed.
3. Реальный запуск: `dataset_validator.log` содержит записи плагинов (`Running plugin: avg_length_main`, `✓ avg_length_main passed`, `train.avg_length_main.avg_length: 489.75`, …).

## Подводные камни и как обходятся

- **Race на `_active_stage`**: невозможен под контрактом sequential stages + GIL.
- **Параллельные стейджи в будущем**: global fallback станет некорректным. Смягчение — `copy_context` уже стоит в 3 ThreadPool'ах; удалить fallback, документ-коммент в logger.py указывает на эту миграцию.
- **Third-party threadpool'ы** (transformers/datasets/tokenizers в обучении): их записи тоже поймаются fallback'ом без правок в чужом коде — это и есть польза варианта A.
- **Async edge cases**: `asyncio.Task` корректно копирует ContextVar → `_current_stage.get()` вернёт правильное имя → fallback не задействуется.
- **`HELIX_NO_FILE_LOGS=1`**: fallback ставится до проверки `_enable_file_logs`, консистентно; file-handlers просто не создаются, filter всё равно корректный.
- **Тестовая изоляция**: существующая фикстура `_fresh_run_logging` в [test_logger.py](../../src/tests/unit/utils/test_logger.py) уже ресетит module state — добавить `_active_stage` в список.

## Порядок реализации

1. Правка `logger.py` (fallback + filter + context manager + test fixture обновление).
2. `dataset_validator.py` — copy_context на submit.
3. `adapter_cache.py` — copy_context.
4. `mlflow/gateway.py` — copy_context.
5. +5 новых тестов.
6. Unit + integration прогон.
7. Коммит: `fix(logging): preserve stage context across ThreadPoolExecutor workers`.

## Ссылки

- [PEP 567 — Context Variables](https://peps.python.org/pep-0567/) — `copy_context()` контракт
- Предыдущие итерации: разделы «Итерация 1» и «Итерация 2» в [luminous-tickling-haven.md](luminous-tickling-haven.md)
- Decision records: `1b6e014c` (iter 1), `57c59143` (iter 2)
