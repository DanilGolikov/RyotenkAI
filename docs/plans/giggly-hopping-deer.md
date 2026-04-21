# Plan: Per-stage log routing — итерация 2 (реализация)

> Полный анализ, дипсинк в 3 итерации и ответы на риски живут в [luminous-tickling-haven.md](luminous-tickling-haven.md), раздел «Итерация 2 — надёжный per-stage capture + нормализация имён». Этот файл — исполнимый summary.

## Контекст

В первой итерации мы вынесли логи в `<attempt>/logs/`, ввели `LogLayout`, `stage_logging_context` и реестр `log_paths` в state. Но на реальном запуске обнаружились два бага:

1. **Имена файлов с пробелами** (`logs/Dataset Validator.log`) — потому что `StageNames.DATASET_VALIDATOR = "Dataset Validator"` и `LogLayout.stage_log` использует имя напрямую.
2. **Per-stage лог неполный** — `_StageContextFilter` висит только на `ryotenkai` логгере (`propagate=False`), поэтому сообщения от `mlflow`, `transformers`, `paramiko`, `httpx` проходят мимо. Пользователь видит их на экране, но в `<stage>.log` их нет.

## Решение (вариант B из таблицы в основном плане)

1. **`_slugify` в `LogLayout`** → `"Dataset Validator"` → `"dataset_validator"`.
2. **`stage_logging_context` — два handler'а на один файл:**
   - Handler A на `ryotenkai` логгер с `_StageContextFilter`
   - Handler B на **root** логгер с `_StageContextFilter` + `_ExcludeRyotenkaiFilter` (защита от дубликата)
3. **`pipeline.log` handler перевесить с `ryotenkai` на root** — чтобы агрегат тоже ловил сторонние логи.
4. **`root.setLevel(_log_level)`** — иначе root logger отсечёт INFO до того, как они дойдут до handler'а.
5. **`_quiet_noisy_libraries`** в `init_run_logging` — httpx/urllib3/filelock/botocore → WARNING.
6. **`log_service._discover_from_state`** — имя файла брать из `Path(rel_path).name`, не из `stage_name` (иначе пробелы возвращаются).

## `print()` — вне скоупа

Выпилка 5 `print()` в `training/pipeline` идёт **отдельной задачей** (spawned task). Там же заменяем на `logger.info/debug`. Это чище, чем костыль tee на stdout.

## Критичные файлы

- [src/utils/logs_layout.py](../../src/utils/logs_layout.py) — `_slugify`, применить в `stage_log`
- [src/utils/logger.py](../../src/utils/logger.py) — `_ExcludeRyotenkaiFilter`, переписать `stage_logging_context`, отдельная функция `_attach_pipeline_file_handler` на root, `_quiet_noisy_libraries`, обновить `set_log_level`
- [src/api/services/log_service.py](../../src/api/services/log_service.py:62) — `out[Path(rel_path).name] = abs_path`
- [src/tests/unit/utils/test_logs_layout.py](../../src/tests/unit/utils/test_logs_layout.py) — тесты `_slugify`, slug-интеграция в `stage_log`
- [src/tests/unit/utils/test_logger.py](../../src/tests/unit/utils/test_logger.py) — тест захвата `logging.getLogger("mlflow")` в stage-файл и pipeline.log
- [src/tests/unit/api/services/test_log_service_registry.py](../../src/tests/unit/api/services/test_log_service_registry.py) — обновить: StageNames с пробелом в ключе `stage_runs`, но slug-путь в log_paths → файл находится корректно

## Verification

1. Unit: `pytest src/tests/unit/utils/test_logs_layout.py src/tests/unit/utils/test_logger.py src/tests/unit/api/services/test_log_service_registry.py -xvs`
2. Restart integration: `pytest src/tests/integration/test_phase_executor_restart_flow.py src/tests/integration/test_strategy_orchestrator_restart_flow.py`
3. Реальный запуск (частично — до stage GPU Deployer): файлы `logs/dataset_validator.log`, `logs/gpu_deployer.log` без пробелов; в gpu_deployer.log есть `paramiko.*`; в pipeline.log есть те же записи сторонних либ.
4. TUI: вкладки стейджей открываются по slug-именам, content полный.

## Риски (детальные 3-итерационные ответы в основном плане)

- **Q3**: pipeline.log перестаёт висеть на `ryotenkai` — решено (на root с полным захватом).
- **Q4**: Старые `Dataset Validator.log` не переименовываем — legacy fallback в log_service покрывает чтение.
- **Q5**: Handler.setLevel = _log_level (единый с pipeline.log).
- **Q6**: Subprocess без PIPE — все наши с PIPE, покрыто через Python reader.
- **Q7**: `_slugify` — чистая функция, race невозможен.
