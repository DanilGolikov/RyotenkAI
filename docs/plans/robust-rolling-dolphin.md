# Robust system-metrics callback refactor

## Context

После предыдущей итерации Phase 14 follow-up `SystemMetricsCallback` включён по умолчанию (`callback_enabled=True`), и при ревью выяснилось:

- **Дублирование.** `GPUMetricsCallback` логирует те же CPU/GPU/RAM, но через `subprocess` к `nvidia-smi` и под параллельным namespace `system/gpu_0_*` — на дашборде MLflow две группы метрик от двух источников, оператору неясно, на что смотреть.
- **Параллельный встроенный поток.** Нативный сэмплер MLflow (`enable_system_metrics_logging`) работает в фоновом потоке, шлёт метрики **мимо** `ResilientMLflowTransport` → теряет данные при офлайн-окнах. Дублирует наш callback и расходится с ним по таймлайну.
- **Баги в `SystemMetricsCallback`:**
  - Multi-GPU: читает только `nvmlDeviceGetHandleByIndex(0)` — на 2×/4× карт остальные невидимы (`system_metrics_callback.py:83`).
  - DDP: нет rank guard → каждая ранк-копия шлёт свои метрики, в MLflow N-кратные дубли.
  - `psutil.cpu_percent(interval=None)` без прайминга всегда возвращает `0.0` на первом вызове.
  - При ошибке чтения температуры пишется `0.0` вместо `NaN` — на дашборде «GPU замёрз до нуля» вводит в заблуждение (`system_metrics_callback.py:124`).
  - Постоянная метаинформация (имя GPU, общий объём VRAM, версия драйвера) пишется метриками каждый шаг — должно быть тегами рана.
- **Лишний config surface.** Из 4 полей `SystemMetricsConfig` (`sampling_interval`, `samples_before_logging`, `callback_enabled`, `callback_interval`) реально нужен только `callback_enabled`. Первые два настраивают нативный сэмплер, который мы выпиливаем; `callback_interval` (throttle на N шагов) не оправдывает себя на 7B-моделях, где шаг 1-3 сек.

**Цель:** один callback, один источник правды, минимальный config, корректное поведение в multi-GPU и DDP, статика — тегами.

---

## Approach

### 1. Снести `GPUMetricsCallback` целиком

- Удалить `src/training/callbacks/gpu_metrics_callback.py` (108 строк).
- Удалить `src/tests/unit/training/callbacks/test_gpu_metrics_callback.py`.
- В `src/training/trainers/factory.py:339-341` убрать импорт + регистрацию `GPUMetricsCallback` в списке коллбэков.
- В `src/training/callbacks/__init__.py` убрать экспорт (если есть).
- В `src/training/callbacks/completion_callback.py` убрать упоминание `GPUMetricsCallback` в docstring.
- В `src/tests/unit/training/test_trainer_factory_callbacks.py` убрать `assert any(isinstance(cb, GPUMetricsCallback) ...)`.

### 2. Выключить нативный MLflow-сэмплер

- В `src/training/managers/mlflow_manager/setup.py:_configure_system_metrics` (строки 270-285) удалить вызовы:
  - `mlflow.enable_system_metrics_logging()` (строка 278)
  - `mlflow.set_system_metrics_sampling_interval(...)` (строка 282)
  - `mlflow.set_system_metrics_samples_before_logging(...)` (строка 283)
  Метод можно превратить в no-op с однострочным INFO-логом «native sampler intentionally disabled — using SystemMetricsCallback».
- В `src/training/orchestrator/phase_executor/mlflow_logger.py` удалить условные `enable_system_metrics_logging()` на строках ~128 и ~214 и их `disable_*` парные вызовы (становятся не нужны).
- Тестовые моки `mlflow.enable_system_metrics_logging = MagicMock()` в ~33 местах `test_phase_executor.py` оставить как есть — они становятся безвредными (не вызываются).

### 3. Сжать `SystemMetricsConfig` до одного поля

Файл: `src/config/integrations/system_metrics.py`

Удалить:
- `sampling_interval` (поле + валидаторы + упоминания в docstring + YAML-примере)
- `samples_before_logging` (то же)
- `callback_interval` (то же)

Оставить только:
```python
class SystemMetricsConfig(StrictBaseModel):
    callback_enabled: bool = Field(
        default=True,
        description=(
            "Enable the HF Trainer SystemMetricsCallback. On by default — "
            "step-aligned gpu/*, cpu/*, ram/* metrics flow through "
            "ResilientMLflowTransport and survive offline windows via "
            "MetricsBuffer. Set False only if pynvml.nvmlInit hangs on "
            "your specific cloud GPU image."
        ),
    )
```

Обновить:
- Module-level docstring (`system_metrics.py:1-53`) — убрать упоминания нативного сэмплера и трёх полей.
- `src/config/integrations/experiment_tracking.py:32-40` — добавить в `_LEGACY_MLFLOW_KEYS`:
  - `"system_metrics_callback_interval"` (уже там нет — добавить)
  - убедиться что `"system_metrics_sampling_interval"` и `"system_metrics_samples_before_logging"` остались (они и сейчас в списке — для миграционного хинта)
- Также добавить в legacy detection логику для **nested**-кейса: если в YAML `experiment_tracking.mlflow.system_metrics.sampling_interval` (новый путь, но удалённое поле) — отдельный понятный ошибка типа «field removed in YYYY-MM, native MLflow sampler disabled».

### 4. Переработать `SystemMetricsCallback`

Файл: `src/training/callbacks/system_metrics_callback.py` (текущие 203 строки)

#### 4.1. Multi-GPU iteration (`_setup`, `_get_gpu_metrics`)

В `_setup` (строки 57-101) заменить:
```python
self._gpu_handle = _pynvml.nvmlDeviceGetHandleByIndex(0)  # текущее
```
на:
```python
self._gpu_count = _pynvml.nvmlDeviceGetCount()
self._gpu_handles = [
    _pynvml.nvmlDeviceGetHandleByIndex(i) for i in range(self._gpu_count)
]
```

В `_get_gpu_metrics` (строки 103-135) — цикл по `self._gpu_handles`, ключи метрик становятся `gpu/{idx}/utilization`, `gpu/{idx}/memory_used_gb`, `gpu/{idx}/memory_percent`, `gpu/{idx}/temperature`.

#### 4.2. CPU prime call (`_setup`)

В конец `_setup` добавить одну строку:
```python
if self._psutil_available:
    self._psutil.cpu_percent(interval=None)  # priming — discard 0.0 first reading
```

#### 4.3. Rank-0 guard (`on_step_end`)

В начале `on_step_end` (строка 168, сразу после `self._setup()`):
```python
if not state.is_world_process_zero:
    return
```
HF Trainer контракт: `state.is_world_process_zero` — стандартный API; в этом репо прецедента в коллбэках нет, но это идиома из HF docs.

#### 4.4. NaN для отсутствующих метрик

В `_get_gpu_metrics` строка 124:
```python
temp = 0.0  # текущее
```
заменить на:
```python
temp = float("nan")
```
Применить тот же паттерн ко всем `except` блокам, которые сейчас возвращают `0.0`/`{}` для отсутствующих метрик. Альтернатива: вообще не включать ключ в payload — MLflow тогда не нарисует точку (приоритетный вариант, чище визуально).

#### 4.5. Статика → теги через `on_train_begin`

Новый метод (после `__init__`, до `_setup`):
```python
def on_train_begin(self, args, state, control, **kwargs):
    self._setup()
    if not state.is_world_process_zero:
        return
    if self._mlflow is None:
        return

    tags: dict[str, str] = {}
    if self._pynvml_available and self._gpu_handles:
        tags["system.gpu.count"] = str(self._gpu_count)
        for i, h in enumerate(self._gpu_handles):
            with contextlib.suppress(Exception):
                tags[f"system.gpu.{i}.name"] = self._pynvml.nvmlDeviceGetName(h)
            with contextlib.suppress(Exception):
                mem = self._pynvml.nvmlDeviceGetMemoryInfo(h)
                tags[f"system.gpu.{i}.memory_total_gb"] = f"{mem.total / (1024**3):.1f}"
        with contextlib.suppress(Exception):
            tags["system.driver.version"] = self._pynvml.nvmlSystemGetDriverVersion()
    if self._psutil_available:
        with contextlib.suppress(Exception):
            tags["system.cpu.count"] = str(self._psutil.cpu_count())

    if tags:
        with contextlib.suppress(Exception):
            self._mlflow.set_tags(tags)
```
Использует тот же `mlflow.set_tags(...)` идиом, что и `phase_executor/mlflow_logger.py:130-135`.

#### 4.6. Удалить `log_every_n_steps`

В `__init__` и `on_step_end`. Аргумент больше не нужен — логируем на каждом шаге. В `factory.py` конструктор будет `SystemMetricsCallback()` без аргументов.

### 5. Обновить тесты

- `src/tests/unit/config/integrations/test_system_metrics.py` — удалить тесты для удалённых полей:
  - `test_sampling_interval_below_1_rejected`, `test_sampling_interval_above_60_rejected`
  - `test_samples_before_logging_below_1_rejected`, `test_samples_before_logging_above_10_rejected`
  - `test_callback_interval_above_100_rejected`
  - `test_sampling_interval_at_min_accepted`, `test_sampling_interval_at_max_accepted`
  - `test_samples_before_logging_at_max_accepted`, `test_callback_interval_at_max_accepted`
  - `test_callback_interval_default_present_when_disabled`
  - Обновить `test_defaults_match_hardcoded_values` — assert только `callback_enabled is True`.
  - Обновить `test_full_construction_keyword_args` — только `callback_enabled=False`.
  - Обновить `test_mlflow_integration_config_accepts_nested_block_in_yaml` — убрать `sampling_interval` из payload.
- `src/tests/unit/test_mlflow_events.py:468-469` — удалить присвоения `mlflow_config.system_metrics_sampling_interval = 5.0` и `.samples_before_logging = 10`.
- Новые тесты для `SystemMetricsCallback`:
  - `test_rank_zero_only_logs` — fake `state.is_world_process_zero = False` → `mlflow.log_metrics` не вызывается.
  - `test_multi_gpu_metrics_collected` — мок `pynvml.nvmlDeviceGetCount() == 2` → ключи содержат `gpu/0/*` и `gpu/1/*`.
  - `test_cpu_percent_primed` — `_psutil.cpu_percent` вызывается дважды до первого `_get_cpu_metrics` (один раз в `_setup`, один раз в реальном чтении).
  - `test_static_info_set_as_tags_on_train_begin` — `mlflow.set_tags` вызывается с ожидаемыми ключами.
  - `test_temperature_nan_when_unavailable` — мок temperature getter raises → метрика NaN или отсутствует в payload.

---

## Critical files to modify

| Файл | Тип |
|---|---|
| `src/training/callbacks/gpu_metrics_callback.py` | **DELETE** |
| `src/tests/unit/training/callbacks/test_gpu_metrics_callback.py` | **DELETE** |
| `src/training/callbacks/system_metrics_callback.py` | major refactor |
| `src/config/integrations/system_metrics.py` | shrink to one field |
| `src/config/integrations/experiment_tracking.py` | extend `_LEGACY_MLFLOW_KEYS` |
| `src/training/managers/mlflow_manager/setup.py` | drop native-sampler calls |
| `src/training/orchestrator/phase_executor/mlflow_logger.py` | drop native-sampler calls |
| `src/training/trainers/factory.py` | remove GPU reg, simplify SystemMetrics ctor |
| `src/training/callbacks/__init__.py` | remove GPU export |
| `src/training/callbacks/completion_callback.py` | docstring cleanup |
| `src/tests/unit/config/integrations/test_system_metrics.py` | delete obsolete tests, update remaining |
| `src/tests/unit/training/test_trainer_factory_callbacks.py` | drop GPU assertion |
| `src/tests/unit/test_mlflow_events.py` | drop legacy field setters |

## Reusable patterns

- **Lazy import + try/except** идиома из текущего `_setup` (строки 57-101).
- **`mlflow.set_tags(...)`** из `phase_executor/mlflow_logger.py:130`.
- **`getattr` fallback на nested config** из текущего `factory.py` (после Phase 14).
- **`contextlib.suppress(Exception)`** для best-effort блоков (уже используется в `system_metrics_callback.py:199`).
- **`state.is_world_process_zero`** — стандартный HF Trainer API (без локального прецедента, но идиоматично).

---

## Verification

1. `pytest src/tests/unit/config/integrations/` — green (~25 тестов вместо 33; часть удалена).
2. `pytest src/tests/unit/training/callbacks/` — green, новые тесты для rank-0 / multi-GPU / NaN / tags.
3. `pytest src/tests/unit/training/managers/test_mlflow_manager*` — green (моки `enable_system_metrics_logging` остаются безвредными).
4. `ruff check . && ruff format --check .` — clean.
5. `mypy src/training/callbacks/system_metrics_callback.py src/config/integrations/system_metrics.py` — clean.
6. Полный `pytest -q src/tests/unit/config/ src/tests/unit/training/callbacks/` — без regression на смежных тестах.
7. **Manual smoke (опционально, multi-GPU box):** запустить SFT с двумя GPU, в MLflow UI проверить:
   - один namespace `gpu/0/*`, `gpu/1/*`, `cpu/*`, `ram/*` — без `system/gpu_0_*` дубля.
   - в Tags panel: `system.gpu.0.name`, `system.gpu.1.name`, `system.gpu.count`, `system.driver.version`, `system.cpu.count`.
   - на DDP-симуляции (`torchrun --nproc_per_node=2`) — нет N-кратных точек на графиках.
   - sleep Mac во время трейна → метрики реплеятся из `MetricsBuffer` после wake.

---

## Addendum: bump MLflow `3.10.1 → 3.11.1` + перезапуск локального стека

### Контекст

Параллельно с callback-чисткой обновляем MLflow. Сейчас залочены версии расходятся:

| Слой | Текущее | Целевое |
|---|---|---|
| Python client (`pyproject.toml`) | `>=3.8.1,<4.0.0` | `>=3.11.1,<4.0.0` |
| Python client (`uv.lock`) | `3.10.1` | `3.11.1` |
| Server image (`docker/mlflow/Dockerfile.mlflow:5`) | `ghcr.io/mlflow/mlflow:v3.8.1` | `ghcr.io/mlflow/mlflow:v3.11.1` |

**Server-client дрифт** между server `3.8.1` и client `3.10.1` уже сейчас — это два минора, ниже официальной поддержки. Бамп серверa догоняет клиент и одновременно поднимает обоих до `3.11.1`.

### Что реально получаем (из релевантных ChangeLog 3.9.0 → 3.11.1)

Самое полезное **уже на 3.10.1** (получили при предыдущем бампе клиента):
- ✅ Fix `IntegrityError` при `log_batch` с дублями метрик в одном батче (#20807) — критично для `MetricsBuffer.replay`.
- ✅ TCP keepalive на HTTP-сессиях (#21514) — критично для `ResilientMLflowTransport`, снижает флапы circuit breaker на длинных трейнах через NAT/load balancer.
- ✅ Fix SQLAlchemy connection pool leak (#19386) — критично для длинных pipeline run-ов.
- Fix optimistic pagination в `_search_runs` (#20547).
- Security: artifact path traversal fix (#19260).

**Сервер на 3.8.1 этих фиксов НЕ имеет.** Бамп сервера до 3.11.1 — это в первую очередь подтянуть **серверную сторону** этих стабилизационных фиксов.

Дополнительно на 3.11.1:
- Поддержка `transformers >= 5.x` в `mlflow.transformers` autolog (future-proof).
- Fix `autolog` затирающего `warnings.showwarning` (#21707) — потенциально могло мешать нашему `get_logger`.
- Pickle-free serialization (`torch.export` / `skops`) — на будущее, если решим логировать модели.
- UV detection для model deps.
- `IS NULL` / `IS NOT NULL` в `search_runs` filter.

**Breaking changes для нашего кода: ноль.** Все API на `mlflow.fluent.*` и `MlflowClient` неизменны.

### Шаги

**1. Бамп Python-клиента**
- `pyproject.toml:31` — `"mlflow>=3.8.1,<4.0.0"` → `"mlflow>=3.11.1,<4.0.0"`.
- `requirements.txt:36` — синхронизировать (`mlflow>=3.11.1,<4.0.0`) и убрать устаревший inline-комментарий.
- `uv lock` — пересобрать lockfile, проверить что `mlflow`, `mlflow-skinny`, `mlflow-tracing` все на `3.11.1`.

**2. Бамп server-образа**
- `docker/mlflow/Dockerfile.mlflow:5` — `FROM ghcr.io/mlflow/mlflow:v3.8.1` → `FROM ghcr.io/mlflow/mlflow:v3.11.1`.
- Убедиться что `entrypoint.mlflow.sh` не использует флагов/переменных, исчезнувших в 3.11 (быстрое чтение скрипта на ~10 строк).

**3. Перезапуск локального стека**
- `cd docker/mlflow && ./start.sh restart` — это `down` + `up -d --build`, пересобирает `mlflow-server:latest` с новым base image, поднимает заново вместе с postgres и minio.
- Postgres схема MLflow умеет апгрейдиться сама на старте через `mlflow db upgrade` (вызывается в `entrypoint.mlflow.sh`). Если в 3.11 есть schema migrations — они применятся автоматически.
- Проверить `./start.sh logs` на ошибки в первые 30 секунд.

**4. Smoke check после перезапуска**
- `curl http://localhost:5002/health` → `OK`.
- Открыть UI, увидеть существующие experiments/runs (миграция БД сохранила данные).
- Запустить минимальный pipeline / тест против локального tracking-server, чтобы убедиться что client `3.11.1` ↔ server `3.11.1` работают вместе.

### Файлы

| Файл | Изменение |
|---|---|
| `pyproject.toml` | bump min version |
| `requirements.txt` | bump min version |
| `uv.lock` | regenerate |
| `docker/mlflow/Dockerfile.mlflow` | bump base image tag |
| `docker/mlflow/entrypoint.mlflow.sh` | check for compatibility (likely no change) |

### Verification

1. `uv sync` — без конфликтов зависимостей.
2. `pytest -q src/tests/unit/training/managers/` — green (моки MLflow API не должны зависеть от конкретной версии).
3. `pytest -q src/tests/unit/pipeline/mlflow_attempt/` — green.
4. `ruff check . && mypy src/` — clean.
5. `docker/mlflow/start.sh restart` — стек поднимается без ошибок.
6. `curl http://localhost:5002/health` → `OK`.
7. End-to-end: запустить короткий тренировочный run, проверить что он появляется в UI, метрики/теги пишутся, артефакты грузятся в MinIO.

### Риски

- **Schema migration на postgres.** Минимальный — MLflow апгрейды обычно forward-compatible на одном major. На всякий случай: бэкапнуть `mlflow_postgres` volume перед `restart` (`docker run --rm -v mlflow_postgres_data:/data alpine tar czf - /data > backup.tar.gz`).
- **Drift entrypoint-скрипта** относительно нового образа. Нивелируется тем, что наш `entrypoint.mlflow.sh` тонкий и использует стандартные `mlflow server`/`mlflow db upgrade` команды.
- **HF Trainer ↔ MLflow autolog API.** Не меняется — `mlflow.transformers` autolog как был, так и остался. Поддержка transformers 5.x — это **расширение**, не изменение существующего поведения для transformers 4.x.

### Sequencing

Можно делать в любом порядке относительно основной части плана (callback cleanup):
- **Вариант A (рекомендованный):** сначала bump MLflow, перезапустить стек, убедиться что всё работает → потом callback cleanup. Изолирует возможные проблемы апгрейда от изменений callback-кода.
- **Вариант B:** одним PR. Можно, но если сломается — труднее найти причину.
