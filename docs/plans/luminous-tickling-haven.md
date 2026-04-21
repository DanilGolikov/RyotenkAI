# Раздельные логи по стейджам пайплайна

> **Active plan: «Итерация 2» (раздел в конце файла)** — актуальная задача: починить несоответствие контента per-stage логов и нормализовать имена файлов.
> Всё выше — история (первый проход, уже реализован).

---



## Контекст

Сейчас в проекте один лог-файл на попытку запуска пайплайна — `pipeline.log` — в который пишут события ВСЕХ стейджей подряд. Пользователь хочет оценить идею дробления: каждый стейдж пишет свой лог-файл.

Аналогичная дискуссия уже проходила в марте 2026 про `pipeline_events.json` (мемори: `mem_287f1e013031`, `mem_076a2067a976`) — там было принято решение пилить монолит на per-stage JSON. Текущий вопрос — про **текстовые логи `logging`**, не про JSON-события.

---

## Текущая архитектура логов

### FS-layout одного run'а

```
runs/run_20260419_095457_869b918b/
├── pipeline_state.json          # [state/store.py] состояние
├── run.lock                      # блокировка
├── tui_launch.log                # лог TUI-запуска, корень run'а
└── attempts/
    └── attempt_1/
        ├── pipeline.log          # ← ЕДИНЫЙ лог всех стейджей (98 KB)
        ├── training.log          # подтянутый с удалённого пода
        ├── experiment_report.md
        ├── evaluation/ … inference/ …
```

### Кто и куда пишет

- [src/utils/logger.py:185](src/utils/logger.py:185) `init_run_logging(run_name, log_dir=attempt_directory)` — ставит один `FileHandler` на корневой logger, `log_dir/pipeline.log`
- [src/pipeline/orchestrator.py](src/pipeline/orchestrator.py) вызывает `init_run_logging` один раз на attempt, на уровне оркестратора; все стейджи пишут через `logging.getLogger(__name__)` и попадают в один handler
- [src/pipeline/stages/managers/log_manager.py](src/pipeline/stages/managers/log_manager.py) — отдельный кейс: тянет `training.log` с удалённого пода через SSH (не тот же поток логов, это производный артефакт тренировки)
- [src/api/services/log_service.py:8](src/api/services/log_service.py:8) `ALLOWED_LOG_FILES = ("pipeline.log", "training.log", "inference.log", "eval.log", "tui_launch.log")` — белый список для TUI/API

### Как читаются логи

- [src/pipeline/live_logs.py](src/pipeline/live_logs.py) `LiveLogTail.read_new_lines()` — tail с оффсетом
- TUI/API отдают по имени файла из `ALLOWED_LOG_FILES`
- MLflow events (`training_events.json`) — отдельный канал, не текстовый лог

---

## Как устроены pipeline_state и resume

Это важно для анализа, потому что логи — часть «следа attempt'а», который resume должен корректно наследовать.

- **Per-attempt изоляция** уже есть: `attempts/attempt_{N}/` — всё, что породила попытка, живёт тут (логи, артефакты)
- [src/pipeline/state/store.py:137](src/pipeline/state/store.py:137) `next_attempt_dir()` — `attempts/attempt_{n}`
- [src/pipeline/orchestrator.py:907](src/pipeline/orchestrator.py:907) `_derive_resume_stage()` — ищет первый незавершённый/упавший стейдж в `state.attempts[-1]`
- [src/pipeline/orchestrator.py:1023](src/pipeline/orchestrator.py:1023) `_restore_reused_context()` — для пройденных стейджей восстанавливает outputs из `current_output_lineage` и помечает `MODE_REUSED` (не пересчитывает)
- [src/pipeline/orchestrator.py:1011](src/pipeline/orchestrator.py:1011) `_invalidate_lineage_from()` — чистит lineage начиная с restart-стейджа
- Новая попытка → новый `attempt_{N+1}/` → новый `pipeline.log` с нуля. **Логи прошлых попыток остаются на месте**, но они *«чужие»* для текущей попытки

Ключевой вывод: resume-машина **не зависит от логов вообще** — state и outputs lineage самодостаточны. Это даёт нам свободу менять layout логов без риска сломать resume.

---

## Дипсинк: оценка идеи сплита логов по стейджам

### Мотивация (что реально болит)

1. **Поиск**: чтобы увидеть, что происходило в `training_monitor`, надо грепать 98KB смеси из 6 стейджей
2. **TUI-вьюер**: сейчас есть per-attempt tail единого файла — невозможно дать вкладку «логи стейджа» без парсинга
3. **Resume-диагностика**: при рестарте с `model_evaluator` интересны логи именно этого стейджа в новой попытке; сейчас они опять смешаны с оркестраторной прелюдией
4. **Архивирование**: удалить «скучные» стейджи, оставить training_monitor, не тривиально
5. **Параллелизм в будущем**: если когда-то появится параллельный стейдж (валидация нескольких датасетов, мульти-eval) — monolithic log становится месивом без per-stage изоляции

### Плюсы сплита

- **Читаемость**: `attempts/attempt_N/stages/training_monitor/stage.log` — чистая история одного стейджа
- **Симметрия с `pipeline_state.json`**: state уже per-stage (`StageRunState`), логи выравниваются с моделью
- **Симметрия с обсуждённым сплитом `pipeline_events.json`**: один вектор миграции — «всё per-stage»
- **Проще rotation/cleanup** по стейджам
- **Подготовка к параллельным стейджам** — не нужно переизобретать, когда понадобится
- **TUI**: прямолинейная вкладка «лог стейджа X» без фильтрации
- **При resume ничего не теряется** — логи прошлых попыток остаются в их `attempt_{K}/stages/{stage}/`, новая попытка пишет в свои. Полная история доступна по `attempts/*/stages/{stage}/stage.log`

### Минусы/риски

1. **«Межстейджевый» шум**: часть логов оркестратора — между стейджами (bootstrap state, config drift, mlflow preflight, lineage invalidation, резюмирование). Для них нужен отдельный «оркестраторный» канал, иначе теряется глобальная линия. Решение: оставить `attempts/attempt_N/orchestrator.log` + per-stage файлы
2. **Хендлер-свитчинг — источник багов**: в Python `logging` нельзя «на лету» переключить только-stage-handler без гонок, если код вне стейджа вдруг логирует через root. Нужен контекстный `ContextFilter`, который по `stage_ctx` роутит запись в нужный handler
3. **Сторонние библиотеки** (transformers, mlflow, ssh, paramiko) пишут через свои логгеры — они тоже попадут в «текущий» stage log, если фильтр глобальный. Это может быть желательно (их спам привязан к стейджу), но важно это осознавать
4. **Хронология теряется**: при разборе инцидента иногда нужна общая лента всех стейджей по времени. Решение: агрегированный `pipeline.log` тоже продолжать писать (dual-write) — «tee»
5. **TUI/API уже завязаны на `ALLOWED_LOG_FILES`** с фиксированными именами — нужен апдейт: `stages/{name}/stage.log` либо реестр per-stage файлов в pipeline_state
6. **log_manager.py качает удалённый `training.log`** в `get_run_log_dir()` — этот путь перестаёт быть «корнем attempt'а», если мы схлопнем его к `stages/training_monitor/`. Нужен явный shim
7. **Backward compat**: старые runs читаются TUI — нельзя резко убрать fallback на одиночный `pipeline.log`
8. **Размер дерева**: +N-1 мелких файлов на attempt — косметический минус

### Альтернатива 1: structured logging + один файл

JSONL-строки с `stage=...` полем в одном `pipeline.log`, на чтении фильтруем. Дёшево, но читаемость «глазами» ухудшается, и TUI всё равно придётся делать парсер/индекс.

### Альтернатива 2: секционирование одного файла

Маркеры `===== STAGE training_monitor START =====` + offsets в state. Быстро прикрутить, но в ре-рестартах и параллельных случаях — хрупко.

### Как это делают индустриальные аналоги

Взял четыре системы, близкие по сути (оркестратор с стейджами/шагами), — чтобы свериться с устоявшейся практикой, а не изобретать велосипед.

**Jenkins** — монолит + маркеры:
- Один console-log на build: `$JENKINS_HOME/jobs/<job>/builds/<n>/log` на все стейджи
- `[Pipeline] stage` — текстовые маркеры в том же файле, Blue Ocean UI их парсит и рисует per-stage вьюер
- `parallel` блоки ломают читаемость файла; сообщество обходит через `tee` в отдельные файлы и `archiveArtifacts` (де-факто самодельный per-stage файл)
- Урок: монолит плохо масштабируется на параллелизм; UI-парсинг маркеров спасает sequential, но не parallel

**GitLab CI** — per-job файлы + секции внутри:
- Каждая job пишется в отдельный файл: `/gitlab-ci/builds/<YYYY_mm>/<project_id>/<job_id>.log`
- После завершения — архивация в object storage (artifacts dir / S3) через Sidekiq
- Внутри job: ANSI-маркеры `section_start:<ts>:<name>` / `section_end:` — collapsible секции в UI, `[collapsed=true]` для дефолтно-свёрнутых
- Incremental logging: chunks в Redis → persistent после завершения (для масштаба)
- Урок: **per-unit файл + секции внутри** — самая чистая модель из массовых CI. Почти ровно то, что я предлагаю в hub+spoke, только у GitLab job ≈ наш stage

**Kubernetes** — per-container + ротация по restart:
- CRI кладёт каждый контейнер pod'а в свой файл: `/var/log/pods/<ns>_<pod>_<uid>/<container>/0.log`
- При рестарте контейнера `0.log` → `1.log`, новый `0.log` — `kubectl logs --previous` для прошлой инкарнации
- `kubectl logs <pod> --all-containers --prefix` — агрегированная лента с префиксом контейнера
- Sidecar-паттерн: если приложение пишет несколько файлов в одном контейнере — sidecar'ы tail'ят shared volume и вываливают каждый поток в свой stdout
- Урок: **ротация по «попытке» встроена на уровне инфраструктуры** (наш `attempt_{N}/` — его прямой аналог), а per-unit (контейнер ≈ стейдж) — единица изоляции; агрегация делается на чтении, не на записи

**Argo Workflows / Tekton** — наследуют K8s + artifact storage:
- Каждый step = отдельный контейнер → per-step лог из коробки (через K8s)
- Artifact repository (MinIO/S3) хранит логи step'ов как артефакты после завершения workflow для долгого retention
- `argo logs @latest`, `tkn taskrun logs --follow` — агрегированный follow по всем шагам
- Kubetail / stern — UI/CLI над раздельными логами, стримят все сразу с префиксами
- Урок: индустрия пришла к **per-unit хранение + агрегатор-утилиты на чтение** — ровно то, к чему стремится hub+spoke

### Выводы из аналогов

1. **Jenkins** выбрал монолит и страдает на параллелизме. У нас сейчас — как Jenkins. Если ждём параллельных стейджей в будущем (multi-dataset validation, multi-eval) — надо уходить от монолита заранее
2. **GitLab CI** — самый близкий аналог: per-job-file + collapsible sections внутри. Наша модель stage ≈ их job, а под-подсобытия (plugin runs у DatasetValidator, phases у TrainingMonitor) — их sections. Сильный повод сделать per-stage файл
3. **K8s** подтверждает, что per-unit + ротация по «попытке» — естественный дизайн. У нас `attempt_{N}/` уже есть, осталось добавить `stages/{name}/` под ним
4. **Argo/Tekton** показывают, что агрегирующую ленту удобно держать отдельно (либо файл-агрегат, либо live aggregator) — наш `pipeline.log` как агрегат — в этой же парадигме

Все четыре системы разделяют логи не глубже чем «unit выполнения» (job/container/step) — внутри unit'а редко дробят. У нас unit = стейдж; дробить глубже (например, per-plugin-run в DatasetValidator) не нужно — это решается секционными маркерами/prefix'ами внутри одного stage-файла.

### Альтернатива 3: «hub+spoke» с плоской папкой `logs/` (рекомендация)

Вся семантика «это логи» выносится в одну директорию `logs/` на уровне attempt'а. Артефакты (`evaluation/`, `inference/`) остаются своими папками — у них другая семантика (не текстовые стримы, а структурированные артефакты стейджа).

```
attempts/attempt_1/
├── logs/
│   ├── pipeline.log               # агрегированная лента (как сейчас, просто переехал)
│   ├── dataset_validator.log
│   ├── gpu_deployer.log
│   ├── training_monitor.log       # локальная лента стейджа training_monitor
│   ├── training.log               # удалённый, подтянутый с пода (переехал сюда)
│   ├── model_retriever.log
│   ├── inference_deployer.log
│   └── model_evaluator.log
├── evaluation/                     # артефакты (JSON/MD/CSV оценки) — НЕ логи
├── inference/                      # артефакты (inference_manifest.json и пр.) — НЕ логи
└── experiment_report.md
```

Правила:
- Всё, что пишется текстовым стримом через `logging`/SSH-tail, живёт в `logs/`
- Всё, что генерируется стейджем как структурированный output (JSON/MD/CSV/картинки) — в своих папках стейджа или MLflow
- Имя файла стейдж-лога = имя стейджа: `{stage_name}.log`
- Удалённый `training.log` остаётся с историческим именем (его ищут старые скрипты и TUI)
- `pipeline.log` переезжает `attempt_1/pipeline.log` → `attempt_1/logs/pipeline.log`; добавить временный symlink/shim на время миграции, если нужно старому потребителю

Почему плоско, а не папки на стейдж:
- Стейджей мало (6–7), имена не конфликтуют — вложенность не оправдана
- `ALLOWED_LOG_FILES` в [log_service.py:8](src/api/services/log_service.py:8) уже работает с плоскими именами — минимум правок в API/TUI
- У `training_monitor` и так два файла (`training_monitor.log` + `training.log`) — рядом они уместны по семантике «логи стейджа тренировки», лишний уровень `stages/training_monitor/` ничего не добавляет
- Артефакты остаются в своих папках — семантика разделения прозрачная: `logs/` = стримы, `evaluation/`/`inference/` = артефакты

Этот вариант:
- Не ломает ни одного существующего потребителя (агрегат остаётся, просто переезжает в `logs/`)
- Даёт чистые per-stage логи новым потребителям (TUI-вкладка, быстрый grep, cleanup)
- Переносит инвариант «per-stage изоляции» из state в логи — архитектурная консистентность
- Семантическая группировка: одна папка для текстовых логов, отдельные для артефактов

---

## Рекомендация

**Делать — в варианте hub+spoke (Альт. 3).** Польза реальная (читаемость + TUI + согласованность со state), стоимость невысокая (один per-stage FileHandler с ContextFilter, контекст уже есть в оркестраторе), резюме-логика не страдает (логи ей не нужны).

**Не делать сейчас**, если приоритеты: стабилизация resume, GRPO/SAPO-ресерч — тогда это косметика и её можно отложить. Но **заложить хук** (context-переменную `current_stage`) стоит уже сейчас — он тривиален и откроет путь и к per-stage логам, и к лучшей маркировке MLflow-событий.

---

## Соответствие SOLID / Clean Architecture

### Проблема без `LogLayout`
Пути логов (`attempt_dir/logs/pipeline.log`, `logs/{stage}.log`, `logs/training.log`) прописываются в 4+ местах: `logger.py`, `orchestrator.py`, `log_manager.py`, `log_service.py`. Это нарушает SRP — «знать раскладку логов» размазано.

### Решение: `LogLayout`

Новый модуль `src/utils/logs_layout.py` — единственный владелец правил раскладки FS-логов:

```python
class LogLayout:
    def __init__(self, attempt_dir: Path) -> None
    @property
    def logs_dir(self) -> Path                              # attempt_dir / "logs"
    @property
    def pipeline_log(self) -> Path                          # logs/pipeline.log
    def stage_log(self, stage_name: str) -> Path            # logs/{stage_name}.log
    @property
    def remote_training_log(self) -> Path                   # logs/training.log
    def stage_log_paths(self, stage_name: str) -> dict[str, str]
        # реестр для StageRunState.log_paths, относительно attempt_dir
```

**SRP**: раскладка изменяется в одном файле. **OCP**: новый стейдж получает лог без правок — `LogLayout.stage_log(name)` работает для любого имени. **DIP**: потребители зависят от `LogLayout`, не от `Path`-литералов. **Clean Arch**: единственный адаптер между domain (имена стейджей) и FS.

### Обновления по SOLID в остальных файлах
- `log_service.py` — `ALLOWED_LOG_FILES` **убирается**. Доступные логи читаются из `state.log_paths` каждого `StageRunState` → OCP соблюдается: новый стейдж = новая запись в state, без правок API-слоя
- Fallback для старых runs (без `logs/`): если `state.log_paths` пуст ИЛИ в state нет `log_paths` (старая схема) → смотрим корень attempt'а как раньше (legacy layout)
- В `logger.py` `init_run_logging(...)` принимает `LogLayout` (или конструирует его из `log_dir`) — не знает о конкретных именах файлов внутри
- В `orchestrator.py` `stage_logging_context(stage_name, layout)` получает `LogLayout` из оркестратора — нет прямого построения путей

---

## План реализации (если решили делать)

### Критичные файлы
- [src/utils/logs_layout.py](src/utils/logs_layout.py) — **новый**, класс `LogLayout` (см. выше), единственный владелец FS-раскладки логов
- [src/utils/logger.py](src/utils/logger.py) — добавить `ContextFilter` (ContextVar `_current_stage`) и `add_stage_handler(stage_name, layout) -> RemovableHandler`, новую функцию `stage_logging_context(stage_name, layout)` (contextmanager); `init_run_logging()` кладёт `pipeline.log` через `LogLayout.pipeline_log`
- [src/pipeline/orchestrator.py](src/pipeline/orchestrator.py) — создаёт `LogLayout(attempt_dir)` в `_run_stateful`, оборачивает выполнение каждого стейджа в `stage_logging_context(stage.name, self._log_layout)`, записывает `log_paths` в `StageRunState` при переходе в COMPLETED/FAILED
- [src/pipeline/stages/managers/log_manager.py:55](src/pipeline/stages/managers/log_manager.py) — принимает `LogLayout` или явный путь из `LogLayout.remote_training_log`
- [src/api/services/log_service.py:8](src/api/services/log_service.py:8) — читать `state.log_paths` вместо хардкода `ALLOWED_LOG_FILES`, legacy-fallback на корень attempt'а для старых runs
- [src/pipeline/state/models.py](src/pipeline/state/models.py) — добавить `log_paths: dict[str, str]` в `StageRunState` (опциональное, default `{}`, миграция через `from_dict`) — реестр путей логов стейджа, относительно attempt_dir (`"stage": "logs/dataset_validator.log"`, `"remote_training": "logs/training.log"`)
- [src/tui/live_logs.py](src/tui/live_logs.py), [src/pipeline/live_logs.py](src/pipeline/live_logs.py) — уже работают с произвольным Path, правок минимум

### Переиспользовать
- `init_run_logging()` + `setup_logger()` — основа, новый хендлер добавляется тем же механизмом
- `atomic_write_json` для реестра в state
- `_save_state()` уже пишет `pipeline_state.json` после каждого стейджа — реестр попадёт туда бесплатно

### Backward compat
- `pipeline.log` продолжает писаться, просто путь меняется с `attempts/attempt_N/pipeline.log` на `attempts/attempt_N/logs/pipeline.log`. Для чтения старых runs (без `logs/`) `log_service` должен фолбэчиться на корень attempt'а
- `from_dict` у `StageRunState`: `log_paths` — default `{}`; старые записи читаются без ошибок
- `training.log` сохраняет имя (с ним работают существующие скрипты и TUI), меняется только путь

### Verification
1. `pytest src/tests/unit/pipeline/ src/tests/unit/utils/test_logger.py -xvs` — unit
2. `pytest src/tests/integration/test_phase_executor_restart_flow.py src/tests/integration/test_strategy_orchestrator_restart_flow.py -xvs` — resume не сломан
3. Запустить `run.sh` с fresh конфигом; убедиться:
   - появился `attempts/attempt_1/logs/{stage_name}.log` на каждый выполненный стейдж
   - `attempts/attempt_1/logs/pipeline.log` содержит записи из всех стейджей (агрегат)
   - `attempts/attempt_1/logs/training.log` подтянут c пода (для успешного training_monitor)
   - `pipeline_state.json` содержит `log_paths` в каждом `StageRunState`
4. Запустить resume после намеренного фейла `model_evaluator`; убедиться:
   - `attempts/attempt_2/logs/model_evaluator.log` создаётся
   - логи прошлой попытки в `attempts/attempt_1/logs/` на месте
   - TUI показывает обе попытки, все per-stage файлы видны
5. TUI: `src/tui/` вручную — вкладка per-stage log открывается, live-tail работает; старый run (без `logs/`) тоже открывается (фолбэк)

### Вне скоупа
- Сплит `pipeline_events.json` на per-stage JSON — отдельная, уже обсуждённая задача (mem_287f1e013031)
- Ротация/архивирование старых attempts
- Реорганизация `evaluation/`/`inference/` — они остаются как есть, это артефакты, не логи

---

# Итерация 2 — надёжный per-stage capture + нормализация имён

## Контекст (проблема)

После выката Итерации 1 при запуске пайплайна обнаружены **две связанные проблемы**:

1. **Имена файлов с пробелами**: `logs/Dataset Validator.log`, `logs/GPU Deployer.log`. Причина: `StageNames` в [src/pipeline/_types.py:29](src/pipeline/_types.py:29) — это enum со значениями вида `"Dataset Validator"` (с пробелом), а `LogLayout.stage_log(name)` берёт имя напрямую. Проблема: ломает CLI-автокомплит, грубит скриптам (`grep`, `awk`, `find` через пробел), в URL API встречается с `%20`.
2. **Логи не соответствуют происходящему в стейдже** — в `Dataset Validator.log` не хватает сообщений от MLflow, transformers, datasets, paramiko, а также от `print()`-вызовов в `training/managers/model_saver.py`, `training/orchestrator/metrics_collector.py` и др.

Обе проблемы лежат в механике capture — текущий `_StageContextFilter` прикреплён к логгеру **`ryotenkai`** (см. [src/utils/logger.py:289](src/utils/logger.py:289)), а `propagate=False` у этого логгера ([src/utils/logger.py:105](src/utils/logger.py:105)) отрезает его от root. Значит записи сторонних библиотек (mlflow, transformers, paramiko, httpx, huggingface_hub) — которые логируются через свои независимые иерархии — НЕ доходят до нашего per-stage handler. Плюс прямые `print()` вообще не идут через `logging`.

---

## Дипсинк: 3 итерации анализа

### Итерация 1 — точная диагностика capture

Карта источников вывода внутри стейджа (по результатам explore):

| Источник | Идёт в | Ловится текущим per-stage handler? |
|---|---|---|
| `get_logger("x")` (50+ мест) → `ryotenkai.x` | `ryotenkai` → наш handler | ✅ |
| `logging.getLogger("ryotenkai")` напрямую (SSH client, health_check, providers, plugin_discovery) | `ryotenkai` | ✅ |
| `logging.getLogger("mlflow")` (MLflow info/warn) | `mlflow` → root; не `ryotenkai` | ❌ |
| `logging.getLogger("transformers")` / `"datasets"` | свой → root | ❌ |
| `logging.getLogger("paramiko")` (SSH) | свой → root | ❌ |
| `print(...)` в `metrics_collector.py:34`, `model_saver.py:58`, `models/loader.py:59`, `data_buffer/events.py:24-25` | `sys.stdout` | ❌ |
| subprocess (SSH `Popen`, `launch/runtime.py:238`) — их stdout | наш Python читает и логирует → `ryotenkai` | ✅ (транзитивно) |
| TQDM прогресс-бары | `sys.__stderr__` (bypasses `sys.stderr`) | ❌ (и правильно — не засоряем файл) |

Итог: текущий filter покрывает ~60% того, что «говорит» стейдж. Сторонние логи + `print()` проходят мимо.

### Итерация 2 — как решают аналоги, что применимо у нас

**GitLab / K8s / Argo** — все полагаются на **process boundary**: каждая job/container/step пишет stdout+stderr → runner через docker `CaptureLogs()` или k8s `Pods().GetLogs()` забирает весь поток целиком ([GitLab MR !3680](https://gitlab.com/gitlab-org/gitlab-runner/-/merge_requests/3680)). Изоляция абсолютная, потому что процесс — отдельный.

Для нас process-per-stage — большой рефактор (ломает MLflow manager, shared context). Берём идею «перехватить весь output» и реализуем **in-process**:

| Вариант | Ловит Python logging | Ловит `print()` | Ловит subprocess stdout | Ломает tqdm |
|---|---|---|---|---|
| A. Handler только на `ryotenkai` (**сейчас**) | только `ryotenkai.*` | нет | нет | нет |
| B. Handler на **root** + `ContextFilter` | **всё logging** | нет | нет | нет |
| C. B + `contextlib.redirect_stdout/stderr` с `Tee` | всё logging | да | нет | нет |
| D. C + `os.dup2` FD-level | всё logging | да | да | рискует |
| E. Subprocess-per-stage (GitLab-way) | всё | да | да | нет (слишком инвазивно) |

**Выбор: B.** `print()` — плохая практика сама по себе; чистое решение — выпилить ~5 `print()` в коде и заменить на `logger.info`, а не страховаться через tee stdout/stderr. Выпилка `print()` ведётся **отдельной задачей** (spawned task). В итерации 2 делаем ТОЛЬКО B — root-handler + ContextFilter. Это чистая архитектура, а не страхующий слой.

Детали [Python Logging Cookbook — filter-based context injection](https://docs.python.org/3/howto/logging-cookbook.html) и [PEP 567 ContextVar](https://peps.python.org/pep-0567/): ContextVar распространяется по asyncio tasks и копируется в ThreadPoolExecutor, что проверено в коде ([dataset_validator.py:248](src/pipeline/stages/dataset_validator.py:248), [adapter_cache.py:60](src/training/orchestrator/phase_executor/adapter_cache.py:60)). Поэтому ThreadPool у нас — ок.

### Итерация 3 — конкретное решение, обходы подводных камней

1. **Slug имени файла в `LogLayout`**
   - Новый приватный `_slugify(name)`: `"Dataset Validator"` → `"dataset_validator"` (regex `[^a-z0-9]+` → `_`).
   - `stage_log(name)` использует slug. `stage_log_registry` пишет slug-путь в реестр.
   - В `log_service._discover_from_state`: имя файла для API = `Path(rel_path).name` (берём из реестра, не пересобираем по stage_name с пробелом).

2. **Handler attach — куда именно**
   - Цель: захватить и `ryotenkai.*`, и `mlflow/transformers/paramiko/httpx/...`.
   - Наивный способ — handler на root + один filter. Но записи из `ryotenkai` **не** пропагируют до root (`propagate=False`).
   - **Рабочий вариант**: вешаем per-stage handler **и на `ryotenkai`, и на root-logger** с двумя filter'ами:
     - на `ryotenkai`-handler: обычный `_StageContextFilter(stage)`.
     - на root-handler: `_StageContextFilter(stage) AND not record.name.startswith("ryotenkai")` — чтобы избежать дубля (ryotenkai идёт своим путём).
   - Оба handler'а пишут в **один** `FileHandler` (общий open file object), чтобы не дублировать FD и упорядочить записи в файле.

3. **`print()` — вне скоупа итерации 2**
   - `print()` не ловим tee-костылём. Вместо этого — отдельная задача: выпилить ~5 `print()` в [metrics_collector.py:34](src/training/orchestrator/metrics_collector.py:34), [model_saver.py:58](src/training/managers/model_saver.py:58), [models/loader.py:59](src/training/models/loader.py:59), [data_buffer/events.py:24-25](src/training/managers/data_buffer/events.py:24), [model_retriever/model_card.py:224](src/pipeline/stages/model_retriever/model_card.py:224) и заменить на `logger.info`. Это чище архитектурно и не требует подмены `sys.stdout`.

4. **Шум от внешних либ**
   - Добавить разово в `init_run_logging` блок «приглушить шумные»: `logging.getLogger("httpx").setLevel(WARNING)`, `urllib3`, `filelock`, `botocore`. Не трогать mlflow/transformers/paramiko (их INFO полезен).

5. **Миграция реестра `log_paths` в state**
   - Новые записи пишутся через slug.
   - В `log_service`: fallback уже работает — если файл не существует по slug-пути, смотрит legacy.

---

## Рекомендация

Делать **B** с slug-нормализацией. Цена: ~40 строк в `logger.py`, +`_slugify` в `logs_layout.py`. Польза: стабильные имена + 100% Python-logging capture без process-рефактора. `print()` — выпиливается отдельной задачей, там же сразу правим архитектурную проблему, а не прячем её tee-костылём.

---

## План реализации

### Критичные файлы
- [src/utils/logs_layout.py](src/utils/logs_layout.py) — добавить `_slugify`, применять в `stage_log` и `stage_log_registry`
- [src/utils/logger.py](src/utils/logger.py) — переписать `stage_logging_context`:
  - открыть **один** `FileHandler` на stage-файл
  - attach на `logging.getLogger("ryotenkai")` с `_StageContextFilter(stage)`
  - attach на `logging.getLogger()` (root) с `_StageContextFilter(stage) + _ExcludeRyotenkaiFilter` (чтобы не задваивать записи из ryotenkai)
  - в `finally` — снять оба handler'а, закрыть файл, сбросить ContextVar
- [src/utils/logger.py](src/utils/logger.py) — в `init_run_logging` разово подкрутить level шумных либ; `pipeline.log` FileHandler перевесить с `ryotenkai` на root (чтобы агрегат был полный)
- [src/api/services/log_service.py](src/api/services/log_service.py) — `_discover_from_state`: имя файла = `Path(rel_path).name` (не `f"{stage_name}.log"`)
- [src/tests/unit/utils/test_logs_layout.py](src/tests/unit/utils/test_logs_layout.py) — добавить тесты `_slugify` и интеграции со `stage_log`
- [src/tests/unit/utils/test_logger.py](src/tests/unit/utils/test_logger.py) — добавить тесты захвата mlflow/transformers logger'ов в stage-файл

### Переиспользовать
- `_StageContextFilter` (уже есть), `_current_stage` ContextVar (уже есть)

### Backward compat
- Старые runs с `Dataset Validator.log` — читаются через legacy fallback в log_service (уже)
- Реестр `log_paths` в старых state.json — при load значения со старыми именами остаются рабочими путями

### Verification
1. `pytest src/tests/unit/utils/test_logs_layout.py src/tests/unit/utils/test_logger.py src/tests/unit/api/services/test_log_service_registry.py -xvs`
2. Запустить `run.sh` до стадии mlflow init; убедиться:
   - файлы названы `dataset_validator.log`, `gpu_deployer.log`, …
   - в `gpu_deployer.log` есть сообщения `paramiko.*` (SSH connect) и `mlflow.*`
   - в `dataset_validator.log` есть сообщения `datasets.*` (load)
   - в `pipeline.log` (новый — на root) есть те же записи от сторонних либ, что и в per-stage
3. `pytest src/tests/integration/test_phase_executor_restart_flow.py src/tests/integration/test_strategy_orchestrator_restart_flow.py`
4. TUI — открыть любой run, убедиться что имена файлов без пробелов, content full

---

## Риски и нерешённые вопросы (зафиксированы)

- **Q3**: Прежняя запись в pipeline.log остаётся? Или нужен отдельный handler для pipeline.log агрегата?
- **Q4**: Миграция старых runs: существующие `Dataset Validator.log` — переименовывать или оставить?
- **Q5**: Level фильтр — ставить `INFO` на stage handler или наследовать из root?
- **Q6**: Subprocess с inherited FD (без PIPE) — попадают?
- **Q7**: Race condition в `_slugify` на nested stages?

---

## Ответы (дипсинк по каждому вопросу)

### Q3. pipeline.log агрегат — нужен ли отдельный handler?

**Итерация 1**: Сейчас `init_run_logging` ставит FileHandler на `ryotenkai` → `pipeline.log`. Это handler **без filter'а** — принимает всё из `ryotenkai.*`. Root-logger туда не пишет.

**Итерация 2**: Если мы начнём ловить mlflow/transformers в root-level stage handler, пользователь ожидает, что pipeline.log ТОЖЕ их содержит (он же «агрегированная лента»). Значит pipeline.log тоже должен быть root-handler, не ryotenkai-handler.

**Итерация 3**: Правильное: **pipeline.log handler** ставится на **root logger**, без ContextFilter'а (пишет всё). Console handler остаётся на `ryotenkai` (propagate=False защищает от шума transformers в консоль — хорошо, не засоряем терминал). Итого:
- console: `ryotenkai` only, чисто
- pipeline.log: root, всё
- stage.log: root + ryotenkai одновременно (через два handler'а с одним filter'ом), фильтр по ContextVar

**Ответ**: перенести pipeline.log handler с `ryotenkai` на root. Console handler остаётся на `ryotenkai` для чистоты терминала.

### Q4. Миграция старых runs с `Dataset Validator.log`

**Итерация 1**: Старые файлы на диске уже есть. Код с slug будет создавать `dataset_validator.log` — параллельно к `Dataset Validator.log` если запускать второй attempt того же run'а.

**Итерация 2**: Для чистоты: если в `logs/` есть и `Dataset Validator.log`, и `dataset_validator.log` — TUI покажет оба. Это минорный косметический шум.

**Итерация 3**: Не переименовывать — это изменение FS-артефактов, риск race. Оставить как есть, legacy fallback в log_service выведет оба. Пользователь сам может удалить вручную при необходимости.

**Ответ**: не переименовываем. Документируем в Backward compat. При запуске resume на run'e со старым attempt — новый attempt создаст slug-файлы в своей папке, старый attempt остаётся нетронутым.

### Q5. Level на stage handler

**Итерация 1**: Если INFO, потеряем DEBUG от MLflow при глубокой диагностике. Если наследовать от root — зависит от root level (обычно WARNING).

**Итерация 2**: Нам важна полнота stage.log. Берём `_log_level` (то же, что у pipeline.log) — обычно INFO. Уровень единообразен с pipeline.log.

**Итерация 3**: Фиксируем: `handler.setLevel(_log_level)`. DEBUG включается через `LOG_LEVEL=DEBUG` env — применяется ко всем трём handler'ам.

**Ответ**: `_log_level` (единый с pipeline.log).

### Q6. Subprocess stdout через inherited FD

**Итерация 1**: Если вызываем `subprocess.run(..., stdout=None)` — child пишет в fd 1 родителя напрямую в терминал, минуя и logging, и наш root-handler.

**Итерация 2**: В нашем коде subprocess везде с `stdout=subprocess.PIPE` (см. `launch/runtime.py:238`, `utils/ssh_client.py`). Python читает stdout построчно и логирует через `logger.info(line)` → `ryotenkai` → stage handler. Покрыто.

**Итерация 3**: Проверить grep'ом: `subprocess.run(...)` без `stdout=PIPE` — если есть, вывод пролетает мимо. Для 100% покрытия нужен `os.dup2` (вариант D), что мы отмели — лишняя сложность ради редкого кейса.

**Ответ**: ограничение документируем. Все текущие subprocess'ы используют PIPE → безопасно. В тесте покрыть случай.

### Q7. Race condition в _slugify на nested stages

**Итерация 1**: `_slugify` — чистая функция строки. Без shared state. Race невозможен.

**Итерация 2**: Вложенный `stage_logging_context` (stage внутри stage, например, DatasetValidator может запустить поделам под-стейджи) — ContextVar token перекрывается, inner stage перехватывает. Slug считается независимо для каждого. Нет конфликта.

**Итерация 3**: Но: если inner stage имеет ТО же имя что outer — оба handler'а будут писать в один файл. В нашем коде такого нет (стейджи имеют уникальные имена).

**Ответ**: безопасно. Добавить assertion `_current_stage.get() is None` на входе в stage_logging_context (защита от забытого выхода), в тестах покрыть nested case.

---

## Вне скоупа итерации 2

- **Выпилка `print()` в пользу `logger.info`** — отдельная задача (spawned). ~5 мест, 15 минут. Чистит архитектурный хлам, а не прячет его.
- FD-level capture (`os.dup2`) — оставлено на будущее, если появится subprocess без PIPE
- process-per-stage (GitLab-way) — рефактор оркестратора, отдельная большая задача
- QueueHandler для async — пайплайн синхронный, не нужен

Sources:
- [GitLab Job logs](https://docs.gitlab.com/administration/cicd/job_logs/)
- [GitLab Runner MR !3680 — capture helper service logs](https://gitlab.com/gitlab-org/gitlab-runner/-/merge_requests/3680)
- [Python Logging Cookbook — contextvars-based filter](https://docs.python.org/3/howto/logging-cookbook.html)
- [PEP 567 — Context Variables](https://peps.python.org/pep-0567/)
