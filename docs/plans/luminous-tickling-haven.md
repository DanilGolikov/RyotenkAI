# Оценка идеи: раздельные логи по стейджам пайплайна

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
