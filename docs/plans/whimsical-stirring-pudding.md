# RunPod Pod Lifecycle: Watchdog Safety-Net + Architecture Cleanup

## Context

Feature B (Pod Auto-Stop, уже в main) положила RunPod-специфичную bash-функцию
`_ryotenkai_stop_pod()` прямо внутрь generic `start_training.sh`, который
генерируется в `src/pipeline/stages/managers/deployment_manager.py:848-884`.
Это работает для happy-path (Python завершился → bash вызывает podStop), но:

1. **Архитектурный долг**: RunPod-логика протекла в generic слой. Deployment
   manager проверяет `provider_name == "runpod"` строкой вместо использования
   провайдера как абстракции. Нарушение SRP и DIP.
2. **Дыра в safety-net**: `_ryotenkai_stop_pod` вызывается **только** если
   bash-скрипт завершился нормально. Не покрыто: SIGKILL, kernel panic, OOM
   killer, hang Python, отказ пользователя от pipeline (закрыл ноутбук и не
   вернулся), неудача всех 3 retry curl'а.
3. **Риск денег**: pod простаивает и тарифицируется, если пользователь отошёл
   от панели, а happy-path не сработал.

**Цель этой фичи** — сделать две связанные вещи:

- **Refactor**: вынести весь RunPod pod-lifecycle код из `deployment_manager.py`
  в `src/providers/runpod/` через новый hook в `IGPUProvider` протоколе.
  Deployment manager становится provider-agnostic.
- **Watchdog**: добавить независимый bash-скрипт на pod'е, который гарантирует
  остановку pod'а после любого логического конца training'а, но **только если
  pipeline неактивен** — пока pipeline активно мониторит через SSH, watchdog
  остаётся в спящем режиме.

Принципы: SOLID, KISS, YAGNI, boy scout rule. Обратная совместимость не нужна.

## Idea Evaluation

### Сильные стороны

- **Независимость процесса**: watchdog стартует через `setsid nohup ... & disown`,
  переживает SIGKILL родительского bash, kernel panic ловится только max-lifetime'ом.
- **Pipeline heartbeat gate**: главная защита от false positives. TrainingMonitor
  уже опрашивает pod каждые 5с (Feature A) — достаточно `touch` одного файла
  на каждой итерации. Пока pipeline активен, watchdog дремлет.
- **Чистая архитектура через hook**: никакой RunPod-специфики в
  `deployment_manager.py`. Новые провайдеры (Lambda, Vast.ai) смогут внедрять
  свои hook'и без модификации generic слоя.
- **Фиксирует существующий антипаттерн** одновременно с добавлением фичи.
- **Инфраструктура уже есть**: `nvidia-smi`, `curl`, доступ к `/workspace`,
  `RUNPOD_API_KEY`.

### Риски и митигации

| # | Риск | Митигация |
|---|------|-----------|
| 1 | **False positive: watchdog убил здоровый pod во время нормального training** (пауза между eval batch'ами даёт провал util). | (a) Pipeline heartbeat gate: пока pipeline мониторит, watchdog не трогает idle detection вообще. (b) Двойной порог `util<5% И mem_pct<30%` одновременно — реально работающий Python держит веса в VRAM. (c) 20 минут непрерывного idle. (d) Startup grace 5 минут. |
| 2 | **Watchdog сам упал** (bash-ошибка) — safety-net отсутствует. | `set -uo pipefail`, heartbeat-файл `/workspace/.watchdog_heartbeat` обновляется каждый tick; `start_training.sh` проверяет heartbeat через 10с и логирует предупреждение если watchdog не стартовал; watchdog.sh проходит через shellcheck в CI. |
| 3 | **Race с happy-path**: оба вызывают `podStop` API одновременно. | `podStop` идемпотентен. Повторный вызов вернёт ошибку "уже остановлен" — ловим и выходим. |
| 4 | **API key leak** в логи. | Никогда не `set -x`. curl без `-v`. stderr curl не попадает в log напрямую. |
| 5 | **Watchdog переживёт legitimate pipeline resume** (laptop sleep → wake → pipeline снова опрашивает pod). | Heartbeat gate: pipeline при resume начнёт снова писать heartbeat → watchdog уйдёт в сон. Если laptop спит дольше `PIPELINE_HEARTBEAT_STALE=10 мин`, watchdog активируется — но видит, что Python реально работает (GPU не idle), и не стопает. Если же training сам упал за время sleep'а, watchdog правильно остановит pod. |
| 6 | **Pod жив вечно** (watchdog сам завис). | Hard max lifetime 48h — безусловный стоп. |
| 7 | **`RUNPOD_KEEP_ON_ERROR=true`** — пользователь отлаживает упавший run. | Watchdog видит `TRAINING_FAILED` + флаг → не стопает, продолжает heartbeat до max lifetime. |
| 8 | **Watchdog не запустился** (ошибка в скрипте). | Проверка `.watchdog_heartbeat` в `start_training.sh` в течение 10с после setsid; лог-warning если не стартовал. Training идёт дальше — watchdog это safety-net, не блокер. |
| 9 | **Watchdog потребляет ресурсы training'а**. | Цикл: 30с sleep, одна `nvidia-smi` (ms), `stat` на 3 файла, `touch` heartbeat. CPU/IO пренебрежимы по сравнению с training'ом. |

**Вердикт**: идея качественная, митигации покрывают все выявленные риски.
Pipeline heartbeat gate — ключевой инсайт, который полностью решает проблему
"не мешать активному run'у".

## Architecture

### Current state (проблемный)

```
deployment_manager.py (generic)
├── _create_env_file()  ← вбивает RUNPOD_* env vars (строки 718-728)
└── _start_training_cloud()
    └── f-string script_content с RunPod _ryotenkai_stop_pod() (строки 848-884)
```

### Target state (чистый)

```
deployment_manager.py (generic, provider-agnostic)
└── _start_training_cloud(ssh_client, context, provider: IGPUProvider)
    ├── hooks = provider.prepare_training_script_hooks(ssh_client, context)
    ├── merge hooks.env_vars into .env
    └── build script_content:
        {hooks.pre_python}     ← RunPod: запуск watchdog
        python -m ...
        {hooks.post_python}    ← RunPod: вызов _runpod_stop_pod (happy path)

src/providers/training/interfaces.py
├── @dataclass TrainingScriptHooks
│   ├── env_vars: dict[str, str]
│   ├── pre_python: str       # bash перед вызовом Python
│   └── post_python: str      # bash после exit_code capture
└── IGPUProvider protocol
    └── prepare_training_script_hooks(ssh_client, context) -> Result[TrainingScriptHooks]

src/providers/runpod/training/provider.py
└── prepare_training_script_hooks():
    ├── проверяет cleanup.auto_stop_after_training
    ├── uploads resources/runpod_stop_pod.sh через ssh_client
    ├── uploads resources/watchdog.sh через ssh_client
    ├── returns hooks:
    │   - env_vars: RUNPOD_API_KEY, RUNPOD_POD_ID, RUNPOD_AUTO_STOP, RUNPOD_KEEP_ON_ERROR
    │   - pre_python: "setsid nohup bash /workspace/watchdog.sh ... & disown"
    │   - post_python: "source /workspace/runpod_stop_pod.sh && _runpod_stop_pod"

src/providers/runpod/training/resources/  (NEW)
├── watchdog.sh           # независимый safety-net процесс
└── runpod_stop_pod.sh    # функция _runpod_stop_pod, shared между watchdog и happy-path

src/providers/training/single_node/provider.py
└── prepare_training_script_hooks() → Ok(empty hooks)  # single_node ничего не делает

src/pipeline/stages/gpu_deployer.py
└── self.deployment.start_training(ssh_client, context, provider=self._provider)

src/pipeline/stages/training_monitor.py  (Feature A — уже модифицирован)
└── в _monitor_training_resilient polling loop (каждые 5с):
    ssh_client.exec_command(f"touch {workspace}/.pipeline_heartbeat", silent=True)
```

### Pipeline heartbeat gate (ключевая идея)

Watchdog имеет два режима:

1. **Dormant** (pipeline активен): `/workspace/.pipeline_heartbeat` свежее
   `PIPELINE_HEARTBEAT_STALE=600` секунд → watchdog только обновляет свой
   heartbeat и спит. **Ничего не проверяет, никого не трогает.**

2. **Active** (pipeline отсутствует): heartbeat старее 10 минут → watchdog
   начинает проверять маркеры и GPU idle. Активируется:
   - Пользователь закрыл ноутбук навсегда.
   - Pipeline упал на host-стороне.
   - Training завершился и pipeline ушёл (happy-path должен был отработать,
     но если нет — watchdog добивает).

Reactivation: если pipeline возвращается (resume после sleep), он снова
начинает `touch`'ать heartbeat → watchdog автоматически уходит в dormant.

### Watchdog поведение (в active режиме)

```
loop:
  touch .watchdog_heartbeat
  uptime = now - start_ts
  if uptime > MAX_LIFETIME: stop("MAX_LIFETIME")

  if pipeline_heartbeat fresh: idle_since=0; sleep; continue  # dormant

  # Active mode
  if TRAINING_COMPLETE or TRAINING_FAILED:
    if TRAINING_FAILED and RUNPOD_KEEP_ON_ERROR: sleep; continue  # debug mode
    sleep LOGICAL_END_GRACE (60s)
    stop("LOGICAL_END")

  if uptime > STARTUP_GRACE (5min):
    util, mem_pct = nvidia-smi  (max across GPUs)
    if util<5 and mem_pct<30:
      idle_since = idle_since or now
      if (now - idle_since) > IDLE_THRESHOLD (20min):
        stop("GPU_IDLE")
    else:
      idle_since = 0

  sleep POLL_INTERVAL (30s)
```

`stop(reason)` пишет `/workspace/STOPPED_BY_WATCHDOG` с reason/timestamp/
uptime/метриками, потом вызывает `_runpod_stop_pod` (та же функция, что и
happy path — через sourced `runpod_stop_pod.sh`).

## Critical files

### New files

- **`src/providers/runpod/training/resources/watchdog.sh`** — независимый
  safety-net, ~100 строк bash. Константы в голове файла: `POLL_INTERVAL=30`,
  `STARTUP_GRACE=300`, `PIPELINE_HEARTBEAT_STALE=600`, `IDLE_THRESHOLD=1200`,
  `IDLE_UTIL_MAX=5`, `IDLE_MEM_MAX_PCT=30`, `LOGICAL_END_GRACE=60`,
  `MAX_LIFETIME=172800`. Структура цикла описана выше.

- **`src/providers/runpod/training/resources/runpod_stop_pod.sh`** — shared
  bash функция `_runpod_stop_pod()` (~30 строк). Вся curl/GraphQL/retry логика
  из текущего `_ryotenkai_stop_pod` в `deployment_manager.py:853-883`
  переезжает сюда **без изменения поведения**. Source'ится из watchdog.sh и
  из `post_python` hook.

- **`src/providers/runpod/training/resources/__init__.py`** — пустой marker для
  importlib.resources доступа.

### Modified files

- **`src/providers/training/interfaces.py`** — добавить `TrainingScriptHooks`
  dataclass и метод `prepare_training_script_hooks(ssh_client, context)` в
  `IGPUProvider` Protocol. Default реализация в single_node provider возвращает
  `Ok(TrainingScriptHooks(env_vars={}, pre_python="", post_python=""))`.

- **`src/providers/runpod/training/provider.py`** — реализовать
  `prepare_training_script_hooks()`:
  - Читает `cleanup = self._config.cleanup` (`src/config/providers/runpod/cleanup.py`).
  - Если `not cleanup.auto_stop_after_training` — возвращает empty hooks.
  - Использует `importlib.resources.files("src.providers.runpod.training.resources")`
    для чтения `runpod_stop_pod.sh` и `watchdog.sh` как текста.
  - Через `ssh_client.exec_command` (here-doc) создаёт оба файла на pod'е
    в `/workspace/` + `chmod +x`.
  - Возвращает hooks:
    - `env_vars`: `RUNPOD_API_KEY`, `RUNPOD_POD_ID`, `RUNPOD_AUTO_STOP=true`,
      `RUNPOD_KEEP_ON_ERROR={true|false}`.
    - `pre_python`: `setsid nohup bash /workspace/watchdog.sh >/workspace/watchdog.log 2>&1 </dev/null & disown` + 10-секундная проверка heartbeat.
    - `post_python`: `source /workspace/runpod_stop_pod.sh && _runpod_stop_pod || true` (не падает если source не нашёл — safety-net сработает).

- **`src/providers/training/single_node/provider.py`** — добавить тривиальную
  реализацию `prepare_training_script_hooks()` возвращающую empty hooks.

- **`src/pipeline/stages/managers/deployment_manager.py`**:
  - `_create_env_file`: **удалить** RunPod-специфичный блок (строки 718-728).
    Вместо этого принимать `extra_env_vars: dict[str, str]` параметром и
    мержить их в `env_vars`.
  - `_start_training_cloud`: принимать `provider: IGPUProvider` параметром.
    Вызывать `hooks = provider.prepare_training_script_hooks(ssh_client, context)`.
    Мержить `hooks.env_vars` в `.env`. В `script_content` f-string:
    - **удалить** `_ryotenkai_stop_pod` функцию и её вызов (строки 848-884).
    - Добавить `{hooks.pre_python}` **перед** блоком выбора PY_BIN.
    - Добавить `{hooks.post_python}` **после** `exit_code=$?`, перед
      генерацией TRAINING_FAILED marker (чтобы post_python видел `exit_code`).
  - `start_training(ssh_client, context)` → `start_training(ssh_client, context, provider)`.

- **`src/pipeline/stages/gpu_deployer.py`** — при вызове
  `self.deployment.start_training(ssh_client, context)` (строка ~220) передать
  `provider=self._provider`.

- **`src/pipeline/stages/training_monitor.py`** (Feature A уже модифицирован,
  добавим heartbeat):
  - В polling loop `_monitor_training_resilient` / `_monitor_training`,
    перед `_check_marker`, добавить вызов:
    `ssh_client.exec_command(f"touch {workspace}/.pipeline_heartbeat", silent=True, background=False, timeout=2)`.
  - Silent=True чтобы не засорять логи; короткий timeout чтобы не блокировать
    polling при лагах SSH.
  - Выполняется только для cloud-провайдеров (проверка через `provider_name
    == "runpod"` в context, или всегда — `touch` на несуществующий путь просто
    молча отработает; YAGNI).

### Tests to update

- **`src/tests/unit/pipeline/test_stages_monitor.py`** — существующие тесты
  уже используют мок `ssh_client.exec_command`. Новый `touch` вызов не должен
  ломать тесты, но нужно проверить: если мок строгий (assert_called_with),
  добавить expectation на `touch`. Если свободный — пройдёт.

- **`src/tests/unit/pipeline/test_stages_deployer.py`** (если есть) — проверить
  что `start_training` вызывается с `provider=...` параметром.

- **`src/tests/unit/providers/runpod/test_training_provider.py`** (создать если
  нет) — unit тест для `prepare_training_script_hooks`:
  - Mock SSH client, проверить что оба файла uploaded.
  - Проверить содержимое hooks (env_vars содержат нужные ключи, pre/post
    python содержат ожидаемые команды).
  - Проверить что при `auto_stop_after_training=False` возвращается empty hooks.

- **`src/tests/unit/providers/training/test_interfaces.py`** — обновить мок-провайдер
  если он реализует IGPUProvider, добавить заглушку метода.

- **Shellcheck**: прогнать `shellcheck watchdog.sh runpod_stop_pod.sh` локально
  перед коммитом. Добавить в CI если есть pre-commit hook для bash.

## Что НЕ делаем (YAGNI)

- Не делаем Python-версию watchdog — bash достаточен, меньше зависимостей.
- Не выносим константы watchdog в pydantic config — hardcoded в bash. Если
  когда-нибудь понадобится tuning, вынесем позже.
- Не делаем generic "pipeline heartbeat" абстракцию — это одна строчка `touch`
  в monitor'е и одна проверка `stat` в bash.
- Не меняем интерфейс `SSHClient` — используем existing `exec_command`.
- Не удаляем `runs_active_pods.json` registry в `cleanup_manager.py` — это
  отдельная фича (host-side cleanup), не пересекается.
- Не трогаем Feature A изменения в `training_monitor.py` кроме добавления
  одного `touch` вызова.

## Verification

### Unit tests
```bash
pytest src/tests/unit/providers/runpod/test_training_provider.py -v
pytest src/tests/unit/pipeline/test_stages_monitor.py -v
pytest src/tests/unit/pipeline/test_stages_deployer.py -v
```

### Static checks
```bash
shellcheck src/providers/runpod/training/resources/watchdog.sh
shellcheck src/providers/runpod/training/resources/runpod_stop_pod.sh
ruff check .
mypy src/providers/ src/pipeline/stages/managers/deployment_manager.py
```

### Manual E2E on real RunPod

Мини-конфиг: 10 training steps, мелкая модель, `auto_stop_after_training=true`.

1. **Happy path (pipeline активен, training завершился нормально)**:
   - Запустить pipeline. Дождаться TRAINING_COMPLETE.
   - Ожидаемо: `post_python` hook → `_runpod_stop_pod` → pod stopped.
   - `watchdog.log` показывает "pipeline heartbeat active, dormant" на
     протяжении всего run'а.
   - Pod остановлен быстро (в течение <1 мин после Python exit).

2. **Pipeline отсутствует, training зависает**:
   - Запустить pipeline, дождаться что watchdog стартовал.
   - Убить pipeline на host'е: `pkill -9 -f "python -m src.main"`.
   - Остановить heartbeat'ы (через ~10 мин heartbeat становится stale).
   - Замочить training Python процесс через `kill -STOP` на pod'е
     (GPU освобождается, процесс alive но не работает).
   - Ожидаемо: через ~25 мин (5 grace + 20 idle) watchdog стопает pod с
     `STOPPED_BY_WATCHDOG=GPU_IDLE`.

3. **Pipeline sleep scenario (resume возможен)**:
   - Запустить pipeline, дождаться нескольких heartbeat'ов.
   - Закрыть ноутбук на 3 минуты (heartbeat свежий), открыть. Pipeline резюмит
     (Feature A).
   - Ожидаемо: watchdog всё время dormant, pod НЕ остановлен.
   - Проверить `watchdog.log`: только heartbeat-тики, ни одной проверки GPU.

4. **Debug mode**:
   - Конфиг `keep_pod_on_error=true`. Заведомо падающее training.
   - Убить pipeline на host'е после падения.
   - Ожидаемо: watchdog видит TRAINING_FAILED + RUNPOD_KEEP_ON_ERROR=true →
     НЕ стопает pod. Pod жив до ручного стопа или 48h max lifetime.

5. **SIGKILL bash (жёсткий)**:
   - Запустить pipeline и training.
   - На pod'е: `kill -9 $(pgrep -f start_training.sh)`. Bash мёртв, Python
     мёртв, GPU освобождён.
   - Убить pipeline на host'е (чтобы heartbeat стал stale).
   - Ожидаемо: watchdog через ~25 мин стопает с `GPU_IDLE`.
   - Проверить `ps -ef | grep watchdog` — watchdog всё ещё жив после смерти
     родителя (setsid + nohup работает).

6. **Architecture validation**:
   - `grep -rn "runpod\|RUNPOD" src/pipeline/stages/managers/deployment_manager.py`
     должен вернуть **пустой результат**. Весь RunPod-код ушёл в провайдер.
