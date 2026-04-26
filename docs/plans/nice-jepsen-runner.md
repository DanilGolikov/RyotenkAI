# План: Job Server — provider-agnostic control plane для удалённого обучения

> Status: **DRAFT — требует одобрения**
> Author: daniil
> Date: 2026-04-26
> Worktree: `nice-jepsen-07d789`

---

## 1. Контекст

Сегодня удалённое обучение (RunPod / single_node) построено на 3 независимых процессах в pod-е, общающихся через **9 marker-файлов** на shared FS, и Mac, который **поллит SSH** каждые 5 секунд.

Анализ выявил 10 системных проблем:

1. Implicit FSM в файлах (`TRAINING_COMPLETE`, `.pipeline_heartbeat`, `STOPPED_BY_WATCHDOG`, ...)
2. Polling вместо push, грубый прогресс
3. `watchdog.sh` (165 строк bash) как критический сервис без тестов
4. Tight coupling через FS-конвенции
5. Дублирование ответственности между `TrainingMonitor` и `watchdog.sh`
6. Mac = SPOF (sleep ноута → watchdog убивает pod)
7. Нет interactive control (stop/snapshot/swap-params)
8. Нет ergonomics для SSH-only окружений
9. Нет audit-канала для решений lifecycle
10. Plugin runtime закрыт от introspection

## 2. Решения и подтверждения от пользователя

| Решение | Подтверждено |
|---|---|
| Auth single-tenant, через SSH-туннель, JWT — позже | ✅ |
| Не "истинный pause", а **detach/reattach клиента** (training продолжается) | ✅ |
| Один **унифицированный** docker image, поле `image_name`/`docker_image` **прибито гвоздями** (не override) | ✅ |
| Single_node режим: Mac → SSH → удалённый docker host (никаких локальных запусков) | ✅ |
| MLflow остаётся, job-server становится прокси-релеем | ✅ |
| Транспорт Mac↔pod: HTTP/WebSocket через `ssh -L` туннель | ✅ |

## 3. Целевая архитектура

```
┌──────────────────── Mac (Control Plane) ────────────────────┐
│                                                             │
│  Web UI (React/Vite, существующий)                          │
│         │                                                   │
│         ▼                                                   │
│  FastAPI (src/api/main.py, существующий)                    │
│         │                                                   │
│  CLI (Typer, существующий + новые подкоманды)               │
│         │                                                   │
│  PipelineOrchestrator (существующий, рефакторится)          │
│         │                                                   │
│  ┌──────┴────────────────────────────────────────┐          │
│  │ ProviderRegistry (расширяемая)                │          │
│  │  ├─ RunPodProvider     (GraphQL pods API)     │          │
│  │  ├─ SingleNodeProvider (SSH + docker run)     │          │
│  │  └─ <FutureProvider>  (Lambda, GCP, ...)      │          │
│  └──────┬────────────────────────────────────────┘          │
│         │ (создал pod, получил SSH endpoint)                │
│         ▼                                                   │
│  JobClient (new) — HTTP/WS клиент для pod-а                 │
│  SSHTunnel (new) — управляет `ssh -L 18080:127.0.0.1:8080`  │
│         │                                                   │
└─────────┼───────────────────────────────────────────────────┘
          │ HTTP/WebSocket поверх SSH-туннеля
          ▼
┌──────────────────── pod (один docker container) ────────────┐
│                                                             │
│  ENTRYPOINT: dumb-init → start.sh →                         │
│    ├─ /usr/sbin/sshd                                        │
│    └─ uvicorn ryotenkai.runner:app --host 127.0.0.1:8080    │
│                                                             │
│  Job Server (новый Python пакет на pod-е, provider-agnostic)│
│  ┌──────────────────────────────────────────────────────┐   │
│  │ JobLifecycleFSM (in-memory)                          │   │
│  │  states: idle → preparing → running → stopping →     │   │
│  │          completed | failed | cancelled              │   │
│  │                                                      │   │
│  │ Components:                                          │   │
│  │  ├─ Supervisor       (subprocess.Popen, signals)     │   │
│  │  ├─ EventBus         (pub/sub, ring buffer 10k)      │   │
│  │  ├─ MLflowRelay      (sync forward)                  │   │
│  │  ├─ IdleDetector     (Python replacement watchdog)   │   │
│  │  ├─ HealthReporter   (GPU/RAM/CPU snapshots)         │   │
│  │  └─ ArtifactIndex    (checkpoints, faulthandler.log) │   │
│  └──────────────────────────────────────────────────────┘   │
│         │ (запускает по команде POST /jobs)                 │
│         ▼                                                   │
│  Trainer subprocess: python -m src.training.run_training    │
│         │                                                   │
│         │ HTTP POST loopback (TrainerCallback)              │
│         │  127.0.0.1:8080/internal/events                   │
│         ▼                                                   │
│  Job Server EventBus                                        │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

## 4. Зоны ответственности

### 4.1. Control Plane (Mac, существующий код)

- Lifecycle ресурса: создание/удаление pod-а через провайдер
- Lineage: pipeline_state.json, attempts, restart points
- Применение presets, валидация конфига
- MLflow root run / parent run (как сейчас)
- Запуск SSH-туннеля
- Mac CLI/Web UI = HTTP-клиент к Job Server

**НЕ знает:** что происходит внутри trainer'а на каждом step'е, GPU-метрики (получает push-events).

### 4.2. ProviderRegistry (Mac, рефакторится)

Каждый провайдер реализует **одинаковый** интерфейс:

```python
class IGPUProvider(Protocol):
    def provision(self, spec: GPURequirement) -> ProvisionResult: ...
    # → ProvisionResult имеет SSH endpoint + workspace_path

    def teardown(self, resource_id: str) -> None: ...

    def status(self, resource_id: str) -> ProviderStatus: ...

    @property
    def runtime_image(self) -> str: ...
    # один и тот же для всех "gold" image, override только для dev
```

Job Server не знает о провайдере. Provider-specific только:
- `RUNPOD_AUTO_STOP=true` env (если pod должен сам себя стопнуть через GraphQL)
- `RUNPOD_API_KEY` / `RUNPOD_POD_ID` (передаются для self-stop)

### 4.3. Job Server (pod, новый код в `src/runner/`)

- Принять `POST /jobs` с описанием работы (config + datasets references)
- Запустить trainer subprocess
- Слушать события от trainer'а (через TrainerCallback → loopback HTTP)
- Хранить in-memory FSM + ring buffer событий (для detach/reattach)
- Стримить события через WebSocket подписчикам
- Detect idle GPU + длительность без heartbeat trainer'а → trigger graceful stop
- При stop: послать SIGTERM trainer'у, дождаться save_checkpoint, потом SIGKILL
- Self-stop pod (через provider-specific hook env, опционально)

**НЕ знает:** какой провайдер под ним, кто ещё запускает работы, что снаружи Mac.

### 4.4. Trainer subprocess (pod, существующий код)

- Делает обучение как сейчас
- Получает дополнительный callback `RyotenkaiEventCallback` который POST-ит в `127.0.0.1:8080/internal/events`
- Получает обработчик SIGTERM, который ставит `TrainerControl.should_training_stop = True`
- Файлы и markers — **больше не пишет** (всё через HTTP события)

## 5. Транспорт и IPC

| Связь | Протокол | Обоснование |
|---|---|---|
| Mac CLI ↔ Job Server | HTTP REST + WebSocket events через SSH `-L` | Стандартно, отлаженно, через туннель прозрачно |
| Trainer ↔ Job Server | loopback HTTP (127.0.0.1:8080/internal/*) | Не плодим транспорт, тот же FastAPI |
| MLflow tracking | как сейчас (HTTP к MLflow серверу) + проксирование через Job Server для офлайн-буфера | Сохраняем существующее поведение |
| Provider ↔ Pod (provisioning) | как у провайдера (RunPod GraphQL, SSH+docker для single_node) | Без изменений |

## 6. Унифицированный Docker образ

`ryotenkai/training-runtime:VERSION` (базируется на текущем):

Содержит:
- CUDA + PyTorch (как сейчас)
- Все Python зависимости для `src/training/*` (как сейчас)
- Job Server код (`src/runner/`) пред-инсталлирован
- `dumb-init` для корректного PID 1
- entrypoint.sh запускает sshd + uvicorn job-server
- exposes: 22 (sshd), 8080 (job-server, **только loopback**)

Поле конфига `image_name`/`docker_image` — **удаляется**. Образ зашит в:
- `src/runner/__about__.py` — `RUNTIME_IMAGE = "ryotenkai/training-runtime:vX.Y.Z"`
- `src/providers/*/training.py` — провайдеры читают это поле, не из конфига

Override **возможен** только через env `RYOTENKAI_RUNTIME_IMAGE_OVERRIDE` (для dev, не документируется).

## 7. Lifecycle и FSM

```
                              POST /jobs
                                  │
                                  ▼
                           [preparing]
                              │   │
                       prep ok│   │ prep failed
                              ▼   ▼
                          [running] ──► [failed]
                              │
                ┌─────────────┼──────────────┐
        natural │             │stop request   │ idle / crash
        finish  │             │               │
                ▼             ▼               ▼
         [completed]    [stopping] ──►   [failed]
                              │
                              ▼
                         [completed | cancelled]
```

Каждое состояние = эвент в EventBus, persistент в `state.jsonl` (для аудита). FSM в памяти job-server, восстанавливается из `state.jsonl` при рестарте контейнера.

## 8. Detach/Reattach (твой ключевой сценарий)

1. Mac запустил `ryotenkai run start config.yaml` → SSH-туннель + POST /jobs → trainer запущен.
2. Mac CLI получает `attempt_no` и сохраняет в `pipeline_state.json`.
3. Mac уходит в сон → SSH рвётся → job-server **продолжает** работу, складывает события в ring-buffer.
4. Mac просыпается → `ryotenkai run resume <attempt_no>` → CLI:
   - читает `pipeline_state.json` для pod endpoint
   - проверяет через провайдер что pod жив (RunPod GraphQL)
   - поднимает SSH-туннель заново
   - GET `/api/jobs/<id>` — текущее состояние FSM
   - WS `/api/jobs/<id>/events?since=<last_offset>` — догоняет события
   - продолжает мониторинг

Ring-buffer ограничен (10k последних событий, configurable). Если Mac спал слишком долго, теряется хвост детальных событий, но **последний state + последние N событий** всегда доступны. Все события дополнительно пишутся в `state.jsonl` на pod-е → при необходимости можно скачать полностью.

## 9. Multi-GPU и multi-node — заделы (НЕ реализуем сейчас)

Архитектура учитывает это **в интерфейсах** без сегодняшнего кода:

```python
class LauncherStrategy(Protocol):
    def build_command(self, config_path: str, world_size: int) -> list[str]: ...

class PythonLauncher(LauncherStrategy):     # сейчас — единственный
    def build_command(...) -> ["python", "-m", "src.training.run_training", ...]

class TorchrunLauncher(LauncherStrategy):   # multi-GPU, future
    ...

class AccelerateLauncher(LauncherStrategy): # будущее
    ...
```

Multi-node:
- Job Server заранее spec-фицирован под "rank-0 = координатор". В мульти-нодовом сценарии один pod = координатор (Mac общается только с ним), остальные = рабы.
- Не делаем сейчас, но FSM и API учитывают это (поле `topology` в `POST /jobs` request).

## 10. Поэтапная реализация

### Фаза 0. Подготовка инфраструктуры (½ дня)

- 0.1. Создать пакет `src/runner/` со скелетом FastAPI
- 0.2. Добавить тестовый стенд: `tests/runner/` + fixtures для mock trainer
- 0.3. Добавить `dumb-init` в Dockerfile.runtime
- 0.4. Скрипт `docker/training/build_and_push.sh` — добавить копирование `src/runner/` в образ

### Фаза 1. Job Server skeleton (1-2 дня)

- 1.1. `src/runner/main.py` — FastAPI app, lifespan
- 1.2. `src/runner/state.py` — JobLifecycleFSM
- 1.3. `src/runner/event_bus.py` — pub/sub + ring buffer
- 1.4. `src/runner/api/jobs.py` — REST endpoints (`POST /jobs`, `GET /jobs/{id}`, `POST /jobs/{id}/stop`)
- 1.5. `src/runner/api/events.py` — WebSocket `/jobs/{id}/events`
- 1.6. `src/runner/api/internal.py` — loopback `/internal/events`
- 1.7. Unit-тесты на FSM transitions, ring buffer, event subscription

### Фаза 2. Supervisor + signals (1 день)

- 2.1. `src/runner/supervisor.py` — subprocess.Popen, capture stdout/stderr → events, two-phase shutdown (SIGTERM → wait → SIGKILL)
- 2.2. SIGTERM handler в Python trainer (через faulthandler-friendly signal hook → `TrainerControl.should_training_stop`)
- 2.3. Ловить native crash: faulthandler уже есть, но supervisor дополнительно отслеживает exit code и signal через `proc.wait()`
- 2.4. Тесты: graceful stop, force-kill после timeout, crash recovery

### Фаза 3. TrainerCallback интеграция (½ дня)

- 3.1. `src/training/callbacks/runner_event_callback.py` — POST на `/internal/events` (epoch/step/loss/eval/save_checkpoint)
- 3.2. Регистрируется условно: если `RYOTENKAI_RUNNER_URL` env установлен (т.е. trainer запущен под supervisor)
- 3.3. Gracefully degrade если runner не отвечает (training не падает)
- 3.4. Тесты: callback events, retry, degradation

### Фаза 4. IdleDetector + HealthReporter (1 день)

- 4.1. `src/runner/idle_detector.py` — Python-замена `watchdog.sh`. Использует `nvidia-ml-py` (pynvml) вместо subprocess `nvidia-smi`. Те же пороги.
- 4.2. `src/runner/health_reporter.py` — периодический snapshot GPU/RAM/CPU, эмитит health events
- 4.3. Self-stop logic: на трансфер в FSM `[completed | cancelled | failed]` → опционально стопает pod через provider-specific hook
- 4.4. Тесты: idle detection, threshold edge cases

### Фаза 5. SSH tunnel + JobClient на Mac (1 день)

- 5.1. `src/orchestration/ssh_tunnel.py` — управляет `ssh -L 18080:127.0.0.1:8080 -N -f` (background tunnel) + cleanup at exit
- 5.2. `src/orchestration/job_client.py` — httpx + websockets клиент к `localhost:18080`
- 5.3. Reconnect logic при сетевом обрыве (exponential backoff)
- 5.4. Тесты: tunnel lifecycle, client reconnect, event ordering

### Фаза 6. Pipeline integration (1-2 дня)

- 6.1. `TrainingDeploymentManager` упрощается: убирается `_start_training_cloud` `_start_training_docker`, вместо них — POST в JobClient. Убирается `start_training.sh` генерация.
- 6.2. `TrainingMonitor` упрощается: вместо SSH-poll → JobClient.subscribe_events()
- 6.3. Удаление: `watchdog.sh`, `_touch_pipeline_heartbeat`, marker file probes, `runpod_stop_pod.sh` остаётся (вызывается из Python supervisor через provider hook)
- 6.4. Удаление поля `image_name`/`docker_image` из конфига + миграция existing configs
- 6.5. Интеграционные тесты на полный happy path (mock pod через локальный uvicorn)

### Фаза 7. CLI + Web UI (1 день)

- 7.1. CLI новые команды: `ryotenkai job status`, `job logs`, `job stop`, `job events --follow`
- 7.2. CLI extended: `run start`, `run resume <attempt>` использует новый стек
- 7.3. Web UI: страница "Live Training" с WS-подпиской через тот же тоннель (обращается на тот же `localhost:18080` что и CLI)
- 7.4. Тесты: CLI commands snapshot, WS reconnect

### Фаза 8. Migration + cleanup (½ дня)

- 8.1. Удалить `watchdog.sh`, `_upload_watchdog_resources`, marker-file logic
- 8.2. Удалить `_start_training_cloud`/`_start_training_docker`, `_create_env_file` упрощается
- 8.3. Конфиг migration script: автоматически удаляет `image_name`/`docker_image` поля при load (с warning)
- 8.4. Документация: `docs/runner-architecture.md`

**Итого: 7-9 рабочих дней.** Можно укладывать в спринт.

## 11. Что удаляется (cleanup checklist)

| Файл/символ | Причина |
|---|---|
| `src/providers/runpod/training/resources/watchdog.sh` | Логика → `IdleDetector` Python |
| Все `*_MARKER` константы | FSM в памяти |
| `TrainingMonitor._touch_pipeline_heartbeat` | Heartbeat не нужен, supervisor сам себе хозяин |
| `TrainingMonitor._check_marker`, `_read_marker_content` | Никаких markers |
| `TrainingDeploymentManager._start_training_cloud` (большая часть) | Заменяется JobClient.start() |
| `TrainingDeploymentManager._start_training_docker` | То же |
| `start_training.sh` f-string генератор | Не нужен |
| Marker probes в `_start_training_cloud` | Заменяется FSM polling через JobClient |
| `image_name` field из `RunPodTrainingConfig` | Прибито гвоздями |
| `docker_image` field из `SingleNodeTrainingConfig` | То же |

## 12. Тестирование

### 12.1. Юнит-тесты Job Server

- FSM transitions (все легальные + все нелегальные)
- EventBus pub/sub, ring buffer overflow
- Supervisor: SIGTERM, SIGKILL, normal exit, crash exit (различение кодов)
- IdleDetector: thresholds, edge cases (no GPU, multiple GPUs)
- API: каждый endpoint, WS connect/disconnect/reconnect

### 12.2. Юнит-тесты Mac Client

- SSHTunnel lifecycle (open, close, leak detection)
- JobClient reconnect
- Events ordering при reconnect

### 12.3. Интеграционные

- E2E: запустить локальный uvicorn job-server → послать POST /jobs → mock trainer (echo subprocess) → события доходят на client
- Detach/reattach: client отключается, события копятся, переподключение → fetch с offset
- Stop flow: client посылает stop → supervisor SIGTERM → mock trainer завершается → FSM в `cancelled`

### 12.4. Существующие тесты

- Не должно быть регрессий в ~3600 unit-тестах
- Существующие тесты `test_deployment_manager.py`, `test_training_monitor.py`, `test_lifecycle_manager.py` — переписать под новую архитектуру

## 13. Open Questions (требуют ответа перед implementation)

| # | Вопрос | Зачем нужно знать | Ответ |
|---|---|---|---|
| OQ-1 | Как обрабатывать reward-плагины из `community/reward/` — копировать на pod или зашивать в образ? | сейчас `REQUIRED_MODULES` не включает `community/`, есть gap | TBD после анализа риска R-7 |
| OQ-2 | Куда складывать `state.jsonl` на pod — `/workspace` или `/var/lib/ryotenkai`? | persistence через рестарт контейнера | TBD |
| OQ-3 | Какой ring buffer event size — 10k? 50k? | trade-off память vs detach window | TBD после моделирования: ~1KB/event × 10k = 10MB → ОК |
| OQ-4 | MLflow relay — пересылать каждое событие синхронно или batch? | latency vs нагрузка на MLflow | TBD; голос: sync для low-frequency events, batch для metrics |
| OQ-5 | Что делать с существующими in-flight runs при роллауте? | backwards compatibility | TBD: или дождаться завершения, или принудительно мигрировать |

## 14. Risks (заполнено в результате 3 итераций анализа — см. § 15-17)

См. § 15.

## 15. Анализ рисков — Итерация 1 (broad surface)

Прошёл по плану кругами, перечисляю что может пойти не так. Пока без mitigations — задача итерации 1 — **выявить**.

| ID | Риск (одной строкой) | Где проявится | Mitigation (черновой) |
|---|---|---|---|
| R-1 | SIGSEGV в bitsandbytes/flash-attn в trainer subprocess | Supervisor зависает или неправильно репортит | exit_code > 128 → signal name |
| R-2 | `community/reward/<id>/` не уезжает на pod (gap в REQUIRED_MODULES) | Reward plugin недоступен → ImportError в trainer | Включить в sync на основе config.training.params.reward_plugin |
| R-3 | SSH tunnel: localhost:18080 уже занят | Mac CLI не поднимет туннель | Динамический выбор свободного порта |
| R-4 | RunPod рестартует контейнер (laptop sleep + pod stop+start) | In-memory FSM теряется, события исчезли | state.jsonl persist + restore |
| R-5 | SIGTERM не доходит до Python trainer | graceful stop не срабатывает, force-kill | `exec` в start.sh, dumb-init |
| R-6 | MLflow упал → синхронный relay блокирует event flow | События застревают, FSM не обновляется | Async write + circuit breaker |
| R-7 | Docker image rebuild при каждом изменении `src/runner/` | Медленный dev cycle | Bind-mount в dev mode |
| R-8 | WebSocket idle-disconnect через SSH | Long-running training теряет stream | WS ping каждые N секунд |
| R-9 | Python version mismatch (conda vs system) | Trainer subprocess не находит deps | Жёстко `/opt/conda/bin/python` в supervisor |
| R-10 | `pynvml` отсутствует на каком-то базовом образе | IdleDetector не работает | Fallback на subprocess `nvidia-smi` |
| R-11 | uvicorn PID 1 не пробрасывает сигналы детям | Trainer не получает SIGTERM при stop pod | dumb-init |
| R-12 | localhost:18080 виден другим юзерам Mac | Безопасность | bind 127.0.0.1, не 0.0.0.0 |
| R-13 | Volume `/workspace` на pod restart теряется | state.jsonl исчезает | На RunPod /workspace = persistent volume — проверить |
| R-14 | Race между Mac `pipeline_state.json` и pod `state.jsonl` | Несогласованное состояние | Чёткое source-of-truth разделение |
| R-15 | In-flight runs при rollout новой архитектуры | Текущие experiments сломаются | Дождаться завершения текущих SAPO experiments |
| R-16 | Контейнер рестартанул mid-training — auto-resume? | Surprise behaviour | NO auto-resume, явный явный явный явный явный явный явный |
| R-17 | Supervisor падает → trainer subprocess orphan | Зомби, GPU занят | Process group + kill on supervisor exit |
| R-18 | API `/jobs/{id}` vs `/job/current` для single-active | API неконсистентен | Enforce single active job |
| R-19 | SSH ControlMaster конфликтует с CLI `-L` тоннелем | Race condition | Отдельный SSH master для tunnel |
| R-20 | Конфиг migration: поля `image_name`/`docker_image` исчезают | Старые YAML сломаются | Auto-strip с warning |

**20 рисков выявлено.** Переходим к итерации 2.

## 16. Анализ рисков — Итерация 2 (детализация и дипсинк)

Дипсинк по 5 наиболее опасным из 20:

### R-2 (community plugins gap) — углубление

Проверил код: [deployment_manager.py:89](src/pipeline/stages/managers/deployment_manager.py:89) `REQUIRED_MODULES` действительно НЕ содержит `community/`. При этом [reward_plugins/factory.py:48](src/training/reward_plugins/factory.py:48) вызывает `catalog.ensure_loaded()` — это сканирует `community/` на pod-е. **Если папки нет → registry пуст → ImportError при `RewardPluginRegistry.create()`.**

Возможные объяснения почему сейчас работает:
- (а) docker image содержит `community/` pre-baked (но Dockerfile.runtime НЕ копирует его — только base deps)
- (б) есть отдельный шаг sync community который я пропустил
- (в) **это латентный баг, который проявится при первом реальном запуске reward plugin'а**

**Mitigation для нового дизайна:** добавить в Job Server logic "при `POST /jobs` принимать `plugins_payload: list[ZipBytes]` и распаковывать в `community/`". Mac упаковывает только нужные плагины (по `config.training.params.reward_plugin` и т.д.) и шлёт их в payload. Это **попутно решает** OQ-1.

### R-4/R-13 (state persistence) — углубление

На RunPod [`api_client.py:71`](src/providers/runpod/training/api_client.py:71) `volume_mount_path = "/workspace"` — это **persistent volume**. Контейнер можно убить и пересоздать, `/workspace` сохраняется. Значит `state.jsonl` в `/workspace/.ryotenkai/state.jsonl` переживает рестарт контейнера на том же поде.

**Caveat:** если pod **уничтожен** (terminate) и создан новый — это новый volume. Но это и так "новый run" с точки зрения lineage Mac.

**Mitigation:** `state.jsonl` в `/workspace/.ryotenkai/`, на старте Job Server делает `restore_or_init()`.

### R-5/R-11 (signals + PID 1) — углубление

[deployment_manager.py:929](src/pipeline/stages/managers/deployment_manager.py:929) сейчас стартует через `nohup ... & disown` — это **отдельный процесс, не дочерний** SSH-сессии. SIGTERM от Mac до trainer-процесса дойти не может.

В новом дизайне: Job Server запускает trainer как **child через subprocess.Popen с `start_new_session=False`**, чтобы они были в одной process group. SIGTERM на process group через `os.killpg(pgid, SIGTERM)` доходит до всех, включая dataloader workers.

Чтобы PID 1 (uvicorn) **передавал** SIGTERM наружу-внутрь, нужен `dumb-init` или `tini` как ENTRYPOINT — это де-факто стандарт для Docker workloads.

### R-7 (slow dev cycle) — углубление

Каждый push в `src/runner/` без bind-mount требует rebuild image (~5-10 мин) + push в Docker Hub (~5-15 мин) + RunPod pull (~2-5 мин). Это **20-30 мин на каждое изменение**.

**Mitigation:** в dev-режиме `JobServer` стартует через `pip install -e .` от bind-mounted кода. Production режим — bake into image. Реализуется через `RYOTENKAI_DEV_MOUNT=/workspace/dev_code` env: если задано, supervisor перед стартом trainer-а кладёт PYTHONPATH из этого пути.

### R-19 (SSH ControlMaster конфликт) — углубление

Текущий [`ssh_client.py:139`](src/utils/ssh_client.py:139) использует ControlMaster=auto в `~/.ssh/control_sockets/<host>:<port>`. Если CLI tunnel `ssh -L 18080:127.0.0.1:8080` идёт через **тот же** master, при разрыве/restart-е master все его dependent connections рвутся.

**Mitigation:** CLI tunnel использует **отдельный** ControlMaster socket (`~/.ssh/control_sockets/runner_<host>:<port>`) — изолирован от провайдер-операций (rsync, exec).

### R-1 (SIGSEGV detection) — углубление

Сейчас [run_training.py:579](src/training/run_training.py:579) `_install_crash_observability` пишет faulthandler в файл. В новой схеме supervisor читает stdout/stderr trainer'а через `proc.stdout` pipe → пушит в EventBus как обычные log events. Native crash будет виден через:
- `proc.returncode > 128` → signal name через `signal.Signals(rc - 128).name`
- Tail из faulthandler.log (читается supervisor-ом и эмитится как event)
- Tail последних N stdout строк из in-memory кольцевого буфера supervisor-а

Это **строго лучше** текущей схемы, где TRAINING_FAILED собирается bash'ем.

---

### Дополнительные риски, выявленные на итерации 2

| ID | Риск | Mitigation |
|---|---|---|
| R-21 | Datasets рассылаются по rsync, могут быть большими (>1GB) | Job Server принимает datasets через `POST /datasets/upload` (multipart streaming) — отдельно от config |
| R-22 | TrainerCallback POST на каждом step может перегрузить FastAPI | Локальный buffer + flush every N steps; events.jsonl на pod как durable log |
| R-23 | Stop request приходит между save_steps → checkpoint неполный | `should_save = True` перед `should_training_stop = True` через TrainerCallback |
| R-24 | Multi-process dataloader workers держат GPU | На stop посылать SIGTERM группе процессов, а не одному |
| R-25 | Долгий cold start (model loading 5+ мин) — Mac не знает что pod нормально работает | Heartbeat events: trainer пушит `model_loading_progress` events |
| R-26 | RunPod podStop может не срабатывать с первого раза (из watchdog видно — 3 retries) | Move retries в Python supervisor с exponential backoff |
| R-27 | Job Server crash во время preparing → pod в "висячем" состоянии | На uvicorn startup делаем `restore_or_cleanup()`: если state=preparing/stopping, переводим в failed |
| R-28 | Web UI открыт в нескольких вкладках → несколько WS подписок | Каждая вкладка = независимая подписка, EventBus broadcast — это норма |
| R-29 | Логи trainer'а очень большие (десятки МБ) → ring buffer переполнится | Ring buffer хранит **events**, не raw stdout. Raw stdout пишется в `training.log` как сейчас, доступен через `GET /jobs/{id}/log?tail=N&follow=true` (chunked transfer) |
| R-30 | Job Server обновился (новая версия image), а runs/in-flight ожидают старую API | Versioned API: `/api/v1/jobs/...`, в response `server_version` field |

**30 рисков (после итерации 2).** Идём в итерацию 3 — приоритизация и закрытие.

## 17. Анализ рисков — Итерация 3 (финальный список)

Шкалы: **S** (severity 1-5) — насколько плохо если случится; **L** (likelihood 1-5) — насколько вероятно. Score = S × L. Top-критичные: score ≥ 12.

| ID | Риск | S | L | Score | Фаза | Mitigation (compact) | Open Q? |
|---|---|---:|---:|---:|---|---|---|
| R-2 | Community plugins gap (reward не уезжает) | 5 | 5 | **25** | Ф1, Ф6 | `POST /jobs` принимает plugins_payload (zip); Mac пакует только нужные по config | OQ-1 → закрыт |
| R-5 | SIGTERM не доходит до Python trainer | 5 | 4 | **20** | Ф2 | `exec` в start.sh; subprocess в той же process group; SIGTERM на pgid; dumb-init | — |
| R-1 | SIGSEGV в bitsandbytes/flash-attn | 5 | 4 | **20** | Ф2 | Supervisor различает rc>128, читает faulthandler.log, эмитит event | — |
| R-11 | uvicorn PID 1 signals | 4 | 4 | **16** | Ф0, Ф2 | `dumb-init` как ENTRYPOINT | — |
| R-15 | In-flight runs при rollout | 5 | 3 | **15** | Ф8 | Дождаться завершения текущих SAPO experiments; feature-flag для канарейки | OQ-5 → закрыт |
| R-22 | TrainerCallback POST overhead | 4 | 4 | **16** | Ф3 | Local buffer + flush каждые N шагов; events.jsonl durable | — |
| R-6 | MLflow синхронный relay блокирует | 4 | 3 | **12** | Ф3 | Async write task; circuit breaker (уже есть `MLflowTransportCircuitBreaker`) | — |
| R-23 | Stop между save_steps → неполный checkpoint | 4 | 3 | **12** | Ф2, Ф3 | `should_save=True` ДО `should_training_stop=True` в TrainerCallback | — |
| R-19 | ControlMaster конфликт CLI ↔ provider ops | 3 | 4 | 12 | Ф5 | Отдельный socket для tunnel | — |
| R-7 | Slow dev cycle (rebuild image) | 3 | 5 | 15 | Ф0 | Bind-mount в dev mode через env var | — |
| R-17 | Supervisor crash → orphan trainer | 4 | 2 | 8 | Ф2 | Process group + SIGTERM-on-exit hook | — |
| R-25 | Cold start без events → Mac думает pod завис | 3 | 4 | 12 | Ф3 | Trainer пушит `model_loading_progress` events | — |
| R-26 | RunPod podStop ненадёжен | 3 | 3 | 9 | Ф4 | Retry с exp backoff; safety-net cron self-stop | — |
| R-27 | Job Server crash во время `preparing` | 4 | 2 | 8 | Ф1 | restore_or_cleanup() на startup → стук в failed | — |
| R-4 | Контейнер restart → in-memory FSM теряется | 4 | 2 | 8 | Ф1 | state.jsonl в /workspace/.ryotenkai (persistent volume) | OQ-2 → закрыт |
| R-13 | /workspace volume не persistent | 5 | 1 | 5 | Ф1 | На RunPod это volume_mount_path, persistent. Документировать. | — |
| R-21 | Большие datasets через POST | 3 | 4 | 12 | Ф6 | Multipart streaming; rsync остаётся для bulk transfer (заодно скип если уже есть) | — |
| R-8 | WebSocket idle disconnect | 3 | 3 | 9 | Ф5 | WS ping каждые 30с; client auto-reconnect с offset | — |
| R-3 | localhost:18080 занят | 2 | 4 | 8 | Ф5 | Авто-выбор порта из range 18080-18099 | — |
| R-12 | localhost:18080 виден другим Mac users | 2 | 3 | 6 | Ф5 | bind 127.0.0.1 явно | — |
| R-9 | Python version mismatch | 4 | 1 | 4 | Ф2 | Жёстко `/opt/conda/bin/python` через PY_BIN env | — |
| R-10 | pynvml отсутствует | 3 | 2 | 6 | Ф4 | Fallback на subprocess `nvidia-smi` | — |
| R-14 | Race Mac state.json ↔ pod state.jsonl | 4 | 2 | 8 | Ф6 | Pod = source of truth для in-flight; Mac = SoT для done | — |
| R-16 | Auto-resume mid-training сюрприз | 3 | 2 | 6 | Ф1 | NO auto-resume; явный `POST /jobs/{id}/resume` | — |
| R-18 | Single-active job vs API design | 2 | 4 | 8 | Ф1 | Enforce single active в FSM; API всё равно принимает {id} | — |
| R-20 | Старые YAML с image_name/docker_image | 2 | 4 | 8 | Ф6, Ф8 | Pydantic alias + warning + auto-strip | — |
| R-24 | Dataloader workers держат GPU при stop | 4 | 2 | 8 | Ф2 | SIGTERM на pgid, не один pid | — |
| R-28 | Multiple WS subscribers | 1 | 5 | 5 | Ф5 | EventBus broadcast — by design | — |
| R-29 | Огромные training.log в ring buffer | 4 | 3 | 12 | Ф1 | Ring buffer хранит events, не raw log; log streamится chunked | — |
| R-30 | API версия конфликта при upgrade | 3 | 2 | 6 | Ф6 | Versioned `/api/v1/...` + `server_version` в response | — |

### 17.1. Топ-7 критичных рисков (score ≥ 15) — обязательно закрываем до Phase 1

1. **R-2** (community plugins gap, score 25) — **БЛОКЕР**. Без решения этого риска reward-плагины не работают на pod-е.
2. **R-5** (SIGTERM, score 20) — без этого нельзя сделать graceful stop.
3. **R-1** (SIGSEGV detection, score 20) — критично для надёжности репортинга падений.
4. **R-11** (PID 1 signals, score 16) — следствие R-5; решается dumb-init.
5. **R-22** (TrainerCallback overhead, score 16) — может убить performance, нужен buffer ДО первого реального запуска.
6. **R-15** (in-flight runs rollout, score 15) — нужна координация с research SAPO/GRPO experiments.
7. **R-7** (slow dev cycle, score 15) — без bind-mount разработка займёт в 3-5 раз больше времени.

### 17.2. Что закрывает Open Questions

| OQ | Резолюция |
|---|---|
| OQ-1 (community plugins) | **Резолюция**: `POST /jobs` принимает `plugins_payload` (zip с manifest+code), Job Server распаковывает в `community/<kind>/<id>/`. Mac упаковывает только нужные плагины по конфигу. |
| OQ-2 (state.jsonl path) | **Резолюция**: `/workspace/.ryotenkai/state.jsonl` — на persistent volume, переживёт container restart. |
| OQ-3 (ring buffer size) | **Резолюция**: 10k events × 1KB ≈ 10MB RAM — приемлемо. Configurable через env `RYOTENKAI_EVENT_BUFFER_SIZE`. |
| OQ-4 (MLflow relay sync vs async) | **Резолюция**: Async через background asyncio task + circuit breaker. Sync для критичных событий (run start/end). |
| OQ-5 (in-flight runs migration) | **Резолюция**: feature flag `RYOTENKAI_RUNNER_V2=true`. Дефолт false до конца текущих SAPO experiments. После того как пользователь подтвердит — flip default. |

---

## 18. Готовность плана к одобрению

- ✅ Архитектурный обзор зафиксирован (§ 3-4)
- ✅ Транспорт и IPC выбраны (§ 5)
- ✅ Docker и образ описаны (§ 6)
- ✅ FSM и lifecycle нарисованы (§ 7-8)
- ✅ Multi-GPU/multi-node заделы (§ 9)
- ✅ Поэтапная реализация в 9 фаз (§ 10), 7-9 рабочих дней
- ✅ Cleanup checklist (§ 11)
- ✅ Тестирование описано (§ 12)
- ✅ Open Questions закрыты (§ 17.2)
- ✅ 30 рисков идентифицированы и приоритизированы (§ 17), top-7 критичных требуют отдельного внимания
- ⏳ **Ожидается одобрение пользователем перед стартом Phase 0**

---

**STATUS: WAITING FOR USER APPROVAL**

После одобрения:
1. Подтвердить координацию с активными SAPO/GRPO experiments (R-15)
2. Стартовать Phase 0 (полдня)
3. Параллельно держать публикацию обновлённого Docker image готовой
