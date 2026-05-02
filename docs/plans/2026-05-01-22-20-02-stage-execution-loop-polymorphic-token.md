# План: Pull-only ground-truth для логов trainer'а

> Status: **IMPLEMENTED** (2026-05-02) — все critical шаги выполнены, 205 unit tests зелёные.
> Deferred (cosmetic): code_syncer/file_uploader/LogManager/gpu_deployer cleanup, e2e regression tests.
> Author: daniil + claude
> Date: 2026-05-01
> Worktree: `dazzling-rosalind-b482fa` (RESEACRH branch)
> Trigger: реальный run, упавший в 2026-05-01 22:20:18 без всякой диагностики:
> ```
> Stage 3/6: Training Monitor
> [MONITOR] Trainer process started (pid=286)
> [MONITOR:POSTMORTEM] non-zero exit detected — collecting pod-side diagnostics
> [MONITOR:POSTMORTEM] 0/6 probes returned data
> Pipeline failed: [TRAINING_FAILED] trainer exited non-zero (exit_code=1, signal=None)
> [DEPLOYER] No training log found on remote
> ```

---

## 1. Context — зачем это нужно

После сегодняшних коммитов (subprocess spawn worker, cooperative cancellation, PodSshWaiter unification, Mac-orchestrated runner launch, thin training image) у нас образовалась **полная слепота на post-mortem trainer'а**. Конкретные симптомы из реального лога 22:20:18:

1. Trainer стартовал PID=286 → через 10 секунд exit_code=1 (это **ImportError-class** crash или near-`main()` failure).
2. SSH-probes от monitor'а нашли **0 байт** в `/workspace/training.log` и не различили это от "файла не существует".
3. log_manager.scp пытался забрать `training.log` по двум путям — оба missing.
4. На Mac в `pipeline.log` нет ни одной строки stderr trainer'а.
5. Operator получает `[TRAINING_FAILED] trainer exited non-zero (exit_code=1, signal=None)` — единственный сигнал — и вынужден ехать в RunPod console руками.

**Root cause** (deep-think + verification):

Trainer пытается **сам про себя** писать логи через `logging.FileHandler`, который привязывается в `_install_training_file_handler()` ([src/training/run_training.py:578-622](src/training/run_training.py:578)) внутри `main()`. Но `main()` вызывается **после** module-load imports на строках [33-55](src/training/run_training.py:33) — включая `from src.workspace.integrations.loader import load_pipeline_config`. Если ImportError ловится здесь — Python кидает SystemExit **до** того, как FileHandler привязался → файл `/workspace/training.log` никогда не создаётся.

Это **timing race**: trainer не может сохранить запись о собственной смерти, потому что умирает раньше, чем доходит до записи. Архитектурно — нарушение SRP: trainer не должен отвечать за свою observability survival.

Дополнительно — мы платим стоимость WS-streaming (Phase 12.B EventBus + EventJournal + 1500 LOC) для трансляции `trainer_log` events, которые на Mac **silenced** (`return None` на [training_monitor.py:923](src/pipeline/stages/training_monitor.py:923)). Cargo cult с момента revert'а EventMirrorWriter в [d5f1cbd](src/pipeline/stages/training_monitor.py).

---

## 2. Зафиксированные решения (согласованы с пользователем)

| Решение | Выбор |
|---|---|
| Backbone для trainer **logs (stdout/stderr)** | **Pull-only ground-truth** (Supervisor пишет файл на pod, LogManager delta-pull на Mac) |
| Pod-side путь | `<workspace>/logs/trainer.stdio.log` (per-run, см. §3.3 layout) |
| training.log судьба | **Удаляем** (вместе с `_install_training_file_handler`) |
| training.faulthandler.log судьба | **Удаляем тоже**. faulthandler перенастраиваем на stderr (default Python поведение) → SEGV/ABRT/CUDA crash traceback идёт в stderr trainer'а → Supervisor pump → `trainer.stdio.log`. Один файл закрывает все trainer-output concerns |
| **Итого файлов-логов на pod** | **Только два** (per-run): `<workspace>/logs/runner.log` (uvicorn) + `<workspace>/logs/trainer.stdio.log` (trainer subprocess output, включая native crashes) |
| **PodLayout abstraction** | Новый frozen dataclass [src/runner/pod_layout.py](src/runner/pod_layout.py) — single source of truth для всех pod-side путей. Provider-agnostic. Заменяет 5+ ad-hoc f-string concat'ов |
| **Per-run isolation** | Все артефакты под `<workspace>/` где `workspace = <provider-root>/runs/<run_id>`. Sequential runs на одном pod'е не collidate'ят |
| EventJournal pull на Mac | **Не делаем** (журнал остаётся pod-side как durability layer для MLflow flap'ов, не для post-mortem видимости) |
| Phasing | Один cohesive PR |
| Stdio layout | Один interleaved файл с per-line `[OUT]`/`[ERR]` префиксом (KISS; раздельные файлы — future option) |
| **Telemetry / MLflow events** | **БЕЗ ИЗМЕНЕНИЙ.** MLflowRelay, MLFLOW_EVENT_KINDS, EventJournal durability для metrics — всё работает как раньше. Этот PR трогает **только** log content в EventBus |
| **WS / EventBus / EventJournal** | Продолжают носить **control + telemetry events**. Убираем оттуда **только** `trainer_log` events |

---

## 3. Архитектура

### 3.0. Pod-side filesystem layout (NEW — единая структура для всех провайдеров)

**Текущее состояние (каша)** — на pod'е сейчас смешаны per-run и global пути:

| Артефакт | Где сейчас | Проблема |
|---|---|---|
| `src/` (rsync) | `<workspace>/src/` | ✅ per-run |
| `pipeline_config.yaml` | `<workspace>/config/` | ✅ per-run |
| dataset | `<workspace>/data/` | ✅ per-run |
| community plugins | `<workspace>/community/` | ✅ per-run |
| EventJournal | `<workspace>/.runner/events/` | ✅ per-run |
| **`runner.log`** | **`/workspace/runner.log`** (хардкод!) | ❌ GLOBAL — collides on resume |
| **`training.log`** | **`/workspace/training.log`** (хардкод!) | ❌ GLOBAL — collides on resume |
| training.faulthandler.log | `<workspace>/training.faulthandler.log` | ⚠ per-run, но удаляем (см. §2) |

**`workspace = <provider-root>/runs/<run_id>`** где:
- RunPod: `/workspace/runs/<run_id>` (Docker volume mount)
- single_node: `<user-config-base>/runs/<run_id>` (e.g. `~/ryotenkai_training/runs/<run_id>`)

**Целевая структура (унифицированная, per-run, без dot-prefix'ов)**:

```
<provider-root>/                          # /workspace для RunPod, <user-base> для single_node
└── runs/                                 # все run'ы под одним namespace
    └── <run_id>/                         # workspace_path — per-run isolated
        ├── src/                          # CodeSyncer rsync target
        ├── config/
        │   └── pipeline_config.yaml      # FileUploader
        ├── data/
        │   └── <dataset>.tar.gz          # FileUploader
        ├── community/                    # PluginUnpacker
        │   ├── reward/<id>/
        │   └── libs/<id>/
        ├── output/                       # checkpoints, model artifacts
        ├── logs/                         # ◄── ВСЕ ЛОГИ ЗДЕСЬ, per-run
        │   ├── runner.log                # ◄── переезжает с /workspace/runner.log
        │   └── trainer.stdio.log         # ◄── NEW (заменяет /workspace/training.log)
        ├── events/                       # ◄── EventJournal (было .runner/events/)
        │   └── events.NNN.jsonl
        └── state/                        # ◄── FSM state (было .ryotenkai/)
            ├── job.json
            └── job.jsonl
```

**Никаких dot-prefix'ов**: все артефакты видны оператору; per-run isolation уже даёт нужную "капсуляцию".

**Mac-side parallel structure** (LogLayout уже описывает):

```
~/.ryotenkai/runs/<run_id>/attempts/<n>/
└── logs/
    ├── pipeline.log                  # Mac orchestrator log
    ├── <stage_name>.log              # per-stage logs
    ├── runner.log                    # ← scp from <pod-workspace>/logs/runner.log
    └── trainer.stdio.log             # ← scp from <pod-workspace>/logs/trainer.stdio.log
```

**Симметрия pod ↔ Mac** — log shipper pattern:
- **Pod**: pisатель → `<workspace>/logs/<file>`
- **Mac**: receiver → `<attempt>/logs/<file>`
- LogManager делает scp с делтой между ними. Простая 1:1 mapping.

### 3.0.1. PodLayout abstraction (новый класс) — где живёт + ownership

**Проблема (DRY violation)**: пути конструируются ad-hoc через `f"{workspace}/foo/bar.log"` в 5+ модулях ([code_syncer.py:149](src/pipeline/stages/managers/deployment/code_syncer.py:149), [file_uploader.py:367](src/pipeline/stages/managers/deployment/file_uploader.py:367), [training_launcher.py:514](src/pipeline/stages/managers/deployment/training_launcher.py:514), [runner_launcher.py:55](src/pipeline/stages/managers/deployment/runner_launcher.py:55), [event_journal.py:109](src/runner/event_journal.py:109), и др.). Нет single source of truth.

**SOLID анализ — где должен жить класс**:

PodLayout используют:
- Mac-side: `code_syncer`, `file_uploader`, `runner_launcher`, `gpu_deployer`
- Pod-side: `supervisor`, `event_journal`, `runner/main.py`

**`src/runner/pod_layout.py` — REJECTED**: pipeline-side (Mac) код будет импортировать из `src/runner/` (pod-side). Нарушает dependency direction.

**`src/providers/pod_layout.py` — REJECTED**: layout одинаковый для всех провайдеров — это не provider concern (provider знает только свой `root`).

**`src/utils/pod_layout.py` — ✅ ВЫБРАН**: shared layer, рядом с парным `src/utils/logs_layout.py` (Mac-side LogLayout). Оба — про "где файлы". Симметричная пара.

**Решение**: новый frozen dataclass `src/utils/pod_layout.py`:

```python
from dataclasses import dataclass
from pathlib import PurePosixPath  # POSIX explicit — pod is always Linux

@dataclass(frozen=True)
class PodLayout:
    """Pod-side filesystem layout — single source of truth.
    
    Provider-agnostic: providers own only the `root` value
    (root = <provider-base>/runs/<run_id>); the directory structure
    under it is identical regardless of provider.
    
    Parallel structure to `src/utils/logs_layout.LogLayout` (Mac side).
    """
    
    root: PurePosixPath  # provider-supplied; = <provider-base>/runs/<run_id>
    
    # --- subdirectories (no dot-prefix — per-run isolation already encapsulates) ---
    @property
    def src_dir(self) -> PurePosixPath:       return self.root / "src"
    @property
    def config_dir(self) -> PurePosixPath:    return self.root / "config"
    @property
    def data_dir(self) -> PurePosixPath:      return self.root / "data"
    @property
    def community_dir(self) -> PurePosixPath: return self.root / "community"
    @property
    def output_dir(self) -> PurePosixPath:    return self.root / "output"
    @property
    def logs_dir(self) -> PurePosixPath:      return self.root / "logs"
    @property
    def events_dir(self) -> PurePosixPath:    return self.root / "events"
    @property
    def state_dir(self) -> PurePosixPath:     return self.root / "state"
    
    # --- specific files ---
    @property
    def config_file(self) -> PurePosixPath:        return self.config_dir / "pipeline_config.yaml"
    @property
    def runner_log(self) -> PurePosixPath:         return self.logs_dir / "runner.log"
    @property
    def trainer_stdio_log(self) -> PurePosixPath:  return self.logs_dir / "trainer.stdio.log"
    @property
    def fsm_state_json(self) -> PurePosixPath:     return self.state_dir / "job.json"
    @property
    def fsm_state_jsonl(self) -> PurePosixPath:    return self.state_dir / "job.jsonl"
    
    # --- factory ---
    @classmethod
    def from_root(cls, root: PurePosixPath) -> "PodLayout":
        return cls(root=root)
    
    def ensure_dirs_command(self) -> str:
        """Single idempotent shell command to create the layout on pod."""
        dirs = [self.src_dir, self.config_dir, self.data_dir, 
                self.community_dir, self.output_dir, self.logs_dir, 
                self.events_dir, self.state_dir]
        return f"mkdir -p {' '.join(str(d) for d in dirs)}"
```

### 3.0.2. Provider ownership — Dependency Inversion правильно

**Проблема (Dependency Inversion violation)**: сейчас `SSHConnectionInfo.workspace_path` — это **строка**, и каждый компонент сам f-string'ит пути из неё. Высокоуровневый код (gpu_deployer) зависит от строкового представления, не от abstraction.

**Решение**: `TrainingProvider` Protocol получает новый method `pod_layout_for_run(run_id) -> PodLayout`. Каждый provider возвращает PodLayout с своим root'ом, **structure под root'ом одинаковая** (PodLayout это гарантирует).

```python
# src/providers/training/interfaces.py — extension Protocol
class TrainingProvider(Protocol):
    # existing methods...
    
    def pod_layout_for_run(self, run_id: str) -> PodLayout:
        """Provider-specific root + universal structure."""
        ...

# src/providers/runpod/training/provider.py
def pod_layout_for_run(self, run_id: str) -> PodLayout:
    return PodLayout.from_root(PurePosixPath(f"/workspace/runs/{run_id}"))

# src/providers/single_node/training/provider.py
def pod_layout_for_run(self, run_id: str) -> PodLayout:
    return PodLayout.from_root(
        PurePosixPath(f"{self.config.workspace_path}/runs/{run_id}")
    )
```

**Что изменилось**: каждый компонент (gpu_deployer, supervisor, runner_launcher) принимает `PodLayout` через DI **от provider'а**, не строит сам. Это closes Dependency Inversion — высокоуровневые модули зависят от abstraction (`PodLayout` Protocol), не от concrete strings.

### 3.0.3. Зоны ответственности (final)

| Component | Owns | Что нет |
|---|---|---|
| **TrainingProvider** (`src/providers/...`) | Provider-specific `root` (где workspace), SSH lifecycle, pod create/destroy | Не owns sub-structure |
| **PodLayout** (`src/utils/pod_layout.py`) | Универсальная структура под root | Не owns root |
| **LogLayout** (`src/utils/logs_layout.py`) | Универсальная структура на Mac (existing) | — |
| **Каждый компонент** (Supervisor, gpu_deployer, ...) | Свою бизнес-логику | Не строит paths сам — берёт из layout через DI |

**SOLID checklist финального решения**:

- **S (Single Responsibility)** ✅ PodLayout = только структура. Provider = только root + lifecycle. Supervisor = только subprocess.
- **O (Open/Closed)** ✅ Новый artifact = новая property в PodLayout. Новый provider = новая реализация `pod_layout_for_run`. Existing code не трогается.
- **L (Liskov)** ✅ Любой TrainingProvider swap'ится — все возвращают PodLayout с своим root'ом, structure идентичная.
- **I (Interface Segregation)** ✅ Provider не имеет лишних методов. PodLayout — небольшой cohesive set properties.
- **D (Dependency Inversion)** ✅ Высокоуровневые модули зависят от Layout / Provider Protocol, не от строк или конкретных провайдеров.

### 3.1. As-is (текущее, сломанное)

```
trainer subprocess
  ├─ stdout/stderr ───PIPE───► Supervisor._pump_stream
  │                              ├─► bus.publish('trainer_log')
  │                              │     ├─► EventJournal (.runner/events/) ◄── никогда не читаем
  │                              │     └─► WS ──► Mac TrainingMonitor ──► silenced (line 923)
  │                              └─► (NOT WRITTEN TO DISK ANYWHERE)
  │
  └─ ryotenkai logger ──► FileHandler (RACE: attaches AFTER imports)
                            └─► /workspace/training.log
                                  └─► log_manager.scp ──► Mac
                                  ⚠ Если crash до attach — файла нет
```

Failure modes:
- Trainer ImportError → no training.log → 0/6 probes → operator слепой
- WS disconnect mid-run → trainer_log events потеряны на Mac (на pod в journal — но мы не пуллим)
- Pod evicted before log_manager scp → даже training.log потерян

### 3.2. To-be (целевое)

Три плоскости (planes) с чёткими транспортами:

```
                              POD                                                    MAC
                                                                                
trainer subprocess                                                              
  ├─ stdout/stderr ──PIPE──► Supervisor._pump_stream                            
  │                            └─► /workspace/trainer.stdio.log     ─SCP delta─► <attempt>/logs/trainer.stdio.log
  │                                ◄── DATA PLANE (logs):                          
  │                                    file is ground-truth,                       
  │                                    LogManager pulls periodic + final           
  │                                                                            
  ├─ ryotenkai logger ──► StreamHandler ──► stdout (captured above)            
  │                       (NO FileHandler — REMOVED)                           
  │                                                                            
  ├─ HF TrainerCallback (RunnerEventCallback)                                  
  │   └─► POST loopback /api/v1/internal/events ──► bus.publish('mlflow_*')   
  │                                                  │                          
  │                                                  ├─► MLflowRelay queue ───► MLflow tracking server (HTTP)
  │                                                  │   ◄── TELEMETRY PLANE:    │
  │                                                  │       buffered async      │
  │                                                  │       circuit-breaker     │
  │                                                  │                           ▼
  │                                                  │                       Mac MLflow client
  │                                                  │                       (graphs, dashboards)
  │                                                  │                                      
  │                                                  └─► EventJournal (durability for flap'ов)
  │                                                  └─► WS ──► Mac UI (live metric charts)
  │                                                                            
  └─ FSM transitions (Supervisor) ──► bus.publish(                              
       'trainer_spawned' | 'trainer_exited' | 'cancellation_*' | 'spawn_failed')
       'health_snapshot' (HealthReporter)                                        
       'idle_detector_triggered' (IdleDetector)                                  
       'pod_terminate_*' (PodTerminator)                                         
                          ◄── CONTROL PLANE: lifecycle, cancellation, FSM sync   
                          ├─► EventJournal (durability)                          
                          └─► WS ──► Mac TrainingMonitor ──► FSM sync, cleanup chain
                                                                                
uvicorn boot                                                                     
  └─ stdout/stderr ──nohup tee──► /workspace/runner.log              ─SCP delta─► <attempt>/logs/runner.log
                                  ◄── DATA PLANE (uvicorn logs):                  
                                      captures pre-import errors                  
                                                                                  
trainer native crash (SEGV/ABRT/BUS)                                              
  └─ Python faulthandler ──► sys.stderr (default)                                   
                              └─► (already captured by Supervisor pump)             
                              └─► /workspace/trainer.stdio.log                      
                              ◄── ОДИН файл ловит и обычный output, и native traceback
```

Три **plane** — это ключевое разделение concerns:

| Plane | Что | Транспорт | Ground-truth | Frequency |
|---|---|---|---|---|
| **Control** | FSM lifecycle, cancellation | EventBus → EventJournal + WS | EventJournal (durable) | ~tens per run |
| **Telemetry** | mlflow_metric/param/tag/run_*, health_snapshot | EventBus → MLflowRelay (queue+CB+retry) → MLflow server; параллельно WS → Mac UI | MLflow tracking server (external) | ~thousands per run |
| **Data (logs)** | trainer stdout/stderr, uvicorn output | **File on pod** → LogManager SCP delta-pull → Mac local file | `<workspace>/{trainer.stdio,runner}.log` | continuous, MB-level |

Зоны ответственности после fix'а:

| Component | Owns | Plane |
|---|---|---|
| **Trainer (subprocess)** | Обучение + публикация HF metrics через `RunnerEventCallback` (telemetry) | Telemetry source |
| **Supervisor** (runner-side) | Subprocess lifecycle (control) + **single-writer** для `trainer.stdio.log` (data) | Control + Data |
| **HealthReporter / IdleDetector / PodTerminator** | Health/idle/lifecycle events (control + telemetry health) | Control + Telemetry |
| **EventBus** (in-pod, RAM ring) | Routing для control + telemetry events. **Без `trainer_log`** | Control + Telemetry |
| **EventJournal** (.runner/events/) | Durability для control + telemetry. Pulled никогда (live replay для late WS subscribers) | Control + Telemetry |
| **MLflowRelay** | Async forward telemetry events в MLflow server, circuit-breaker | Telemetry |
| **LogManager** (Mac) | Delta-pull логов-файлов: `runner.log`, `trainer.stdio.log` (NEW). Plus Mac-локальные pipeline.log/stage logs (как сейчас) | Data |
| **TrainingMonitor** (Mac) | WS subscriber для control + telemetry events; post-mortem orchestration. **Не трогает log content** | Control + Telemetry |
| **gpu_deployer._download_remote_logs** | Final pull data-plane artifacts (logs) перед `provider.disconnect()` | Data |

Покрытие failure modes (после fix'а):

| Scenario | runner.log | trainer.stdio.log |
|---|:---:|:---:|
| uvicorn boot crash | ✅ | ❌ |
| Missing `RYOTENKAI_RUNTIME_PROVIDER` | ✅ | ❌ |
| Trainer ImportError на module-load | ❌ | ✅ |
| Trainer падает на `load_pipeline_config()` | ❌ | ✅ |
| OOM-killer SIGKILL trainer'а на step 5 | ❌ | ✅ |
| **Native SEGV** (CUDA/bitsandbytes/flash-attn) | ❌ | ✅ (faulthandler → stderr → pump) |
| Trainer штатно отрабатывает | ❌ | ✅ |

**Два файла, два разных concern'а** (uvicorn vs trainer subprocess), без дублирования и без "третьего лога для SEGV".

---

## 4. Конкретные изменения (file:line)

### 4.0. NEW: `src/utils/pod_layout.py` — PodLayout abstraction

Создать `PodLayout` frozen dataclass (см. §3.0.1) в **`src/utils/`** (рядом с `logs_layout.py` — парная Mac-side структура). Single source of truth для pod-side путей. Использовать `PurePosixPath` (pod всегда Linux). Тесты pin'ят invariants (см. §6.4):
- `PodLayout.from_root(...).runner_log` всегда `<root>/logs/runner.log`
- Все per-run директории взаимно непересекаются между разными `run_id`.
- Mac↔Pod symmetry: `pod_layout.runner_log.name == log_layout.remote_runner_log.name`.

### 4.0.1. NEW: `TrainingProvider.pod_layout_for_run(run_id)` — Dependency Inversion

Расширить Protocol [src/providers/training/interfaces.py](src/providers/training/interfaces.py) методом `pod_layout_for_run(run_id: str) -> PodLayout`. Реализовать в:
- [src/providers/runpod/training/provider.py](src/providers/runpod/training/provider.py): root = `/workspace/runs/{run_id}`
- [src/providers/single_node/training/provider.py](src/providers/single_node/training/provider.py): root = `{config.workspace_path}/runs/{run_id}`

Все консьюмеры PodLayout (gpu_deployer, runner_launcher, code_syncer, file_uploader, etc.) получают layout через provider, не f-string'ят сами.

### 4.1. `src/runner/supervisor.py` — добавить tee-write

- **Constructor**: добавить параметр `stdio_log_path: Path | None = None` (DI). Источник — `layout.trainer_stdio_log`. Открыть файл в `_spawn` ([line 230](src/runner/supervisor.py:230)) **до** первой строки subprocess'а, чтобы файл существовал даже если trainer падает на первом print.
- **`_pump_stream`** ([line 417-442](src/runner/supervisor.py:417)):
  - **Удалить** `self._bus.publish("trainer_log", ...)` (line 442). Зачем — ниже в §4.5.
  - **Добавить** запись каждой строки в `stdio_log` через line-buffered `loop.run_in_executor(None, self._stdio_log.write, line + "\n")`. Ошибка disk-full / FS — `contextlib.suppress(OSError)` чтобы не убить pump.
  - Обе пампы (stdout / stderr) пишут в **один** файл (interleaved) — добавляем prefix `[OUT]` / `[ERR]` per line чтобы можно было grep'ать.
- **`_spawn`** ([line 230](src/runner/supervisor.py:230)): открыть `self._stdio_log = open(stdio_log_path, "ab", buffering=0)` сразу после `create_subprocess_exec`. `mode="ab"` — append-binary, idempotent, byte-level. `parents=True, exist_ok=True` для mkdir.
- **`shutdown`** ([line 365](src/runner/supervisor.py:365)): закрыть `_stdio_log` после reap'а, в finally-блоке.
- **Тесты** ([src/tests/unit/runner/test_supervisor.py](src/tests/unit/runner/test_supervisor.py)): новый `TestStdioCapture` класс, см. §6.

### 4.2. `src/runner/main.py` — конструировать PodLayout, передать в Supervisor

- В lifespan startup ([near line 159](src/runner/main.py:159)):
  ```python
  workspace = Path(os.getcwd())  # uvicorn запущен с cwd=workspace_path
  layout = PodLayout.from_workspace(str(workspace))
  ```
- Передать `layout` в Supervisor: `Supervisor(..., stdio_log_path=Path(str(layout.trainer_stdio_log)))`.
- EventJournal получает: `EventJournal(directory=Path(str(layout.events_dir)))`.
- Перед startup: `subprocess.run(["sh", "-c", layout.ensure_dirs_command()])` чтобы все директории существовали.

### 4.3. `src/pipeline/stages/managers/log_manager.py` — убрать хардкоды, принимать layout

- **Удалить** `DEFAULT_REMOTE_PATH = "/workspace/training.log"` (line 65) — это хардкод глобального пути, **источник bug'а с collision на resume**.
- **Удалить** `LOCAL_LOG_NAME = "training.log"` (line 69).
- Constructor LogManager: оставить `remote_path` и `local_path` как обязательные параметры (без default'ов — caller обязан передать). Это force каждого caller'а думать через PodLayout/LogLayout.
- В каждом call site:
  - На pod side: `LogManager(ssh_client, remote_path=str(pod_layout.runner_log), local_path=mac_log_layout.remote_runner_log)`
  - На pod side: `LogManager(ssh_client, remote_path=str(pod_layout.trainer_stdio_log), local_path=mac_log_layout.remote_trainer_stdio_log)`
- Без backwards-compat — пользователь явно сказал "от старого избавляемся".

### 4.4. `src/utils/logs_layout.py` — переименовать публичные имена

- `REMOTE_TRAINING_LOG_NAME = "training.log"` → `REMOTE_TRAINER_STDIO_LOG_NAME = "trainer.stdio.log"` ([line 37](src/utils/logs_layout.py:37)).
- `REMOTE_TRAINING_LOG_PATHS_KEY = "remote_training"` → `REMOTE_TRAINER_STDIO_LOG_PATHS_KEY = "remote_trainer_stdio"` ([line 42](src/utils/logs_layout.py:42)).
- Property `remote_training_log` → `remote_trainer_stdio_log` ([line 84](src/utils/logs_layout.py:84)).
- Обновить consumers (grep по `remote_training_log` / `REMOTE_TRAINING_LOG`).
- Обновить docstring (line 10) — описание нового layout'а.

### 4.5. `src/runner/supervisor.py` — выпиливаем `trainer_log` events

- **Строка 442**: `self._bus.publish("trainer_log", {"kind": kind, "line": line})` — **удалить**. Это data plane, она теперь в файле. WS/EventBus не носят log content.
- Update event taxonomy в docstring: явно сказать что bus принимает control + telemetry events, log content идёт в файл.

⚠ **Важно**: удаление `trainer_log` events НЕ затрагивает MLflow events (`mlflow_metric`, `mlflow_param`, `mlflow_tag`, `mlflow_run_started`, `mlflow_run_ended`). Они публикуются **через другой путь** — trainer'овский `RunnerEventCallback` шлёт HTTP loopback на `POST /api/v1/internal/events` ([src/runner/api/internal.py:90](src/runner/api/internal.py:90)), который вызывает `bus.publish(body.kind, body.payload)` для них. MLflowRelay фильтрует по `MLFLOW_EVENT_KINDS` ([src/runner/mlflow_relay.py:74](src/runner/mlflow_relay.py:74)) — `trainer_log` там не упомянут. Регрессия metrics flow невозможна (по построению white-list).

⚠ **Также не затрагивает**: `health_snapshot` (HealthReporter), `idle_detector_triggered` (IdleDetector), `pod_terminate_*` (PodTerminator), FSM events (`trainer_spawned`, `trainer_exited`, `cancellation_*`). Все они продолжают публиковаться без изменений.

### 4.6. `src/pipeline/stages/training_monitor.py` — две упрощающие правки

#### 4.6.1. Удалить мёртвую ветку `trainer_log`
- **Строки 923-929** ([training_monitor.py:923](src/pipeline/stages/training_monitor.py:923)): `if kind == "trainer_log": return None` — **удалить целиком**. Событие больше не публикуется.
- В docstring _dispatch_event (line 893-906) убрать упоминание `trainer_log`.

#### 4.6.2. Постмортем — упрощаем кардинально

**Принцип**: trainer output уже на Mac (`<attempt>/logs/trainer.stdio.log` + `<attempt>/logs/runner.log` — скачаны LogManager'ом). SSH probes для trainer-output **избыточны** — читаем локальные файлы.

SSH probes остаются **только для environment signals** (то, что нельзя получить из файлов trainer'а):
- dmesg slices (OOM-killer, NVIDIA xid)
- nvidia-smi current state

**Новый flow `_collect_death_diagnostics`**:

1. **Читаем локальные файлы** (3 источника):
   - `<attempt>/logs/trainer.stdio.log` — последние 30 строк → `[MONITOR:POSTMORTEM:trainer] <line>`. Если файл missing/empty → explicit token.
   - `<attempt>/logs/runner.log` — последние 30 строк → `[MONITOR:POSTMORTEM:runner] <line>`. Аналогично.
2. **SSH probes** (3 environment-level — только то, чего нет в файлах):
   - `dmesg | tail -n 80` → ядерные сообщения (RUNNING pod scope)
   - `dmesg | grep -iE 'oom|kill|memory|nvrm|xid|nvidia' | tail -n 30` → consolidated kernel signals
   - `nvidia-smi --query-gpu=name,utilization.gpu,memory.used,memory.total --format=csv,noheader` → current GPU state

**Результат**: `[MONITOR:POSTMORTEM]` блок состоит из:
- 30 строк trainer.stdio.log (или `<<EMPTY>>` / `<<MISSING>>`)
- 30 строк runner.log (или `<<EMPTY>>` / `<<MISSING>>`)
- kernel/GPU snapshot

**Удаляем probes**:
- `faulthandler` (файла больше нет — faulthandler в stderr → stdio.log)
- `training_log_tail` (файла больше нет — есть trainer.stdio.log, читаем локально)

**Workspace path**: больше не нужен в monitor (probes не ходят в workspace). Удаляем хардкод `workspace = "/workspace"` ([line 719](src/pipeline/stages/training_monitor.py:719)).

**Унификация**: все артефакты trainer'а лежат в **одном месте** на pod-е (`/workspace/`), на Mac — в **одном месте** (`<attempt>/logs/`). Один источник правды per concern. Probes минимальны, делают только то, что нельзя получить иначе.

### 4.7. `src/pipeline/stages/gpu_deployer.py` — использовать PodLayout

- **`_download_remote_logs`** ([line 398](src/pipeline/stages/gpu_deployer.py:398)):
  - Конструировать `PodLayout.from_workspace(self.deployment.workspace)` в начале.
  - Channel 1 (runner.log): `remote_path = str(pod_layout.runner_log)` → `<workspace>/logs/runner.log` (new per-run path).
  - Channel 2 (trainer.stdio.log): `remote_path = str(pod_layout.trainer_stdio_log)` → `<workspace>/logs/trainer.stdio.log`.
  - **Удалить fallback paths** (`logs_base/<subdir>/pipeline.log`, `logs_base/training.log`) — это legacy от thin-image migration. Pod layout гарантирует один canonical путь.
  - Update comments для отражения per-run layout.
- **Тесты** ([src/tests/unit/pipeline/test_stages_deployer.py](src/tests/unit/pipeline/test_stages_deployer.py)) — обновить mock paths.

### 4.8. `src/training/run_training.py` — выпилить FileHandler + перенастроить faulthandler

- **Строка 575-622**: `_DEFAULT_TRAINING_LOG_PATH` + `_install_training_file_handler` — **удалить целиком**. Это исчезающий концерн (trainer не отвечает за свою survival-видимость).
- **`_install_crash_observability`** ([line 625-735](src/training/run_training.py:625)): **упрощаем**. Удаляем попытку открыть отдельный файл для faulthandler. Заменяем на:
  ```python
  import faulthandler
  faulthandler.enable(all_threads=True)  # пишет в sys.stderr по умолчанию
  ```
  SEGV/ABRT/CUDA traceback теперь идёт в stderr trainer subprocess'а → Supervisor pump → `trainer.stdio.log`. Один путь, без отдельного файла.
- atexit logging flush в `_install_crash_observability` — оставляем (orthogonal, полезный).
- **Backward compat alias** `train_v2 = run_training` ([line 572](src/training/run_training.py:572)) — оставляем (это про API, не про observability).
- Если в `main()` вызывается `_install_training_file_handler()` — удалить вызов.

### 4.9.1. `src/pipeline/stages/managers/deployment/runner_launcher.py` — убрать хардкод runner.log

- **Текущее**: `RUNNER_LOG_PATH = "/workspace/runner.log"` (хардкод, line 55) — **источник resume-collision bug'а**.
- **Новое**: принимать `layout: PodLayout` или `runner_log_path: str` параметром в `launch_runner()`. Использовать `layout.runner_log` вместо хардкода.
- В command construction: `nohup ... > {layout.runner_log} 2>&1` — per-run path.
- Workspace path параметр (`workspace_path`) уже передаётся — на его основе можно построить layout прямо здесь.
- Тесты обновляются (mock на новый path).

### 4.9.2. `src/pipeline/stages/managers/deployment/code_syncer.py` — использовать PodLayout

- **Текущее**: `f"{self._workspace}/{module}"` (line 149) — ad-hoc concat.
- **Новое**: `str(pod_layout.src_dir / module_relative_path)`.
- DRY: один источник правды для путей "куда rsync'ить".

### 4.9.3. `src/pipeline/stages/managers/deployment/file_uploader.py` — использовать PodLayout

- **Текущее**: `f"{self._workspace}/config/pipeline_config.yaml"` (line 367).
- **Новое**: `str(pod_layout.config_file)`.

### 4.9.4. `src/runner/event_journal.py` — events/ без dot-prefix

- **Текущее**: hardcoded `EVENTS_DIR_REL = ".runner/events"` ([line 109](src/runner/event_journal.py:109)).
- **Новое**: удалить `EVENTS_DIR_REL` constant. Конструктор принимает `directory: Path` явно — caller (runner/main.py) передаёт `Path(str(layout.events_dir))`. EventJournal больше не знает про относительные пути в workspace.

### 4.9.5. `src/runner/state.py` — state/ без dot-prefix

- **Текущее**: `state_dir = workspace_dir / ".ryotenkai"` ([line 279](src/runner/state.py:279)), files `state.json` + `state.jsonl`.
- **Новое**: убрать хардкод `.ryotenkai`. Конструктор принимает `state_dir: Path` явно — caller (runner/main.py) передаёт `Path(str(layout.state_dir))`. Файлы переименовываются в `job.json` + `job.jsonl` (явное имя — имя того, что хранится).

### 4.9. `src/pipeline/stages/managers/deployment/training_launcher.py` — убрать env vars

- **Строка 527**: `env["RYOTENKAI_TRAINING_LOG_PATH"] = LogManager.DEFAULT_REMOTE_PATH` — **удалить**. Trainer его больше не читает.
- **Строка 514**: `env["PYTHONFAULTHANDLER_PATH"] = f"{workspace_env}/training.faulthandler.log"` — **удалить тоже**. Faulthandler теперь пишет в stderr (default).
- **Строка 513**: `env["PYTHONFAULTHANDLER"] = "1"` — **оставляем** (это активация faulthandler от Python stdlib; теперь без custom path → stderr).
- **Drift-guard test** [test_training_launcher_v2.py:158-166](src/tests/unit/pipeline/stages/managers/deployment/test_training_launcher_v2.py:158) — **удалить** (env var больше не существует, инвариант не actionable).

### 4.10. `src/runner/event_bus.py` / `src/runner/event_journal.py` — без изменений

EventJournal продолжает работать. Просто `trainer_log` события туда больше не попадают (см. §4.5). Disk pressure на pod падает на 2-3 порядка (сейчас ~30 events/sec из-за trainer_log, после — ~1-2 events/sec для control + telemetry, в основном mlflow_metric).

EventJournal остаётся **durability layer для control + telemetry events** — это нужно для:
- WS subscriber late join / Mac reconnect after sleep
- MLflowRelay survive trainer crash + restart (queue persistence)

EventJournal **НЕ скачивается на Mac** (пользователь явно: "ивенты не собираем"). Это in-pod replay storage.

### 4.10.1. MLflow flow — БЕЗ ИЗМЕНЕНИЙ

Не трогаем:
- [src/runner/mlflow_relay.py](src/runner/mlflow_relay.py) — buffered async forward
- [src/runner/api/internal.py:90-97](src/runner/api/internal.py:90) — loopback endpoint, MLflowRelay submit
- [src/training/callbacks/runner_event_callback.py](src/training/callbacks/runner_event_callback.py) — trainer-side push
- [src/training/managers/mlflow_manager/](src/training/managers/mlflow_manager/) — trainer-side resilient transport

Регрессия проверяется отдельным тестом в §6.7.

### 4.11. Frontend — `web/src/components/LogDock.tsx`

- **Строка 5**: `const FILES = ['pipeline.log', 'training.log', 'inference.log', 'eval.log']` → `['pipeline.log', 'trainer.stdio.log', 'runner.log', 'inference.log', 'eval.log']`.
- Backend endpoint `GET /api/v1/runs/<id>/attempts/<n>/logs?file=...` — нужно подтвердить во время implementation:
  - Если whitelist на стороне FastAPI router — обновить.
  - Если path traversal guard уже работает по directory listing — добавить нечего.
- `pipeline.log` (Mac orchestrator log) — **остаётся как есть**, изменения не трогают Mac-side aggregated log.

### 4.12. Документация

- `docs/architecture/log-collection.md` или эквивалент — обновить (если есть).
- `docs/plans/harmonic-rolling-crayon.md` § 4 (target architecture) — добавить short note про эту revision.
- `CONFIG_REFERENCE.md` — убрать `RYOTENKAI_TRAINING_LOG_PATH` если он там описан.

---

## 5. Migration sequence (один cohesive PR, логические шаги)

Шаги внутри PR — каждый коммит должен оставлять репо в **зелёном** состоянии (тесты проходят, ruff/mypy clean).

Правильный порядок: **Supervisor stdio capture ЗАМЕНЯЕТ `bus.publish('trainer_log')` атомарно** (один коммит) — нет промежуточного состояния где данные пишутся в два места.

1. **NEW: PodLayout abstraction** — создать `src/utils/pod_layout.py` (frozen dataclass рядом с `logs_layout.py`, factory `from_root`, properties для всех путей, helper `ensure_dirs_command()`). Unit tests pin invariants (per-run isolation, structure, factory, Mac↔Pod symmetry).
2. **NEW: TrainingProvider.pod_layout_for_run** — добавить method в Protocol [interfaces.py](src/providers/training/interfaces.py), реализовать в RunPod + single_node providers. Закрывает Dependency Inversion violation.
3. **Mac LogLayout rename** — `REMOTE_TRAINING_LOG_NAME` → `REMOTE_TRAINER_STDIO_LOG_NAME = "trainer.stdio.log"`. Add property `remote_trainer_stdio_log`. Coverage tests обновляются вместе.
4. **runner/main.py wires PodLayout**: lifespan получает root от env var (или argparse) и создаёт `layout = PodLayout.from_root(root)`, передаёт в Supervisor (stdio_log_path = layout.trainer_stdio_log), EventJournal (directory = layout.events_dir), JobLifecycleFSM (state_dir = layout.state_dir). Pre-startup `mkdir -p` через `ensure_dirs_command`.
4. **Supervisor: stdio capture replacing trainer_log events (атомарный шаг)**:
   - Добавить tee-write в `_pump_stream` → `<workspace>/logs/trainer.stdio.log`, append-only, prefix `[OUT]`/`[ERR]`.
   - Удалить `bus.publish('trainer_log', ...)`.
   - Открытие файла в `_spawn` до первой строки subprocess'а.
   - Обновить event-taxonomy docstring.
5. **runner_launcher.py**: убрать хардкод `RUNNER_LOG_PATH = "/workspace/runner.log"`. Принять `layout` параметром, использовать `layout.runner_log`. Закрывает resume-collision bug.
6. **code_syncer.py + file_uploader.py**: заменить ad-hoc f-string concat на `pod_layout.<property>`. DRY.
7. **LogManager**: удалить `DEFAULT_REMOTE_PATH` constant + `LOCAL_LOG_NAME`. Constructor требует explicit `remote_path` и `local_path`. Все callers через PodLayout/LogLayout.
8. **Monitor: cleanup dead branch** — удалить `if kind == "trainer_log": return None` в `_dispatch_event`.
9. **gpu_deployer**: `_download_remote_logs` использует PodLayout. Удалить legacy fallback paths.
10. **trainer side cleanup**:
   - Удалить `_install_training_file_handler` + вызов в `main()`.
   - В `_install_crash_observability`: упростить faulthandler setup до `faulthandler.enable(all_threads=True)` (без custom file). atexit flush — оставить.
11. **training_launcher**: удалить `env["RYOTENKAI_TRAINING_LOG_PATH"]` и `env["PYTHONFAULTHANDLER_PATH"]`. Drift-guard test удалить.
12. **Postmortem rewrite** (§4.6.2):
   - `_collect_death_diagnostics`: читает локальный `trainer.stdio.log` + `runner.log`, плюс 3 SSH probes (dmesg, dmesg-grep, nvidia-smi).
   - Удалить probe `faulthandler`, `training_log_tail`.
   - Удалить хардкод `workspace = "/workspace"`.
13. **Frontend LogDock**: `FILES = ['pipeline.log', 'trainer.stdio.log', 'runner.log', 'inference.log', 'eval.log']`. Backend endpoint whitelist (если есть) — обновить.
14. **Docs** — обновить `docs/plans/harmonic-rolling-crayon.md` § 4 (target architecture), `CONFIG_REFERENCE.md` (убрать env vars), `docs/architecture/log-collection.md` (если есть). Add ADR на `PodLayout` если есть decision-record система.
15. **Regression test** — `test_stdio_capture_e2e.py` для 22:20:18 scenario + `test_resume_no_log_collision.py` (sequential runs на одном pod не collidate'ят логи).

---

## 6. Test plan (7 категорий)

Все тесты в `src/tests/unit/` или `src/tests/integration/`, каждый шаг требует свой набор:

### 6.1. Positive
- **trainer_normal_run** (integration): полный run с fake-trainer, файл `trainer.stdio.log` создан, скачан на Mac в `<attempt>/logs/trainer.stdio.log`.
- **stdio_capture_writes_lines** (unit): trainer пишет 3 строки stderr → файл содержит 3 строки с `[ERR]` префиксом.
- **probe_with_data**: `<<MISSING>>` отсутствует, видны actual lines.

### 6.2. Negative
- **trainer_import_crash** (integration): fake-trainer делает `raise ImportError("boom")` сразу. trainer.stdio.log должен содержать traceback. На Mac виден через `_download_remote_logs`.
- **trainer_first_print_then_exit**: один `print("hello")` + `sys.exit(1)`. Файл содержит "hello".
- **trainer_native_segv** (integration, ключевой): fake-trainer делает `import faulthandler; faulthandler.enable(); os.kill(os.getpid(), signal.SIGSEGV)`. Файл содержит `Fatal Python error: Segmentation fault` + traceback (faulthandler пишет в stderr → Supervisor pump → stdio.log).
- **trainer_zero_output_silent_exit**: `sys.exit(1)` сразу без print. Файл существует, но пустой → postmortem печатает `<<EMPTY>>` для trainer.stdio.log.

### 6.3. Boundary
- **stdio_log_zero_byte**: trainer crashed до first print → файл существует, размер 0 байт → postmortem печатает `[trainer]: <<EMPTY>>`.
- **stdio_log_missing_on_mac**: scp не успел до cleanup (network flake) → локальный файл missing → postmortem печатает `[trainer]: <<MISSING>>`.
- **stdio_log_huge_single_line**: одна строка 4097 байт без `\n` (превышает PIPE_BUF) → assert chunked write через Lock не теряет порядок.
- **stdio_log_disk_full**: write fails с OSError → pump продолжает работать (не падает), warning emitted в runner.log.
- **runner_log_missing**: только trainer.stdio.log есть → postmortem печатает trainer + `[runner]: <<MISSING>>`.

### 6.4. Invariants
- **podlayout_per_run_isolation** (property test): для любых двух разных `run_id` пути PodLayout полностью disjoint. Ни один path-A.startswith(path-B) при разных run'ах.
- **podlayout_provider_agnostic**: PodLayout с `root="/workspace/runs/abc"` (RunPod) и `root="/home/user/ryotenkai/runs/abc"` (single_node) производят одинаковую sub-tree структуру.
- **layout_pod_mac_symmetry**: `pod_layout.runner_log.name == mac_log_layout.remote_runner_log.name == "runner.log"`. Аналогично для trainer.stdio.log. Гарантия что scp 1:1 mapping не сломается.
- **single_writer**: только один Supervisor пишет в trainer.stdio.log. Concurrent test с двумя Supervisor'ами должен fail explicitly или сериализовать.
- **append_only**: stat файла после second write `>= ` first write size. Нет seek backwards.

### 6.5. Dependency-error
- **ssh_disconnect_during_pull**: scp прерывается mid-byte → next iteration видит partial local file, picks up size delta correctly.
- **probe_ssh_returns_error**: `ok=False` → state = "ssh_error", не маскируется как `<<EMPTY>>`.
- **logmanager_no_remote_file**: trainer.stdio.log существует на Mac но не на pod (pod evicted) → log_manager `download_full` returns False, warning logged, no exception.

### 6.6. Regression
- **2026_05_01_22_20_18_scenario** (integration): воссоздать exact setup лога — fake trainer падает с ImportError через 10s, pod alive — assert:
  - `<attempt>/logs/trainer.stdio.log` содержит traceback
  - postmortem НЕ говорит "all probes empty" (теперь говорит `trainer_stdio_tail: <data>`)
  - operator видит причину crash'а в run-level `pipeline.log`
- **resume_no_log_collision** (integration): запустить run 1 на pod-A (генерирует logs), потом resume на тот же pod-A (новый run_id) — assert run-2 logs НЕ перезаписали run-1 logs. Pre-fix: оба run'а пишут в `/workspace/runner.log`. Post-fix: каждый в `<workspace>/runs/<run_id>/logs/`.
- **provider_layout_parity**: тот же fake-trainer на mock RunPod и mock single_node — assert файлы оказались в правильных местах для каждого провайдера, через PodLayout.

### 6.7. Specific logic
- **trainer_log_event_not_published**: запустить fake-trainer пишущий в stderr → assert на pod-side EventBus НЕТ `trainer_log` events (только control + telemetry).
- **monitor_dispatch_no_trainer_log_branch**: feed event `{kind: "trainer_log", payload: {...}}` to `_dispatch_event` → returns None silently AND logs unknown-kind debug (поскольку ветки больше нет).
- **mlflow_events_still_flow** (regression): trainer публикует `mlflow_metric` через RunnerEventCallback → assert событие попало в EventBus, MLflowRelay queue получил submit, EventJournal записал. **Регрессия metrics flow** не должна быть.
- **health_snapshot_still_flows**: HealthReporter публикует snapshot → assert на Mac WS subscriber получил, monitor рендерит [MONITOR] ALIVE. Регрессия health monitoring не должна быть.
- **fsm_events_still_flow**: trainer_spawned / trainer_exited пушатся → Mac TrainingMonitor видит → FSM sync работает. Регрессия cancellation chain не должна быть.

### 6.8. Combinatorial
Матрица: `{trainer-pre-import-crash, trainer-mid-init-crash, trainer-mid-train-crash, trainer-OOM-SIGKILL, trainer-clean-exit} × {pod-alive-at-cleanup, pod-evicted-before-cleanup} × {ssh-stable, ssh-flaky}` — итого 5×2×2 = 20 cells. В каждой cell assert как минимум один из {trainer.stdio.log, runner.log} содержит данные на Mac.

---

## 7. Risk register

| # | Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|---|
| R1 | Supervisor stdio write блокирует _pump_stream → backpressure → trainer SIGPIPE | Low | High | `loop.run_in_executor` для write; suppress OSError; pump продолжает |
| R2 | Race: stdout pump и stderr pump пишут в один файл одновременно → interleaved garbage | Medium | Medium | `flock`-style serialization через single asyncio.Lock в Supervisor, ИЛИ append O_APPEND atomic line-write (буфер ≤ PIPE_BUF=4KB) |
| R3 | Disk full на pod → файл не растёт → trainer logs lost during incident | Low | High | Pod-side disk monitoring (отдельный feature); этот PR — KISS, not solving disk pressure |
| R4 | Path drift повторяется: кто-то добавит ещё один путь | Low | Medium | Property test §6.4 — invariant guard. Single LogLayout как SSOT |
| R5 | PII / secrets в stdio dump (HF_TOKEN, RUNPOD_API_KEY если trainer print env) | Medium | High | Redaction filter в Supervisor: regex `hf_[A-Za-z0-9]+`, `sk-[A-Za-z0-9]+`, `RUNPOD_[A-Z_]+=...`. Документация: операторы добавляют свои паттерны через env var |
| R6 | trainer.stdio.log растёт неограниченно (12-часовой run × 4 KB/s = ~170 MB) | Medium | Low | Acceptable для текущей нагрузки. Rotation — follow-up |
| R7 | Удалили training.log, ломаем существующих consumers (CLI grep'ы, dashboards) | Medium | Medium | Search-and-replace по всей кодовой базе + frontend LogDock + docs |
| R8 | Удаление `bus.publish('trainer_log')` ломает что-то, на что я не посмотрел | Low | Medium | Grep `trainer_log` по всему репо до commit |
| R9 | atomic_fs.atomic_write не подходит для append; нужен раздельный helper | Low | Low | Использовать обычный `open(mode="ab")` — атомарность строк гарантируется PIPE_BUF |
| R10 | Случайно сломали MLflow metrics forward вместе с удалением trainer_log | Low | High | MLFLOW_EVENT_KINDS — explicit white-list ([mlflow_relay.py:74](src/runner/mlflow_relay.py:74)), trainer_log там не упомянут. Plus regression test §6.7 (`mlflow_events_still_flow`) |
| R11 | Mac LogDock пытается прочитать `training.log` после rename → 404 / empty | Medium | Low | Шаг 8 в migration sequence + e2e test после rename |
| R12 | Tests blowing up из-за renamed constants в LogLayout | Medium | Low | Шаг 1 в migration sequence — изначальный mass rename catches все обновления |
| R13 | PodLayout abstraction добавляет complexity (extra class, lookup) | Low | Low | Frozen dataclass — zero-cost over PurePosixPath. Тесты на invariants pin'ят contract. Indirection пользителя — `layout.runner_log` понятнее чем `f"{ws}/logs/runner.log"` |
| R14 | Single_node provider workspace_path может конфликтовать с user files | Medium | Medium | single_node already creates `runs/<run_id>/` под user-config-base. PodLayout не меняет это поведение, только дисциплинирует sub-paths |
| R15 | Pod после rename не имеет директории `logs/` → write fails | High если забыли | Medium | `runner/main.py` lifespan startup делает `layout.ensure_dirs_command()` через `subprocess.run(["sh","-c", cmd])`. Идемпотентно. Тест: spawn runner на pristine workspace → все директории созданы |

**Где мой дизайн может тихо сломаться** (худший сценарий):
- Supervisor сам крашится до того, как пайпы закроются. Stdio в kernel-pipe-buffer (~64KB) теряется. Mitigation: не вижу простого — kernel-level capture (через `script(1)`) out of scope.
- Mac never connects (network partition full run) → видим только final pull. Без streaming это нормально, но operator не имеет live view. Acceptable trade-off: "одна стабильная реализация" пользователь явно выбрал.

**Open questions** (нерешённые, требуют деппsink или экспериментa):
- **PIPE_BUF на macOS / Linux**: документация говорит 4096 байт минимум на Linux, 512 на macOS. Если строка длиннее — `write()` может раздробить atom. Mitigation: assert line length ≤ 4096 в `_pump_stream`, иначе chunked write с lock.

---

## 8. Что мы явно НЕ делаем (rejected alternatives)

1. **EventJournal pull на Mac** — пользователь явно сказал "пока ивенты не собираем". Журнал остаётся pod-side artefact для in-pod replay. Если в будущем понадобится — добавим scp `<workspace>/.runner/events/` recursive.
2. **Streaming-only** (revert d5f1cbd, добавить EventMirrorWriter, удалить LogManager) — отвергнуто после deep-think: streaming не survives pod evict / Mac sleep / WS partition надёжно. Pull — индустриальный стандарт для long-running batch jobs ([LLM Observability 2025](https://www.getmaxim.ai/articles/llm-observability-best-practices-for-2025/)).
3. **Hybrid** (оба механизма параллельно) — пользователь явно: "одна стабильная реализация".
4. **Раздельные stdout/stderr файлы** (`trainer.stdout.log` + `trainer.stderr.log`) — отложено как follow-up. KISS first; если empirically возникнет need в filtering — выделим. На сегодня interleaved с prefix хватает.
5. **`<workspace>/.runner/stdio/` namespace** — отвергнуто в пользу плоского `/workspace/trainer.stdio.log` (минимум миграции, рядом с `runner.log`).
6. **Restore EventMirrorWriter** — отвергнуто (был revert'нут не зря: 100x log inflation от health snapshots).
7. **Rotation файла на pod** — отложено (acceptable size при текущей нагрузке).
8. **Trainer FileHandler сохранить как supplement** — пользователь явно: "только runner и stdio". Удаляем training.log целиком.

---

## 9. Verification (end-to-end после implementation)

### 9.1. Unit / integration suite
```bash
# Все тесты должны быть зелёными после implementation
pytest src/tests/unit/runner/test_supervisor.py -v
pytest src/tests/unit/pipeline/stages/test_training_monitor.py -v
pytest src/tests/unit/pipeline/test_stages_deployer.py -v
pytest src/tests/integration/runner/test_stdio_capture_e2e.py -v  # NEW
```

### 9.2. Static checks
```bash
ruff check .
ruff format --check .
mypy .                 # должен оставаться 0 errors
```

### 9.3. Manual regression (то что сорвалось 22:20:18)
1. Намеренно ломаем import в `src/training/run_training.py` (например, `from src.workspace.does_not_exist import x`).
2. Запускаем pipeline через CLI на RunPod.
3. Trainer должен упасть в первые 10 секунд.
4. **Acceptance**: в `runs/<id>/attempts/<n>/logs/trainer.stdio.log` лежит `ModuleNotFoundError: No module named 'src.workspace.does_not_exist'` traceback. В `pipeline.log` есть `[MONITOR:POSTMORTEM] trainer_stdio_tail: <первые лайны traceback'а>`.

### 9.4. Live pipeline run
1. Восстанавливаем правильный config, запускаем штатный run на 10 минут.
2. На Mac в `<attempt>/logs/trainer.stdio.log` — растёт периодически (delta-pull раз в 5s).
3. На terminal — final flush, файл полный.
4. WS subscriber получает только control events (нет trainer_log spam'а).
5. EventJournal на pod после run'а — компактный (~100 events, не миллионы).

### 9.5. MCP-driven verification
- `mcp__repowise__get_risk(targets=["src/runner/supervisor.py","src/pipeline/stages/training_monitor.py"])` — проверить что hotspot score не вырос катастрофически.
- `mcp__repowise__get_dead_code()` — убедиться что после удаления `_install_training_file_handler` нет orphaned references.

---

## 10. Best practices alignment (research)

Решение проверено против индустриальных рекомендаций для long-running training jobs (2025):

- ✅ **File redirection over pipes** для long-running jobs — survives crashes and parent termination ([Capturing Linux Process STDOUT and STDERR Output](https://tech-champion.com/linux/capturing-linux-process-stdout-and-stderr-output-a-comprehensive-guide/))
- ✅ **Local file copies as ground truth**, control signals via streaming separately ([LLM Observability Best Practices for 2025](https://www.getmaxim.ai/articles/llm-observability-best-practices-for-2025/))
- ✅ **Line-buffering** (через `PYTHONUNBUFFERED=1` который уже стоит) для prompt log appearance
- ✅ **Correlation IDs** на каждой строке — у нас prefix `[OUT]/[ERR]` + timestamp в каждом логгер-вызове, plus `job_id` в `<attempt>/logs/` directory layout
- ✅ **Capture environment context automatically** at crash time — postmortem probes (faulthandler, dmesg, nvidia-smi) уже есть, теперь дополнительно гарантированный stdio dump
- ✅ **Avoid blameless postmortem timelines without evidence** — после fix'а каждый probe имеет explicit state (missing/empty/data), нет догадок
- ✅ **Three-plane separation** (control / telemetry / data) — индустриально стандарт. Kubernetes сделал это явно: control plane (FSM/lifecycle через etcd+API server), telemetry (metrics через Prometheus pull), data plane (kubelet container stdout → файл на ноде, `kubectl logs` читает файл)

Наше решение **точно повторяет паттерн `kubectl logs`**: kubelet пишет container stdout/stderr в файл на ноде (`/var/log/containers/...`), `kubectl logs` читает файл через kubelet API. Никакого streaming как backbone — streaming поверх (для `kubectl logs -f` follow), но source-of-truth — файл.

Telemetry plane отдельно: Prometheus делает scrape (pull) метрик из приложения по HTTP — это аналог нашего MLflowRelay async forward. Streaming events (как у нас WS) — это паттерн "live UI updates", не source-of-truth.

Data plane (logs) and telemetry plane (metrics) **никогда** не объединяются в один транспорт в индустрии — это первый принцип observability в 2025. Логи — high-volume, free-form, для post-mortem. Метрики — structured, low-volume, для realtime monitoring. Их природа диктует разные транспорты.

### 10.1. Filesystem layout best practices alignment

PodLayout структура соответствует индустриальным стандартам:

- ✅ **Per-run isolation** — каждый run в своей директории. Соответствует [Kubernetes /var/log/pods/<pod_uid>](https://kubernetes.io/docs/concepts/cluster-administration/logging/) pattern (per-pod isolated directory).
- ✅ **Convention-based directories** — `logs/`, `output/`, `config/`, `data/`. Соответствует [Azure ML special folders pattern](https://learn.microsoft.com/en-us/azure/machine-learning/how-to-set-up-training-targets?view=azureml-api-1) (./outputs, ./logs).
- ✅ **Single source of truth для путей** — PodLayout как frozen dataclass. Эквивалент `LogLayout` Mac-side. Никаких string concat'ов в domain коде.
- ✅ **Provider-agnostic** — different `root` values, identical structure. Соответствует [AI/ML on Kubernetes Best Practices](https://www.wiz.io/academy/ai-security/ai-ml-kubernetes-best-practices) (decouple storage from pod lifecycle, abstract via PVC).
- ✅ **Mac↔Pod symmetry** — `pod_layout.runner_log.name == mac_log_layout.remote_runner_log.name`. Log shipper pattern (kubelet symlinks).
- ✅ **Idempotent setup** — `ensure_dirs_command()` mkdir -p. Безопасно при resume / restart.
- ✅ **Sequential runs не collidate'ят** — different `run_id` → different sub-tree → нет race на logs / events / output.

Отложенные best-practices (future work, не в этом PR):
- ⚪ **TTL / rotation** — Azure ML pattern for old run cleanup. У нас manual cleanup на Mac side; pod-side disk pressure можно решать в будущем.
- ⚪ **Tiered storage** — hot NVMe vs cold S3. Не применимо для нашего scale.
- ⚪ **Symlink directory** — kubelet style `/var/log/containers/...`. Излишне для нашего single-pod-per-run scenario.

---

## 11. Critical files (для implementation)

**NEW files**:
```
src/utils/pod_layout.py                                             # PodLayout abstraction (рядом с logs_layout.py)
src/tests/unit/utils/test_pod_layout.py                             # invariants + provider parity + Mac↔Pod symmetry
src/tests/integration/runner/test_stdio_capture_e2e.py              # 22:20:18 regression
src/tests/integration/runner/test_resume_no_log_collision.py        # per-run isolation
```

**Modified files**:
```
src/providers/training/interfaces.py                                # add pod_layout_for_run method to Protocol
src/providers/runpod/training/provider.py                           # implement pod_layout_for_run
src/providers/single_node/training/provider.py                      # implement pod_layout_for_run
src/runner/supervisor.py                                            # stdio capture, remove trainer_log publish
src/runner/main.py                                                  # construct PodLayout, wire to Supervisor + EventJournal + FSM
src/runner/event_journal.py                                         # remove EVENTS_DIR_REL hardcode, accept directory
src/runner/state.py                                                 # remove .ryotenkai hardcode, accept state_dir, rename files (state.* → job.*)
src/pipeline/stages/training_monitor.py                             # postmortem rewrite + remove dead branch
src/pipeline/stages/gpu_deployer.py                                 # download via PodLayout, drop fallbacks
src/pipeline/stages/managers/log_manager.py                         # remove DEFAULT_REMOTE_PATH, require explicit paths
src/pipeline/stages/managers/deployment/runner_launcher.py          # use layout.runner_log instead of hardcode
src/pipeline/stages/managers/deployment/code_syncer.py              # use pod_layout.src_dir
src/pipeline/stages/managers/deployment/file_uploader.py            # use pod_layout.config_file
src/pipeline/stages/managers/deployment/training_launcher.py        # remove env vars, drift-guard test
src/training/run_training.py                                        # remove _install_training_file_handler, simplify faulthandler
src/utils/logs_layout.py                                            # rename remote_training_log → remote_trainer_stdio_log
web/src/components/LogDock.tsx                                      # update FILES list
```

**Test updates**:
```
src/tests/unit/runner/test_supervisor.py                            # new TestStdioCapture
src/tests/unit/pipeline/stages/test_training_monitor.py             # postmortem trichotomy
src/tests/unit/pipeline/test_stages_deployer.py                     # PodLayout mocks
src/tests/unit/pipeline/stages/managers/deployment/test_training_launcher_v2.py  # drift-guard removed
```

---

## 12. Phase 2 (отложено, не делаем сейчас)

После того, как PR1 land'нется и поработает в проде:
- **Раздельные stdout/stderr файлы** если operator UX скажет, что фильтровать interleaved неудобно.
- **Rotation** trainer.stdio.log по размеру (когда хитнут лимит).
- **PII redaction policy** — конфигурируемые regex через env var.
- **Удаление WS streaming для logs** — после того, как весь fleet мигрирует, можно почистить EventJournal-related disk pressure dampers (если останутся).

---

## 13. Open / unresolved questions & audit trail

### 13.1. Open questions (требуют решения во время implementation)

- **PIPE_BUF size на Linux/macOS**: Linux 4096B, macOS 512B. Если строка длиннее — `write()` может раздробить atom между двумя пампами. Mitigation: assert в `_pump_stream` если len(line) > 4096 → chunk через asyncio.Lock. Проверить empirically на typical trainer output (HuggingFace progress bar строки могут быть 200-300 символов — OK).
- **Frontend logs endpoint whitelist**: подтвердить во время implementation, что `GET /api/v1/runs/<id>/attempts/<n>/logs?file=trainer.stdio.log` принимается backend'ом (router whitelist или path traversal guard).

### 13.2. Audit trail (3 итерации ревизии плана, выполнено до утверждения)

**Итерация 1** — terminology + zones:
- ✅ Найдено: "WS только для control plane" — неточно, не учитывает telemetry plane (MLflow events). Исправлено в §2 и §3.2.
- ✅ Найдено: диаграмма не показывает MLflowRelay path. Исправлено — добавлена three-plane diagram.
- ✅ Найдено: §3.2 zones table говорил "EventBus — control-plane events ONLY" — это неверно (MLflow events тоже идут через bus). Исправлено.
- ✅ Найдено: faulthandler.log не упомянут в зонах — добавлен как третий лог-файл.

**Итерация 2** — проверка не сломаются ли metrics:
- ✅ Verified: MLFLOW_EVENT_KINDS — explicit white-list ([mlflow_relay.py:74](src/runner/mlflow_relay.py:74)), `trainer_log` там не упомянут → удаление trainer_log не повлияет на forward.
- ✅ Verified: `RunnerEventCallback` пушит `mlflow_*` events через HTTP loopback, не через Supervisor pipe → независимый путь.
- ✅ Verified: `MLflowRelayCircuitBreaker` + queue retention — trainer crash + restart не теряет metrics (queue persistence via EventJournal).
- ✅ Added regression test §6.7 `mlflow_events_still_flow`.

**Итерация 3** — production failure modes:
- ✅ Найдено: атомарный шаг 2 в migration sequence (replace, не parallel write) — устранена возможность double-write race.
- ✅ Найдено: R10 / R11 / R12 добавлены в risk register (MLflow regression, LogDock 404, test rename).
- ✅ Verified: `pipeline.log` (Mac orchestrator log) не задевается — отдельная сущность, не remote pull.
- ✅ Verified: BACKWARD COMPATIBILITY alias `train_v2 = run_training` ([run_training.py:572](src/training/run_training.py:572)) — это про API surface, не observability — оставляем.

**Итерация 4** (post-pivot, после введения PodLayout abstraction):
- ✅ Найдено: `runner.log` хардкоден на `/workspace/runner.log` ([runner_launcher.py:55](src/pipeline/stages/managers/deployment/runner_launcher.py:55)) — глобальный путь, **resume bug**. Перенос под `<workspace>/logs/runner.log`.
- ✅ Найдено: `training.log` хардкоден на `/workspace/training.log` (LogManager.DEFAULT_REMOTE_PATH) — **тот же resume bug**. Перенос на per-run `<workspace>/logs/trainer.stdio.log`.
- ✅ Найдено: 5+ ad-hoc f-string concat'ов в [code_syncer.py](src/pipeline/stages/managers/deployment/code_syncer.py:149), [file_uploader.py](src/pipeline/stages/managers/deployment/file_uploader.py:367), [training_launcher.py](src/pipeline/stages/managers/deployment/training_launcher.py:514), [runner_launcher.py](src/pipeline/stages/managers/deployment/runner_launcher.py:55), [event_journal.py](src/runner/event_journal.py:109). Решение: PodLayout abstraction.
- ✅ Найдено: PodLayout даёт **4 преимущества**: (1) closes resume-collision bug, (2) DRY single-source-of-truth, (3) provider-agnostic abstraction для будущих провайдеров, (4) Mac↔Pod парность через `LogLayout` (Mac) ↔ `PodLayout` (pod).
- ✅ Verified: `LogLayout` (Mac side) и `PodLayout` (pod side) — парные структуры. 1:1 mapping `runner.log/trainer.stdio.log` гарантирован invariant test.
- ✅ R13 / R14 / R15 добавлены в risk register (PodLayout complexity, single_node user-files, missing dirs).
- ✅ Verified против индустриальных best-practices: Kubernetes per-pod isolation, Azure ML special folders, AWS EKS decouple-from-pod-lifecycle. См. §10.1.

### 13.3. Three-plane summary (TL;DR для review'еров)

| Plane | Что | Транспорт ДО | Транспорт ПОСЛЕ |
|---|---|---|---|
| **Control** | FSM, lifecycle, cancellation | EventBus + WS + EventJournal | **БЕЗ ИЗМЕНЕНИЙ** |
| **Telemetry** | mlflow_*, health, idle | EventBus + WS + EventJournal + MLflowRelay→MLflow server | **БЕЗ ИЗМЕНЕНИЙ** |
| **Data (logs)** | trainer stdout/stderr | trainer FileHandler → training.log + bus.publish('trainer_log') → silenced на Mac | Supervisor → trainer.stdio.log file → LogManager scp → Mac |

Этот PR трогает **только** Data plane. Control/Telemetry — orthogonal.

