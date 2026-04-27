# Phase 12 — Metrics never lost

> Status: **12.A.1 ✅ DONE; 12.A.2 + 12.B + 12.C pending**
> Author: daniil + agent
> Date: 2026-04-27
> Worktree: `nice-jepsen-07d789`
> Predecessors: Phase 9.A/B/C (stop semantics), Phase 11.A/B/C/D (sleep + resume)
> Theme: close the **two remaining durability gaps** Phase 11 left open
> Migration policy: NO BACKWARDS COMPATIBILITY (carry-over)
> Out of: § 11.11 placeholder in [`harmonic-rolling-crayon.md`](./harmonic-rolling-crayon.md)
>
> Commits:
> * `ffa0beb` — Phase 12 plan committed (this file).
> * `f05584f` — Phase 12.A.1: metrics buffer retrieval + Mac-side MLflow replay (54 new tests, 491/491 cross-phase regression pass).

---

## 0. TL;DR

Phase 11 решила «pod не удаляется зря» и «Mac на wake поднимает pod
обратно». Но **два класса данных** всё ещё могут пропасть:

| Что | Где живёт | Как пропадает сегодня |
|---|---|---|
| **MLflow buffered metrics** (loss, accuracy, eval_loss) | `/workspace/metrics_buffer.jsonl` на поде | Mac в сне → pod-side flush падает с timeout → buffer остаётся на диске → ModelRetriever грабит **только** `/workspace/output/` → `cleanup_pod` → файл пропал |
| **EventBus UI events** (step progress, GPU history, lifecycle) | RAM пода (deque cap 10k) | Mac в сне > ~5 минут → ring buffer прокрутился → `BufferTruncatedError` → WS close 4410 → Mac fallback на terminal-snapshot → за весь sleep window пустота в UI |

Phase 12 закрывает оба:

| Sub-phase | Closes | Effort | Risk |
|---|---|---|---|
| **12.A** Metrics buffer retrieval + Mac-side replay + config-driven decimation | Gap 1 (MLflow data) | ~8h | low-medium |
| **12.B** EventBus durability via JSONL journal + WS replay | Gap 2 (UI events) | ~10h | medium |
| **12.C** Observability + GC + docs | Polish | ~3h | low |
| **Total** | | **~21h** | |

**SLO targets after Phase 12:**

| Metric | Target |
|---|---|
| MLflow run completeness when Mac asleep всю тренировку | > 99.9% (vs ~50% сегодня для длинных runs) |
| Live UI event continuity across Mac sleep | 100% lossless (replay из disk) |
| `events.jsonl` disk footprint cap per attempt | ≤ 500 MB |
| Disk-replay path overhead vs ring buffer | p95 < 200 ms additional latency |

---

## 0.1 Contracts & responsibility zones (что система гарантирует, а что — нет)

Перед детализацией — фиксируем **явный контракт** что мы покрываем
auto-recovery и что лежит за границей. Mandate from user: «пользователь
может делать что угодно в процессе обучения, но нужна граница когда
мы перестаём предугадывать каждое действие».

### 🟢 Полностью покрываем (auto-recovery без действий пользователя)

| User action / system event | Recovery mechanism | Status |
|---|---|---|
| Mac sleep / wake (lid close+open) | OS freeze process → unfreeze; WS auto-reconnect; events from disk replay | Phase 1, 11, 12.B |
| Network drop (Wi-Fi off/on) | Same as Mac sleep — TCP timeout → reconnect → resume | Phase 1, 12.B |
| MLflow upstream temporarily down | ResilientMLflowTransport circuit breaker → buffer.jsonl → flush on recovery / replay on Mac wake | Phase 9, 11.A, 12.A |
| Pod stop (Phase 11.B podStop) | `/workspace` preserved → resume via PodAvailabilityProbe → ModelRetriever | Phase 11.B/C |
| Long Mac sleep (>5 минут — ring buffer overflow) | Events на disk на поде → WS server transparently replays from disk | Phase 12.B |
| Long Mac sleep + training completion overlap | metrics_buffer.jsonl на поде → ModelRetriever extension fetches + replays into MLflow | Phase 12.A |
| Trainer crash (segfault, OOM) | FSM → failed; PodStopper decision matrix → pod kept alive если KEEP_ON_ERROR (debug) или podStop (Mac в сне) | Phase 9, 11.B |

### 🟡 Покрываем с user-visible action (one click / CLI)

| User action | Recovery mechanism | Phase |
|---|---|---|
| **Закрыли терминал orchestrator'а** | `ryotenkai run resume <run>` + Web UI Resume button → resume pod, continue pipeline | Phase 11.C-2 |
| **Reboot Mac** (orchestrator process killed) | Same: explicit resume command on next launch | Phase 11.C-2 + 13 (improved persist) |
| **Force kill orchestrator** (`kill -9`) | Same as Reboot Mac | Phase 13 |
| **Network volume corrupt + retry** | Restart-from-checkpoint via `ryotenkai run restart --from-stage <stage>` | Phase 11.C-1 (PodAvailability.GONE state) |

### 🔴 НЕ покрываем (out of scope — пользователь сам)

| User action | Reasoning | Workaround |
|---|---|---|
| **Mac data loss** (диск умер, потерянная папка `runs/`) | Mac state — пользователь его обязан backup'ить (как любой dev tool) | Restart-from-checkpoint в новый run_id из MLflow checkpoint URL |
| **Pod hardware failure** (RunPod node crash mid-training) | Outside our control; RunPod responsibility | Restart from latest checkpoint в new pod |
| **MLflow server полностью лост** (DB corrupted etc.) | Outside our scope; user manages MLflow infra | Reseed MLflow run from local checkpoints |
| **`/workspace` volume manually deleted в RunPod console** | User explicit action — мы не реверсим | Restart from scratch |
| **Training config изменён mid-run** | По дизайну resume blocked при config drift (Phase 6 hash check) | Either revert config или start fresh run |
| **Active malicious интерференция** (someone else has SSH access to pod) | Security domain, не application-layer | RunPod security boundary |

### Граница "когда мы перестаём предугадывать действия"

**Принцип**: автоматически восстанавливаемся от **типичных пользовательских
действий** (sleep, network blip, terminal close). Просим **explicit
permission** для **операций с потерей** (restart-from-scratch, delete
run). Не пытаемся реверсить **destructive system events** (data loss,
hardware failure) — пользователь делает new run.

| Действие | Кто решает |
|---|---|
| Resume from sleep | **Auto** |
| Resume from podStop | **Auto** (one click) |
| Resume from terminated pod | User explicit (`run restart --from-stage`) |
| Delete run + artefacts | User explicit (`runs delete`) — already Phase 11 |
| Override config drift | User explicit (revert config или start fresh) |

Пользователь явно заявил: "имею право перезагрузить Mac". Это
покрывается Phase 11.C-2 (CLI resume + Web UI кнопка) + Phase 13
(persistent orchestrator state on Mac, чтобы resume команда
автоматически подхватывала прерванный run без потери hot state).

---

## 1. Problem statement (что именно сломано)

### 1.1 Gap 1 — MLflow metrics never reach MLflow

**Сценарий**: 1.5h training с Mac в сне всё время.

```
t=0    Mac wakes briefly, ryotenkai run start
       trainer стартует, MLflow up, ResilientTransport buffer attached
t=5s   Mac закрывается (lid close)
       SSH tunnel collapses → trainer's MLflow circuit breaker opens →
       все mlflow.log_metric calls идут в metrics_buffer.jsonl (decimated)
t=1.5h trainer hits max_steps → CompletionCallback.on_train_end →
       flush_buffer() с 5s budget → MLflow upstream НЕДОСТУПЕН →
       breaker остаётся open → НИЧЕГО не drains →
       completion.marker написан с flushed_count=0 →
       trainer process exits → FSM=completed →
       PodTerminator: mac_alive=False, persistent → podStop
       Pod state EXITED, /workspace сохранён
t=2h   Mac wakes
       Pod resume через PodAvailabilityProbe + resume_pod_with_retry
       SSH up → ModelRetriever грабит /workspace/output (адаптеры) →
       hf_uploader.download_directory(...) → success
       Mac TrainingMonitor reconciles completion.marker:
         RUNNING → FINISHED (но metrics ZERO)
       cleanup_pod (provider) → pod TERMINATED
       /workspace/metrics_buffer.jsonl GONE → metrics ПОТЕРЯНЫ permanently
```

**Root cause**: `ModelRetriever.download_directory(/workspace/output/)`
не trogает `/workspace/metrics_buffer.jsonl` — он на уровень выше.

### 1.2 Gap 2 — EventBus events truncated на длинном Mac sleep

**Расчёт**: `DEFAULT_BUFFER_SIZE = 10_000` events.

При training rate 50 steps/min (~1-3 events/step) + HealthReporter
(1 event/30s) + lifecycle events ≈ **150 events/min** → ring buffer
fills в **~67 минут**. После — `BufferTruncatedError`, WS close 4410,
JobClient fallback на `_fallback_to_status` (только terminal snapshot),
**весь sleep window — пустота** в UI.

Lifecycle events (cancellation, pod_terminal_decision) тоже могут
пропасть если они fired в окне сна и ring overflowed.

---

## 2. Architectural decisions (с обоснованием)

### 2.1 JSONL writer: in-process, sync append, NO sidecar

**Решение**: writer живёт inside `EventBus.publish()`, sync open/append/close per call.

| Альтернатива | Почему отвергнуто |
|---|---|
| Sidecar process (fluent-bit-style) | YAGNI: добавляет deployment complexity, IPC, ещё одну точку отказа в pod, не ускоряет publish |
| `asyncio.Queue` + background writer task | Не выживает crash: SIGKILL между `publish()` и flush queue → events lost. Тот же класс багов который мы и закрываем |
| Memory-mapped append | Premature optimization. JSONL append ~50µs/event, fsync ~5ms — всё ещё ниже event rate (≤ 30 events/sec average) |
| Async writer в `loop.run_in_executor` | Усложняет ordering + cancellation safety. Сегодня deque.append атомарен — ничего не теряется. Async writer = pending writes пропадают на shutdown |

**Вывод**: simplest possible — sync inline в `publish()`. KISS.

### 2.2 fsync policy: batched (every N events OR T ms)

**Решение**: накапливаем `_unflushed_count`. fsync когда:
- `_unflushed_count >= EVENTS_FSYNC_BATCH` (default 50), OR
- `(now - _last_fsync_ms) >= FSYNC_INTERVAL_MS` (default 1000), OR
- `close()` / lifespan shutdown / explicit `fsync_now()`

| Policy | Throughput | Window of loss on crash | Complexity |
|---|---|---|---|
| fsync per-write | ~200 events/s (HDD) / ~2k/s (SSD) | 0 events | minimal |
| **batched 50/1s** | **~unlimited** | **≤ 50 events ИЛИ ≤ 1s** | +1 timer |
| no fsync (OS flush) | unlimited | up to 30s (OS dirty page) | minimal but unsafe |

**Обоснование**: typical event rate 10-30/s. fsync per-write добавит
5ms × 30 = 150ms/s wasted CPU+IO. Batched: 1 fsync/s amortizes.
1s gap на crash << 5.5min gap от ring overflow сегодня.

### 2.3 Storage layout: `/workspace/.runner/events/`

```
/workspace/
├── .ryotenkai/                    (Phase 6 — FSM state, untouched)
│   ├── state.jsonl
│   └── state.json
├── .runner/                       (NEW Phase 12)
│   └── events/
│       ├── events.000.jsonl       (oldest still-present)
│       ├── events.001.jsonl
│       ├── events.002.jsonl
│       ├── events.003.jsonl
│       └── events.004.jsonl       (current — newest)
├── metrics_buffer.jsonl           (Phase 9 — MLflow buffer)
├── completion.marker              (Phase 11.A)
├── cancelled.marker               (Phase 9.C)
└── output/                        (trainer output, retrieved сегодня)
```

**Обоснование**:
- НЕ `/workspace/.ryotenkai/events/` — `.ryotenkai/` уже занят FSM state с другой life-time
- НЕ `/workspace/events/` (top-level) — collision risk с user training output
- Hidden dir (dot-prefix) — `tar` exclude patterns по умолчанию пропускают, мы explicitly include

### 2.4 Rotation: numeric-suffix, fixed window 5 files

```
events.000.jsonl   100 MB cap → rotate
events.001.jsonl   100 MB cap → rotate
...
events.004.jsonl   current
about to open events.005.jsonl: delete events.000.jsonl FIRST
```

**Расчёт**: 100 MB / ~1 KB per event ≈ 100k events/file × 5 = 500k
events total. При 30 events/s = ~4.6 hours retention. Покрывает
overnight sleep если training НЕ заполнило 500 MB ДО сна.

**Зачем fixed window**: deterministic disk footprint (operator знает
worst case 500 MB), provisioning predictable.

### 2.5 Schema versioning: per-record `v` field

```json
{"v":1,"offset":N,"ts":"...","kind":"...","payload":{...}}
```

На read: `v` отсутствует → treat as v1. Unknown keys ignored
(JSON-permissive). Reject `v > MAX_SUPPORTED` с warning.

**Альтернатива** (file header) отвергнута: усложняет append, файлы
не безопасно мерж-конкатить. Per-record self-describing → diff в
4 bytes/record (negligible).

### 2.6 Disk full policy: **drop oldest + emit `events_disk_pressure`**

| Policy | Pros | Cons |
|---|---|---|
| stop publishing | safe | live UI freezes; breaks Phase 11 mac_alive heartbeat (no events → no WS yields → wrong terminal decision) |
| **drop oldest (rotation)** | preserves recent events | older context lost — но они уже out of ring buffer |
| fail trainer | fail-stop simplest | catastrophic — kills run из-за disk warn |

**Choice**: drop oldest. Trainer NOT failed. `events_disk_pressure`
event emitted (rate-limited 1/min). Если даже это не получается —
stderr only.

### 2.7 Replay merger logic для MLflow (12.A): idempotency by invariant

**Проблема**: pod-side flush мог partially drain ДО Mac sleep. После
sleep retrieval может содержать events которые УЖЕ в MLflow.

**Решение** (KISS): **per-run side-marker file** на pod.

После каждого successful `flush_buffer()` drain ResilientTransport
пишет `/workspace/.runner/buffer.flush_offset` JSON
`{"v":1,"drained_count":N,"drained_at_ms":...}` через
`atomic_write_json`. MetricsBuffer на disk хранит **только entries
которые currently в файле** (flush убирает оттуда — это уже работает
в Phase 9).

На Mac replay:
1. Retrieve `/workspace/metrics_buffer.jsonl` AND `.flush_offset`
2. Read all entries из retrieved JSONL
3. **All of them = "не были flushed"** (по invariant: flush стирает оттуда)
4. For each: `mlflow_client.log_metric(run_id, key, value, step, ts_ms)`
5. MLflow дедуплицирует identical `(key, step, ts_ms)` upstream

**Обоснование**: file сам — ideal source of truth. Если запись в
файле, она НЕ flushed. Никаких сложных id-сетов не нужно.

**Edge case** (race): pod-side flush in progress когда Mac retrieve.
**Mitigation**: retrieval happens **AFTER** pod в EXITED state
(Phase 11.B podStop). Stopped pod = no live processes = no concurrent
writers.

### 2.8 Single-writer assumption holds

EventBus.publish() уже sync, called только из FastAPI single event
loop (Phase 1 mandate). Supervisor's stdout pump cross-thread
использует `loop.call_soon_threadsafe` → serialized.

**Не используем** `fcntl` file lock — overhead на каждый publish без
benefit (single writer guaranteed).

**Multi-process**: один runner process per pod. Restart внутри pod
(lifespan re-init) → новый EventBus читает existing `events/`,
picks max sequence number, продолжает. Race "old + new одновременно"
невозможна (FastAPI lifespan single-threaded shutdown).

### 2.9 EventBus offset reconciliation после crash recovery (NEW — итерация 2)

**Проблема (OQ-9)**: runner restart in mid-attempt. На init нужно:
- Read journal `newest_persisted_offset()` → set `bus._next_offset = persisted + 1`
- Иначе на restart bus начнёт с 0 → duplicate offsets vs disk records

**Решение**: в `EventBus.__init__(journal=...)`:
```python
if journal is not None:
    persisted = journal.newest_persisted_offset()
    if persisted is not None:
        self._next_offset = persisted + 1
```

Test: spawn EventBus с непустым journal → assert next publish offset = persisted + 1.

### 2.10 UTF-8 + non-serializable payloads (NEW — итерация 2)

**Решение**:
```python
json.dumps(record, separators=(",", ":"), ensure_ascii=False, default=str)
```

- `ensure_ascii=False` — кириллические сообщения (например, error от
  пользователя на русском) пишутся compact UTF-8 а не `\uXXXX`
- `default=str` — `datetime`, `Path`, `Enum` объекты coerce-ятся к
  string, не падают

Файл открывается binary append (`open(path, "ab")`) + ручной encode.
Mirrors what `metrics_buffer.py:103` уже делает (`separators=(",",":")
` без `ensure_ascii` — мы делаем lock-step но более defensive).

---

## 3. Phase 12.A — Metrics buffer retrieval + Mac-side replay (~6h, low-medium risk) ✅ DONE

> Commit `f05584f` — 13 files changed, 2715 insertions; 54 new tests
> (16 replay + 15 retriever + 13 wire-up + 10 marker), all 7-cat
> covered. Drive-by fix for pre-existing fragile `time.time` mock in
> the upstream HF-upload-skip test. 200/200 phase-touching tests
> green; 491/491 across slim-venv compatible Phase 9/11/12 surfaces.

### 3.1 Goal

После natural completion + Mac wake + ssh resume: **полностью** перенести
`metrics_buffer.jsonl` с пода на Mac, replay в локально-сконфигурированный
MLflow run **прежде чем** provider `cleanup_pod` удалит volume.

### 3.2 Components

#### NEW: `src/pipeline/stages/model_retriever/metrics_replay.py`

```python
"""Phase 12.A — replay buffered MLflow metrics from a retrieved
metrics_buffer.jsonl into the same MLflow run that was active during
training.

Idempotency by invariant: the buffer file ONLY contains un-flushed
records (MetricsBuffer.flush() removes drained entries). So replay is
"ship every record exactly once" — no dedup logic needed.
"""

@dataclass(frozen=True)
class ReplayResult:
    replayed: int
    failed: int
    first_step: int
    last_step: int
    duration_ms: int
    errors: list[str]  # capped at 10

class BufferedMetricsReplay:
    def __init__(self, mlflow_client: MlflowClient): ...
    def replay(self, *, buffer_path: Path, run_id: str) -> ReplayResult: ...
        # 1. Read all JSONL entries
        # 2. Sort by (step, timestamp) — same invariant as MetricsBuffer.flush
        # 3. For each: client.log_metric(run_id, key, value, step, ts_ms)
        # 4. Return ReplayResult; buffer file left in place — caller decides
```

#### NEW: `src/pipeline/stages/model_retriever/metrics_buffer_retriever.py`

```python
"""Phase 12.A — retrieve metrics_buffer.jsonl from pod over SSH."""

class MetricsBufferRetriever:
    REMOTE_BUFFER_PATH: ClassVar[str] = "/workspace/metrics_buffer.jsonl"
    REMOTE_FLUSH_OFFSET_PATH: ClassVar[str] = "/workspace/.runner/buffer.flush_offset"
    MAX_BUFFER_SIZE_BYTES: ClassVar[int] = 100 * 1024 * 1024  # 100 MB safety cap

    def __init__(self, ssh_client: SSHClient, workspace_path: str = "/workspace"): ...

    def fetch(self, *, local_dir: Path) -> FetchResult:
        # 1. ssh `test -f {remote}` → missing → FetchResult(missing=True)
        # 2. ssh `stat -c %s {remote}` for size; if > MAX_BUFFER_SIZE → warn + skip
        # 3. download_file → local_dir/metrics_buffer.jsonl
        # 4. Optionally fetch flush_offset marker (best-effort)
        # 5. Return FetchResult(local_path, size_bytes, line_count)
```

#### EXTEND: `src/utils/ssh_client.py`

Add `download_file(remote_path, local_path) -> Result[None, ProviderError]`
single-file `scp`-style download. Today only `download_directory`.

#### EXTEND: `src/pipeline/stages/model_retriever/retriever.py`

```python
def _execute_retrieval(self, context):
    # ... existing HF upload + local download ...

    # Phase 12.A — best-effort metrics buffer retrieval + replay.
    # Fails open: warning logs, doesn't block stage return.
    if not self.mock_mode:
        self._retrieve_and_replay_metrics_buffer(context)

def _retrieve_and_replay_metrics_buffer(self, context):
    attempt_dir = Path(context[PipelineContextKeys.ATTEMPT_DIRECTORY])
    run_id = context.get("mlflow_run_id") or _from_training_stage(context)

    if not run_id:
        logger.info("[METRICS_REPLAY] no mlflow_run_id; skipping")
        return

    retriever = MetricsBufferRetriever(self._ssh_client, self._workspace_path)
    fetch = retriever.fetch(local_dir=attempt_dir)

    if fetch.missing:
        logger.info("[METRICS_REPLAY] buffer absent on pod — flush succeeded")
        if self._callbacks.on_metrics_buffer_retrieved:
            self._callbacks.on_metrics_buffer_retrieved(replayed=0, missing=True)
        return

    from mlflow.tracking import MlflowClient
    replayer = BufferedMetricsReplay(MlflowClient())
    result = replayer.replay(buffer_path=fetch.local_path, run_id=run_id)

    if self._callbacks.on_metrics_buffer_retrieved:
        self._callbacks.on_metrics_buffer_retrieved(
            replayed=result.replayed, line_count=fetch.line_count,
            size_bytes=fetch.size_bytes, missing=False,
        )

    logger.info(
        "[METRICS_REPLAY] replayed %d/%d metrics (run=%s, steps %d→%d)",
        result.replayed, fetch.line_count, run_id,
        result.first_step, result.last_step,
    )
```

#### EXTEND: `ModelRetrieverEventCallbacks` — add `on_metrics_buffer_retrieved`

#### EXTEND: `ResilientMLflowTransport` — write `.flush_offset` marker after each successful drain (для retrieval correctness — § 2.7 invariant)

### 3.3 Tests (7-cat) — `test_metrics_buffer_retriever.py` + `test_metrics_replay.py`

| Cat | Cases |
|---|---|
| **Positive** | fetch existing file → correct size+line_count; replay 100 entries → 100 log_metric calls с правильным (key, value, step, ts) |
| **Negative** | fetch missing → FetchResult(missing=True), no exception; SSH fails → ProviderError; malformed JSONL line → skipped + error captured; mlflow.log_metric raises mid-stream → continues, errors capped at 10 |
| **Boundary** | empty buffer → replayed=0; 1 entry → replayed=1; 10k entries → all replayed, perf < 5s; mixed step ordering → sorted correctly |
| **Invariants** | replay order monotonic step at MLflow side; fetch never overwrites без warning; idempotent — running twice = no-op (buffer moved out of pod after first) |
| **Dependency-errors** | SSH client raises → ProviderError surfaced; MLflow client raises → каждый exception captured |
| **Regressions** | Old buffer format без `step` field → step defaults to 0; `mock_mode=True` → replay skipped entirely |
| **Logic-specific** | pod-side flush ran partially before Mac sleep → buffer = un-flushed only → no overlap with MLflow upstream; HF upload fails + replay still happens (independent paths); replay on terminal MLflow run — assert handle |

### 3.4 Integration test — `test_model_retriever_metrics_replay.py`

Simulate full sleep-and-resume:
1. Build `MetricsBuffer` с 50 entries (decimated, mixed steps)
2. Boot fake MLflow tracking (pytest-mlflow fixture)
3. Mock SSH client returning the buffer file content
4. Run `ModelRetriever._execute_retrieval` (mock HF upload)
5. Assert: 50 metrics в MLflow run; buffer file в `attempts/<n>/`; callback fired

### 3.5 Risk + mitigation (12.A)

| Risk | Severity | Mitigation |
|---|---|---|
| MLflow run_id not in context | High | Defensive: warning + no-op (current behaviour без 12.A — same outcome). Unit test exercises missing run_id |
| Replay duplicates if MLflow doesn't dedup | Medium | KISS: invariant — buffer = un-flushed only |
| SSH fetch fails (pod уже terminated by user) | Low | Best-effort: warning, no exception, continue |
| Replay timing — pod cleanup races ahead | Medium | Replay блокирует stage return; cleanup_pod is provider-side AFTER stage return |
| Mac disk full retrieving large buffer | Low | `MAX_BUFFER_SIZE_BYTES = 100 MB` cap; warn + skip if remote larger |

### 3.6 Estimate: 6h (3h impl + 2h tests + 1h smoke)

---

## 3.A — Phase 12.A.2: config-driven metrics decimation (~2h, low risk)

### 3.A.1 Mandate (от user)

> "Это можно вынести в конфиг, в блок training, там дальше сам
> продумай как это оформить понятно в конфиге, соблюдая общую
> стилистику. И нужно это понятно оформить. Не забудь
> актуализировать описание конфига. По умолчанию давай сделаем без
> ограничений, пускай каждую метрики сохраняем."

### 3.A.2 Что сейчас (Phase 9 hard-coded policy)

`src/training/mlflow/metrics_buffer.py` использует **hard-coded
decimation policy**:

```python
# Phase 9 - hardcoded
if elapsed_minutes < 10:
    keep_every = 1   # all
elif elapsed_minutes < 30:
    keep_every = 2   # every 2nd step
else:
    keep_every = 5   # every 5th step
```

Это OK для коротких runs, но:
- На очень коротких (5 мин) runs decimation не нужен (всё помещается)
- На очень длинных (24h) runs можно decimate агрессивнее
- Пользователь не имеет контроля

### 3.A.3 Решение

**Вынести в config + default keep_all=true**.

#### Config schema (`src/config/training/metrics_buffer.py` — NEW Pydantic model):

```yaml
# В training section pipeline_config.yaml
training:
  # ... existing fields ...
  metrics_buffer:
    # Phase 12.A.2 — controls how MetricsBuffer decimates records
    # when MLflow upstream is unreachable. Buffer is drained on
    # natural completion (Phase 11.A) OR replayed on Mac wake
    # (Phase 12.A). Decimation reduces disk + network volume but
    # loses precision in metric history.
    keep_all: true                  # Default: keep every metric (loseless)
    # Below kicks in only when keep_all=false. Three windows by
    # elapsed training time, each with its own keep_every-N step.
    # Example default below mirrors Phase 9 hard-coded policy
    # (active ONLY when keep_all=false):
    decimation:
      window_first_minutes: 10      # 0-10 min: keep every step
      window_first_keep_every: 1
      window_mid_minutes: 30        # 10-30 min: keep every Nth step
      window_mid_keep_every: 2
      # >30 min: keep every Nth step
      window_late_keep_every: 5
```

#### Pydantic model

```python
class MetricsBufferConfig(BaseModel):
    """Phase 12.A.2 — controls MetricsBuffer decimation policy.

    By default, ``keep_all=True`` — every metric is preserved
    losslessly. Set to False to enable time-windowed decimation
    for very long runs where disk / replay overhead matters more
    than per-step precision.
    """
    keep_all: bool = Field(
        default=True,
        description=(
            "If True (default), keep every metric — no decimation. "
            "Set False for adaptive decimation by training duration."
        ),
    )
    decimation: DecimationWindowConfig = Field(
        default_factory=DecimationWindowConfig,
        description=(
            "Decimation parameters (active only when keep_all=False). "
            "Three time-windowed thresholds with per-window keep_every-N."
        ),
    )


class DecimationWindowConfig(BaseModel):
    window_first_minutes: int = Field(default=10, ge=1)
    window_first_keep_every: int = Field(default=1, ge=1)
    window_mid_minutes: int = Field(default=30, ge=1)
    window_mid_keep_every: int = Field(default=2, ge=1)
    window_late_keep_every: int = Field(default=5, ge=1)
```

#### Changes:
- `src/config/training/__init__.py` — export `MetricsBufferConfig`
- `src/config/training/main.py` (or where TrainingConfig lives) — add
  `metrics_buffer: MetricsBufferConfig = Field(default_factory=MetricsBufferConfig)`
- `src/training/mlflow/metrics_buffer.py:_should_keep_step()` — replace
  hard-coded thresholds with `if self._config.keep_all: return True`
  → else windowed lookup
- `MetricsBuffer.__init__` accepts `MetricsBufferConfig` (default
  factory; backwards-compat: existing callers don't break since
  keep_all=True changes behaviour to "more permissive")
- `src/training/managers/mlflow_manager/manager.py` — pass config
  through to MetricsBuffer

### 3.A.4 Documentation update

- `docs/configuration.md` (or equivalent project config docs):
  - New section "Metrics buffer" под "Training config"
  - Объясняет когда decimate имеет смысл (long runs > 1h, низкий
    interest в per-step precision); когда нет (short runs, debug runs)
  - Default = keep_all=true с обоснованием user mandate

### 3.A.5 Tests (`test_metrics_buffer_decimation_config.py`)

| Cat | Cases |
|---|---|
| Positive | keep_all=true → all entries retained; keep_all=false → time windows applied per config |
| Negative | invalid config (negative minutes) → Pydantic ValidationError на load |
| Boundary | exactly window_first_minutes elapsed → switches to mid window; window_mid_keep_every=1 → keeps all in that window (effectively keep_all but slower) |
| Invariants | keep_all=true ALWAYS returns True regardless of elapsed time |
| Dependency-errors | config not provided → uses default factory (keep_all=true, all metrics) |
| Regressions | Old runs without metrics_buffer config field → defaults applied → behaves identically to keep_all=true (more permissive than Phase 9 hard-coded) |
| Logic-specific | switching keep_all=false → keeps original step counts; back to keep_all=true via config edit (user has to restart run) |

### 3.A.6 Migration policy

Phase 9 had hard-coded decimation; switch к keep_all=true changes
behaviour for **existing configs**. This is a one-shot widening of
default behaviour:
- Old runs replayed: behave as before (file already on disk, decimation
  already happened)
- New runs: keep_all=true unless user opts in via config

User mandate explicitly requested keep_all=true default — no
backwards-compat issue per project policy.

### 3.A.7 Estimate: 2h (45min config + 30min wire-through + 30min tests + 15min docs)

---

## 4. Phase 12.B — EventBus durability + WS replay (~10h, medium risk)

### 4.1 Goal

`EventBus.publish()` persists each event. WS replay серверит **либо**
ring buffer (fast path), **либо** disk file scan (когда `since` <
oldest in ring). JobClient consumes obeying same iteration contract —
disk replay **transparent**.

### 4.2 NEW: `src/runner/event_journal.py`

Self-contained file IO module. NO async — pure sync/threadable.

```python
"""Phase 12.B — durable on-disk JSONL journal for EventBus.

Append-only sequence of rotated files under
`<workspace>/.runner/events/events.NNN.jsonl`.

Capacity:
    - file size cap: 100 MB → rotate
    - total file count cap: 5 → on rotate beyond, drop oldest

Schema (per line):
    {"v":1,"offset":N,"ts":"...","kind":"...","payload":{...}}

Single-writer. fsync batched: every 50 writes OR 1s elapsed.
Plus forced fsync on close().
"""

EVENTS_DIR_REL = ".runner/events"
EVENTS_FILE_FMT = "events.{seq:03d}.jsonl"
DEFAULT_FILE_SIZE_CAP = 100 * 1024 * 1024
DEFAULT_MAX_FILES = 5
DEFAULT_FSYNC_BATCH = 50
DEFAULT_FSYNC_INTERVAL_MS = 1000
SCHEMA_VERSION = 1

@dataclass(frozen=True)
class JournalRecord:
    v: int
    offset: int
    ts: str
    kind: str
    payload: dict

class EventJournal:
    def __init__(self, *, root_dir, file_size_cap=..., max_files=...,
                 fsync_batch=..., fsync_interval_ms=...): ...

    def append(self, *, offset, ts, kind, payload) -> None: ...
    def fsync_now(self) -> None: ...
    def close(self) -> None: ...
    def iter_records(self, *, since: int) -> Iterator[JournalRecord]: ...
    def oldest_persisted_offset(self) -> int | None: ...
    def newest_persisted_offset(self) -> int | None: ...
    def total_bytes(self) -> int: ...
    def file_count(self) -> int: ...
    def reset(self) -> None: ...  # GC, called on FSM final-final
```

#### Critical: `append()` body sketch

```python
def append(self, *, offset, ts, kind, payload):
    record = {"v": SCHEMA_VERSION, "offset": offset, "ts": ts,
              "kind": kind, "payload": payload}
    line = json.dumps(record, separators=(",", ":"),
                      ensure_ascii=False, default=str) + "\n"
    line_bytes = line.encode("utf-8")

    if self._current_size + len(line_bytes) > self._file_size_cap:
        self._rotate()  # close current, open events.NNN+1.jsonl,
                        # delete oldest if file_count > max_files

    self._current_fh.write(line_bytes)
    self._current_size += len(line_bytes)
    self._unflushed_count += 1

    now_ms = int(time.monotonic() * 1000)
    if (self._unflushed_count >= self._fsync_batch
        or now_ms - self._last_fsync_ms >= self._fsync_interval_ms):
        self._fsync_now_locked()
```

#### Crash recovery on init

`__init__` walks `self._root_dir`:
- Empty dir → start seq=0, open `events.000.jsonl` append mode
- Existing files → pick `max(seq)` → resume appending. Inspect file
  size; if `> file_size_cap` (partial rotate before crash) → force
  rotate next append
- Compute `_oldest_seq`, `_newest_seq` from listing
- **Truncated last line** (write interrupted mid-record) → tail
  truncated on read in `iter_records`

### 4.3 Modify: `src/runner/event_bus.py`

```python
def __init__(self, capacity=None, *, journal: EventJournal | None = None):
    ...
    self._journal = journal
    # Phase 12.B § 2.9 — offset reconciliation after crash
    if journal is not None:
        persisted = journal.newest_persisted_offset()
        if persisted is not None:
            self._next_offset = persisted + 1

def publish(self, kind, payload, *, timestamp=None) -> Event:
    ...
    event = Event(...)
    self._buffer.append(event)
    self._next_offset += 1

    # Phase 12.B — durable persistence
    if self._journal is not None:
        try:
            self._journal.append(
                offset=event.offset, ts=event.timestamp,
                kind=event.kind, payload=dict(event.payload),
            )
        except Exception as exc:
            self._signal_disk_pressure(exc)  # rate-limited 1/60s

    # ... existing wakeup signal logic
    return event
```

`close()` calls `journal.close()` if set.

### 4.4 Modify: `src/runner/main.py` lifespan

```python
workspace = _resolve_workspace()
events_dir = workspace / ".runner" / "events"
events_dir.mkdir(parents=True, exist_ok=True)
journal = EventJournal(root_dir=events_dir)
bus = EventBus(journal=journal)
app.state.journal = journal
```

### 4.5 Modify: `src/runner/api/events.py` — disk replay fallback

```python
async def _subscribe_with_disk_fallback(
    bus: EventBus, journal: EventJournal | None, since: int,
) -> AsyncIterator[Event]:
    """Yield events from disk first (if since older than ring),
    then continue from ring. Transparent to caller.
    """
    oldest_in_ring = bus.oldest_offset
    if (oldest_in_ring is not None and since < oldest_in_ring
        and journal is not None):
        oldest_on_disk = journal.oldest_persisted_offset()
        if oldest_on_disk is None or since < oldest_on_disk:
            raise DiskJournalExhausted(
                requested=since,
                oldest_in_ring=oldest_in_ring,
                oldest_on_disk=oldest_on_disk,
            )
        # Disk replay phase
        for record in journal.iter_records(since=since):
            if record.offset >= oldest_in_ring:
                break
            yield Event(offset=record.offset, timestamp=record.ts,
                        kind=record.kind, payload=record.payload)
        # Hand off to ring (subscribe at exactly oldest — no overlap)
        async for event in bus.subscribe(since=oldest_in_ring):
            yield event
    else:
        # Fast path
        async for event in bus.subscribe(since=since):
            yield event
```

`DiskJournalExhausted` — new exception class. WS handler catches it
+ closes 4410 (close code unchanged, semantic clearer).

### 4.6 Mac side changes

**`JobClient.subscribe_events`**: NO behavior change — disk-replay
path is server-side opaque. `current_since` cursor advances correctly
through disk records (offset field uniform across ring + disk).

**`TrainingMonitor._fallback_to_status`**: today fires on
`ReplayTruncatedError`. After 12.B, this fallback fires ONLY on
`DiskJournalExhausted` (truly impossible cursor). Verify catch site.

### 4.7 Tests (7-cat) — `test_event_journal.py` + extend `test_event_bus.py`

| Cat | Cases |
|---|---|
| **Positive** | append → record persists; readback via iter_records matches; rotation: 101 MB written → 2 files; batched fsync: 50 events → 1 fsync call |
| **Negative** | corrupted JSON line mid-file → iter skips + warn, continues; partial last line → tail truncated; disk full simulation → events_disk_pressure published, publish() doesn't raise |
| **Boundary** | empty journal → oldest=None; exactly 100 MB → rotates на 100,000,001-th byte; max_files=5 + write file 6 → file 0 deleted, sequence numbers continue (NOT recycled) |
| **Invariants** | offset monotonic across files (file N's max < file N+1's min); iter_records strictly increasing offsets; close() always fsync's; oldest ≤ newest |
| **Dependency-errors** | readonly fs → publish doesn't raise; init on permission denied → fail-fast on boot |
| **Regressions** | Old EventBus tests pass with `journal=None`; bus.subscribe still raises BufferTruncatedError when no journal |
| **Logic-specific** | WS subscribe since=0 + ring rolled + journal full → all events in order, no gaps; subscribe beyond ring AND beyond journal → DiskJournalExhausted → 4410; crash recovery: kill mid-write, re-init → max seq resumed без data corruption; fsync interval timer: 49 events slow + 1.1s elapsed → fsync triggered by interval not count |

### 4.8 Integration test — `test_event_bus_disk_replay.py`

```python
# 1. Boot real EventBus + EventJournal в tmp dir
# 2. Publish 15,000 events (overflow ring 10k)
# 3. Subscribe with since=0 over actual WebSocket (TestClient)
# 4. Assert: receive ALL 15,000 events in order
# 5. Verify on disk: 1+ rotation happened, total bytes < 500 MB cap
```

### 4.9 Risk + mitigation (12.B)

| Risk | Severity | Mitigation |
|---|---|---|
| Disk full mid-write — current event lost | Medium | events_disk_pressure published; ring still has it; resume after pressure clears |
| Schema evolution breaks readers | Low | Per-record `v`; readers ignore unknown keys; reject `v > MAX_SUPPORTED` with warning |
| Concurrent runner restart reads partial last file | Medium | __init__ scans max_seq, validates last file size; partial last line truncated on read |
| fsync latency spikes affect publish | Low | Batched 1s/50 events; sync write ~50µs; only affects 1 event/sec at most |
| Replay duplicates: since=N matches both disk + ring on overlap | Medium | _subscribe_with_disk_fallback explicitly stops at ring oldest; subscribe at exactly that offset |
| Numeric overflow on sequence: 1000s rotations | Low | 5-file cap → never need > 999. Document; bump format on rollover |
| Pod disk lifecycle vs Mac retrieval | Medium | Same as 12.A: pod EXITED при retrieval, no concurrent writers |
| **WS reader vs rotation race (NEW R-16)** | Low | iter_records reads complete files (oldest_persisted_offset до ring_oldest); current file rotation doesn't intersect with reader's file set |
| **bus.publish() sync overhead (NEW R-17)** | Low | ~50µs per event measured; batched fsync amortizes; ≤1% throughput impact at typical rate |
| **Permissions on .runner/events init (NEW R-15)** | Low | mkdir(mode=0o755, exist_ok=True); on PermissionError → fail-fast on boot with clear error |

### 4.10 Estimate: 10h (3h journal module + 2h EventBus integration + 3h tests + 1h integration + 1h observability)

---

## 5. Phase 12.C — Observability + GC + docs (~3h, low risk)

### 5.1 New event kinds

Add to `src/runner/cancellation_telemetry.py` (or split into
`src/runner/durability_telemetry.py` if grows; same naming convention):

| Kind constant | Emitted by | Carries | Purpose |
|---|---|---|---|
| `EVENTS_DISK_PRESSURE` = `"events_disk_pressure"` | EventJournal via EventBus | `error_code`, `total_bytes`, `file_count`, `last_offset` | Operator-visible disk pressure |
| `EVENTS_ROTATED` = `"events_rotated"` | EventJournal | `from_seq`, `to_seq`, `file_size_bytes`, `oldest_remaining_seq` | File rotation telemetry |
| `EVENTS_GC_RAN` = `"events_gc_ran"` | EventJournal | `deleted_seqs`, `deleted_bytes` | Auto deletion |
| `METRICS_BUFFER_RETRIEVED` = `"metrics_buffer_retrieved"` | ModelRetriever (Mac) | `replayed`, `line_count`, `size_bytes`, `missing` | 12.A retrieval outcome |

Add `DURABILITY_EVENT_KINDS` frozenset for dashboard wiring (separate
from cancellation telemetry — semantic split).

### 5.2 Background GC — runner side

Lightweight task in `src/runner/main.py` lifespan, every 60s:

```python
async def _periodic_journal_health_check(journal, bus, *, interval_s=60):
    while True:
        try:
            total = journal.total_bytes()
            files = journal.file_count()
            threshold = int(0.9 * DEFAULT_FILE_SIZE_CAP * DEFAULT_MAX_FILES)  # 450 MB
            if total > threshold:
                bus.publish(EVENTS_DISK_PRESSURE, {
                    "total_bytes": total, "file_count": files,
                    "threshold_bytes": threshold,
                })
        except Exception as exc:
            logger.warning("journal health check failed: %s", exc)
        await asyncio.sleep(interval_s)
```

**Crash recovery (init-time GC)**: in `EventJournal.__init__`, scan
workdir for `events.*.jsonl.tmp` (interrupted writes). Delete them.
Log via `EVENTS_GC_RAN`.

### 5.3 Docs update — new section в `docs/runner-architecture.md`

```markdown
## Durability semantics (Phase 12)

The runner persists two streams to pod disk so a Mac sleep window
longer than the ring-buffer's capacity (~5.5 min at typical event
rate) doesn't lose data.

### EventBus journal — `<workspace>/.runner/events/`

Append-only JSONL files, rotated at 100 MB, capped at 5 files (500 MB
total). Schema versioned per record (`v=1` today). On crash recovery,
runner picks up highest-numbered file and continues appending. fsync
batched: every 50 events OR 1s, whichever first.

WebSocket subscribers transparently get disk replay when their
`since=N` cursor is older than the in-memory ring's oldest offset
but still present on disk. The 4410 close code now fires only when
the requested offset is older than the oldest persisted record (a
truly impossible cursor).

### MLflow metrics buffer — `/workspace/metrics_buffer.jsonl`

Trainer's resilient transport buffers metric calls when the MLflow
upstream is unreachable. On natural completion the
CompletionCallback drains the buffer (5s budget). If MLflow is still
asleep at that point — which is the entire reason we buffered in the
first place — the buffer file remains.

After Mac wake + pod resume, ModelRetriever fetches the buffer file
alongside the model artefacts and replays each record into the
**same MLflow run_id** that was active during training. The buffer
file only contains records the trainer never managed to flush, so
replay is safe-by-construction (no dedup required).

### Storage layout summary

| Path | Owner | Lifecycle | Retrieved by |
|---|---|---|---|
| `/workspace/.ryotenkai/state.jsonl` | FSM | append-only, never GC'd | n/a (Mac doesn't read) |
| `/workspace/.runner/events/events.NNN.jsonl` | EventBus journal | rotation cap 500 MB | n/a (replayed via WS only) |
| `/workspace/metrics_buffer.jsonl` | Resilient MLflow transport | drained on flush | ModelRetriever (Phase 12.A) |
| `/workspace/.runner/buffer.flush_offset` | Resilient transport | last successful drain marker | (informational only) |
| `/workspace/output/` | Trainer | grows during training | ModelRetriever (existing) |
| `/workspace/completion.marker` | CompletionCallback | one-shot per attempt | TrainingMonitor (Phase 11.A) |
| `/workspace/cancelled.marker` | CancellationCallback | one-shot per attempt | TrainingMonitor (Phase 9.C) |
```

### 5.4 Configuration knobs

| Env var | Where | What |
|---|---|---|
| `RYOTENKAI_EVENT_JOURNAL_FILE_CAP_BYTES` | Pod | Override 100 MB per-file cap |
| `RYOTENKAI_EVENT_JOURNAL_MAX_FILES` | Pod | Override 5-file count cap |
| `RYOTENKAI_EVENT_JOURNAL_FSYNC_BATCH` | Pod | Override 50-event fsync batch |
| `RYOTENKAI_EVENT_JOURNAL_FSYNC_INTERVAL_MS` | Pod | Override 1000 ms fsync interval |
| `RYOTENKAI_METRICS_REPLAY_DISABLE` | Mac | Set 1 to skip retrieval+replay (debug) |

### 5.5 Tests

- 1 integration test `test_runner_durability.py`: combine 12.A + 12.B
  end-to-end on simulated Mac sleep
- 3 unit tests for telemetry constants (kind strings stable)

### 5.6 Estimate: 3h (1h kinds+GC + 1h docs + 1h reconciliation)

---

## 6. Cross-phase risk register (full)

| ID | Description | Severity | Phase | Mitigation |
|---|---|---|---|---|
| R-1 | JSONL append + concurrent publishers race | Medium | 12.B | Single-writer assumption (FastAPI single loop); no fcntl needed; documented |
| R-2 | fsync latency on each publish costs > 5% throughput | Low | 12.B | Batched 50/1s; perf assertion (10k events < 5s) |
| R-3 | Mac asleep > 4.6h → even disk journal rolls over | Medium | 12.B | Acceptable: target overnight sleep BEFORE training fills 500 MB; document caveat |
| R-4 | Replay duplicates (12.A) | Low | 12.A | MetricsBuffer.flush invariant (only un-flushed in file) |
| R-5 | Schema evolution breaks readers | Low | 12.B/C | Per-record `v`; ignore unknowns; reject `v > MAX_SUPPORTED` |
| R-6 | Disk full = trainer fails | High → low | 12.B | Drop-oldest rotation; trainer NOT affected; events_disk_pressure |
| R-7 | Backwards compat: Phase 11 already shipped | Medium | 12.B | New `journal=None` parameter is opt-in; existing tests don't break |
| R-8 | Pod cleanup races ahead of retrieval | Medium | 12.A | Stage blocks until both download + replay complete; cleanup_pod after |
| R-9 | Multiple WS subscribers → file IO contention | Low | 12.B | Each gets own RO file handle; no contention |
| R-10 | Empty/corrupt buffer file on retrieval | Low | 12.A | Negative test cat; ReplayResult(replayed=0); file kept locally |
| R-11 | mlflow_client(run_id) on terminal run | Medium | 12.A | Mock-test both paths; if upstream rejects log_metric on terminal → log warn, file kept; user manually reopens run |
| R-12 | Mac disk full retrieving large buffer | Low | 12.A | `MAX_BUFFER_SIZE_BYTES = 100 MB` cap; warn + skip |
| **R-13 NEW** | Schema mismatch ResilientMLflowTransport ↔ BufferedMetricsReplay | Medium | 12.A | Pin schema version в metrics_buffer.jsonl format; replay falls back gracefully on unknown fields |
| **R-14 NEW** | mlflow_run_id=None (no tracking server configured) | Low | 12.A | replay no-op + buffer file copied to attempts/<n>/ for forensics |
| **R-15 NEW** | Permissions on `.runner/events/` mkdir | Low | 12.B | mkdir(mode=0o755, exist_ok=True); fail-fast on boot if denied |
| **R-16 NEW** | WS reader vs rotation race | Low | 12.B | iter_records reads complete files only; current rotation doesn't overlap reader's set |
| **R-17 NEW** | bus.publish() sync overhead | Low | 12.B | ~50µs/event; batched fsync amortizes; ≤1% impact at typical rate |
| **R-18 NEW** | iter_records linear scan для 100MB файла на slow consumer | Low | 12.B | Acceptable 5-10s on disk replay; future: index file `.idx` per 1000 records (premature optimization) |
| **R-19 NEW** | Slow WS consumer backpressure → server memory growth | Low | 12.B | Iterator-based (one event at a time); send_json yields; verified — no buffer growth |

---

## 7. Open questions resolved (3 итерации deepsink)

| OQ | Question | Resolution |
|---|---|---|
| OQ-1 | fsync interval — per-write или batched? | **Batched**: 50 events OR 1s (§ 2.2) |
| OQ-2 | Schema versioning approach | **Per-record `v` field** (§ 2.5) |
| OQ-3 | Когда удалять events.jsonl после run completion? | **Never automatically while pod alive**. Pod terminate (provider cleanup_pod) deletes volume entirely. Avoids race с Mac retrieval (§ 4.7) |
| OQ-4 | Storage layout root | `/workspace/.runner/events/` (§ 2.3) |
| OQ-5 | Disk full policy | **Drop oldest + emit event**. Trainer NEVER fails (§ 2.6) |
| OQ-6 | Replay merger logic для overlap | **No overlap by invariant** — flush убирает entries из buffer file (§ 2.7) |
| OQ-7 | `RYOTENKAI_EVENT_BUFFER_SIZE` всё ещё нужен? | YES — ring buffer = fast path; disk = fallback only. Knob unchanged |
| OQ-8 | DiskJournalExhausted vs BufferTruncatedError | **DiskJournalExhausted** new (semantic — disk fallback failed too); BufferTruncatedError остаётся для ring-only callers; WS handler emits DiskJournalExhausted |
| **OQ-9 NEW** | EventBus offset reconciliation после crash | **`_next_offset = journal.newest_persisted_offset() + 1`** в `__init__` (§ 2.9) |
| **OQ-10 NEW** | UTF-8 encoding с не-ASCII payloads | **`ensure_ascii=False`** + binary append (§ 2.10) |
| **OQ-11 NEW** | Non-JSON-serializable payloads (datetime, Path, etc.) | **`default=str`** в json.dumps coerce-ит к string (§ 2.10) |

---

## 8. Verification (end-to-end happy path)

```bash
# 1. Build new image
./docker/training/build_and_push.sh --bump minor

# 2. Start a long run config (90+ minutes)
ryotenkai run start config/sapo_helixql_long.yaml

# 3. Close laptop lid после 30 секунд (simulate sleep ВСЁ training)

# 4. Wait для natural completion (~90 min)

# 5. Open laptop, check Web UI
# Expected:
#   - run shows "stopped" badge + "Resume" button
#   - Live tab events: ВСЕ events of training включая весь sleep window
#     (because of disk replay через WS)

# 6. Click Resume (or `ryotenkai run resume <run_id>`)
# Expected order:
#   - Pod resumes (Phase 11)
#   - SSH up, ModelRetriever fires
#   - download model adapters from /workspace/output → success
#   - download metrics_buffer.jsonl from /workspace → ~500 KB
#   - BufferedMetricsReplay → "[METRICS_REPLAY] replayed 487/487 metrics"
#   - HF upload completes
#   - completion.marker reconciliation → MLflow RUNNING → FINISHED
#   - cleanup_pod via provider → pod terminated

# 7. Verify
test ! "$(ryotenkai-cli pod-list | grep <pod_id>)"  # pod gone
mlflow ui  # run status = FINISHED, metrics history covers WHOLE training
ls attempts/<n>/                                     # adapter present
ls attempts/<n>/metrics_buffer.jsonl                 # local copy preserved
grep events_rotated pipeline.log                     # rotation observed
```

---

## 9. Critical files (для имплементации)

**NEW**:
- `src/runner/event_journal.py` — full impl: append, rotate, GC, iter_records
- `src/pipeline/stages/model_retriever/metrics_buffer_retriever.py` — SSH fetch
- `src/pipeline/stages/model_retriever/metrics_replay.py` — replay JSONL → MLflow

**EXTEND**:
- `src/runner/event_bus.py` — accept journal в __init__, hook publish
- `src/runner/api/events.py` — disk-replay fallback в WS handler; DiskJournalExhausted
- `src/runner/main.py` — wire journal в lifespan + periodic health check
- `src/pipeline/stages/model_retriever/retriever.py` — wire metrics replay after upload
- `src/utils/ssh_client.py` — `download_file` single-file helper
- `src/training/mlflow/resilient_transport.py` — write `.flush_offset` marker after drain
- `src/runner/cancellation_telemetry.py` — durability event kinds (or split into `durability_telemetry.py`)
- `docs/runner-architecture.md` — Durability semantics section

---

## 10. Effort estimate

| Sub-phase | Effort | Risk | Tests | Manual smoke |
|---|---|---|---|---|
| 12.A — Metrics retrieval+replay | 6h | low-medium | 7 unit + 1 integration | 5 min mock |
| 12.B — EventBus durability | 10h | medium | 7 unit + extend bus tests + 1 integration | 15 min real RunPod |
| 12.C — Observability + GC + docs | 3h | low | smoke + reconcile | 5 min |
| **Total** | **~19h** | — | — | — |

Spread over 3 release windows. Каждая sub-phase landed как standalone
commit с production observation 1-2 weeks между фазами (Phase 9/11
rollout pattern).

---

## 11. Migration & rollback

**Migration**: zero. Phase 11 already shipped — Phase 12 adds:
- New `journal=None` optional parameter in EventBus.__init__ →
  без journal behaves identically to Phase 11 (backwards-internal compat)
- New best-effort retrieval in ModelRetriever → failure falls back
  to current behaviour (no buffer retrieve = same as Phase 11)
- New env knobs default to OFF or sensible defaults

**Rollback**: revert sub-phase commit. Each is self-contained:
- Revert 12.C → docs revert; events still persist
- Revert 12.B → events stop persisting; revert to ring-only (Phase 11 behaviour)
- Revert 12.A → metrics replay disabled; revert to Phase 11 behaviour

No DB migrations, no config schema changes.

---

## Phase 13+ — out of scope, future placeholders

### 🟢 Already in this plan (after user feedback)

- **Metrics decimation config** → moved into Phase 12.A.2 (this plan, § 3.A)
  — user mandate: configurable + default keep_all=true.

### 🟡 Phase 13 — Mac orchestrator persistence (separate plan, ~12h)

User said: "пользователь имеет право перезагрузить mac... но нужна
граница когда мы перестаём предугадывать". Phase 13 will own the
Mac-side process-recovery story:

* `ryotenkai run attach <run_id>` — restart killed orchestrator from
  persistent on-disk state. Covers: terminal closed, Mac reboot,
  `kill -9`, crash.
* Persistent orchestrator state: serialise hot-state (active stage,
  in-flight artefacts) to `attempts/<n>/orchestrator.state.json` on
  every meaningful transition. Use the same atomic_write_json
  pattern as FSM state.
* Boundary: Mac data loss = NOT recoverable (per § 0.1 contracts).

Not in Phase 12 because it's an independent recovery axis (process
lifetime vs data durability). Phase 12 lays the data-durability
foundation that Phase 13 then builds on (e.g. attached orchestrator
reads metrics_buffer from local cache that 12.A populated).

### 🔴 Truly out of scope (YAGNI / not our zone)

- **EventJournal `.idx` file** для fast random access at large sizes.
  Disk replay sequential scan ~5-10s acceptable. Open if SLO
  measurements show consistent > 30s replay latency.
- **Multi-run dashboard** with cross-run event comparison —
  visualisation feature, not a durability one.
- **Adaptive decimation by training duration** — beyond config-driven
  (12.A.2). Could auto-tune by measuring disk pressure. YAGNI until
  user reports specific issue.
- **Pod hardware failure recovery** — RunPod-side concern; user opens
  fresh run from latest checkpoint.
