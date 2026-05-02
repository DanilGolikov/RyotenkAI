# План: Fail-fast prevention + log visibility hardening

> Status: **PLANNED** (2026-05-02)
> Author: daniil + claude (deep-think пас)
> Worktree: `dazzling-rosalind-b482fa`
> Trigger: 15 consecutive crashes на одной и той же ошибке после merge'а Phase 1 (pull-only ground-truth) в RESEACRH. Последний run — `run_20260502_101003_6uebn`.
> Связанный план: [2026-05-01-22-20-02-stage-execution-loop-polymorphic-token.md](2026-05-01-22-20-02-stage-execution-loop-polymorphic-token.md) (Phase 1 — IMPLEMENTED).

---

## 1. Context — почему делаем

После merge'а Phase 1 (PodLayout / pull-only ground-truth / postmortem trichotomy `<<MISSING>>`/`<<EMPTY>>`/data) реальные run'ы продолжают падать с **полностью пустой post-mortem диагностикой**. Phase 1 дал operator'у **названный токен** видимости (`<<MISSING>>` вместо silent 0/6 probes) — но **не разорвал саму причинно-следственную цепочку**: trainer падает раньше, чем LogManager успевает забрать stdio.log, и Mac получает `<<MISSING>>` как единственный сигнал.

### 1.1. Свежий лог `run_20260502_101003_6uebn` — anatomy of failure

Из [pipeline.log](runs/run_20260502_101003_6uebn/attempts/attempt_1/logs/pipeline.log):

```
17:13:32  Trainer process started (pid=225)                        ← T+0s
17:13:43  POSTMORTEM: trainer.stdio.log <<MISSING>>                ← T+11s
17:13:43  POSTMORTEM: runner.log <<MISSING>>                       ← T+11s
17:13:45  [DEPLOYER] No trainer stdio log found on remote          ← T+13s
          (reason='cleanup'); ... pod was evicted by the platform
          before download — check the postmortem section.
17:13:46  [PROVIDER:DISCONNECT] Pod y5v090632t7gga terminate call  ← T+14s
          returned an error: "pod not found to terminate.
          The pod may already be gone (platform-side eviction)"
```

**0 байт** забрали с pod'а. **15 раз подряд**. Operator вынужден идти в RunPod console.

### 1.2. Карта дефектов (3 независимых архитектурных проблемы)

| Defekt | Что | Symptom |
|---|---|---|
| **A** | rsync rc=0 ≠ "все модули доехали и importable" | trainer падает с `ModuleNotFoundError` через 5-15s после spawn (см. code_syncer.py:86-93 — failure mode уже задокументирован, но не закрыт) |
| **B** | `LOG_DOWNLOAD_INTERVAL_DEFAULT = 30s` > trainer-life ~11s | periodic LogManager polling **ни разу не выстреливает** до crash'а |
| **C** | `decide_terminal_outcome(failed, mac_alive=True) → TERMINATED_SAFETY` (0s grace) | pod terminate'ится раньше, чем Mac успевает финальный SCP (race) |

Платформенная eviction (RunPod kills pod без уведомления) — **частный случай дефекта C**: pod исчезает быстрее, чем Mac успевает забрать.

---

## 2. Решение — 3 PR в строгом порядке

Принцип: **дешевле всего предотвратить crash, чем гнаться за ним по таймлайну**.

| PR | Цель | LOC | Закрывает |
|---|---|---|---|
| **PR-A**: Fail-fast prevention | Не запускать trainer если код не importable на pod | ~200 | Defekt A — root cause 15 крашей |
| **PR-B**: Visibility hardening | Гарантировать stderr_tail на Mac даже при 5-сек crash | ~150 | Defekt B — polling не работает + платформенная eviction |
| **PR-C**: Race elimination | Дать Mac grace period для финального pull | ~100 | Defekt C — pod terminate до scp |

PR-A решает 95% случаев (15/15 текущих крашей — это recurring import error). PR-B+C нужны для **future unknown** crash'ей и платформенных eviction'ов. Каждый PR landит самостоятельно (зелёные тесты), но композируются.

---

## 3. PR-A: Post-sync importability gate

### 3.1. Архитектурное обоснование

Аналог индустриального паттерна **post-deployment readiness probe**:
- [Kubernetes readinessProbe](https://kubernetes.io/docs/concepts/configuration/liveness-readiness-startup-probes/) — verify deployment health before serving traffic
- [AWS CodeDeploy ValidateService](https://docs.aws.amazon.com/codedeploy/latest/userguide/reference-appspec-file-structure-hooks.html) — lifecycle hook for smoke testing после deploy

В нашем контексте:
- **Deployment artifact** = синкнутый src/* tree на pod
- **Smoke test** = `python -c "from src.workspace.integrations.loader import load_pipeline_config; from src.providers import ..."`
- **Failure mode** = ModuleNotFoundError → fail deployment, **не** spawn trainer

Closes Defekt A: gate валидирует **тот же import path что и trainer** до spawn'а. Если smoke прошёл — trainer не упадёт на module-load. Если smoke fail — мы знаем имя missing-module **до** того, как trainer'у вообще дали шанс.

### 3.2. Имплементация

#### 3.2.1. Расширяем `docker/training/runtime_check.py`

Файл **уже** baked в training image как `/opt/helix/runtime_check.py` и проверяет pip packages. Добавляем второй режим — `--check-source`:

```python
# docker/training/runtime_check.py — extension

_REQUIRED_SRC_MODULES: list[str] = [
    "src.workspace.integrations.loader",  # config loader
    "src.config",                          # pydantic schemas
    "src.providers",                       # provider lifecycle (recurring failure)
    "src.training.run_training",           # trainer entrypoint
    "src.runner.main",                     # runner entrypoint (shipped via thin-image)
    "src.utils.config",                    # config façade
]

def check_source_importable() -> int:
    """Verify each required src.* module is import-able on this pod.
    Returns rc=0 on success, rc=2 with offending module list on failure.
    """
    failed: list[tuple[str, str]] = []
    for mod_name in _REQUIRED_SRC_MODULES:
        try:
            importlib.import_module(mod_name)
        except Exception as exc:
            failed.append((mod_name, f"{type(exc).__name__}: {exc}"))

    if not failed:
        print("OK")
        for m in _REQUIRED_SRC_MODULES:
            print(f"{m}=importable")
        return 0

    print("FAILED")
    for mod_name, err in failed:
        print(f"{mod_name}=NOT_IMPORTABLE ({err})")
    return 2

def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check-source", action="store_true")
    args = parser.parse_args()
    if args.check_source:
        return check_source_importable()
    return check_pip_packages()  # existing behaviour
```

#### 3.2.2. Где вызываем — `CodeSyncer.sync()`

Расширяем [code_syncer.py:125-182](src/pipeline/stages/managers/deployment/code_syncer.py:125) — ПОСЛЕ rsync, ПЕРЕД return Ok:

```python
# code_syncer.py — после _clear_pycache, перед logger.info("Source code synced")

# PR-A — post-sync importability gate. rsync rc=0 не гарантирует
# что все src.* importable (см. run_20260429_171726_49j32). Запускаем
# smoke test на pod в том же python interpreter, который потом run'ит trainer.
gate_result = self._verify_importability(ssh_client)
if gate_result.is_failure():
    return gate_result  # AppError с именем offending module

logger.info(f"Source code synced ({len(existing_modules)} modules) + importable")
return Ok(None)


def _verify_importability(self, ssh_client: SSHClient) -> Result[None, AppError]:
    """Run /opt/helix/runtime_check.py --check-source on pod."""
    cmd = (
        f"cd {self._workspace} && "
        f"PYTHONPATH={self._workspace} python3 /opt/helix/runtime_check.py --check-source"
    )
    success, stdout, _ = ssh_client.exec_command(
        command=cmd, timeout=DEPLOYMENT_VERIFY_TIMEOUT, silent=True,
    )

    if success and stdout.startswith("OK"):
        return Ok(None)

    failed_modules = [
        line.split("=", 1)[0]
        for line in stdout.splitlines()
        if "=NOT_IMPORTABLE" in line
    ]
    error_msg = (
        f"Post-sync import check failed. Modules not importable on pod: "
        f"{failed_modules or '<unknown>'}. Action: ensure these modules exist "
        f"in your local checkout before deploy. Full pod output:\n{stdout[:500]}"
    )
    logger.error(f"❌ {error_msg}")
    return Err(AppError(category="DEPLOYMENT", code="IMPORT_GATE_FAILED",
                        message=error_msg))
```

**Что блокируется**: pipeline останавливается на Stage 1 (Deployment) с **explicit named module** в error. Operator видит:

```
[DEPLOYER] ❌ Post-sync import check failed. Modules not importable on pod:
           ['src.providers']. Action: ensure these modules exist in your
           local checkout before deploy.
```

Vs текущее `<<MISSING>>` через 5 минут попыток + потраченные RunPod минуты на pod, который никогда не запустится.

#### 3.2.3. Side benefit: milestone logging для unknown crashes

В `src/training/run_training.py:main()` добавляем три explicit print'а в **начале** функции — они идут в stderr → stdio.log:

```python
# src/training/run_training.py — main() top
def main(argv: list[str] | None = None) -> int:
    print("[TRAINER:M1] Python interpreter started, argv parsed", file=sys.stderr, flush=True)
    args = _parse_args(argv)
    print(f"[TRAINER:M2] Loading config from {args.config}", file=sys.stderr, flush=True)
    config = load_pipeline_config(args.config)
    print("[TRAINER:M3] Config validated, starting MLflow setup", file=sys.stderr, flush=True)
    # ... existing flow ...
```

Postmortem видит:
- Только M1 → import failure после argv (covered by gate, не должно случаться)
- M1+M2 → config validation failed
- M1+M2+M3 → MLflow / dataset / model load failed
- Ничего → catastrophic Python startup failure (CUDA driver, libc mismatch)

Three states give operator **immediate localization** без чтения 200 строк traceback'а.

#### 3.2.4. Drift guard — invariant test

Список `_REQUIRED_SRC_MODULES` может разъехаться с реальными top-level imports trainer'а. Добавляем CI-test [src/tests/unit/training/test_required_modules_drift.py](src/tests/unit/training/test_required_modules_drift.py):

```python
# AST-scan для top-level imports run_training.py.
# Assert: набор top-level imports ⊆ _REQUIRED_SRC_MODULES (с учётом package/module).
# CI fails если кто-то добавил `from src.foo.bar import x` в trainer без обновления списка.
```

### 3.3. Файлы (PR-A)

```
docker/training/runtime_check.py                                # +50 LOC: --check-source
src/pipeline/stages/managers/deployment/code_syncer.py          # +30 LOC: _verify_importability
src/training/run_training.py                                    # +3 LOC: M1/M2/M3 milestones
src/tests/unit/pipeline/stages/managers/deployment/test_code_syncer_import_gate.py   # NEW
src/tests/unit/training/test_required_modules_drift.py          # NEW
src/tests/integration/test_2026_05_02_import_gate_regression.py # NEW (15-crash regression)
```

---

## 4. PR-B: Push stderr_tail в trainer_exited event

### 4.1. Архитектурное обоснование

Аналог [Kubernetes /dev/termination-log](https://kubernetes.io/docs/tasks/debug/debug-application/determine-reason-pod-failure/): terminating container пишет финальное сообщение в well-known файл, kubelet читает и embed'ит в `Pod.status.containerStatuses[].lastState.terminated.message`. **API consumer** видит причину смерти **в самом status'е**, не делая отдельный `kubectl logs` (которого может уже не быть, если pod gc'нут).

В нашем контексте: trainer_exited event currently carries только `{exit_code, signal, cancellation_requested}`. Расширяем до `schema_version=2`:

```python
{
    "exit_code": rc,
    "signal": signal_name,
    "cancellation_requested": ...,
    "stderr_tail": "<last 50 lines, ≤10KB>",      # NEW
    "stdout_tail": "<last 50 lines, ≤10KB>",      # NEW
    "stdio_log_path": "<absolute path on pod>",   # NEW (для SCP fallback)
    "schema_version": 2,                          # NEW
}
```

**Почему push, а не pull**: на момент `trainer_exited` Supervisor **уже владеет** content'ом stdio.log (он его сам записал). Read из локального файла = O(10KB) на pod, без SSH round-trip. Mac получает event через WS, который **уже работает** (он pre-fetched, его не задевает pod terminate). Pod может умереть после publish — message доехал.

Closes Defekt B (+ страхует от платформенной eviction): tail приходит на Mac **синхронно с trainer_exited**, до любых LogManager polls. **Гарантия видимости** даже если pod исчезнет через секунду.

### 4.2. Имплементация

#### 4.2.1. `Supervisor._read_stdio_tail` + extended publish

В [supervisor.py:540-575](src/runner/supervisor.py:540), перед `bus.publish('trainer_exited', ...)`:

```python
# supervisor.py — extension
def _read_stdio_tail(self, max_lines: int = 50, max_bytes: int = 10240) -> tuple[str, str]:
    """Read last N lines from trainer.stdio.log, split by [OUT]/[ERR] prefix.
    Returns (stderr_tail, stdout_tail). Empty strings on missing/error.
    """
    if self._stdio_log_path is None:
        return "", ""
    try:
        size = self._stdio_log_path.stat().st_size
        with self._stdio_log_path.open("rb") as f:
            if size > max_bytes:
                f.seek(-max_bytes, 2)  # seek-from-end
                f.readline()           # skip partial first line
            tail_bytes = f.read()
    except OSError:
        return "", ""

    text = tail_bytes.decode("utf-8", errors="replace")
    lines = text.splitlines()[-max_lines:]
    err_lines = [ln[6:] for ln in lines if ln.startswith("[ERR] ")]
    out_lines = [ln[6:] for ln in lines if ln.startswith("[OUT] ")]
    return "\n".join(err_lines), "\n".join(out_lines)


# In _reap_subprocess — replace existing publish:
stderr_tail, stdout_tail = _redact_secrets(*self._read_stdio_tail())
self._bus.publish(
    "trainer_exited",
    {
        "exit_code": rc,
        "signal": signal_name,
        "cancellation_requested": self._cancellation_requested,
        "stderr_tail": stderr_tail,
        "stdout_tail": stdout_tail,
        "stdio_log_path": str(self._stdio_log_path) if self._stdio_log_path else None,
        "schema_version": 2,
    },
)
```

`_redact_secrets()` — централизованный фильтр (см. §6.RP4): regex по `hf_[A-Za-z0-9]+`, `sk-[A-Za-z0-9]+`, `RUNPOD_[A-Z_]+=...` etc. Тот же helper используем в Phase 1 redaction (R5).

#### 4.2.2. Mac-side consumer

[training_monitor.py](src/pipeline/stages/training_monitor.py) — `_dispatch_event` для `trainer_exited`:

```python
def _handle_trainer_exited(self, payload: dict) -> None:
    # ... existing rc / signal handling ...
    schema_version = payload.get("schema_version", 1)
    if schema_version >= 2:
        stderr_tail = payload.get("stderr_tail", "")
        stdout_tail = payload.get("stdout_tail", "")
        if stderr_tail:
            for line in stderr_tail.splitlines()[-30:]:
                logger.info(f"[TRAINER:STDERR] {line}")
        if stdout_tail:
            for line in stdout_tail.splitlines()[-10:]:
                logger.info(f"[TRAINER:STDOUT] {line}")
    # else: schema v1 → fallback to LogManager.download() (legacy path)
```

Backward-compat: legacy v1 consumers видят payload идентичный pre-PR-B (новые поля просто игнорируются как unknown).

#### 4.2.3. Periodic runner.log pull (бонусный fix)

Текущий periodic downloader в [training_monitor.py:581-587](src/pipeline/stages/training_monitor.py:581) пуллит **только** trainer.stdio.log. Если runner crashed (pre-import error в uvicorn), runner.log не пуллится до postmortem. Добавляем второй LogManager в loop:

```python
# training_monitor.py — _periodic_log_download_loop extension
async def _periodic_log_download_loop(self) -> None:
    while not self._stop_event.is_set():
        await asyncio.sleep(TRAINING_MONITOR_LOG_DOWNLOAD_INTERVAL)
        with contextlib.suppress(Exception):
            self._trainer_log_manager.download(silent=True)
            self._runner_log_manager.download(silent=True)  # NEW
```

И уменьшаем `LOG_DOWNLOAD_INTERVAL_DEFAULT` в [src/constants.py:158](src/constants.py:158) с **30s → 5s**. Trade-off: 6× SSH traffic (но `stat -c%s` cheap, ~10ms per call), но catches crashes happening между T+5s и T+30s.

### 4.3. Файлы (PR-B)

```
src/runner/supervisor.py                                        # +30 LOC: _read_stdio_tail + payload v2
src/pipeline/stages/training_monitor.py                         # +25 LOC: schema v2 + dual LM
src/constants.py                                                # 1 LOC: LOG_DOWNLOAD_INTERVAL_DEFAULT 30→5
src/utils/secret_redaction.py                                   # NEW (~30 LOC) — shared redaction filter
src/tests/unit/runner/test_supervisor_stdio_tail.py             # NEW
src/tests/unit/pipeline/stages/test_training_monitor_v2_payload.py # NEW
src/tests/unit/utils/test_secret_redaction.py                   # NEW
```

---

## 5. PR-C: PodTerminator diagnostic grace для failed runs

### 5.1. Архитектурное обоснование

Текущая [decide_terminal_outcome](src/runner/pod_terminator.py:151) асимметрична:

| terminal_state | mac_alive | Decision | Grace |
|---|---|---|---|
| completed | True | STOPPED_FOR_RESUME_SHORT_GRACE | **60-600s** |
| completed | False | STOPPED_FOR_RESUME | immediate |
| **failed** | **True** | **TERMINATED_SAFETY** | **0s** ⚠ |
| failed | False | STOPPED_FOR_RESUME | immediate |

`failed + mac_alive` — единственный квадрант **без** grace. Это обратная архитектурная иерархия: failed run заслуживает **больше** диагностического времени, не меньше.

Решение: добавить новый decision `TERMINATED_AFTER_DIAGNOSTIC_GRACE` для `failed + mac_alive`. Grace shorter (default 30s) — достаточно для Mac SCP цикла, не блокирует resource cleanup надолго.

Аналог индустриальных паттернов:
- [Kubernetes terminationGracePeriodSeconds](https://kubernetes.io/docs/concepts/workloads/pods/pod-lifecycle/) — pod gets graceful shutdown window перед SIGKILL.
- **Borg / Omega lameduck mode** — Google-internal: terminating tasks enter "lameduck" state for N seconds before kill, allowing observers to scrape final state. Identical pattern.

### 5.2. Имплементация

```python
# src/runner/pod_terminator.py
class PodTerminalOutcome(StrEnum):
    # ... existing ...
    TERMINATED_AFTER_DIAGNOSTIC_GRACE = "terminated_after_diagnostic_grace"
    """Failed + Mac alive: brief grace (30s default) for SCP, then terminate."""


def decide_terminal_outcome(
    *,
    terminal_state: str,
    mac_alive: bool,
    volume_kind: str,
    keep_on_error: bool,
) -> str:
    if terminal_state == "cancelled":
        return PodTerminalOutcome.TERMINATED_USER_STOP
    if terminal_state == "failed" and keep_on_error:
        return PodTerminalOutcome.KEPT_ALIVE_FOR_DEBUG
    if volume_kind == "network":
        return PodTerminalOutcome.TERMINATED_SAFETY

    if terminal_state == "failed":
        if mac_alive:
            return PodTerminalOutcome.TERMINATED_AFTER_DIAGNOSTIC_GRACE  # CHANGED
        return PodTerminalOutcome.STOPPED_FOR_RESUME

    if terminal_state == "completed":
        if mac_alive:
            return PodTerminalOutcome.STOPPED_FOR_RESUME_SHORT_GRACE
        return PodTerminalOutcome.STOPPED_FOR_RESUME

    return PodTerminalOutcome.KEPT_ALIVE_FOR_DEBUG
```

В `PodTerminator.decide_and_act` — action dispatch для нового outcome:

```python
elif decision == PodTerminalOutcome.TERMINATED_AFTER_DIAGNOSTIC_GRACE:
    # Brief wait for Mac's postmortem SCP. Heartbeat-aware: если Mac
    # disconnects mid-grace, terminate immediately (no point waiting).
    deadline = time.monotonic() + self._diagnostic_grace_seconds
    while time.monotonic() < deadline:
        if not await self._heartbeat.is_alive():
            break
        await self._sleep(self._grace_tick)
    return await self._terminate(...)
```

Constant: `DIAGNOSTIC_GRACE_SECONDS = 30.0` (class-level, configurable). Достаточно для одного SCP roundtrip + декодирование stdio.log на Mac.

### 5.3. Файлы (PR-C)

```
src/runner/pod_terminator.py                                    # +30 LOC: new outcome + dispatch
src/tests/unit/runner/test_pod_terminator_diagnostic_grace.py   # NEW
src/tests/unit/runner/test_pod_terminator_decision.py           # update existing combinatorial test (24→32 cells)
```

---

## 6. Risk register

| # | Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|---|
| RP1 | Import gate adds 1-3s latency к каждому deploy (smoke import time) | High | Low | Acceptable — ~1% от total run time. Cache PYC в /opt/helix layer чтобы первый import был warm. |
| RP2 | Import gate false-positive: модуль не в _REQUIRED_SRC_MODULES, но trainer его import'ит и крашится | Medium | Medium | _REQUIRED_SRC_MODULES — explicit list, easy to extend. **Drift test** §3.2.4 — AST-scan trainer's actual top-level imports, assert subset. CI fails если добавили import без обновления списка. |
| RP3 | runtime_check.py не bake'нуто в старые images → существующие pods ломаются | Low | Medium | Detect rc=127 (command not found) → return Err with actionable "image is too old, rebuild ryotenkai-training-runtime image". NOT skip silently. |
| RP4 | stderr_tail в WS payload содержит PII / secrets (HF_TOKEN если trainer print env) | Medium | High | Centralized `_redact_secrets()` helper в [src/utils/secret_redaction.py](src/utils/secret_redaction.py): regex `hf_[A-Za-z0-9]+`, `sk-[A-Za-z0-9]+`, `RUNPOD_[A-Z_]+=...`. Same redaction list as Phase 1 R5. Operators могут extend через env var `RYOTENKAI_REDACT_PATTERNS`. |
| RP5 | Schema version bump (v1→v2) ломает existing Mac clients | Low | Medium | Forward-compat by design: новые fields optional, v1 clients ignore unknown keys. Test §7.6 `trainer_exited_v1_consumer_compat`. |
| RP6 | LOG_DOWNLOAD_INTERVAL 30→5s → 6× SSH traffic | Medium | Low | `stat -c%s` ~10ms на call. 6×/min × 60min run = 360 calls = ~3.6s aggregate SSH time. Negligible. Если станет issue — adaptive interval (5s during first 60s, 30s after). |
| RP7 | DIAGNOSTIC_GRACE_SECONDS=30 задерживает resource cleanup → платим 30s pod time на каждый failed run | Medium | Low | RunPod billing — секунды. 30s ≈ $0.001-0.01 per failed run. Acceptable. Test что grace abort'ится early если Mac disappear (см. §7.5). |
| RP8 | Heartbeat-aware grace в PR-C может замаскировать stale heartbeat (Mac thinks alive but actually dead) | Low | Medium | Heartbeat retry logic existing уже handles это с `HEARTBEAT_RETRY_ATTEMPTS=3 × 10s = 30s`. Переиспользуем same logic. |
| RP9 | PR-A blocks legitimate runs если у user'а старая версия repo (без src/providers/ — пользовательское состояние) | Medium | Medium | **Это feature, не bug**. Operator видит **named** error → точно знает что fix'ить. Лучше fail-fast чем crash trainer'а 11 секунд после spawn × 15 раз. |
| RP10 | PR-B leak file descriptor если `_read_stdio_tail` не закрывает файл | Low | Low | `with` statement в реализации. Linting catches. |
| RP11 | Платформенная RunPod eviction случается во время grace window (pod исчезает сам) | Medium | Medium | PR-B push-tail уже ушёл в `trainer_exited` event ДО eviction'а — Mac получил данные. PR-C grace абортится через heartbeat retry. PR-B страхует PR-C. |
| RP12 | _REQUIRED_SRC_MODULES drift с реальными trainer imports | Medium | Medium | Drift test §3.2.4 (AST-scan + CI). |

---

## 7. Test plan (7 категорий, общая для PR-A+B+C)

### 7.1. Positive
- **import_gate_passes_clean_workspace** (PR-A): rsync src/* полный → runtime_check --check-source rc=0 with all modules → CodeSyncer.sync returns Ok.
- **trainer_exited_v2_with_tail** (PR-B): trainer пишет 30 строк stderr → reap → trainer_exited payload содержит non-empty stderr_tail.
- **diagnostic_grace_allows_scp** (PR-C): trainer fails, mac_alive=True → PodTerminator waits 30s → simulated SCP (sleep 5s) completes → pod terminates after.

### 7.2. Negative
- **import_gate_blocks_missing_providers** (PR-A): rsync без src/providers/ → runtime_check rc=2 with "src.providers=NOT_IMPORTABLE" → CodeSyncer returns Err(IMPORT_GATE_FAILED). **Регрессия для 15 крашей**.
- **import_gate_blocks_syntax_error** (PR-A): инжектируем syntax error в src/providers/__init__.py → gate видит SyntaxError → blocks deployment.
- **trainer_exited_v2_handles_empty_stdio** (PR-B): stdio.log пустой → stderr_tail="", payload still well-formed.
- **diagnostic_grace_aborts_on_mac_dead** (PR-C): mid-grace Mac heartbeat dies → grace breaks early → pod terminates без полного wait'а.

### 7.3. Boundary
- **import_gate_handles_pyc_only_module** (PR-A): на pod есть .pyc но нет .py → importlib still работает (Python стандарт) → gate passes.
- **stdio_tail_truncates_at_10kb** (PR-B): stdio.log 1MB → tail возвращает ровно последние ≤10KB, не 1MB.
- **stdio_tail_handles_partial_first_line** (PR-B): seek-from-end попадает в середину строки → readline() пропускает partial → последующие строки целые.
- **diagnostic_grace_min_max_bounded** (PR-C): grace_tick=10, deadline=30 → max 3 ticks, bounded.

### 7.4. Invariants
- **import_gate_uses_same_python_as_trainer** (PR-A property): smoke test invoked через `/opt/helix/runtime_check.py --check-source` использует **тот же** python interpreter, что и `python -m src.training.run_training`. Иначе gate passes но trainer fails.
- **schema_version_monotonic** (PR-B): новые fields в trainer_exited payload **только** под schema_version >= 2. v1 consumers видят payload идентичный pre-PR-B (forward-compat).
- **decision_table_completeness** (PR-C, exhaustive): для всех 4×2×2×2 = 32 комбинаций (state×alive×volume×keep_on_error) есть **ровно один** outcome. Property test через hypothesis.
- **redaction_helper_centralized** (RP4): один `_redact_secrets()` на все code paths (Supervisor stdio + Phase 1 trainer log redaction). Grep гарантирует отсутствие inline regex'ов.

### 7.5. Dependency-error
- **import_gate_ssh_timeout** (PR-A): SSH execution timeouts → gate result is_failure → CodeSyncer returns Err. Не маскируется как passed.
- **import_gate_runtime_check_missing** (PR-A): `/opt/helix/runtime_check.py` не существует на image → ssh exec rc=127 → gate returns Err with "image is too old, rebuild image". Actionable.
- **stdio_tail_disk_full** (PR-B): `_read_stdio_tail` ловит OSError → returns ("", "") → publish без crash'а.
- **diagnostic_grace_provider_terminate_fails** (PR-C): grace ok but provider.terminate raises → existing retry/backoff logic catches (not regressed).

### 7.6. Regression
- **2026_05_02_17_13_32_scenario** (PR-A integration): создаём fake repo без src/providers/ → запускаем pipeline → assert: pipeline останавливается на Stage 1 (Deployment) с **named** error "src.providers=NOT_IMPORTABLE". НЕ доходит до Stage 3 (Training Monitor). НЕ создаёт pod (или terminate'ит cleanly).
- **trainer_exited_v1_consumer_compat** (PR-B): legacy Mac client читающий schema v1 payload (без tail) → корректно обрабатывает (новые поля игнорируются).
- **mlflow_events_unaffected** (PR-B): trainer публикует mlflow_metric → MLflowRelay forward работает. Регрессия metrics flow не должна быть.
- **kept_alive_for_debug_unchanged** (PR-C): keep_on_error=True + failed → KEPT_ALIVE_FOR_DEBUG (не diagnostic_grace). Old behaviour preserved.
- **platform_eviction_with_push_tail** (PR-B+C integration): mock platform eviction at T+5s → assert Mac уже получил stderr_tail через `trainer_exited` event до eviction'а.

### 7.7. Combinatorial
Матрица для PR-C: `{completed,failed,cancelled,unknown} × {alive,asleep} × {persistent,network} × {keep,no-keep}` = 32 cells. В каждом cell assert decision matches спецификации §5.1 table. Existing test [test_pod_terminator_decision.py](src/tests/unit/runner/test_pod_terminator_decision.py) покрывает 24 cells (без TERMINATED_AFTER_DIAGNOSTIC_GRACE) — расширяем до 32.

---

## 8. Best-practices alignment

PR-A:
- ✅ **Kubernetes readinessProbe / startupProbe** — verify deployment health before serving traffic. "Fail fast at deploy, not at runtime" — индустриальный стандарт.
- ✅ **AWS CodeDeploy ValidateService hook** — lifecycle hook for smoke testing после deploy. Same paradigm.
- ✅ **Argo Rollouts AnalysisRun** — automatic rollback if post-deploy probe fails. Мы делаем static gate (no rollout), но идея identical.

PR-B:
- ✅ **Kubernetes /dev/termination-log** — terminating container writes final message, kubelet embeds в Pod.status. **Точно** наш паттерн.
- ✅ **OpenTelemetry exception events** — span с exception attribute (type, message, stacktrace) embed'ится в trace data. Same idea: error context **с** terminal event, не отдельный pull.
- ✅ **Sentry breadcrumbs** — last N events leading up to error embedded в error report. Наши milestone'ы M1/M2/M3 — это breadcrumbs для cold-start failures.

PR-C:
- ✅ **Kubernetes terminationGracePeriodSeconds** — pod gets graceful shutdown window перед SIGKILL.
- ✅ **AWS Lambda Provisioned Concurrency** — grace для pre-warming reduces cold-start. Мы делаем grace для **post-mortem** vs **lifecycle**.
- ✅ **Borg / Omega lameduck mode** — Google-internal: terminating tasks enter "lameduck" state, allowing observers to scrape final state. Identical pattern.

Best-practices что мы НЕ делаем (отложено в follow-up):
- ⚪ **Distributed tracing** через OpenTelemetry — heavy weight для нашего scale. Local logging достаточно.
- ⚪ **Retention tier'ing** для stdio.log (hot S3 → cold archive) — нужно когда disk pressure станет проблемой.
- ⚪ **Anomaly detection** на trainer_exited rate — Datadog/Grafana — отдельный observability project.

---

## 9. Migration sequence

Каждый PR landит самостоятельно, тесты зелёные. Order matters только для **psychological** value (видимый эффект → больше confidence):

1. **PR-A** (import gate): immediate visible win — все 15 текущих крашей превращаются в clean Stage-1 fail с named module. **0 wasted RunPod minutes**.
2. **PR-B** (push tail): future-proofs unknown crashes. Operator получает stderr **в pipeline.log** для каждого `trainer_exited`.
3. **PR-C** (diagnostic grace): защищает финальный SCP от race с PodTerminator. Помогает когда PR-A не угадал и trainer всё-таки крашится после spawn.

Итого: 3 PR в ту же RESEACRH ветку, ~450 LOC + ~250 LOC tests.

---

## 10. Verification

### 10.1. Manual regression (текущий 15-crash bug)

**Setup**: ребейзим worktree без src/providers/ (или временно `mv src/providers /tmp/`), запускаем pipeline.

**Pre-PR-A**: trainer crashes T+11s, postmortem `<<MISSING>>` × 15.

**Post-PR-A**: pipeline останавливается на Stage 1 с
```
[DEPLOYER] ❌ Post-sync import check failed.
            Modules not importable on pod: ['src.providers'].
            Action: ensure these modules exist in your local checkout before deploy.
```
Pod не создан (или cleanly terminated). 0 wasted RunPod minutes.

### 10.2. Automated regression suite

```bash
# PR-A
pytest src/tests/unit/pipeline/stages/managers/deployment/test_code_syncer_import_gate.py -v
pytest src/tests/unit/training/test_required_modules_drift.py -v
pytest src/tests/integration/test_2026_05_02_import_gate_regression.py -v

# PR-B
pytest src/tests/unit/runner/test_supervisor_stdio_tail.py -v
pytest src/tests/unit/utils/test_secret_redaction.py -v
pytest src/tests/unit/pipeline/stages/test_training_monitor_v2_payload.py -v

# PR-C
pytest src/tests/unit/runner/test_pod_terminator_diagnostic_grace.py -v
pytest src/tests/unit/runner/test_pod_terminator_decision.py -v  # combinatorial 32 cells

# Full regression (Phase 1 + Phase 2)
pytest src/tests/unit/ src/tests/integration/ -v
ruff check . && mypy .
```

### 10.3. End-to-end live run (после всех 3 PR)

1. Намеренно ломаем `src/training/run_training.py` (синтаксис error).
2. Запускаем pipeline.
3. **Acceptance criteria**:
   - PR-A: blocks at Stage 1 deployment with named "src.training.run_training=NOT_IMPORTABLE"
   - PR-B (если import gate отключить для теста): trainer_exited на Mac содержит stderr_tail с traceback'ом в pipeline.log
   - PR-C: pod terminate'ится через ~30s после trainer_exited, успешный SCP в этом окне

---

## 11. Audit trail (3 итерации risk audit)

**Итерация 1** — root cause vs symptom:
- ✅ Phase 1 дал visibility token (`<<MISSING>>`) но не предотвратил root cause. PR-A добавлен.
- ✅ 15 крашей одинаковые → deterministic bug → нужен gate, не diagnostic. PR-A приоритизирован.
- ✅ code_syncer.py:86-93 уже **в коде задокументировал** failure mode (run_20260429_171726_49j32) → закрываем документированный TODO.

**Итерация 2** — race conditions:
- ✅ Defekt C (PodTerminator 0-grace для failed) — race detected через timeline analysis. PR-C добавлен.
- ✅ PR-B push-tail eliminates dependency на post-mortem SSH (которая race'ит с pod terminate). Архитектурно более robust.
- ✅ Schema v2 backward-compat — потенциальный break для legacy clients. RP5 + test §7.6.
- ✅ Платформенная eviction (RunPod kills pod) — частный случай Defekt C. RP11 + integration test §7.6 `platform_eviction_with_push_tail`.

**Итерация 3** — production failure modes:
- ✅ Import gate может false-positive если _REQUIRED list out-of-sync. RP2 + drift test §3.2.4.
- ✅ Redaction для PII в stderr_tail — критично, иначе HF_TOKEN утечёт через WS. RP4 + centralized helper в `src/utils/secret_redaction.py` (переиспользуется Phase 1).
- ✅ Heartbeat staleness в diagnostic_grace может waste pod time. RP8 + heartbeat-aware abort.
- ✅ PR-A решает 15/15 текущих крашей. PR-B+C — для unknown future crashes + платформенной eviction.

---

## 12. Что мы явно НЕ делаем (rejected alternatives)

1. **Полный pre-deploy CI** — отвергнуто. Slow (5-10 min) и не ловит pod-specific issues (CUDA driver mismatch, libc на pod's image). Smoke import on pod — minimal viable.
2. **Восстановление training.log как fallback** — отвергнуто. Phase 1 explicitly removed это, причины не изменились.
3. **Continuous SCP streaming** (rsync --inplace в loop) — отвергнуто. Adds complexity, не решает race с pod terminate.
4. **Pod-side Sentry/Datadog agent** — отвергнуто. Heavy, third-party dependency, overkill для current scale. Local stderr_tail достаточно.
5. **Feature flag для PR-A (`RYOTENKAI_ENABLE_IMPORT_GATE`)** — отвергнуто. YAGNI: gate либо correct (always run), либо buggy (rollback). Никаких scenarios где opt-out имеет смысл.
6. **CodeSyncer на full-tree push** (без --include filters) — отвергнуто. Шире blast radius (~50MB вместо ~3MB), не решает root cause.
7. **EventJournal pull на Mac для post-mortem** — отвергнуто (то же решение, что и в Phase 1: журнал — pod-side replay layer, не Mac visibility).

---

## 13. Critical files (summary)

**NEW**:
```
docs/plans/2026-05-02-fail-fast-prevention-and-log-visibility.md  # этот файл
src/utils/secret_redaction.py
src/tests/unit/utils/test_secret_redaction.py
src/tests/unit/pipeline/stages/managers/deployment/test_code_syncer_import_gate.py
src/tests/unit/training/test_required_modules_drift.py
src/tests/unit/runner/test_supervisor_stdio_tail.py
src/tests/unit/runner/test_pod_terminator_diagnostic_grace.py
src/tests/unit/pipeline/stages/test_training_monitor_v2_payload.py
src/tests/integration/test_2026_05_02_import_gate_regression.py
```

**Modified**:
```
docker/training/runtime_check.py                                  # +50 LOC: --check-source
src/pipeline/stages/managers/deployment/code_syncer.py            # +30 LOC: _verify_importability
src/training/run_training.py                                      # +3 LOC: M1/M2/M3 milestones
src/runner/supervisor.py                                          # +30 LOC: _read_stdio_tail + payload v2
src/pipeline/stages/training_monitor.py                           # +25 LOC: schema v2 + dual LM
src/constants.py                                                  # 1 LOC: 30→5
src/runner/pod_terminator.py                                      # +30 LOC: TERMINATED_AFTER_DIAGNOSTIC_GRACE
src/tests/unit/runner/test_pod_terminator_decision.py             # update combinatorial 24→32
```

---

## 14. Phase 3 (отложено, не делаем сейчас)

После того как PR-A/B/C land'нутся и поработают в проде:
- **Adaptive LOG_DOWNLOAD_INTERVAL** (5s during first 60s of run, 30s after) — если 5s static станет noticeable load.
- **Stdio rotation** trainer.stdio.log по размеру — когда disk pressure станет проблемой.
- **PII redaction policy** — конфигурируемые regex через env var.
- **Platform-eviction detection** — hook на `pod not found to terminate` → пометить run как `evicted` отдельно от `failed` для honest metrics.
