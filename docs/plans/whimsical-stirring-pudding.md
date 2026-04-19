# Silent Training Death: Observability Hardening + PEFT Double-Apply Fix

## Context

Четвёртый подряд GRPO пилот (`run_20260410_144204/attempt_4`) снова умер молча.
Симптомы **ровно те же**, что и в `run_20260410_130601`, но теперь с новым
набором улик — предыдущий пакет мер (grad checkpointing, halve bs, halve gen_bs,
`_collect_death_diagnostics`) дожил до step 52/432 вместо step 8, но **причину
смерти мы всё ещё не видим**. Это блокер. Пользователь прямо сказал:
«особо важно изучи проблему почему ошибка не логгируется в логи».

### Что видно (факты)

Из `runs/run_20260410_144204/attempts/attempt_4/`:

| Сигнал | Значение | Вывод |
|---|---|---|
| `training.log` последняя строка | `12%  52/432 [05:01<22:22, 3.53s/it]` | обрыв посреди прогресс-бара |
| Python traceback | **отсутствует** | Python exception НЕ долетел до logger'а |
| TRL метрики (`{'loss': ..., 'reward': ...}`) | **отсутствуют** | `logging_steps=10` дошли до 52 — должно было быть 5 логов, нет ни одного |
| `[MONITOR:POSTMORTEM] dmesg_oom` | `<empty>` | kernel НЕ убивал процесс (нет OOM-kill) |
| `[MONITOR:POSTMORTEM] nvidia_smi` | `10562 MiB, 24564 MiB, 20%` | VRAM был стабильно 43% — **не ресурсы** |
| `[MONITOR:POSTMORTEM] workspace_markers` | `<empty>` | Ни `TRAINING_COMPLETED`, ни `TRAINING_FAILED` не создан |
| MONITOR ALIVE | VRAM 10.3/24 GB, GPU 18–24% на протяжении 5+ минут | Тренировка реально шла |
| `Trainable parameters: 0 (0.00%)` + warning `Already found a peft_config attribute` | См. training.log стр. 83, 174–177 | **PEFT применяется дважды** |
| REWARD_DEBUG 1–3 | сэмпл 1 = пустая строка, сэмпл 2 = испаноязычная галлюцинация, сэмпл 3 = английская проза | Модель не генерит HelixQL → reward = −1 константа → advantage = 0 (отдельная корректностная проблема) |

### Что мы НЕ знаем (и должны узнать)

1. **Что именно убило процесс.** Не OOM, не Python exception, не watchdog.
   Остаются: native C crash (segfault в bnb/flash-attn/torch), `abort()` из
   CUDA kernel, SIGTERM из неясного источника, Python exit без traceback.
2. **Почему TRL не логгировал `logs` dict** на шагах 10/20/30/40/50.
3. **Связаны ли PEFT-double-apply + 0 trainable params с молчаливой смертью.**

### Почему мы слепые

Проблема **на трёх уровнях**, ни один не закрыт:

**Уровень 1: Shell-wrapper (`deployment_manager.py:844-880`).**
```bash
exec >{log_file} 2>&1
...
"$PY_BIN" {module_args}
exit_code=$?
```
Нет `PYTHONUNBUFFERED=1`, нет `PYTHONFAULTHANDLER=1`, нет `stdbuf`. Block-
буферизация файла означает: если процесс упал до flush'а, хвост stderr
теряется. Exit-код есть в shell-переменной, но в файл **не записывается** —
есть только условное создание `TRAINING_FAILED` с шаблонным текстом
«Training failed early». Это неинформативно.

**Уровень 2: Python entry (`src/training/run_training.py:579-657`).**
```python
def main() -> int:
    ...
    try:
        output_path = run_training(...)
        return 0
    except Exception as e:
        logger.error(f"Training failed: {e}")
        return 1
```
- **Нет `faulthandler.enable()`** — native crash (SIGSEGV, SIGABRT, SIGBUS)
  не оставляет ни Python, ни C stacktrace.
- `except Exception` ловит только Python-исключения; native crash падает
  мимо.
- Нет `atexit` flush'а logging handlers.
- `notifier.notify_failed()` вызывается только внутри `except Exception` в
  `run_training()` (стр. 516–521). Native crash его миновал → нет маркера.

**Уровень 3: Monitor postmortem (`training_monitor.py:741-794`).**
Текущие probe'ы (добавлены в прошлой сессии):
1. `dmesg -T | grep -i oom` ← был пуст, бесполезно
2. `nvidia-smi` ← был пуст (процесс уже мёртв)
3. `ls TRAINING_* STOPPED_BY_WATCHDOG` ← был пуст

**Не хватает**:
- `tail -n 200 training.log` (чтобы видеть хвост перед обрывом не только в
  UI, но и в postmortem-секции pipeline.log с явными метками)
- `cat TRAINING_EXIT_CODE` (которого пока нет, см. фикс уровня 1)
- `cat training.faulthandler.log` (которого пока нет, см. фикс уровня 2)
- `dmesg -T | tail -80` (без фильтра `oom` — видеть все kernel-события:
  segfaults, cgroup kills, nvidia driver errors)
- Recent dmesg NVRM/XID errors (`grep -iE 'nvrm|xid|nvidia'`)

### Вторичная проблема: PEFT double-apply → 0 trainable params

Training log:
```
trainer_builder:117 INFO - PEFT target_modules: all-linear
trainer_builder:186 INFO - LoRA config created: r=16, alpha=32, type=qlora
peft/mapping_func.py:72: UserWarning: You are trying to modify a model with PEFT for a second time.
peft/tuners_utils.py:285: UserWarning: Already found a `peft_config` attribute in the model.
```

Источник — `src/training/trainers/factory.py:269-274`:
```python
if hasattr(config.training, "type") and config.training.type in ("qlora", "lora", "adalora"):
    peft_config = create_peft_config(config)
    if peft_config is not None:
        trainer_kwargs["peft_config"] = peft_config
```

Это добавляет `peft_config` безусловно. Если модель уже обёрнута PEFT
(например, `Tranium/helixql-research-sft-pilot-1.5b` — SFT-чекпоинт, который
может публиковаться с `adapter_config.json`), TRL пытается применить
адаптер поверх существующего, что ломает `requires_grad` маску и даёт 0
trainable params. Это **не причина молчаливой смерти**, но это означает, что
даже если бы процесс дожил, обучение всё равно было бы no-op. Починить
обязательно перед следующим раном.

---

## Цели

1. **Гарантировать, что любая следующая смерть (native crash, signal,
   Python exception) оставит читаемый след** либо в `training.log`, либо в
   сиблинг-файле, который монитор заберёт перед cleanup'ом.
2. Починить PEFT double-apply, чтобы у LoRA-адаптера были ненулевые
   trainable params.
3. Не ломать существующие happy-path'ы (SFT pilot уже отработал нормально).

Отдельной целью **НЕ является**: починить reward-константу и галлюцинации
модели. Это отдельная ветка работы (prompt builder / system prompt для
GRPO-пути). Откроем после того, как увидим реальную причину смерти.

---

## План изменений

### Fix 1: Shell wrapper — unbuffered Python + exit code + fault log path
**Файл**: `src/pipeline/stages/managers/deployment_manager.py`
**Метод**: `_start_training_cloud`, секция `script_content` (стр. ~844–880)

**Изменения в сгенерированном bash-скрипте**:
1. После `exec >{log_file} 2>&1` добавить экспорт env vars:
   ```bash
   export PYTHONUNBUFFERED=1
   export PYTHONFAULTHANDLER=1
   export PYTHONFAULTHANDLER_PATH={workspace}/training.faulthandler.log
   ```
   `PYTHONFAULTHANDLER=1` активирует `faulthandler` автоматически, но пишет
   в stderr. Мы также передадим путь через env var и явно активируем в
   Python (Fix 2).
2. Записывать exit-код ПОСЛЕ `python`:
   ```bash
   echo "$exit_code $(date -Iseconds)" > {workspace}/TRAINING_EXIT_CODE
   ```
3. Улучшить ранний `TRAINING_FAILED`:
   ```bash
   if [ $exit_code -ne 0 ] && [ ! -f {workspace}/TRAINING_FAILED ] \
        && [ ! -f {workspace}/TRAINING_COMPLETE ]; then
     {
       echo "exit_code=$exit_code"
       echo "timestamp=$(date -Iseconds)"
       echo "signal_name=$(kill -l $((exit_code - 128)) 2>/dev/null || echo unknown)"
       echo "--- last 50 lines of training.log ---"
       tail -n 50 {workspace}/training.log 2>/dev/null || true
     } > {workspace}/TRAINING_FAILED || true
   fi
   ```

Это дёшево, не ломает существующую логику (условие `! -f TRAINING_FAILED`
сохранено), но даёт постmortem'у явный exit-код и сигнал.

### Fix 2: Python entry — faulthandler + atexit flush
**Файл**: `src/training/run_training.py`
**Метод**: `main()` (стр. 579)

1. В самом начале `main()` (до `argparse`):
   ```python
   import atexit
   import faulthandler

   # Persistent fault log — survives native crashes (SEGV, ABRT, BUS, FPE).
   _fault_log_path = os.environ.get(
       "PYTHONFAULTHANDLER_PATH",
       "training.faulthandler.log",
   )
   try:
       _fault_log = open(_fault_log_path, "w", buffering=1)
       faulthandler.enable(file=_fault_log, all_threads=True)
   except OSError:
       faulthandler.enable(all_threads=True)  # fallback: stderr

   # Emergency flush of all logging handlers on any exit path.
   def _flush_logging() -> None:
       for h in list(logger.handlers):
           try:
               h.flush()
           except Exception:
               pass
   atexit.register(_flush_logging)
   ```
2. Ничего не менять в `run_training()` — существующая try/except/finally
   цепочка корректна для Python-исключений.

`faulthandler.enable(file=..., all_threads=True)` — штатный способ
захватить SIGSEGV/SIGFPE/SIGABRT/SIGBUS/SIGILL с Python-стеком ВСЕХ потоков.
Запись идёт напрямую через `write(2)` → переживает Python runtime crash.

### Fix 3: Monitor postmortem — собрать все новые артефакты
**Файл**: `src/pipeline/stages/training_monitor.py`
**Метод**: `_collect_death_diagnostics` (стр. 741)

Расширить список `probes` (порядок важен — от самого информативного к
вспомогательному):

```python
probes: list[tuple[str, str, int]] = [
    ("exit_code", f"cat {workspace}/TRAINING_EXIT_CODE 2>/dev/null", 5),
    ("faulthandler", f"tail -n 200 {workspace}/training.faulthandler.log 2>/dev/null", 5),
    ("training_log_tail", f"tail -n 120 {workspace}/training.log 2>/dev/null", 10),
    ("training_failed_marker", f"cat {workspace}/TRAINING_FAILED 2>/dev/null", 5),
    ("dmesg_tail", "dmesg -T 2>/dev/null | tail -80", 5),
    ("dmesg_oom", "dmesg -T 2>/dev/null | grep -iE 'oom|kill|memory' | tail -40", 5),
    ("dmesg_nvidia", "dmesg -T 2>/dev/null | grep -iE 'nvrm|xid|nvidia' | tail -40", 5),
    ("nvidia_smi", "nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu --format=csv,noheader", 5),
    ("workspace_markers", f"ls -la {workspace}/TRAINING_* {workspace}/STOPPED_BY_WATCHDOG 2>/dev/null", 5),
]
```

Каждый probe уже обёрнут в try/except внутри метода — это контракт
«diagnostics must never block main error path», не ломаем.

Также в докстринге метода обновить описание: теперь это 9 probes,
первыми идут exit-код и faulthandler.

### Fix 4: PEFT double-apply guard
**Файл**: `src/training/trainers/factory.py` (стр. 269–274)

Перед добавлением `peft_config` проверить, не обёрнута ли модель уже:

```python
if hasattr(config.training, "type") and config.training.type in ("qlora", "lora", "adalora"):
    from src.training.trainer_builder import create_peft_config

    model_already_peft = hasattr(model, "peft_config") or hasattr(model, "base_model")
    if model_already_peft:
        logger.warning(
            "[TF:PEFT_GUARD] Model already has PEFT adapter "
            "(peft_config=%s, base_model=%s). Skipping peft_config to avoid "
            "double-apply. If this is a resume/adapter-cache path this is "
            "expected; if this is a fresh run from a HF SFT checkpoint, the "
            "checkpoint was uploaded with adapter_config.json and should be "
            "merged or reloaded as base.",
            hasattr(model, "peft_config"),
            hasattr(model, "base_model"),
        )
    else:
        peft_config = create_peft_config(config)
        if peft_config is not None:
            trainer_kwargs["peft_config"] = peft_config
```

Это **не** магическим образом возвращает trainable params — но это
**фиксирует контракт** и даёт чёткое предупреждение. Настоящая починка
(если SFT-чекпоинт действительно уже peft) идёт отдельно: либо мерджить
адаптер при публикации SFT на HF, либо в loader распознавать и раскрывать.
Для текущего ран-а `Trainable parameters: 0` — это именно то, что сейчас
тихо наблюдалось и делало обучение no-op.

---

## Критические файлы

| Файл | Что меняем | Риск |
|---|---|---|
| `src/pipeline/stages/managers/deployment_manager.py` | bash-скрипт в `_start_training_cloud` | низкий — bash остаётся `set -euo pipefail`, новые строки идемпотентны |
| `src/training/run_training.py` | `main()` — faulthandler + atexit | очень низкий — только добавление, ничего не удаляем |
| `src/pipeline/stages/training_monitor.py` | `_collect_death_diagnostics` — расширить probes | очень низкий — каждый probe уже в try/except |
| `src/training/trainers/factory.py` | PEFT double-apply guard | средний — меняет поведение для случаев, где модель пришла уже обёрнутой; раньше тихо ломалось, теперь явно предупреждает и пропускает |

**НЕ трогаем**:
- `grpo_pilot.yaml` / `sapo_pilot.yaml` / `dpo_pilot.yaml` — конфиги уже
  обновлены в прошлой сессии (`cloud_type ALL`, grad_checkpointing, halved
  bs/gen_bs, max_prompt 512, max_completion 192).
- Reward plugin / chat-template — отдельная работа после того, как увидим
  реальную причину смерти.
- Watchdog / runpod_stop_pod.sh — работают.

---

## Верификация

### Unit
1. `src/tests/unit/pipeline/test_stages_monitor.py::TestCollectDeathDiagnostics`
   — обновить ожидание количества probe'ов (3 → 9), добавить проверки на
   новые labels (`exit_code`, `faulthandler`, `training_log_tail`).
2. `src/tests/unit/pipeline/managers/test_deployment_manager.py` — если
   есть тест генерации `script_content`, обновить чтобы покрыть новые
   строки (`TRAINING_EXIT_CODE`, env exports). Если теста нет — не
   создаём (не ломаем scope).
3. `src/tests/unit/training/test_run_training.py` (или аналогичный) —
   smoke-тест, что `main()` не падает на `faulthandler.enable` даже если
   `open(_fault_log_path, "w")` выбрасывает (проверить fallback на stderr).
4. `src/tests/unit/training/trainers/test_factory.py` — тест на PEFT
   guard: если `model.peft_config` уже выставлен, `peft_config` не
   добавляется в `trainer_kwargs`.

Прогон: `pytest src/tests/unit/pipeline/test_stages_monitor.py src/tests/unit/training/trainers/ -x`

### Lint / Typecheck
- `ruff check src/pipeline/stages/managers/deployment_manager.py src/training/run_training.py src/pipeline/stages/training_monitor.py src/training/trainers/factory.py`
- `ruff format --check` тех же файлов
- `mypy src/pipeline/stages/training_monitor.py src/training/run_training.py src/training/trainers/factory.py` (deployment_manager.py исторически шумит, не трогаем если не добавили type issues)

### End-to-end
Следующий GRPO pilot запуск. Критерии приёмки:

1. **Если умрёт снова молча**, в `pipeline.log` секции
   `[MONITOR:POSTMORTEM]` должны быть НЕ пустыми как минимум:
   - `exit_code` — с числом и timestamp'ом
   - `faulthandler` — если был native crash
   - `training_log_tail` — хвост файла
   - `dmesg_tail` — kernel события
2. **Если умрёт из-за Python exception**, `training.log` должен содержать
   traceback (прилетает через `logger.exception` + atexit flush).
3. **Если PEFT-guard сработает**, в `training.log` должно быть
   `[TF:PEFT_GUARD]` warning и `Trainable parameters > 0`.

Без end-to-end верификации (без реального RunPod pod'а) мы не узнаем,
правда ли fault handler переживёт конкретный crash. Но faulthandler —
штатный механизм CPython, рассчитанный ровно на этот сценарий; доверяем
стандартной библиотеке.

---

## Последовательность исполнения

1. Fix 3 (monitor probes) + его unit-тесты — самый изолированный.
2. Fix 2 (faulthandler в run_training) + unit smoke.
3. Fix 1 (deployment_manager bash) — самый «грязный», делаем после чтобы
   не пересобирать тестовое окружение дважды.
4. Fix 4 (PEFT guard) + unit — независимый.
5. Прогнать ruff/mypy/pytest на затронутых файлах.
6. Коммит. Следующий ран — с реальным pod'ом, с глазами.
