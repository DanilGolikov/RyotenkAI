# Pipeline Orchestrator — Полная Декомпозиция (Facade + Specialized Managers)

## Context

Файл [src/pipeline/orchestrator.py](src/pipeline/orchestrator.py) — **2062 LOC**, 44 метода в одном классе `PipelineOrchestrator`. Hotspot score 99%, risk_type="bug-prone", +2377 добавленных строк против 338 удалённых за 90 дней — файл растёт, а не стабилизируется. Тесты уже жёстко завязаны на 14+ приватных методов (`test_orchestrator_stateful_helpers.py` и др.), что делает каждое изменение хрупким.

Частичная декомпозиция **уже начата** (decision `Extract ValidationArtifactManager`, active): извлечены [artifact_manager.py](src/pipeline/validation/artifact_manager.py) (244 LOC) и [transitioner.py](src/pipeline/state/transitioner.py) (144 LOC). Документ [orchestrator-decomposition-analysis.md](docs/plans/orchestrator-decomposition-analysis.md) (2026-04-10) устарел — отражает старое состояние на 2266 LOC.

**Цель**: довести до конца facade-паттерн по образцу [strategy_orchestrator.py](src/training/orchestrator/strategy_orchestrator.py) — оркестратор ~400-500 LOC как тонкий координатор, вся логика в 7 специализированных компонентах. Публичный API (`run()`, `list_stages()`, `list_restart_points()`, `get_stage_by_name()`, `notify_signal()`) **не меняется**.

**Outcome**: testable-in-isolation компоненты, снижение bus-factor риска, подготовка почвы для устранения 5 circular dependencies (scc-178).

## Текущее состояние (на 2026-04-21)

| Методов в orchestrator | LOC | Уже вынесено |
|---|---|---|
| 44 метода + 1 функция `run_pipeline` | 2062 | `ValidationArtifactManager`, функции `mark_stage_*`/`finalize_attempt_state` в `transitioner.py` |

**Группы методов (что ещё в god-файле):**

| Группа | Методов | LOC | Приоритет |
|---|---|---|---|
| MLflow Integration (`_setup_mlflow*`, `_ensure_mlflow_preflight`, `_teardown_mlflow_attempt`, `_get_mlflow_run_id`, `_open_existing_root_run`) | 6 | ~210 | Фаза 1 |
| Stage Planning (`_get_stage_index`, `_compute_enabled_stage_names`, `_normalize_stage_ref`, `_derive_resume_stage`, `_forced_stage_names`, `_validate_stage_prerequisites`) | 6 | ~100 | Фаза 1 (pure) |
| Lineage & Context Restore (`_invalidate_lineage_from`, `_restore_reused_context`, `_sync_root_context_from_stage`, `_extract_restart_outputs`, `_fill_from_context`, `_get_stage_skip_reason`) | 6 | ~180 | Фаза 2 |
| Config Drift (`_build_config_hashes`, `_validate_config_drift`, `_bootstrap_pipeline_state`, `_record_launch_rejection_attempt`) | 4 | ~180 | Фаза 2 |
| Stage Info Logging (`_log_stage_specific_info`) | 1 | ~120 | Фаза 2 |
| Reporting (`_aggregate_training_metrics`, `_collect_descendant_metrics`, `_generate_experiment_report`, `_print_summary`) | 4 | ~310 | Фаза 2 |
| Cleanup & Resources (`_cleanup_resources`, `_maybe_early_release_gpu`, `_is_inference_runtime_healthy`, `_flush_pending_collectors`) | 4 | ~130 | Фаза 3 |
| Main Loop (`_run_stateful`, `_init_stages`, `_init_collectors`, `__init__`, public API, `list_restart_points`) | 13 | ~830 | Фаза 3 |

## Целевая архитектура

```
src/pipeline/
├── orchestrator.py                          ← ~450 LOC (facade)
├── state/transitioner.py                    ← уже существует (+ lineage функции)
├── validation/artifact_manager.py           ← уже существует
├── mlflow_attempt/
│   ├── bootstrap.py                         ← MLflowBootstrap (one-shot, init)
│   └── attempt_manager.py                   ← MLflowAttemptManager (per-attempt)
├── executor/
│   ├── stage_planner.py                     ← StagePlanner (pure: index/enabled/resume/forced/prereq/normalize)
│   └── run_executor.py                      ← StatefulRunExecutor (main loop + cleanup coord)
├── context/
│   ├── propagator.py                        ← ContextPropagator (sync/fill/extract_restart_outputs)
│   └── stage_info_logger.py                 ← StageInfoLogger (только _log_stage_specific_info)
├── reporting/
│   └── summary_reporter.py                  ← ExecutionSummaryReporter (metrics+report+print_summary)
└── config_drift/
    └── drift_validator.py                   ← ConfigDriftValidator (hashes+drift+bootstrap)
```

**Принципы контрактов между компонентами:**

1. **Context как DTO-параметр, не `self.context`**. Каждый менеджер принимает `context: dict[str, Any]` явно — никто, кроме orchestrator'а, не владеет context'ом. Это убирает hidden coupling через `self.context`.
2. **`Result[T, AppError]` везде, никаких raise**. Как в [strategy_orchestrator.py](src/training/orchestrator/strategy_orchestrator.py).
3. **Менеджеры не берут `PipelineRunLock`** — это invariant orchestrator'а (deadlock risk).
4. **Protocol interfaces** для мокирования в тестах: `IMLflowAttemptManager`, `IStagePlanner` и т.д.
5. **Никаких новых импортов из orchestrator в менеджеры** — обратная сторона Facade, гарантирует отсутствие новых circular dependencies.

## Поэтапный план

### PR-серия (10 PR, каждый обратим независимо)

**Фаза 1 — Pure + изолированные (LOW RISK, HIGH ROI)**

- **PR-1: `StagePlanner` (pure functions)** — извлечь `_get_stage_index`, `_compute_enabled_stage_names`, `_normalize_stage_ref`, `_derive_resume_stage`, `_forced_stage_names`, `_validate_stage_prerequisites` в `src/pipeline/executor/stage_planner.py`. Stateless класс, тесты в `tests/unit/pipeline/executor/test_stage_planner.py`. Thin delegates в orchestrator.
- **PR-1a: миграция тестов PR-1** на прямой API `StagePlanner`, удаление thin delegates из orchestrator.
- **PR-2: `MLflowBootstrap`** — `_setup_mlflow`, `_ensure_mlflow_preflight` (one-shot lifecycle). Вызывается из `__init__`/`_bootstrap_pipeline_state`. Переносим в `src/pipeline/mlflow_attempt/bootstrap.py`.
- **PR-3: `MLflowAttemptManager`** — `_setup_mlflow_for_attempt`, `_teardown_mlflow_attempt`, `_get_mlflow_run_id`, `_open_existing_root_run` в `src/pipeline/mlflow_attempt/attempt_manager.py`. Критично: обернуть `setup_for_attempt` в try/finally (митигирует double-close — см. Риск 2.2).
- **PR-3a: миграция тестов MLflow** — `test_pipeline_orchestrator.py` assertions `orchestrator._mlflow_manager` → `orchestrator._mlflow_attempt.manager`.

**Фаза 2 — Функциональные группы (MEDIUM RISK)**

- **PR-4: `ContextPropagator`** — `_sync_root_context_from_stage`, `_fill_from_context`, `_extract_restart_outputs`, `_get_stage_skip_reason`. Context передаётся явно параметром. Файл: `src/pipeline/context/propagator.py`.
- **PR-5: `StageInfoLogger`** — вынести массивный `_log_stage_specific_info` (120 LOC) в `src/pipeline/context/stage_info_logger.py`. Зависит от context+mlflow_manager — оба передаются параметрами.
- **PR-6: `ConfigDriftValidator`** — `_build_config_hashes`, `_validate_config_drift`, часть `_bootstrap_pipeline_state` (drift-проверка), `_record_launch_rejection_attempt`. Файл: `src/pipeline/config_drift/drift_validator.py`.
- **PR-7: lineage функции → transitioner.py** — `_invalidate_lineage_from`, `_restore_reused_context` переезжают к уже существующим `mark_stage_*` в `transitioner.py` (они мутируют state — единое место).
- **PR-8: `ExecutionSummaryReporter`** — `_aggregate_training_metrics`, `_collect_descendant_metrics`, `_generate_experiment_report`, `_print_summary`. Вызывается только после finalize. Файл: `src/pipeline/reporting/summary_reporter.py`.

**Фаза 3 — Ядро оркестратора (HIGH RISK, HIGH VALUE)**

- **PR-9: `StatefulRunExecutor`** — вынести `_run_stateful` (~380 LOC) в `src/pipeline/executor/run_executor.py`. Orchestrator становится конструктором DI-графа + тонкие публичные методы.
  - Остаются в orchestrator: `run()`, `list_stages()`, `list_restart_points()`, `get_stage_by_name()`, `notify_signal()`, `__init__`, `_init_stages`, `_init_collectors`.
  - `_cleanup_resources`, `_maybe_early_release_gpu`, `_is_inference_runtime_healthy`, `_flush_pending_collectors` — переезжают в `StatefulRunExecutor` (они вызываются из его finally/try).
- **PR-10: миграция тестов + cleanup** — финальная чистка: удаление всех оставшихся thin delegates, обновление `test_orchestrator_stateful_*.py`, добавление Protocol interfaces для всех менеджеров.

**Итог**: orchestrator.py ≈ 450 LOC, каждый менеджер 100-300 LOC, полный test isolation.

## Риски — 3 итерации анализа

### Итерация 1 — Структурные

| # | Риск | Уровень | Митигация |
|---|---|---|---|
| 1.1 | `self.context` (dict) — hidden coupling между группами | **HIGH** | Контракт: менеджеры принимают `context: dict` параметром, НЕ пишут в него напрямую. Orchestrator делает `context.update(outputs)` после stage.run(). |
| 1.2 | Порядок инициализации в `__init__` (119 LOC) — 5+ менеджеров создадут ад зависимостей | MEDIUM | Компоненты, требующие stages (callback registration), получают их через отдельный `attach_stages(stages)` после `_init_stages()`. Phase 3 — вынести init в `PipelineOrchestratorBuilder` если станет > 150 LOC. |
| 1.3 | Tests доступают приватные атрибуты (14+ мест) | MEDIUM | Thin delegates + миграция тестов в PR-*a parallel-series; финал в PR-10. |

### Итерация 2 — Runtime

| # | Риск | Уровень | Митигация |
|---|---|---|---|
| 2.1 | Signal propagation — `_shutdown_signal_name` не проверяется в stage loop | **HIGH** | **Вне scope декомпозиции** — отдельный fix-TODO. Но: `StatefulRunExecutor.__init__` получает `signal_source: Callable[[], str \| None]` — это откроет путь для fix без ещё одного рефакторинга. |
| 2.2 | MLflow nested run double-close при ошибке в setup | **HIGH** | PR-3: `setup_for_attempt` в `try/finally`, который cleanup частично открытых runs. Unit-тест специально на это. |
| 2.3 | Exceptions из `finally` блока `_run_stateful` теряются | MEDIUM | PR-9: каждый cleanup-шаг в свой try/except с `logger.exception`, без ре-рейза. |
| 2.4 | `PipelineRunLock` держится `_run_stateful` — менеджер может случайно попытаться взять lock повторно | MEDIUM | Documented invariant: «no manager acquires run_lock». Добавить assert в debug builds. |

### Итерация 3 — Тестовые/Backward-compat

| # | Риск | Уровень | Митигация |
|---|---|---|---|
| 3.1 | 6 тестовых файлов завязаны на приватные методы orchestrator'а | MEDIUM | Thin delegates на время фазы; миграция тестов — отдельные PR-*a сразу после extraction PR. Удаление делегатов — только после миграции тестов. |
| 3.2 | `_init_stages` регистрирует validation callbacks на `DatasetValidator` — callback wiring неявный | MEDIUM | Документировать в `ValidationArtifactManager.__init__` требование "после init stages вызвать `register_validation_callbacks(dataset_validator_stage)`". |
| 3.3 | Новые модули `src/pipeline/mlflow_attempt/`, `executor/`, `context/`, `reporting/`, `config_drift/` — не вызвать новых circular deps (есть scc-178 на 156 файлов) | LOW | Jarr: новые модули импортируют только `state/`, `artifacts/`, `stages/` (низкоуровневые) + stdlib. Никаких обратных импортов. Добавить import-linter rule в PR-1. |
| 3.4 | Config drift hash fallback (legacy state без `model_dataset_config_hash`) | LOW | PR-6: сохранить текущий fallback в `ConfigDriftValidator`, задокументировать, unit-тест на оба пути. |

## Нерешённые вопросы → ответы

**Q1. DI через конструктор или фабрики?**  
A. Прямая инстанциация в `__init__`. Фабрики overhead без гейна (training orchestrator делает так же). Для тестов — Protocol interfaces и monkey-patch.

**Q2. Сохранять ли self.context в orchestrator, или полностью DTO (AttemptContext dataclass)?**  
A. **Phase 1-2 — остаётся `self.context` (dict)**, менеджеры принимают через параметры. **Phase 3 (PR-9)** — ввести `AttemptContext` dataclass в `StatefulRunExecutor` для type-safety; orchestrator продолжает использовать dict для backward compat stages.

**Q3. StagePlanner — класс или набор функций?**  
A. **Класс** (stateless) с `self.stages: list[PipelineStage]` — потому что все методы оперируют одним списком, функции с параметром `stages=` делали бы вызовы многословными.

**Q4. Что делать с `run_pipeline()` на уровне модуля (строка 2026)?**  
A. Оставить как есть — это CLI-entry-point, не часть god-класса.

**Q5. `StatefulRunExecutor` (PR-9) — где должны жить `_maybe_early_release_gpu` и `_is_inference_runtime_healthy`?**  
A. Внутри `StatefulRunExecutor` как приватные хелперы — они вызываются только из loop'а, не имеют смысла вне его контекста.

**Q6. Нужно ли обновлять `orchestrator-decomposition-analysis.md`?**  
A. **Нет** — он устарел. PR-1 включит: (а) пометку `DEPRECATED — см. parallel-snacking-quilt.md` в верхней части файла, (б) создание decision `update_decision_records(action="create")` ссылающегося на этот план.

**Q7. Decision records?**  
A. После PR-1 создать decision "Full decomposition of PipelineOrchestrator into Facade+Managers" (status=active). После каждого PR-N — `update_decision_records(action="update", consequences=[...])`.

## Критические файлы

**Изменяемые в каждом PR:**
- [src/pipeline/orchestrator.py](src/pipeline/orchestrator.py) — целевой god-файл
- [src/tests/unit/pipeline/test_orchestrator_stateful_flow.py](src/tests/unit/pipeline/test_orchestrator_stateful_flow.py)
- [src/tests/unit/pipeline/test_orchestrator_stateful_helpers.py](src/tests/unit/pipeline/test_orchestrator_stateful_helpers.py)
- [src/tests/unit/test_pipeline_orchestrator.py](src/tests/unit/test_pipeline_orchestrator.py)
- [src/tests/unit/test_pipeline_orchestrator_missing_lines.py](src/tests/unit/test_pipeline_orchestrator_missing_lines.py)
- [src/tests/unit/pipeline/test_restart_policy.py](src/tests/unit/pipeline/test_restart_policy.py)
- [src/tests/unit/test_early_pod_release.py](src/tests/unit/test_early_pod_release.py)

**Reference (не меняем, используем как образец):**
- [src/training/orchestrator/strategy_orchestrator.py](src/training/orchestrator/strategy_orchestrator.py) — Facade + Components pattern
- [src/training/orchestrator/chain_runner.py](src/training/orchestrator/chain_runner.py) — чистый loop-компонент
- [src/pipeline/validation/artifact_manager.py](src/pipeline/validation/artifact_manager.py) — уже извлечённый пример
- [src/pipeline/state/transitioner.py](src/pipeline/state/transitioner.py) — уже извлечённый пример (функции-мутаторы)

**Переиспользуемые утилиты:**
- `Result[T, AppError]` из [src/utils/result.py](src/utils/result.py) — обязательно во всех новых компонентах
- `logger, console` из [src/utils/logger.py](src/utils/logger.py)
- `LogLayout` из [src/utils/logs_layout.py](src/utils/logs_layout.py) — для всех менеджеров, пишущих логи
- `utc_now_iso` из [src/pipeline/artifacts/base.py](src/pipeline/artifacts/base.py) — timestamps
- Existing `StageNames`, `PipelineContextKeys` из [src/pipeline/stages/__init__.py](src/pipeline/stages/__init__.py)

## Verification

**Для каждого PR-N:**

1. **Unit-тесты нового компонента** — coverage > 90% для каждого extracted манаджера. Директория: `src/tests/unit/pipeline/<subdir>/test_<component>.py`.
2. **Smoke-test orchestrator'а**: `pytest src/tests/unit/pipeline/test_orchestrator_stateful_flow.py -v` — должен проходить неизменным (пока не дойдём до PR-N-a миграции тестов).
3. **E2E**: `pytest src/tests/e2e/test_full_pipeline_e2e.py` — критический smoke полного pipeline'а.
4. **Lint + typecheck**: `ruff check src/pipeline/ && mypy src/pipeline/`.
5. **Import-linter (PR-1 вводит)**: проверка, что новые модули не создают циклов.

**После PR-10 (финал):**

1. `wc -l src/pipeline/orchestrator.py` → ожидаем ~450 LOC (verification против цели).
2. `get_risk(["src/pipeline/orchestrator.py"])` через Repowise MCP — ожидаем hotspot_score < 70%, risk_type перестаёт быть "bug-prone".
3. `get_dead_code(min_confidence=0.7)` — проверка, что ничего из extraction'а не осталось dead.
4. Полный test-suite `pytest src/tests/` — зелёный.
5. `update_decision_records(action="update_status", decision_id="...", status="completed")` для "Extract ValidationArtifactManager" с ссылкой на новый decision.

## Метрики успеха

| Метрика | До | Цель |
|---|---|---|
| orchestrator.py LOC | 2062 | ≤ 500 |
| Методов в PipelineOrchestrator | 44 | ≤ 12 |
| Публичный API | 5 методов | 5 методов (неизменно) |
| Testable-in-isolation компоненты | 2 (только вынесённые) | 9 |
| Hotspot score | 0.985 | < 0.70 |
| Bus factor | 1 | 1 (не улучшается рефакторингом, но код становится проще читать) |
| Duration (оценка) | — | 5-7 дней итеративно, 10 PR |
