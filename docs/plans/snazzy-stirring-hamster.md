# Аудит `dict[str, Any]` → DTO в `src/`

## Context

В `src/` найдено **~160 сигнатур** с `dict[str, Any]` в 50 файлах. Проект имеет зрелые конвенции типизированных данных:
- **Pydantic v2** (`StrictBaseModel`) — конфиги и внешние контракты (91+ класс в `src/config/`).
- **`@dataclass`** — runtime state, domain entities, frozen immutable (`src/pipeline/state/models.py`, `src/providers/runpod/models.py`).
- **`TypedDict`** — JSON-артефакты с известной схемой (`src/pipeline/artifacts/schemas.py`).

Однако типизация непоследовательна: ряд «узлов» пайплайна продолжает передавать `dict[str, Any]` через границы модулей, хотя их ключевой набор фиксирован. Это снижает безопасность типов, затрудняет рефакторинг (нет IDE-автокомплита и статического контроля ключей) и маскирует неявные контракты. Цель аудита — выявить и приоритизировать кандидатов для замены на явные DTO, следуя существующим конвенциям.

**Скоп:** только `src/`. Критерий: (1) функции с `dict[str, Any]` в сигнатурах, (2) JSON-артефакты и payload'ы. Рефакторинг реализуется отдельными PR'ами — текущий артефакт является только аудитом.

## Конвенции замены (следовать текущим)

| Куда | Что использовать | Примеры из кода |
|------|------------------|----------------|
| Конфиг (YAML/ENV → Python) | `Pydantic BaseModel` (наследуется от `StrictBaseModel`) | [src/config/base.py](src/config/base.py), [src/config/training/schema.py](src/config/training/schema.py) |
| Внешний API ответ/запрос (RunPod, OpenAI) | `Pydantic BaseModel` с `model_config = {"extra": "allow"}` | нет прецедента — надо ввести |
| Runtime state (изменяемый) | `@dataclass` с `to_dict()/from_dict()` | [src/pipeline/state/models.py](src/pipeline/state/models.py), [src/training/managers/data_buffer/state_models.py](src/training/managers/data_buffer/state_models.py) |
| Domain entity (immutable) | `@dataclass(frozen=True, slots=True)` | [src/providers/runpod/models.py](src/providers/runpod/models.py), [src/pipeline/domain/run_context.py](src/pipeline/domain/run_context.py) |
| JSON-артефакт (контракт writer↔reader) | `TypedDict` | [src/pipeline/artifacts/schemas.py](src/pipeline/artifacts/schemas.py) |
| Плагинный `metadata`/`extra`/`params` (динамика) | **оставить `dict[str, Any]`** | все validation/evaluation plugins |
| TRL `**kwargs` pass-through | **оставить `dict[str, Any]`** | reward plugins `build_config_kwargs` |

## Реестр кандидатов

Всего в реестре **~45 значимых мест**. Разделены на HIGH / MEDIUM / LOW.

### HIGH — пересекают границы модулей, фиксированный набор полей

| # | Место | Текущий тип | Предлагаемый DTO | Форма | Обоснование |
|---|-------|-------------|------------------|-------|-------------|
| H1 | [src/training/orchestrator/metrics_collector.py:42](src/training/orchestrator/metrics_collector.py:42) `MetricsCollector.extract_from_trainer` | `dict[str, Any]` | `TrainingMetricsSnapshot` | `@dataclass` | Набор фикс. полей: `train_loss`, `eval_loss`, `learning_rate`, `train_runtime`, `global_step`, `epoch`, `peak_memory_gb`, throughput. Передаётся в `DataBuffer`, MLflow, отчёты. |
| H2 | [src/training/orchestrator/metrics_collector.py:142](src/training/orchestrator/metrics_collector.py:142) `MetricsCollector.aggregate_phases` | `dict[str, Any]` | `PhasesMetricsAggregate` | `@dataclass` | `total_phases`, `total_steps`, `total_runtime_seconds`, `final_loss`, `per_phase: list[TrainingMetricsSnapshot]`. |
| H3 | [src/training/managers/data_buffer/state_models.py](src/training/managers/data_buffer/state_models.py) `PhaseState.metrics: dict[str, Any]` | `dict[str, Any]` | `TrainingMetricsSnapshot` | `@dataclass` | Тот же набор, что и H1 — консолидировать. |
| H4 | [src/providers/runpod/sdk_adapter.py](src/providers/runpod/sdk_adapter.py) `get_pod`/`list_pods` (возвращает dict) | `dict[str, Any]` / `list[dict[str, Any]]` | `RunPodPodResponse` | `Pydantic BaseModel` с `extra="allow"` | Ключи: `id`, `desiredStatus`, `gpuCount`, `portMappings`, `publicIp`, `machine.gpuDisplayName`. Уже есть связанный `PodSnapshot` (domain frozen dataclass) — Pydantic модель берёт роль «сырого ответа SDK» перед нормализацией в `PodSnapshot`. |
| H5 | [src/providers/runpod/sdk_adapter.py](src/providers/runpod/sdk_adapter.py) `create_pod_from_payload(payload)` | `dict[str, Any]` | `RunPodCreatePayload` | `TypedDict` | Ключи: `name`, `imageName`, `gpuTypeIds`, `gpuCount`, `cloudType`, `supportPublicIp`, `dataCenterId`, `env`, `networkVolumeId`, `ports`, `volumeInGb`. Payload собирается несколькими слоями — контракт must. |
| H6 | [src/providers/runpod/inference/pods/api_client.py](src/providers/runpod/inference/pods/api_client.py) `list_network_volumes` / `get_network_volume` / `create_network_volume` | `dict[str, Any]` | `NetworkVolumeResponse` + `NetworkVolumeCreatePayload` | Pydantic + TypedDict | Ключи стабильные. |
| H7 | [src/providers/inference/interfaces.py:97](src/providers/inference/interfaces.py:97) `InferenceArtifacts.manifest: dict[str, Any]` | `dict[str, Any]` | `InferenceManifest` | `@dataclass` | JSON-артефакт `inference_manifest.json`, читается/пишется несколькими провайдерами (RunPod, single_node). Нужно разнести на подструктуры: `runpod`, `ssh`, `model`, `serve`, `vllm`. |
| H8 | [src/reports/domain/entities.py](src/reports/domain/entities.py) `ExperimentData.{validation,evaluation,training,deployment,model,inference}_results: dict \| None` | `dict[str, Any]` | `Union[ValidationArtifactData, EvalArtifactData, …]` per-поле | уже есть TypedDict в [src/pipeline/artifacts/schemas.py](src/pipeline/artifacts/schemas.py) | Адаптер читает эти dict'ы — типизируем под существующие схемы. |
| H9 | [src/reports/domain/entities.py](src/reports/domain/entities.py) `PhaseData.config: dict[str, Any]` | `dict[str, Any]` | `PhaseConfigSnapshot` | `@dataclass` | Нормализованный конфиг фазы: `learning_rate`, `batch_size`, `epochs`, `strategy_params`. |
| H10 | [src/pipeline/artifacts/base.py:37](src/pipeline/artifacts/base.py:37) `StageArtifactEnvelope.data: dict[str, Any]` | `dict[str, Any]` | `StageArtifactEnvelope[T: TypedDict]` | `@dataclass` + `Generic[T]` | Есть типизированные TypedDict, но envelope обобщённый. Ввести Generic-параметр или discriminated union по `stage`. |

### MEDIUM — локально повторяющиеся, пересекают ≤2 модуля

| # | Место | Предлагаемый DTO | Форма |
|---|-------|------------------|-------|
| M1 | [src/providers/runpod/pod_control.py:53](src/providers/runpod/pod_control.py:53) `get_ssh_info` → `{host, port}` | `SSHEndpoint` | `TypedDict` (или переиспользовать существующий `SshEndpoint` frozen dataclass в [src/providers/runpod/models.py](src/providers/runpod/models.py)) |
| M2 | [src/providers/runpod/inference/pods/pod_session.py:132](src/providers/runpod/inference/pods/pod_session.py:132) `vllm_cfg: dict[str, Any]` | `VLLMRuntimeConfig` | `Pydantic BaseModel` (переиспользовать/согласовать с [src/config/inference/engines/vllm.py](src/config/inference/engines/vllm.py)) |
| M3 | [src/providers/single_node/inference/artifacts.py](src/providers/single_node/inference/artifacts.py) `_build_ssh_args(ssh_cfg: dict)` | `SSHConnectionConfig` | переиспользовать существующий `Pydantic SSHConfig` в [src/config/providers/ssh.py](src/config/providers/ssh.py) |
| M4 | [src/evaluation/model_client/openai_client.py](src/evaluation/model_client/openai_client.py) `payload: dict[str, Any]` | `ChatCompletionRequest` / `ChatCompletionResponse` | `TypedDict` |
| M5 | [src/evaluation/plugins/llm_judge/cerebras_judge.py](src/evaluation/plugins/llm_judge/cerebras_judge.py) парсинг Judge-ответа | `JudgeResponse` (`reasoning`, `score`) | `TypedDict` |
| M6 | [src/evaluation/runner.py:44](src/evaluation/runner.py:44) `plugin_meta: dict[str, Any]` | `PluginMetadata` (`plugin_name`, `description`, `params`, `thresholds`) | `TypedDict` |
| M7 | [src/training/orchestrator/strategy_orchestrator.py](src/training/orchestrator/strategy_orchestrator.py) `get_summary() → dict` | `OrchestratorSummary` | `@dataclass` |
| M8 | [src/training/orchestrator/resume_manager.py](src/training/orchestrator/resume_manager.py) `get_interrupt_info() → dict` | `InterruptInfo` (`reason`, `phase_idx`, `checkpoint_path`, `timestamp`) | `@dataclass` |
| M9 | [src/training/orchestrator/shutdown_handler.py](src/training/orchestrator/shutdown_handler.py) `get_shutdown_info() → dict` | `ShutdownInfo` (`signal`, `timestamp`, `was_graceful`) | `@dataclass` |
| M10 | [src/training/mlflow/event_log.py](src/training/mlflow/event_log.py) `log_event(**kwargs)` | `EventLogEntry` (`category`, `source`, `phase_idx`, `strategy`, `status`, …) | `@dataclass` — kwargs фактически структурные |
| M11 | [src/reports/adapters/mlflow_adapter.py](src/reports/adapters/mlflow_adapter.py) `_normalize_params(params)` | `NormalizedExperimentParams` | `TypedDict` |
| M12 | [src/reports/adapters/mlflow_adapter.py](src/reports/adapters/mlflow_adapter.py) `_load_json_events()` | `TrainingEventRecord` | `@dataclass` |
| M13 | [src/reports/adapters/mlflow_adapter.py](src/reports/adapters/mlflow_adapter.py) `_extract_gpu_info` / `_extract_model_info` | `GPUInfo` / `ModelInfo` | `@dataclass` (использовать как общий, если уже есть в utils) |
| M14 | [src/reports/adapters/mlflow_adapter.py:427](src/reports/adapters/mlflow_adapter.py:427) `phase_config: dict[str, Any]` | `PhaseConfigDict` | `TypedDict` |
| M15 | [src/reports/models/report.py](src/reports/models/report.py) `effective_config: dict[str, Any]` | `ConfigSnapshot` | `@dataclass` |
| M16 | [src/tui/adapters/run_catalog.py:26](src/tui/adapters/run_catalog.py:26) `run_catalog: dict[str, object]` | `RunCatalogRow` (`name`, `status`, `created_at`, …) | `@dataclass(frozen=True)` |

### LOW — оставить `dict[str, Any]` (осознанно) или локальный лёгкий рефакторинг

| # | Место | Почему оставить |
|---|-------|-----------------|
| L1 | [src/training/reward_plugins/factory.py:14](src/training/reward_plugins/factory.py:14) `RewardPluginResult.config_kwargs` / `trainer_kwargs` | Чистый `**kwargs` pass-through в TRL `*Config` и `Trainer`. Структурирование ломает расширяемость плагинов. |
| L2 | [src/training/reward_plugins/base.py](src/training/reward_plugins/base.py) `build_config_kwargs` / `build_trainer_kwargs` | То же — pass-through. |
| L3 | [src/data/validation/base.py](src/data/validation/base.py) `ValidationResult.params` / `.thresholds` | Плагин-специфичные, динамические по типу плагина. |
| L4 | [src/evaluation/plugins/base.py](src/evaluation/plugins/base.py) `EvalSample.metadata`, `EvalResult.metrics` | Плагин-специфичные поля; извлекаются из JSONL с произвольной схемой. |
| L5 | [src/providers/runpod/inference/pods/pod_session.py:110](src/providers/runpod/inference/pods/pod_session.py:110) `extra: dict[str, Any]` | Extensibility-сумочка сессии. |
| L6 | [src/providers/training/factory.py:26](src/providers/training/factory.py:26) `ProviderConstructor` принимает `dict[str, Any]` | Protocol, передаёт произвольные provider kwargs. |
| L7 | [src/providers/runpod/inference/pods/api_client.py](src/providers/runpod/inference/pods/api_client.py) `_request_json(params, payload)` | Generic HTTP слой. |
| L8 | [src/pipeline/orchestrator.py](src/pipeline/orchestrator.py) `context: dict[str, Any]` | Lookup-table по `PipelineContextKeys` — строковые ключи из Enum-подобного класса. Замена потребует полной перестройки механизма stage-inter-communication. Отдельная задача. |
| L9 | [src/utils/memory_manager.py](src/utils/memory_manager.py), [src/utils/environment.py](src/utils/environment.py) | Низкоуровневые системные снапшоты с переменным набором ключей (зависит от платформы). |
| L10 | `save_stage_artifact(context: dict[str, Any], …)` [src/pipeline/artifacts/base.py:188](src/pipeline/artifacts/base.py:188) | Принимает orchestrator context (L8). |

### Итоговая статистика

- HIGH: 10 кандидатов — ядро рефакторинга.
- MEDIUM: 16 кандидатов.
- LOW (не трогать / отдельная задача): 10 мест.
- Всего в аудите: ~36 обсуждаемых мест из 160 сигнатур (остальные — локальные/однократные).

## Топ-3 рекомендуемых «быстрых победы»

1. **`TrainingMetricsSnapshot`** — одно место в `MetricsCollector`, но влияет на `DataBuffer`, MLflow-logger и отчёты. Самый высокий ROI (H1+H2+H3 в одной задаче).
2. **`RunPodPodResponse` + `RunPodCreatePayload`** — ясный контракт с SDK, уменьшит KeyError'ы при изменениях API RunPod (H4+H5).
3. **`InferenceManifest`** — JSON-артефакт пересекает RunPod/single_node/inference клиенты; типизация упростит работу с manifest (H7).

## Риски и нерешённые вопросы

*Зафиксированы после 3 итераций анализа плана.*

### Риски

- **R1. Обратная совместимость JSON state.** `StageArtifactEnvelope` и `PipelineState` уже имеют сохранённые JSON'ы с `from_dict`. Ввод Generic или смена формы ломает чтение старых state-файлов.
  **Митигация:** сохранить backward-compat в `from_dict` — добавить версионирование и fallback-ветку для «сырых» dict.

- **R2. Pydantic для RunPod SDK отстаёт от реальной схемы.** RunPod API добавляет поля без предупреждения.
  **Митигация:** ВСЕ внешние response-модели создавать с `model_config = ConfigDict(extra="allow")` и `populate_by_name=True`. Не наследовать от `StrictBaseModel`.

- **R3. Потеря гибкости в reward plugins.** Попытка типизировать `config_kwargs`/`trainer_kwargs` в `RewardPluginResult` сломает плагины.
  **Митигация:** явно зафиксировано в LOW, не рефакторить.

- **R4. Каскад изменений вокруг `ExperimentData`.** 6 dict-полей читаются из артефактов в `MLflowAdapter` и рендерятся в `ReportBuilder`. Любая смена типов затронет обе стороны.
  **Митигация:** миграция полями по одному, начиная с уже типизированного `training_results` → `TrainingArtifactData`.

- **R5. Дубликат `PipelineState`.** Класс с одинаковым именем в [src/pipeline/state/models.py](src/pipeline/state/models.py) и [src/training/managers/data_buffer/state_models.py](src/training/managers/data_buffer/state_models.py). Два разных назначения (pipeline JSON-persist vs training runtime), но риск путаницы при импорте.
  **Митигация:** out of scope для этого аудита, но зафиксировать как follow-up: переименовать один из классов (напр., `TrainingBufferState`).

- **R6. Opt-значения в `TrainingMetricsSnapshot`.** `peak_memory_gb` и throughput-метрики опциональны (None если нет CUDA). Нужна явная семантика `Optional[float]`.
  **Митигация:** все optional поля — `field(default=None)` с явным `| None` в аннотации.

- **R7. `to_dict/from_dict` vs `model_dump()`.** Смешение dataclass-ручной сериализации и Pydantic приведёт к двум разным API сериализации в кодовой базе.
  **Митигация:** dataclass'ы используют `dataclasses.asdict` + ручной `from_dict`; Pydantic — `model_dump()` / `model_validate()`. Не смешивать в одном объекте.

- **R8. Discriminated union для `StageArtifactEnvelope[T]`.** Generic over TypedDict на runtime не валидируется (Python не знает TypedDict на etime). Может привести к ложному ощущению типобезопасности.
  **Митигация:** либо использовать Literal-discriminator поле `stage` + `match/case` в reader'е, либо оставить `dict[str, Any]` с типизацией только на стороне writer'а. Решение — «оставить как есть» и типизировать на writer-стороне через TypedDict (фактически уже сделано частично).

### Нерешённые вопросы

- **Q1.** Pydantic v2 strict vs `extra="allow"` — какой режим для **внешних API** (RunPod, OpenAI)? **Ответ:** `extra="allow"` для external (R2), `extra="forbid"` для internal конфигов (уже `StrictBaseModel`).
- **Q2.** Нужно ли вводить `from_dict` на Pydantic-моделях или использовать `model_validate`? **Ответ:** использовать `model_validate` — нативно и с валидацией. Не имитировать dataclass-API на Pydantic.
- **Q3.** Куда класть новые DTO? **Ответ (по конвенции):**
  - Training runtime → `src/training/models/` (создать новую директорию) или `src/training/orchestrator/metrics_models.py` рядом с collector'ом.
  - RunPod external → `src/providers/runpod/api_models.py` (отдельно от доменных `models.py`).
  - Inference manifest → `src/providers/inference/manifest.py`.
  - Reports entities → в существующий `src/reports/domain/entities.py`.
- **Q4.** Миграция существующих JSON'ов — нужна ли? **Ответ:** нет — только backward-compat в reader'ах (см. R1). Новые запуски будут писать новый формат, старые — читаться через fallback.

## Следующие шаги (вне этого аудита)

1. Утвердить реестр с командой.
2. Сформировать backlog: 1 PR на 1 DTO (или кластер тесно связанных), начать с H1+H2+H3 (`TrainingMetricsSnapshot`).
3. Перед каждым PR — мини-план с конкретными файлами и сигнатурами.
4. Покрыть тестами сериализацию/десериализацию новых DTO.

## Верификация аудита

Этот артефакт — только аудит, код не меняется. Проверить полноту можно так:

```bash
# Пересчитать общее число сигнатур в src/
rg -c -t py 'dict\[str, Any\]|Dict\[str, Any\]' src | awk -F: '{s+=$2} END {print s}'
# Ожидание: ~160+ (подтверждение масштаба)

# Список файлов с max вхождений — проверить, что все HIGH/MEDIUM попали в реестр
rg -c -t py 'dict\[str, Any\]' src | sort -t: -k2 -n -r | head -30
```

Полнота реестра — по признаку, что топ-30 файлов по количеству `dict[str, Any]` покрыты таблицами HIGH/MEDIUM/LOW.
