# Plugin System Architecture Analysis for Community Refactor

## Executive Summary

Проанализирована вспомогательная система плагинов RyotenkAI для подготовки к рефакторингу на `community/` папку. Документ содержит полный обзор конфигурации, загрузки, тестирования и интеграции плагинов всех трёх типов (валидация, эвалюэйшн, реварды).

---

## A. Конфигурация плагинов со стороны пользователя

### A1. YAML-конфигурация активных плагинов

**Dataset Validation Plugins**
- Файл конфига: `src/config/datasets/validation.py`
- YAML структура:
```yaml
datasets:
  my_dataset:
    source_type: local
    train_path: "data/train.jsonl"
    validations:
      mode: "fast" | "full"
      critical_failures: <int>  # 0 = never stop
      plugins:
        - id: "unique_id_1"
          plugin: "plugin_name"  # registered in ValidationPluginRegistry
          apply_to: ["train", "eval"]
          params: {<key>: <value>}
          thresholds: {<key>: <value>}
```
- Источник: `src/config/datasets/schema.py` → `DatasetConfig` → `validations: DatasetValidationsConfig`
- Класс конфига: `DatasetValidationPluginConfig` (`src/config/datasets/validation.py:14-39`)
  - Поля: `id`, `plugin`, `apply_to`, `params`, `thresholds`
  - Валидация: `validate_unique_plugin_ids()` (line 72-83)

**Evaluation Plugins**
- Файл конфига: `src/config/evaluation/schema.py`
- YAML структура:
```yaml
evaluation:
  enabled: true
  dataset:
    path: "data/eval/eval.jsonl"
  evaluators:
    plugins:
      - id: "syntax_check_1"
        plugin: "helixql_syntax"
        enabled: true
        save_report: false
        params: {}
        thresholds:
          min_valid_ratio: 0.80
```
- Классы конфига:
  - `EvaluationConfig` (line 76-99)
  - `EvaluatorPluginConfig` (line 43-57)
    - Поля: `id`, `plugin`, `enabled`, `save_report`, `params`, `thresholds`
    - Валидация: `validate_unique_plugin_ids()` (line 65-73)

**Training Reward Plugins**
- Указываются в стратегии обучения, НЕ в глобальном конфиге
- YAML структура (в `training.strategies[].params`):
```yaml
training:
  strategies:
    - strategy_type: "grpo"
      params:
        reward_plugin: "helixql_compiler_semantic"
        reward_params:
          key1: value1
```
- Загрузка: `src/training/reward_plugins/factory.py:38-46`
  - `phase_config.params.get("reward_plugin")` → название плагина
  - `phase_config.params.get("reward_params", {})` → параметры

### A2. Связь с плагином: path, name, version?

**Актуальный механизм связи: `name` (зарегистрированное имя)**
1. Плагин регистрируется с уникальным `name` (ClassVar):
   ```python
   @ValidationPluginRegistry.register
   class MyPlugin(ValidationPlugin):
       name = "my_plugin"  # ← это ключ для поиска
   ```
2. В конфиге указывается это `name`:
   ```yaml
   - plugin: "my_plugin"  # ← прямая ссылка на ClassVar.name
   ```
3. Registry lookup:
   ```python
   plugin = ValidationPluginRegistry.get_plugin("my_plugin", params=..., thresholds=...)
   ```

**Не используется:**
- `path` — плагины находятся по discovery (рекурсивное сканирование `src/{data/validation,evaluation,training/reward_plugins}/plugins/`)
- `version` — есть в `ClassVar.version`, но не проверяется при загрузке (информационно)

### A3. Примеры конфигов в `configs/presets/`

Проверено: `configs/presets/01-small.yaml`, `02-medium.yaml`, `03-large.yaml`
- **Результат:** В текущих пресетах НЕ используются плагины валидации/эвалюэйшн/реварды
- Примеры — только в документации (`src/data/validation/README.md:189-201`, `src/evaluation/plugins/README.md:194-211`)

---

## B. Тесты плагинов

### B1. Структура тестов

**Текущий паттерн: тесты рядом с реализацией**
```
src/data/validation/plugins/
├── base/
│   ├── min_samples.py
│   └── test_avg_length.py          ← тест для avg_length.py (в другой папке!)
├── dpo/
│   ├── helixql_preference_semantics.py
│   └── test_helixql_preference_semantics.py  ← РЯДОМ с реализацией
├── sapo/
│   ├── helixql_sapo_prompt_contract.py
│   └── test_helixql_sapo_prompt_contract.py  ← РЯДОМ
├── sft/
│   ├── helixql_gold_syntax_backend.py
│   └── test_helixql_gold_syntax_backend.py   ← РЯДОМ
```

**Обнаруженные тесты в `src/`:**
- `src/data/validation/plugins/dpo/test_helixql_preference_semantics.py` ← рядом
- `src/data/validation/plugins/sapo/test_helixql_sapo_prompt_contract.py` ← рядом
- `src/data/validation/plugins/sft/test_helixql_gold_syntax_backend.py` ← рядом
- `src/evaluation/plugins/semantic/test_helixql_semantic_match.py` ← рядом
- `src/evaluation/plugins/syntax_check/test_helixql_generated_syntax_backend.py` ← рядом

**Unit-тесты в отдельной структуре:**
- `src/tests/unit/data/validation/plugins/base/` — для base/ плагинов
- `src/tests/unit/data/validation/plugins/dpo/` — для dpo/ плагинов
- `src/tests/unit/data/validation/plugins/sft/` — для sft/ плагинов
- `src/tests/unit/evaluation/` — для evaluation плагинов
- `src/tests/unit/training/` — для reward плагинов

### B2. Последствия для PyTest Discovery

**Проблема:** При переезде в `community/` папку:
1. **Тесты рядом с реализацией** (в `src/`) будут обнаружены pytest
2. **Дублирующиеся unit-тесты** в `src/tests/unit/` также будут найдены
3. Возможны конфликты при запуске `pytest` без фильтрации

**Рекомендация:**
- Переносить плагины И их встроенные тесты в `community/{type}/{category}/{plugin_name}.py` + `community/{type}/{category}/test_{plugin_name}.py`
- **ИЛИ** удалить встроенные тесты и опираться только на `src/tests/unit/...`
- **ИЛИ** исключить `community/` из pytest при запуске unit-тестов (но оставить для e2e)

---

## C. Загрузка из архивов

### C1. Текущий механизм загрузки

**Используется:** Стандартный Python `importlib` + путь на диск
```python
# src/utils/plugin_discovery.py:33-69
def discover_modules(package_name: str, *, recursive: bool = True, ...) -> list[str]:
    package = importlib.import_module(package_name)
    package_root = Path(package.__file__).resolve().parent
    # рекурсивное сканирование *.py файлов
    # исключение __pycache__, test_* файлов
    return sorted(set(modules))
```

### C2. ZIP/TAR поддержка

**Текущий статус:** НЕ поддерживается
- Нет использования `zipfile`, `tarfile` в исходном коде
- `importlib` работает только с диском или namespace packages

**Что потребуется для архивов:**
1. Распаковка архива в временную папку перед import
2. Добавление пути в `sys.path`
3. Модификация `discover_modules()` для работы с распакованными модулями

---

## D. Внешний API и webhooks

### D1. API методы для плагинов

**Файл:** `src/api/routers/plugins.py`
```python
@router.get("/{kind}", response_model=PluginListResponse)
def list_plugins(kind: PluginKind) -> PluginListResponse:
    # kind: "reward" | "validation" | "evaluation"
```

**Сервис:** `src/api/services/plugin_service.py`
```python
def list_plugins(kind: PluginKind) -> PluginListResponse:
    # Агрегирует manifests из трёх registry
    # Возвращает PluginListResponse
```

### D2. Возвращаемые данные

**API возвращает NOT пути на диск, а manifests:**
```python
# Пример ответа:
{
  "kind": "validation",
  "plugins": [
    {
      "id": "min_samples",
      "name": "min_samples",
      "version": "1.0.0",
      "priority": 10,
      "description": "Checks minimum sample count",
      "category": "basic",
      "stability": "stable",
      "params_schema": {...},
      "thresholds_schema": {...},
      "suggested_params": {...},
      "suggested_thresholds": {...}
    }
  ]
}
```

**Путей на диск НЕ содержит** — это важно для изоляции.

---

## E. Интеграция с pipeline стадиями

### E1. Dataset Validator

**Файл:** `src/pipeline/stages/dataset_validator.py:90-180`

Получение плагинов:
```python
from src.data.validation.registry import ValidationPluginRegistry
from src.data.validation.discovery import ensure_validation_plugins_discovered

ensure_validation_plugins_discovered()  # ← lazy discovery
for plugin_cfg in dataset_config.validations.plugins:
    plugin = ValidationPluginRegistry.get_plugin(
        plugin_cfg.plugin,
        params=plugin_cfg.params,
        thresholds=plugin_cfg.thresholds
    )
    result = plugin.validate(dataset)
```

### E2. Model Evaluator

**Файл:** `src/pipeline/stages/model_evaluator.py:67-120`

Делегирует `EvaluationRunner`:
```python
from src.evaluation.runner import EvaluationRunner
from src.evaluation.plugins.discovery import ensure_evaluation_plugins_discovered

ensure_evaluation_plugins_discovered()
runner = EvaluationRunner(eval_config, secrets, callbacks)
result = runner.run(context)
```

### E3. Training: Reward Plugins

**Файл:** `src/training/reward_plugins/factory.py:24-68`

```python
from src.training.reward_plugins.discovery import ensure_reward_plugins_discovered
from src.training.reward_plugins.registry import RewardPluginRegistry

ensure_reward_plugins_discovered()
plugin = RewardPluginRegistry.create(plugin_name, reward_params)
plugin.setup()
config_kwargs = plugin.build_config_kwargs(...)
trainer_kwargs = plugin.build_trainer_kwargs(...)
```

### E4. Registry Access Pattern

**Все три системы используют одинаковый паттерн:**
1. `ensure_*_plugins_discovered()` — вызывает `discover_and_import_modules()`
2. `*PluginRegistry.get_plugin() / create()` — возвращает инстанс
3. Plugin lifecycle: `__init__()` → `setup()` → exec → `teardown()`

---

## F. Manifest-шные поля существующих плагинов

### F1. BasePlugin - базовый контракт

Все плагины наследуют **`BasePlugin`** (`src/utils/plugin_base.py:15-72`):

```python
class BasePlugin:
    name: ClassVar[str] = ""           # ← уникальный ID
    priority: ClassVar[int] = 50       # ← порядок выполнения
    version: ClassVar[str] = "1.0.0"   # ← semver
    
    MANIFEST: ClassVar[dict[str, Any] | None] = None
    
    @classmethod
    def get_manifest(cls) -> dict[str, Any]:
        # Возвращает нормализованный dict с all fields
```

### F2. Таблица: какие плагины какие поля имеют

| Плагин | Файл | name | version | priority | MANIFEST.description | MANIFEST.category | MANIFEST.stability | MANIFEST.params_schema | MANIFEST.thresholds_schema | MANIFEST.suggested_* |
|--------|------|------|---------|----------|----------------------|-------------------|-------------------|----------------------|---------------------------|----------------------|
| **min_samples** | `src/data/validation/plugins/base/min_samples.py:22-58` | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| **avg_length** | `src/data/validation/plugins/base/avg_length.py` | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| **empty_ratio** | `src/data/validation/plugins/base/empty_ratio.py` | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| **diversity_score** | `src/data/validation/plugins/base/diversity.py` | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| **deduplication** | `src/data/validation/plugins/sft/deduplication.py` | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| **preference_format** | `src/data/validation/plugins/dpo/preference_format.py` | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| **identical_pairs** | `src/data/validation/plugins/dpo/identical_pairs.py` | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| **helixql_gold_syntax_backend** | `src/data/validation/plugins/sft/helixql_gold_syntax_backend.py` | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| **helixql_preference_semantics** | `src/data/validation/plugins/dpo/helixql_preference_semantics.py` | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| **helixql_sapo_prompt_contract** | `src/data/validation/plugins/sapo/helixql_sapo_prompt_contract.py` | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ? | ? | ✓ |
| **helixql_syntax** | `src/evaluation/plugins/syntax_check/helixql_syntax.py` | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| **helixql_generated_syntax_backend** | `src/evaluation/plugins/syntax_check/helixql_generated_syntax_backend.py` | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| **helixql_semantic_match** | `src/evaluation/plugins/semantic/helixql_semantic_match.py` | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| **cerebras_judge** | `src/evaluation/plugins/llm_judge/cerebras_judge.py` | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| **helixql_compiler_semantic** | `src/training/reward_plugins/plugins/helixql_compiler_semantic.py` | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |

### F3. Стандартные fields в MANIFEST

```python
# BasePlugin.get_manifest() возвращает:
{
    "id": cls.name,                          # ← from name
    "name": cls.name,                        # ← from name
    "version": cls.version,                  # ← from version
    "priority": cls.priority,                # ← from priority
    "description": "",                       # ← from MANIFEST or ""
    "category": "",                          # ← from MANIFEST or ""
    "stability": "stable",                   # ← from MANIFEST or default
    "params_schema": {},                     # ← from MANIFEST or {}
    "thresholds_schema": {},                 # ← from MANIFEST or {}
    "suggested_params": {},                  # ← from MANIFEST or {}
    "suggested_thresholds": {},              # ← from MANIFEST or {}
}
```

### F4. Additional Attributes (не в BasePlugin)

| Плагин тип | Дополнительные поля |
|-----------|-------------------|
| **ValidationPlugin** | `expensive: bool`, `required_fields: list[str]`, `supports_streaming: bool` |
| **EvaluatorPlugin** | `requires_expected_answer: bool` |
| **RewardPlugin** | `params: dict` (от конфига), `_secrets: dict` (опционально) |

---

## G. Документация по написанию плагинов

### G1. Существующие README

**Validation plugins:**
- `src/data/validation/README.md` ← **ПОЛНАЯ документация**
  - Architecture diagram
  - Existing plugins table (lines 36-49)
  - Adding a new plugin (lines 95-202)
  - Priority guidelines (lines 228-236)
  - Plugin secrets (lines 51-93)
  - Testing guidelines (lines 254-289)

**Evaluation plugins:**
- `src/evaluation/plugins/README.md` ← **ПОЛНАЯ документация**
  - How It Works
  - Core Types
  - Built-in Plugins table (lines 24-31)
  - Dataset Metadata (lines 33-80)
  - Plugin Secrets (lines 82-125)
  - Creating a Custom Plugin (lines 129-244)
  - Plugin API Reference (lines 213-244)

**Training/Reward plugins:**
- `src/training/README.md` ← существует, но NO плагин docs
- **Отсутствует** полная документация по reward плагинам

### G2. BasePlugin документация

- Встроенная в `src/utils/plugin_base.py:15-38` (docstring)
- Объясняет: `name`, `priority`, `version`, `MANIFEST`

### G3. Примеры кода в README

**Validation:**
- Example plugin template (lines 106-175 в `src/data/validation/README.md`)
- Testing example (lines 254-285)

**Evaluation:**
- Example plugin template (lines 131-181 в `src/evaluation/plugins/README.md`)

**Reward plugins:**
- Минимум документации — только base class в `src/training/reward_plugins/base.py:14-75`

---

## Summary: Ключевые выводы для рефакторинга

### 1. Конфигурация остаётся неизменной
- Плагины идентифицируются по `name` (ClassVar), а не по пути
- При переезде в `community/` нужно обновить только `discovery` path

### 2. Discovery нужно переплатить
- Текущий: `discover_and_import_modules("src.data.validation.plugins", ...)`
- Новый: добавить search в `community/data_validation/`, `community/evaluation/`, `community/reward/`

### 3. Тесты — потенциальная ловушка
- Встроенные тесты в `src/` могут дублировать `src/tests/unit/`
- При переезде нужно рассмотреть консолидацию

### 4. API не возвращает пути
- `list_plugins()` возвращает только manifests
- После рефакторинга фронту не нужны никакие изменения

### 5. Документация неполная
- Validation и Evaluation имеют хорошие README
- Reward plugins требуют документирования

---

## Files Referenced

- `src/utils/plugin_base.py` — base class contract
- `src/data/validation/base.py` — ValidationPlugin ABC
- `src/data/validation/registry.py` — ValidationPluginRegistry
- `src/data/validation/discovery.py` — discovery orchestration
- `src/evaluation/plugins/base.py` — EvaluatorPlugin ABC
- `src/evaluation/plugins/registry.py` — EvaluatorPluginRegistry
- `src/evaluation/plugins/discovery.py` — discovery orchestration
- `src/training/reward_plugins/base.py` — RewardPlugin ABC
- `src/training/reward_plugins/registry.py` — RewardPluginRegistry
- `src/training/reward_plugins/discovery.py` — discovery orchestration
- `src/training/reward_plugins/factory.py` — plugin instantiation
- `src/config/datasets/validation.py` — validation config schema
- `src/config/evaluation/schema.py` — evaluation config schema
- `src/api/routers/plugins.py` — API endpoint
- `src/api/services/plugin_service.py` — API service
- `src/utils/plugin_discovery.py` — generic discovery framework
- `src/pipeline/stages/dataset_validator.py` — validation stage
- `src/pipeline/stages/model_evaluator.py` — evaluation stage
- `src/training/trainer_builder.py` — trainer builder
- `src/data/validation/README.md` — validation plugin docs
- `src/evaluation/plugins/README.md` — evaluation plugin docs
