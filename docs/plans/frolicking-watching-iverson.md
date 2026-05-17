# Unsloth Integration as a Pluggable Training Engine

## Context

**Зачем.** RyotenkAI сейчас грузит модели через `AutoModelForCausalLM` и применяет LoRA через TRL+PEFT. Это работает, но проигрывает Unsloth по двум осям: (1) **скорость** обучения 1.5-2× ниже из-за отсутствия Triton-kernels, (2) **VRAM** на 50-70% выше, что заставляет арендовать более дорогие GPU и закрывает доступ к обучению на длинных контекстах для моделей которые архитектурно их поддерживают (Llama 3.1, gpt-oss, Qwen3 и т.д.). Для типичного run на RunPod (Qwen2.5-7B QLoRA SFT) это даёт ~$2.85 экономии на каждом обучении. Без интеграции пользователи будут уходить в unsloth-notebooks за скоростью, разрывая workflow с нашей оркестрацией.

**Реалистичные ожидания по выигрышу (не маркетинг).** В типичных нашем сценарии (SFT/CPT на 4K-32K контексте, QLoRA, single-GPU): ×1.5-2 скорость, -50% VRAM. Возможность тренировать на длинных контекстах (>100K, до 500K на 80GB H100 для специально подготовленных моделей в QLoRA + batch=1 режиме) — **бонус для нишевых задач** (long-document QA, обучение на code-репозиториях целиком), не дефолт. Tiled MLP для extreme-контекста замедляет один step, но позволяет физически уместиться в одну GPU там, где раньше требовался кластер.

**Что делаем.** Добавляем новый pluggable слой — `TrainingEngine` — параллельный существующему слою `Provider` (но НЕ являющийся провайдером). **Provider** = ГДЕ железо (single_node SSH, RunPod cloud). **Engine** = КАК грузим модель и инициализируем trainer на этом железе. Реализуем два engine: `HuggingfaceEngine` (текущее поведение: HF transformers + TRL + PEFT) и `UnslothEngine` (новый).

**Engine выбирается явно.** Поле `training.engine` — обязательное в схеме, без default. Это убирает скрытый выбор и заставляет пользователя осознанно указать backend. Существующие YAML конфиги без `engine:` начнут падать на `config validate` с понятным сообщением и подсказкой `add 'engine: huggingface' to your training section`.

**Стратегические границы (из helixir памяти).** RyotenkAI сужает позиционирование до control-plane для воспроизводимого post-training с **backend-agnostic execution over TRL/Unsloth/Axolotl**. Этот PR — первый шаг к этому позиционированию. Сохраняем 100% дифференциаторов: cloud orchestration, multi-phase chaining, plugins, MLflow, evaluation, SAPO, declarative YAML, CLI/Web UI.

**Скоуп MVP (после трёх итераций deep think с пользователем).**
- Уровень: полная engine abstraction + UnslothEngine (~5 дней работы).
- Стратегии в первой волне: **только SFT + CPT** (минимум подвижных частей: нет ref_model для DPO/ORPO, нет reward_funcs для GRPO/SAPO).
- LoRA flow: **engine отвечает за LoRA целиком** — `UnslothEngine.apply_peft()` сам зовёт `FastLanguageModel.get_peft_model`, `peft_config=None` в trainer.
- DPO/ORPO/GRPO/SAPO под Unsloth — **запрещены валидатором** до отдельного PR с интеграционным тестом.
- AdaLoRA + Unsloth — запрещено (Unsloth не поддерживает).
- Multi-GPU + Unsloth — запрещено (Unsloth single-GPU only в текущей версии).
- Full FT + Unsloth — запрещено (Unsloth не покрывает FSDP/DeepSpeed надёжно).

---

## Архитектурное решение

### Новый слой `src/training/engines/`

Симметрия с `src/providers/training/` для согласованности онбординга и ревью:

```
src/training/engines/
├── __init__.py            — auto_register_engines() + публичные ре-экспорты
├── interfaces.py          — Protocol ITrainingEngine
├── factory.py             — TrainingEngineFactory (по образцу GPUProviderFactory)
├── constants.py           — ENGINE_HUGGINGFACE, ENGINE_UNSLOTH
├── huggingface/
│   ├── __init__.py        — register("huggingface", HuggingfaceEngine)
│   └── engine.py          — HuggingfaceEngine (текущее поведение: HF transformers + TRL + PEFT)
└── unsloth/
    ├── __init__.py        — conditional register если unsloth установлен
    └── engine.py          — UnslothEngine
```

### Интерфейс `ITrainingEngine` (Protocol)

```python
class ITrainingEngine(Protocol):
    name: str

    def load_model(self, config: PipelineConfig) -> tuple[PreTrainedModel, PreTrainedTokenizer]: ...
    def apply_peft(self, model: PreTrainedModel, config: PipelineConfig) -> PreTrainedModel: ...
    def create_trainer(self, strategy_phase: StrategyPhaseConfig, model, tokenizer, train_ds, eval_ds, peft_config) -> Trainer: ...
    def save_checkpoint(self, model: PreTrainedModel, path: Path) -> None: ...
    def load_checkpoint(self, base_model: PreTrainedModel, adapter_path: Path, config: PipelineConfig) -> PreTrainedModel: ...
    def supports(self, capability: str) -> bool: ...
    def validate_runtime_environment(self, gpu_info: GpuInfo) -> Result[None, EngineError]: ...
    def resolve_model_name(self, requested_name: str) -> str: ...
```

**Capabilities** (string flags для `supports()`): `lora`, `qlora`, `adalora`, `full_ft`, `multi_gpu`, `fsdp`, `deepspeed`, `fp8`, `vision`, `long_context_500k`, `gguf_export`, `low_vram_optimizations`, `rl_grpo`, `rl_sapo`, `dpo`, `orpo`, `cpt`, `cot`, `sft`.

### Конфиг-схема (`src/config/training/schema.py`)

В `TrainingOnlyConfig` (после `provider:` на строке 55):

```python
engine: Literal["huggingface", "unsloth"] = Field(
    ...,  # REQUIRED — без default. Заставляет пользователя осознанно выбрать backend.
    description="Training engine (REQUIRED). 'huggingface' — стандартный путь HF transformers + TRL + PEFT (используйте для current behaviour). 'unsloth' — 1.5-2× быстрее, -50% VRAM, single-GPU only, no AdaLoRA, SFT/CPT only в v1.",
)
engine_options: dict[str, Any] = Field(
    default_factory=dict,
    description="Engine-specific kwargs. Unsloth: max_seq_length, fast_inference, allow_repo_substitution.",
)
```

**Migration для существующих YAML.** При `engine` отсутствует — `config validate` бросает `MissingEngineError` с подсказкой:

```
Field 'training.engine' is required (no default).
To preserve current behaviour, add to your config:
    training:
      engine: huggingface

To opt into the optimized path (1.5-2× faster, -50% VRAM):
    training:
      engine: unsloth
```

Все встроенные example-configs (`src/config/pipeline_config.yaml`, `tests/fixtures/*.yaml`) обновляются в одном PR с добавлением `engine: huggingface`.

Validator (model_validator) блокирует:
- `engine="unsloth"` + `type="adalora"` → fail
- `engine="unsloth"` + любая стратегия из `{dpo, orpo, grpo, sapo}` (на v1) → fail
- `engine="unsloth"` + `type="full_ft"` → fail

### Точки врезания (минимум) — критические файлы

| Файл | Что меняется |
|---|---|
| **[src/config/training/schema.py](src/config/training/schema.py)** | + поля `engine`, `engine_options`; + model_validator с проверкой совместимости |
| **[src/training/models/loader.py](src/training/models/loader.py)** | `load_model_and_tokenizer()` становится тонкой обёрткой: `engine = container.get_engine(); return engine.load_model(config)` |
| **[src/training/trainer_builder.py](src/training/trainer_builder.py)** | `create_trainer()` делегирует engine; `create_peft_config()` — только для huggingface engine, для unsloth возвращает None (LoRA уже в модели) |
| **[src/training/orchestrator/resume_manager.py:167](src/training/orchestrator/resume_manager.py:167)** | Заменить `PeftModel.from_pretrained()` → `engine.load_checkpoint()` |
| **[src/training/orchestrator/phase_executor/adapter_cache.py:195](src/training/orchestrator/phase_executor/adapter_cache.py:195)** | Заменить `PeftModel.from_pretrained()` → `engine.load_checkpoint()` |
| **[src/utils/container.py](src/utils/container.py)** | + `engine_factory.create(config.training.engine)` как ленивый сингтлон в TrainingContainer |
| **[src/training/managers/data_buffer/](src/training/managers/data_buffer)** (state model) | + поля `engine: str`, `resolved_model_name: str` в pipeline_state.json |
| **[src/training/run_training.py](src/training/run_training.py)** | Side-effect: `auto_register_engines()` ПЕРВОЙ строкой после imports (gates Unsloth import-order) |
| **[pyproject.toml](pyproject.toml)** | + optional dependency group `[project.optional-dependencies] unsloth = [...]` |
| **[docker/training/](docker/training)** | + новый sibling `docker/training-unsloth/Dockerfile`, наследует от training/Dockerfile, ставит unsloth со своими pinned deps |
| **[web/src/components/ConfigBuilder/FieldRenderer.tsx](web/src/components/ConfigBuilder/FieldRenderer.tsx)** | Conditional rendering: при `engine=unsloth` скрыть adalora из type, скрыть несовместимые strategy_type, показать info-banner с tradeoffs |

**НЕ ТРОГАЕМ:** ChainRunner, PhaseExecutor, MemoryManager (минимальное изменение — engine-aware threshold), strategies/*, MLflow, evaluation, dataset validation, reports, providers. Это держит blast radius минимальным.

### Deployment — как Unsloth попадает на target и запускается

**Unsloth — обычный Python pip-пакет**, не CLI и не сервис. Доставка происходит на трёх уровнях:

#### 1. Dev-машина (локальная разработка)

В `pyproject.toml` — optional dependency group:

```toml
[project.optional-dependencies]
unsloth = ["unsloth>=2026.4.0,<2026.6.0"]
```

- `pip install -e .` — базовая установка, **без** unsloth.
- `pip install -e ".[unsloth]"` — с unsloth для локального тестирования.

CI прогоняет обе матрицы (`pytest` для базового стека, `pytest -m unsloth` для unsloth-aware тестов).

#### 2. Docker-образы (training runtime)

Создаём **sibling image** — не трогаем существующий:

```
docker/
├── training/
│   └── Dockerfile                  ← существующий (huggingface engine)
└── training-unsloth/
    └── Dockerfile                  ← новый, FROM training/, добавляет unsloth
```

`docker/training-unsloth/Dockerfile`:
```dockerfile
FROM ryotenkai/ryotenkai-training-runtime:latest
RUN pip install "unsloth==2026.4.6" "unsloth_zoo==2026.4.8" \
    --extra-index-url https://download.pytorch.org/whl/cu126
```

**Почему отдельный image:** unsloth тянет pinned стек (`trl==0.24.0`, `peft==0.19.1`, `xformers==0.0.31`). Установка в основной image может сломать пользователей `engine: huggingface`. Изоляция → безопасность, проще rollback.

CI билдит и пушит оба образа в Docker Hub при изменении соответствующих Dockerfile.

#### 3. Provider — авто-резолв образа по engine

Provider уже принимает `docker_image` в YAML. Добавляем тонкий resolver: если `docker_image` не задан явно, provider берёт image по engine.

Новый модуль `src/training/engines/docker.py` (~30 строк):

```python
DEFAULT_IMAGES = {
    "huggingface": "ryotenkai/ryotenkai-training-runtime:latest",
    "unsloth":     "ryotenkai/ryotenkai-training-runtime-unsloth:latest",
}

def resolve_docker_image(engine: str, provider_config: dict) -> str:
    if explicit := provider_config.get("docker_image"):
        return explicit  # пользователь задал явно — уважаем
    return DEFAULT_IMAGES[engine]
```

Вызывается из provider до deployment контейнера. Изменение в [src/providers/single_node/training/provider.py](src/providers/single_node/training/provider.py) и [src/providers/runpod/training/](src/providers/runpod/training) — одна строка каждый.

Пользователь **не правит provider секцию** — image выбирается автоматически.

#### 4. Что меняется при запуске обучения

| Слой | Изменение | Затрагивает пользователя? |
|---|---|---|
| `pyproject.toml` | + optional group `unsloth` | Только при локальной dev-установке |
| `docker/training-unsloth/` | + новый Dockerfile | Нет (CI билдит автоматически) |
| Provider resolver | Авто-выбор image по engine | Нет (всё прозрачно) |
| `src/training/run_training.py` | + `auto_register_engines()` первой строкой (gates Unsloth import order) | Нет |
| `loader.py` / `trainer_builder.py` | Делегаты engine | Нет |
| **YAML конфиг** | + обязательное `training.engine` | **Да — одна строка** |
| Команды CLI | Без изменений | `ryotenkai run start/resume/inspect/...` идентичны |

**Команда запуска НЕ меняется** — `ryotenkai run start --config my.yaml` работает для обоих engines.

#### 5. UX-сценарии

**Существующий проект, переход на unsloth:**

```yaml
# Diff в my_config.yaml — одна строка
 training:
+  engine: unsloth
   type: qlora
   hyperparams: { ... }
```

`ryotenkai run start --config my_config.yaml` → pipeline видит engine, провайдер тянет unsloth-image, обучение в 2× быстрее, всё остальное (мониторинг, MLflow, evaluation, reports) работает идентично.

**Legacy YAML без engine:**

```bash
$ ryotenkai config validate --config legacy.yaml
ERROR: Field 'training.engine' is required (no default).
To preserve current behaviour, add to your config:
    training:
      engine: huggingface

To opt into the optimized path (1.5-2× faster, -50% VRAM):
    training:
      engine: unsloth
```

#### 6. CI/CD изменения

| Шаг | До | После |
|---|---|---|
| Build images | `training` | `training` + `training-unsloth` (parallel jobs) |
| Pytest | один прогон | matrix: `[base, unsloth]`, последний с extras |
| Smoke tests | qwen-0.5b-sft | + qwen-0.5b-sft-unsloth (skip если нет GPU runner) |
| Renovate/Dependabot | `unsloth` версия — новая запись | autoUpdate с CI gate |

---

### Существующие паттерны и утилиты для переиспользования

- **[src/providers/training/factory.py](src/providers/training/factory.py)** — образец для `TrainingEngineFactory` (register/unregister/create/auto_register). Копировать структуру, не наследовать.
- **[src/training/strategies/base.py](src/training/strategies/base.py)** — паттерн `StrategyMetadata` для self-describing engines (создать аналогичный `EngineMetadata`).
- **[src/utils/result.py](src/utils/result.py)** — `Result[T, Err]` для всех методов engine, согласованно с проектом.
- **[src/utils/container.py](src/utils/container.py)** — DI pattern, не вводить global state.
- **MemoryManager.auto_configure** — добавить engine-aware ветку, не дублировать логику.

---

## Riski-ledger (3 итерации deep think)

| # | Риск | Тяжесть | Mitigation |
|---|---|---|---|
| 1 | **Import order**: `import unsloth` должен быть до `import trl`, иначе patches не применяются молча | КРИТ | `auto_register_engines()` первой строкой в `run_training.py`; `engines/unsloth/__init__.py` делает `import unsloth` ДО любых других imports. Тест: pytest fixture проверяет sys.modules ordering. |
| 2 | **Скрытая подмена модели**: Unsloth молча меняет `meta-llama/X` → `unsloth/X-bnb-4bit` (GitHub issue #2407), ломает model_dataset_config_hash | КРИТ | (а) логировать `model.config.name_or_path` до и после; (б) сохранить `resolved_model_name` в pipeline_state.json; (в) сверять при resume; (г) опция `engine_options.unsloth.allow_repo_substitution: bool = false` — fail-fast по умолчанию. |
| 3 | **save/load checkpoints**: `PeftModel.from_pretrained()` ломается на Unsloth-saved adapter (`AttributeError: 'apply_qkv'`) | КРИТ | Добавить `ITrainingEngine.load_checkpoint()`. Заменить 2 callsites: [resume_manager.py:167](src/training/orchestrator/resume_manager.py:167) и [adapter_cache.py:195](src/training/orchestrator/phase_executor/adapter_cache.py:195). UnslothEngine использует `FastLanguageModel.from_pretrained(adapter_path)`. **Между фазами IN-MEMORY** ([chain_runner.py:151](src/training/orchestrator/chain_runner.py:151)) — риск там нулевой. |
| 4 | **Resume engine drift**: смена engine между запусками сломает чекпойнты | КРИТ | Engine pinned в pipeline_state.json при первом запуске; fail-fast при mismatch. Engine входит в `model_dataset_config_hash` (см. helixir mem ISSUE-5) → автоматически новый hash на смене engine. |
| 5 | **Reward plugins для GRPO/SAPO**: совместимость с Unsloth-патченым GRPOTrainer не подтверждена | СРЕД | На v1 запрещено валидатором. После MVP: integration test с reward_funcs + Unsloth GRPOTrainer на 0.5B модели. Снятие ограничения — отдельный PR. |
| 6 | **Pinned deps конфликты**: unsloth тянет trl==0.24.0, peft==0.19.1, xformers и т.д. | СРЕД | (а) Optional dependency group в pyproject.toml; (б) Отдельный Docker image `ryotenkai-training-runtime-unsloth`; (в) Provider выбирает image по `training.engine`. |
| 7 | **MemoryManager thresholds**: с Unsloth -50% VRAM пресеты тиров неверны | СРЕД | `MemoryManager.auto_configure()` принимает engine; если `engine.supports("low_vram_optimizations")` → margin/2, critical_pct+5%. Логировать tier+margin в MLflow. |
| 8 | **MLflow autolog дубли**: Unsloth тоже цепляет MLflowCallback | СРЕД | UnslothEngine.create_trainer() удаляет встроенный MLflowCallback из callback_handler. Тест: inspect MLflow UI на отсутствие дубликатов. |
| 9 | **Provider/engine matrix**: multi-GPU pod + unsloth = не поддержано | СРЕД | Pre-flight check после connect к target: `nvidia-smi --query-gpu=count`. `engine.validate_runtime_environment(gpu_info)` → fail-fast если несовместимо. |
| 10 | **Frontend Literal handling**: openapi.json регенерация + ConfigBuilder rendering | МЕЛК | Проверить FieldRenderer — `training.type` уже Literal-подобный, паттерн есть. Добавить conditional disabling + info-banner. Storybook test. |
| 11 | **RoPE scaling rassync training↔inference**: long-context training с rope_scaling требует совпадающего max_model_len в vLLM, иначе контекст обрезается при inference | СРЕД | (а) Engine при сохранении чекпойнта пишет `rope_scaling` в config.json (стандартное HF поведение, нужно убедиться что Unsloth не забывает); (б) Pipeline-валидатор сравнивает `training.engine_options.max_seq_length` vs `inference.vllm.max_model_len` — warning если inference < training; (в) Документация: что long-context training имеет смысл только для моделей которые архитектурно его поддерживают (gpt-oss, Llama 3.1+, Qwen3); (г) Тест: long-context run → inference deploy → отправить prompt длиной >128K → проверить корректность ответа. |

**Открытых вопросов после трёх итераций — нет.** Все риски имеют конкретный mitigation план.

---

## Implementation phases

### Phase 0 — Smoke test compatibility (0.5 дня)
Локальный эксперимент: unsloth==2026.4.6, Qwen2.5-0.5B, SFT 50 шагов, `model.save_pretrained()`, затем reload через `FastLanguageModel.from_pretrained(adapter_path)` — подтвердить работоспособность. Документировать pinned deps что Unsloth тянет.

### Phase 1 — Engine abstraction layer (1.5 дня)
- Создать `src/training/engines/` с interfaces, factory, constants.
- HuggingfaceEngine — обёртка вокруг текущего кода loader.py + trainer_builder.py (existing behaviour, без изменений в логике).
- UnslothEngine — `FastLanguageModel.from_pretrained` + `get_peft_model` + lazy import unsloth.
- Conditional auto-register в `engines/unsloth/__init__.py` (skip если пакет не установлен).
- Unit-тесты factory (register, conflict, missing engine).

### Phase 2 — Wire-in (1 день)
- `loader.py` и `trainer_builder.py` рефакторятся в тонкие делегаторы engine.
- `resume_manager.py` и `adapter_cache.py` — заменить `PeftModel.from_pretrained` на `engine.load_checkpoint`.
- `container.py` — добавить engine factory + DI.
- `pipeline_state.json` schema — поля `engine`, `resolved_model_name`.

### Phase 3 — Config + validation (1 день)
- `TrainingOnlyConfig` — поля `engine`, `engine_options`.
- Model validator — все incompat комбинации с понятными error messages.
- `model_dataset_config_hash` включает engine.
- `MemoryManager.auto_configure` engine-aware.

### Phase 4 — Infrastructure (0.5 дня)
- `pyproject.toml` — optional dependency group `unsloth`.
- `docker/training-unsloth/Dockerfile` — наследник training image с pinned unsloth stack.
- Provider config — выбор docker_image по engine.

### Phase 5 — Tests (1 день)
- Unit: engine factory, capability checks, validators.
- Integration: SFT chain CPT→SFT с обоими engines, сравнить loss curves (фиксированный seed) — макс расхождение <5%.
- Resume test: kill mid-phase, restart, проверить engine pinning + resolved_model_name match.
- MLflow inspect — нет дубликатов.

### Phase 6 — Frontend (0.5 дня)
- Регенерация openapi.json.
- ConfigBuilder/FieldRenderer — conditional disabling, info-banner с tradeoffs.
- Compatibility matrix tooltip.

### Phase 7 — Docs (0.5 дня)
- README — секция Training Engines.
- CONFIG_REFERENCE.md — engine + engine_options.
- docs/architecture/engines.md — новый файл с диаграммой слоёв.

**Итого: 5-6 дней.** GGUF export — отдельный PR после v1 (опциональный bonus).

---

## Verification

### Unit-тесты (`pytest src/tests/unit/training/engines/`)
```bash
pytest src/tests/unit/training/engines/test_factory.py
pytest src/tests/unit/training/engines/test_huggingface_engine.py
pytest src/tests/unit/training/engines/test_unsloth_engine.py  # skip if unsloth not installed
pytest src/tests/unit/config/test_engine_validators.py
```

### Integration: smoke test обоих engines
```bash
# Baseline (engine: huggingface)
ryotenkai run start --config tests/fixtures/qwen-0.5b-sft-hf.yaml

# С unsloth (тот же YAML + engine: unsloth)
ryotenkai run start --config tests/fixtures/qwen-0.5b-sft-unsloth.yaml

# Сверить loss curves через MLflow:
ryotenkai runs report runs/<run_id_huggingface>
ryotenkai runs report runs/<run_id_unsloth>
# diff loss values, max delta <5%
```

### Resume test
```bash
ryotenkai run start --config tests/fixtures/cpt-sft-chain.yaml &
sleep 60 && kill -INT $!
ryotenkai run resume runs/<run_id>  # должен корректно подхватить через engine.load_checkpoint
```

### Validator test
```bash
# Должен fail с ясным сообщением
ryotenkai config validate --config tests/fixtures/unsloth-with-adalora.yaml
ryotenkai config validate --config tests/fixtures/unsloth-with-grpo.yaml
ryotenkai config validate --config tests/fixtures/unsloth-with-fullft.yaml

# Missing engine field — должен fail с подсказкой
ryotenkai config validate --config tests/fixtures/legacy-no-engine.yaml
# expect: "Field 'training.engine' is required (no default). Add 'engine: huggingface' to preserve current behaviour."
```

### MLflow inspect
- Запустить unsloth run, открыть MLflow UI, проверить:
  - Нет дублированных метрик loss/grad_norm
  - Tags содержат `engine=unsloth`, `resolved_model_name=...`
  - GPU metrics залогированы (наш callback работает)

### Frontend (Web UI)
```bash
cd web && npm run dev
# Открыть :5173, ConfigBuilder
# 1. engine dropdown появляется
# 2. при unsloth → adalora скрыт, GRPO/SAPO disabled с tooltip
# 3. info-banner отображается
```

### Pre-flight test
- На single_node без CUDA попытаться запустить engine=unsloth → fail-fast
- На multi-GPU pod (2× GPU) попытаться → fail-fast с понятным сообщением

---

## Acceptance criteria

- [ ] `engine: huggingface` работает без регрессий — все существующие интеграционные тесты зелёные после добавления `engine: huggingface` в их fixtures.
- [ ] `engine: unsloth` для SFT и CPT успешно проходит chain CPT→SFT на Qwen2.5-0.5B.
- [ ] Loss curves huggingface vs unsloth сходятся в пределах 5% при том же seed.
- [ ] YAML без `engine:` падает на `config validate` с понятной подсказкой добавить `engine: huggingface`.
- [ ] Все встроенные example/fixture конфиги обновлены и проходят валидацию.
- [ ] Время обучения с unsloth измеримо ниже на ≥30% (типичный run на A100/L40S).
- [ ] Resume after crash работает с обоими engines.
- [ ] Config validator блокирует все 4 запрещённые комбинации с ясными сообщениями.
- [ ] Pre-flight check на multi-GPU + unsloth даёт понятный fail-message.
- [ ] MLflow UI: нет дубликатов метрик; tags содержат engine + resolved_model_name.
- [ ] Web UI ConfigBuilder корректно рендерит engine + engine_options + conditional fields.
- [ ] Документация обновлена: README, CONFIG_REFERENCE, docs/architecture/engines.md.

---

## Out of scope для v1 (следующие PR)

1. **DPO/ORPO + Unsloth** — после verification ref_model patching.
2. **GRPO/SAPO + Unsloth** — после verification reward_funcs совместимости.
3. **GGUF export** stage — `model.save_pretrained_gguf()` как отдельный pipeline stage.
4. **MLX engine** для Apple Silicon — на образце UnslothEngine (у MLX-Tune Unsloth-compatible API).
5. **Multi-GPU Unsloth** — когда официальный релиз станет stable.
6. **FP8 training** — отдельный config flag когда наберём примеры.
