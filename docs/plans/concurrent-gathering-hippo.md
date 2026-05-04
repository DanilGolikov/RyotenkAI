# Provider abstraction refactor — manifest-driven registry + capability advertisement

**Date:** 2026-05-04
**Author role:** Senior MLOps architect
**Branch target:** RESEACRH (after 2 PRs)
**Backwards compatibility:** **NOT** preserved — bad code is deleted; new code is the contract.

---

## 1. Context

После Phase B пакетизации в монорепо (`packages/{shared,community,providers,pod,control}`) система провайдеров (RunPod / single_node) находится в полусобранном состоянии:

- **Два разных стиля factory.** Training (`GPUProviderFactory`) — нормальный registry pattern. Inference (`InferenceProviderFactory` в [inference/factory.py:37-45](packages/providers/src/ryotenkai_providers/inference/factory.py:37)) — `if/elif` на имя провайдера. Несоответствие масштабируется хуже линейно: третий провайдер потребует ещё одну ветку.
- **Phase 14.D+F миграция «string-dispatch → capability flags» не завершена.** В control-plane коде остались **11 string-checks** на `provider_name == "runpod" / "single_node"`. Самый яркий — [`training_monitor.py:682`](packages/control/src/ryotenkai_control/pipeline/stages/training_monitor.py:682) `if self._provider_name != "runpod"` — recovery работает только для одного провайдера, без флага.
- **Pod-side registry vs Mac-side factory — два source of truth.** [`provider_registry.py:77-80`](packages/pod/src/ryotenkai_pod/runner/runtime/provider_registry.py:77) дублирует список провайдеров, ручная синхронизация. **Нет теста парности**: новый провайдер с `supports_lifecycle_actions=True` без pod-side записи → silent `BootstrapConfigError` через 4 часа в проде.
- **13-15 файлов** для добавления нового провайдера. Нет генератора, нет manifest, role (training/inference/both) — implicit.
- **Конструкторы провайдеров несоответствующие**: training принимает `(config: dict, secrets)`, inference — `(*, config: PipelineConfig, secrets)`. Невозможно унифицировать registry без выравнивания.
- **Дублирование методов через 3 Protocols** (`provider_name`, `provider_type`, `get_capabilities`).

**Цель рефакторинга:** свести абстракцию провайдеров к одному manifest-driven registry с явной декларацией ролей и способностей через `provider.toml`. Удалить **все** остатки string-dispatch. Снизить порог входа нового провайдера до **1 manifest + 1-3 класса + автоматически сгенерированные тесты**.

**Ожидаемый результат:**
- Добавить нового провайдера = `python packages/providers/scripts/new_provider.py <id>` + заполнить TODO в manifest. Никаких правок в registry/factory/startup_validator/resume_service вручную.
- 0 string-checks по `provider_name` в business-logic коде. Все диспатчи — через capability advertisement.
- Cross-process parity (Mac↔pod) гарантирована schema-validator'ом + invariant-тестом.
- Один canonical Protocol-набор: `IProviderBase` + role-specific `IGPUProvider/IInferenceProvider` + capability-specific micro-Protocols.

---

## 2. Подтверждение архитектурного паттерна (research-driven)

**Capability Advertisement** — индустриально стандартный паттерн для решения проблемы «одна общая логика, разные подмножества способностей у разных backend'ов»:

| Система | Паттерн | Отношение |
|---|---|---|
| **MCP (Model Context Protocol)** | `Capabilities` объект, обмен при `initialize` handshake. Caller проверяет `capabilities.tools` перед вызовом `tools/list`. | Прямой аналог нашего case |
| **LSP (Language Server Protocol)** | Client/server announce `capabilities` (`ServerCapabilities`). Каждая method ссылается на capability path. | То же |
| **Crossplane v2.2** | `RunFunctionRequest.Capabilities` advertises `CAPABILITY_REQUIRED_SCHEMAS` etc. Function проверяет членство перед использованием feature. | Идентичный pattern |
| **dbt adapters** | Required vs optional macros + dispatch pattern. Adapter override с feature flags (`dbt-spark` Delta support). | Похожий подход |
| **Существующий `ITerminalActionProvider`** | Capability-flag (`supports_lifecycle_actions: bool`) ↔ Protocol membership (`isinstance(p, ITerminalActionProvider)`) — invariant test pins parity. | Уже сделано в проекте |

**Вывод:** проект уже идёт правильным курсом (Phase 14.A с `ITerminalActionProvider`). Нужно **добить миграцию** этим же паттерном для оставшихся способностей и формализовать через manifest.

---

## 3. Целевая архитектура

### 3.1 Иерархия Protocol'ов

```
IProviderBase (NEW)                      # identity + capabilities
├── IGPUProvider (training role)         # extends IProviderBase
└── IInferenceProvider (inference role)  # extends IProviderBase

# Capability micro-Protocols (separate, opt-in inheritance):
ITerminalActionProvider     # уже есть; supports_lifecycle_actions
IPauseResumeProvider (NEW)  # has_pause_resume (было capability-only)
IRecoveryProbeProvider (NEW)            # supports_recovery_probe — закрывает training_monitor.py:682
ICapacityErrorClassifier (NEW)          # supports_capacity_error_detection — закрывает resume_service.py:374

# Pod-side (async, отдельная часовая) — НЕ трогаем:
IPodLifecycleClient   # async, в shared.infrastructure.lifecycle.protocol
```

**Decision:** capability methods **выносятся в отдельные Protocol'ы**, не в `IGPUProvider` (research confirms — это правильнее с точки зрения SOLID/ISP, симметрично с уже существующим `ITerminalActionProvider`, и мypy ловит ошибочные вызовы статически). RunPod наследует `IGPUProvider, ITerminalActionProvider, IPauseResumeProvider, IRecoveryProbeProvider, ICapacityErrorClassifier`. SingleNode — только `IGPUProvider`.

### 3.2 Унифицированный конструктор: `ProviderContext`

```python
@dataclass(frozen=True, slots=True)
class ProviderContext:
    provider_id: str                       # canonical id из manifest
    pipeline_config: PipelineConfig        # full config
    provider_block: Mapping[str, Any]      # config.providers[provider_id]
    secrets: Secrets
```

Все провайдер-классы: `def __init__(self, ctx: ProviderContext) -> None`. Training читает только `ctx.provider_block`; inference — `ctx.pipeline_config.inference` + `ctx.provider_block`. **Один signature → один `registry.create_*(provider_id, ctx)` вызов**.

### 3.3 ProviderRole enum

```python
class ProviderRole(StrEnum):
    TRAINING  = "training"
    INFERENCE = "inference"
```

`BOTH` **не моделируется** как enum-значение — manifest декларирует `roles = ["training", "inference"]` (multi-valued list); loader нормализует в `frozenset[ProviderRole]`. Полное соответствие user-доступной декларации провайдера.

---

## 4. `provider.toml` manifest schema

Один файл рядом с пакетом провайдера:

- `packages/providers/src/ryotenkai_providers/runpod/provider.toml`
- `packages/providers/src/ryotenkai_providers/single_node/provider.toml`

### 4.1 Полный пример (RunPod)

```toml
schema_version = 1

[provider]
id          = "runpod"
name        = "RunPod"
version     = "1.0.0"
roles       = ["training", "inference"]
description = "Cloud GPU pods via RunPod GraphQL API."
author      = "RyotenkAI Core"
stability   = "stable"                    # stable | beta | experimental

[capabilities]
provider_type                     = "cloud"
is_local                          = false
supports_multi_gpu                = true
supports_spot_instances           = false
supports_lifecycle_actions        = true
has_pause_resume                  = true
supports_recovery_probe           = true   # NEW — закрывает training_monitor.py:682
supports_capacity_error_detection = true   # NEW — закрывает resume_service.py:374
supports_log_download             = true
volume_kind                       = "persistent"
runner_workspace_root             = "/workspace"
max_runtime_hours                 = null   # null = unlimited

[entry_points.training]
module = "ryotenkai_providers.runpod.training.provider"
class  = "RunPodProvider"

[entry_points.inference]
module = "ryotenkai_providers.runpod.inference.pods.provider"
class  = "RunPodPodInferenceProvider"

[entry_points.pod_lifecycle_client]
module = "ryotenkai_providers.runpod.runtime.lifecycle_client"
class  = "RunPodPodLifecycleClient"

[entry_points.resume_factory]                # optional; for IPauseResumeProvider
module = "ryotenkai_providers.runpod.training.provider"
classmethod = "RunPodProvider.from_resume_metadata"

# Pydantic-модель для блока ``providers.runpod`` в pipeline_config.yaml.
# Loader валидирует YAML против неё на этапе config load — опечатка / неверный
# тип ловится ДО запуска первого этапа пайплайна. Заменяет текущий untyped
# ``dict[str, Any]`` подход (см. §4.4).
[entry_points.config_schema]
module = "ryotenkai_providers.runpod.config"
class  = "RunPodProviderConfig"

[[required_env]]
name        = "RUNPOD_API_KEY"
description = "RunPod GraphQL API key."
secret      = true
required_when_role = "training"             # role-scoped — pre-factory dispatch source
```

### 4.2 Pydantic schema (`ProviderManifest`)

Живёт в `packages/providers/src/ryotenkai_providers/manifest.py`. Зеркалит структуру [`ryotenkai_community.manifest.PluginManifest`](packages/community/src/ryotenkai_community/manifest.py).

**Schema-level invariants** (в `model_validator(mode="after")`):

1. `roles` non-empty subset `ProviderRole`.
2. Для каждой роли в `roles` — соответствующий `entry_points.<role>` ОБЯЗАТЕЛЕН.
3. `capabilities.supports_lifecycle_actions == True` → `entry_points.pod_lifecycle_client` ОБЯЗАТЕЛЕН. **Это и есть schema-level Mac↔pod parity** (закрывает риск из аудита #5).
4. `capabilities.has_pause_resume == True` → `capabilities.supports_lifecycle_actions == True`.
5. `capabilities.is_local == True` → `capabilities.supports_lifecycle_actions == False` AND `capabilities.volume_kind == "local_disk"`.
6. `capabilities.supports_capacity_error_detection == True` → `provider_type == "cloud"` (capacity errors — это cloud-specific).
7. `required_env[].required_when_role` ∈ `roles`.

### 4.3 Manifest как single source of truth для capabilities

**Decision:** `provider.get_capabilities()` возвращает `ProviderCapabilities`, **derived from manifest** (не hand-coded в классе). Loader при регистрации устанавливает `cls._manifest_capabilities` class-attribute, базовый `IProviderBase` mixin даёт реализацию `get_capabilities()` читающую этот attr.

Это устраняет риск R10 (две источники truth — flag в манифесте vs hand-coded в `get_capabilities()`).

### 4.4 Provider-specific config schemas (typed YAML blocks)

**Текущая проблема:** `config.providers` — это `dict[str, dict[str, Any]]`. Поля разных провайдеров кардинально отличаются (single_node имеет `workspace_path` и `ssh.alias`; runpod имеет `cloud_type`, `container_disk_gb`, `volume_disk_gb`, `ssh.key_path`). Сейчас провайдер вызывает `MyConfig.from_dict(provider_block)` сам, и валидация запускается **после** loading pipeline_config — опечатка в YAML обнаруживается через минуту работы пайплайна, не на старте.

**Decision:** каждый провайдер декларирует свою Pydantic-модель и указывает её в manifest через `entry_points.config_schema` (см. §4.1). Структура модели:

```python
# packages/providers/src/ryotenkai_providers/runpod/config.py
class RunPodConnectConfig(BaseModel):
    ssh: RunPodSSHConfig                       # key_path, host, port

class RunPodTrainingConfig(BaseModel):
    gpu_type: str
    cloud_type: Literal["ALL", "SECURE", "COMMUNITY"]
    container_disk_gb: int = 100
    volume_disk_gb: int = 20
    ports: str = "8888/http,22/tcp"

class RunPodInferenceConfig(BaseModel):
    volume: NetworkVolumeConfig | None = None
    pod: InferencePodConfig | None = None

class RunPodCleanupConfig(BaseModel):
    auto_delete_pod: bool = True
    keep_pod_on_error: bool = False
    on_interrupt: bool = True

class RunPodProviderConfig(BaseModel):
    """Strict typed schema для блока ``providers.runpod`` в YAML."""
    model_config = ConfigDict(extra="forbid")
    connect: RunPodConnectConfig
    training: RunPodTrainingConfig | None = None       # required iff "training" in roles
    inference: RunPodInferenceConfig | None = None     # required iff "inference" in roles
    cleanup: RunPodCleanupConfig
```

**Loader integration**: `PipelineConfig`'s validator делает один проход:

1. Загружает `providers: dict[str, dict]` из YAML.
2. Для каждого `<id>, <block>`: `manifest = registry.get_manifest(id); ConfigCls = registry.get_config_class(id); typed_block = ConfigCls.model_validate(block)`.
3. Меняет `config.providers[id]` на `typed_block` (типизированный объект, не dict).
4. Если manifest требует training в roles, но `typed_block.training is None` → `ValidationError`. Аналогично inference.

После этого `ProviderContext.provider_block` становится `<provider>ProviderConfig` instance (не dict). `RunPodProvider.__init__(ctx)` обращается к `ctx.provider_block.training.gpu_type` с full type-checking — mypy видит правильный тип.

**Bonus: auto-generated Web UI forms.** Pydantic v2 даёт `RunPodProviderConfig.model_json_schema()` — JSON Schema со всеми типами/дефолтами/валидаторами. Web UI (`packages/web/...`) делает `GET /api/providers/<id>/config-schema` → получает schema → рендерит form динамически. **Убирает дублирование «Python-схема ↔ TypeScript form»**, которое сегодня есть в [ConfigBuilder/FieldRenderer.tsx](web/src/components/ConfigBuilder/FieldRenderer.tsx). Тот же паттерн уже работает для community plugins (`manifest.toml.params`).

**Schema-level invariants** (поверх §4.2):

- 8. Класс из `entry_points.config_schema.class` ОБЯЗАН существовать, импортироваться, быть subclass of `pydantic.BaseModel`.
- 9. `RunPodProviderConfig` ОБЯЗАН иметь `connect`, `cleanup` поля; `training` обязательно если `"training" in roles`; `inference` обязательно если `"inference" in roles`.
- 10. `model_config = ConfigDict(extra="forbid")` — обязательно (опечатки YAML не молчат).

---

## 5. `ProviderRegistry` — единая точка доступа

Живёт в `packages/providers/src/ryotenkai_providers/registry.py`. Заменяет:

- ❌ `packages/providers/src/ryotenkai_providers/training/factory.py` (`GPUProviderFactory`) — удаляется.
- ❌ `packages/providers/src/ryotenkai_providers/inference/factory.py` (`InferenceProviderFactory`) — удаляется.
- ❌ `auto_register_providers()` — удаляется.
- ❌ Hand-rolled `_LIFECYCLE_CLIENT_LOCATORS` в pod registry — удаляется (заменяется projected sub-manifest, см. §6).

### 5.1 Public surface

```python
class ProviderRegistry:
    @classmethod
    def from_filesystem(cls, *, roots: Sequence[Path] | None = None) -> ProviderRegistry: ...

    # Discovery
    def list(self, role: ProviderRole | None = None) -> tuple[str, ...]: ...
    def get_manifest(self, provider_id: str) -> ProviderManifest: ...
    def has_role(self, provider_id: str, role: ProviderRole) -> bool: ...

    # Capability introspection (pre-factory — без instantiation)
    def capabilities(self, provider_id: str) -> ProviderCapabilities: ...
    def required_secrets(self, provider_id: str, *,
                         role: ProviderRole | None = None) -> tuple[str, ...]: ...

    # Config schema (для PipelineConfig validator + Web UI)
    def get_config_class(self, provider_id: str) -> type[BaseModel]: ...
    def get_config_json_schema(self, provider_id: str) -> dict[str, Any]: ...

    # Construction
    def create_training(self, provider_id: str,
                        ctx: ProviderContext) -> Result[IGPUProvider, ProviderError]: ...
    def create_inference(self, provider_id: str,
                         ctx: ProviderContext) -> Result[IInferenceProvider, ProviderError]: ...
    def create_resume_provider(self, provider_id: str,
                               metadata: ResumeMetadata) -> Result[IPauseResumeProvider, ProviderError]: ...

    # Pod-side (только class — pod сам конструирует с runtime args)
    def resolve_pod_lifecycle_client_cls(self, provider_id: str
                                         ) -> type[IPodLifecycleClient]: ...
```

### 5.2 Поведение

- **Auto-discovery**: `from_filesystem()` walks `packages/providers/src/ryotenkai_providers/*/provider.toml` (+ extra roots). Eagerly валидирует manifest, **lazily** importlib-resolve'ит entry-point классы (RunPod SDK не пагулит на single-node-only env).
- **Role mismatch**: `create_inference("training_only_provider", ctx)` → `Err(code="PROVIDER_ROLE_MISMATCH")` — clean skip, не raise.
- **Defensive load** (per [community.loader](packages/community/src/ryotenkai_community/loader.py) pattern + Python.org best-practices): один плохой manifest не валит весь registry. `LoadFailure` собирается в отдельный список — surface-able через CLI / API.
- **Singleton**: `get_registry()` lazy module-level. Тесты — собственный instance с custom `roots`.

---

## 6. Pod-side: projected sub-manifest

**Decision:** ship НЕ полный `provider.toml` на pod, а **projected sub-manifest** (only what pod needs).

### 6.1 Projection

Чистая функция `project_to_pod_manifest(manifest: ProviderManifest) -> PodProviderManifest`. Output:

```toml
schema_version = 1
[provider]
id = "runpod"
[capabilities]
supports_lifecycle_actions = true   # gate flag
[entry_points.pod_lifecycle_client]
module = "ryotenkai_providers.runpod.runtime.lifecycle_client"
class  = "RunPodPodLifecycleClient"
```

Build-step: новый CLI `python packages/providers/scripts/compile_pod_manifests.py` walks все `provider.toml`, проецирует, пишет в `packages/pod/src/ryotenkai_pod/runner/runtime/pod_manifests/<id>.toml`. Запускается из `Makefile` цели `pre-pod-build` или вручную в migration PR.

### 6.2 Pod-side runtime

[`packages/pod/src/ryotenkai_pod/runner/runtime/provider_registry.py`](packages/pod/src/ryotenkai_pod/runner/runtime/provider_registry.py):

```python
def resolve_lifecycle_client_from_env(env: Mapping[str, str]) -> IPodLifecycleClient:
    name = env.get(RUNTIME_PROVIDER_ENV_VAR)
    pod_manifest = _load_pod_manifest(name)            # reads projected toml
    if not pod_manifest.capabilities.supports_lifecycle_actions:
        return _BuiltinNoOpClient()                    # no entry-point lookup needed
    locator = pod_manifest.entry_points.pod_lifecycle_client
    cls = importlib.import_module(locator.module).__dict__[locator.class_name]
    return cls(...)  # constructed with provider-specific env
```

`_BuiltinNoOpClient` — единственный "noop" remaining (10 строк, в pod side). `single_node/runtime/lifecycle_client.py` **удаляется** (был отдельный класс ради 1 noop метода — выкидываем по KISS).

### 6.3 Pod manifests ship via code_syncer

`pod_manifests/*.toml` живут в `packages/pod/src/ryotenkai_pod/runner/runtime/pod_manifests/` — auto-shipped через уже существующий `PROVIDED_PACKAGES = [..., ("packages/pod/src/ryotenkai_pod", "ryotenkai_pod")]` в [code_syncer.py](packages/control/src/ryotenkai_control/pipeline/stages/managers/deployment/code_syncer.py:84). **Никаких новых rsync-целей.**

---

## 7. Удаление 11 string-dispatch leftovers

| # | Локация | Сейчас | После |
|---|---|---|---|
| 1 | `startup_validator.py:52,54` (`_resolve_required_secrets_for_provider`) | `if provider == PROVIDER_RUNPOD: return ("RUNPOD_API_KEY",)` | `registry.required_secrets(provider, role=ProviderRole.TRAINING)` — manifest-driven |
| 2 | `provider_config.py:38` (`is_single_node_provider`) | `name == PROVIDER_SINGLE_NODE` | `registry.capabilities(provider).is_local`. Helper удаляется. |
| 3 | `dependency_installer.py:65` | `is_single_node_provider(self.config)` | То же — `caps.is_local` |
| 4 | `training_launcher.py:500` (fallback branch) | mixed `is_local` / `is_single_node_provider` | Просто `caps.is_local`; fallback удаляется (registry гарантирует наличие провайдера) |
| 5 | `api/services/config_service.py:98` | imports `_resolve_required_secrets_for_provider` | `registry.required_secrets(...)` |
| 6 | `resume_service.py:141,149` | `if PROVIDER_RUNPOD: from ... import RunPodProvider` | `registry.create_resume_provider(name, metadata)` — провайдер сам декларирует resume_factory в manifest |
| 7 | `resume_service.py:374` (`is_capacity_error_message`) | `if metadata.provider == PROVIDER_RUNPOD: ...` | `provider = registry.create_training(...).unwrap()`; `if isinstance(provider, ICapacityErrorClassifier): provider.is_capacity_error(msg)` |
| 8 | `resume_service.py:434,441` | duplicates of #6 | То же |
| 9 | `training_monitor.py:682` | `if self._provider_name != "runpod"` | `if not isinstance(self._provider, IRecoveryProbeProvider): return None`; recovery вызывает `provider.attempt_recovery(resource_id)` |

**Итог:** 0 string-checks `provider_name == ...` в business-logic коде. Все диспатчи через capability flag (manifest) + Protocol membership (mypy-проверяемое).

---

## 8. Scaffolder: dev-script (НЕ CLI subcommand)

Per user's clarification: провайдеры добавляются только разрабами, не пользователями. → НЕ extension `ryotenkai plugin scaffold`, НЕ subcommand `ryotenkai`, НЕ console_script.

### 8.1 Местоположение

```
packages/providers/scripts/
├── new_provider.py         # генератор скелета
└── compile_pod_manifests.py # build-step для pod sub-manifests (см. §6)
```

### 8.2 `new_provider.py` invocation

```bash
python packages/providers/scripts/new_provider.py <provider_id> \
    [--roles training,inference] \
    [--with-lifecycle] \
    [--with-recovery-probe] \
    [--with-capacity-classifier]
```

### 8.3 Что генерирует

Под `packages/providers/src/ryotenkai_providers/<provider_id>/`:

```
provider.toml                           # manifest со всеми TODO-полями
__init__.py                              # пустой (registration через manifest auto-discovery)
training/
  __init__.py
  provider.py                            # IGPUProvider skeleton (если "training" в roles)
inference/
  __init__.py
  provider.py                            # IInferenceProvider skeleton (если "inference" в roles)
runtime/                                 # только если --with-lifecycle
  __init__.py
  lifecycle_client.py                    # IPodLifecycleClient skeleton
```

Тесты под `packages/providers/tests/unit/providers/<provider_id>/`:

```
conftest.py                              # фикстура _mk_<provider> (yields provider_id, factory)
test_capabilities.py                     # пинит cap flags из manifest
test_manifest_load.py                    # asserts manifest validates
test_lifecycle.py                        # только если --with-lifecycle (smoke с mock-API)
```

### 8.4 Что НЕ генерируется (auto-discovered)

- Запись в `test_factory_capability_invariant.py`: тест **переписывается** на auto-discovery через `ProviderRegistry.from_filesystem().list()` + parametrize. Новый провайдер автоматически попадает в матрицу.
- Запись в startup_validator/resume_service/etc: эти файлы **больше не дискриминируют** по провайдеру — registry всё знает.
- Запись в pod registry: `compile_pod_manifests.py` projects sub-manifest автоматически (плюс CI-step проверяет idempotence).

### 8.5 Шаблоны

Reuse `dump_manifest_toml(..., todo_fields=...)` style из [scaffold_template.py](packages/community/src/ryotenkai_community/scaffold_template.py:62). Skeleton classes используют `raise NotImplementedError("TODO: ...")` (НЕ `pass`) — mypy не жалуется на empty bodies, runtime сразу ругается если разраб забыл реализовать (R21).

### 8.6 CI smoke check для свеже-сгенерированного провайдера

`scripts/test_scaffolder.sh` (CI step): `new_provider.py test_provider --roles training` → `mypy` + `pytest packages/providers/tests/unit/providers/test_provider/` должны пройти. Catches scaffolder regression.

---

## 8.7 Tooling: validation + projection scripts (manifest-code consistency)

Между `provider.toml` и Python-кодом могут разъехаться: имена классов, capability flags vs Protocol membership, required env vars, config schema class. **Двунаправленный sync — anti-pattern** (теряется single source of truth). Подход: manifest = источник правды, **код производный**, drift ловится валидаторами **до коммита**.

Три скрипта в `packages/providers/scripts/`, все **single-direction**, idempotent:

### 8.7.1 `check_manifests.py` — drift detector (главный)

```bash
python packages/providers/scripts/check_manifests.py [--strict] [--verbose]
```

Запускается локально (pre-commit hook), в CI, и через scaffolder `--check` flag. **Ничего не генерирует**. Проверяет инварианты:

| Invariant | Что проверяет |
|---|---|
| Class exists | `entry_points.training.class` существует, импортируется (importlib lazy resolve) |
| Class identity | Class из `entry_points.training` наследует `IGPUProvider` (runtime-checkable + explicit method presence) |
| Inference parity | Class из `entry_points.inference` наследует `IInferenceProvider` |
| Lifecycle parity | `entry_points.pod_lifecycle_client` class наследует `IPodLifecycleClient` |
| Config schema parity | `entry_points.config_schema` class — subclass of `pydantic.BaseModel`, имеет `connect/cleanup` поля |
| Cap ↔ Protocol parity | `has_pause_resume=true` ⇔ `IPauseResumeProvider` в bases (true ↔ false drift) |
| Cap ↔ Protocol parity | `supports_recovery_probe=true` ⇔ `IRecoveryProbeProvider` в bases |
| Cap ↔ Protocol parity | `supports_capacity_error_detection=true` ⇔ `ICapacityErrorClassifier` в bases |
| Cap ↔ Protocol parity | `supports_lifecycle_actions=true` ⇔ `ITerminalActionProvider` в bases |
| Required env coverage | Каждый `[[required_env]] name = "X"` ∈ `provider.required_secrets()` (нет orphan ни в одну сторону) |
| Roles ↔ entry_points | `roles = ["training", "inference"]` ⇔ оба `entry_points.training` и `entry_points.inference` присутствуют |
| Provider id consistency | `manifest.provider.id == provider_class.provider_id` (instance-level) |
| Pod manifest projection | `pod_manifests/<id>.toml` совпадает с `project_to_pod_manifest(provider_manifest)` |

**Output при drift** — человекочитаемое описание с file:line + рекомендация как починить:

```
❌ packages/providers/src/ryotenkai_providers/runpod/provider.toml:14
   capabilities.has_pause_resume = true
   но class RunPodProvider (training/provider.py:91) не наследует
   IPauseResumeProvider.

   Fix: либо добавь IPauseResumeProvider в bases, либо убери flag.

❌ packages/providers/src/ryotenkai_providers/runpod/provider.toml:31
   entry_points.training.class = "RunPodProvider2"
   но класс не найден в module ryotenkai_providers.runpod.training.provider.

   Fix: переименуй класс в Python или обнови manifest.

✓ 12 invariants checked across 2 manifests, 0 drift detected.
```

Внутри — те же инварианты что в pytest (см. §10.4), но как отдельный CLI: быстрее (~1s vs pytest startup), фокусированный output, человекочитаемая диагностика. **CI прогоняет через pytest**; **разраб локально** — через скрипт.

### 8.7.2 `compile_pod_manifests.py` — pod sub-manifest projection

Уже описан в §6.1. Pure function, idempotent, with CI-step `git diff --exit-code` после прогона.

### 8.7.3 `generate_config_json_schemas.py` — JSON Schema export для Web UI

```bash
python packages/providers/scripts/generate_config_json_schemas.py
```

Walks все manifests, для каждого: `ConfigCls = registry.get_config_class(id); schema = ConfigCls.model_json_schema(); write to packages/providers/src/ryotenkai_providers/<id>/config_schema.generated.json`.

**Зачем:** Web UI ([ConfigBuilder/FieldRenderer.tsx](web/src/components/ConfigBuilder/FieldRenderer.tsx)) сегодня хардкодит fields для каждого провайдера в TypeScript отдельно от Python-схем. С generation: UI делает `import schema from "<id>/config_schema.generated.json"` или `GET /api/providers/<id>/config-schema` (бэкенд отдаёт runtime-resolved). Single source of truth — Pydantic.

Опционально (можно landить отдельным малым PR-3 после Web UI integration).

### 8.7.4 `scripts/test_scaffolder.sh` — end-to-end smoke

CI-step. Вызывает `new_provider.py test_provider --roles training,inference --with-lifecycle` → `check_manifests.py` → `mypy packages/providers/src/ryotenkai_providers/test_provider/` → `pytest packages/providers/tests/unit/providers/test_provider/`. После — cleanup. Catches любой regression в цепочке scaffolder → manifest → Python код → тесты.

### 8.7.5 Что **не** автоматизируем

- **Codegen Python из TOML** (manifest → Pydantic class) — антипаттерн, разраб хочет писать Python руками.
- **Codegen TOML из Python** (Python AST → manifest) — антипаттерн, manifest становится derived data, теряет смысл декларации.
- **Bidirectional sync** — два источника правды, расхождения молчат.

---

## 9. Migration plan — 2 PRs (per user choice)

### PR-1: «Provider registry + capability advertisement migration»

**Touches:** ~30 файлов. Атомарный cut-over всей factory + всех string-checks.

#### Adds

1. `packages/providers/src/ryotenkai_providers/manifest.py` — Pydantic `ProviderManifest`, `LATEST_PROVIDER_SCHEMA_VERSION = 1`, `ProviderRole` enum, schema-level validators (§4.2).
2. `packages/providers/src/ryotenkai_providers/registry.py` — `ProviderRegistry`, `ProviderContext`, `get_registry()` singleton.
3. `packages/providers/src/ryotenkai_providers/training/interfaces.py`:
   - `IProviderBase` (3 props: `provider_id`, `provider_name`, `provider_type`, `get_capabilities`).
   - `IPauseResumeProvider`, `IRecoveryProbeProvider`, `ICapacityErrorClassifier` (NEW micro-Protocols).
   - `IGPUProvider` extends `IProviderBase` (drops duplicated declarations).
   - `ProviderCapabilities` gets `supports_recovery_probe`, `supports_capacity_error_detection`.
4. `packages/providers/src/ryotenkai_providers/inference/interfaces.py`:
   - `IInferenceProvider` extends `IProviderBase`.
5. `packages/providers/src/ryotenkai_providers/runpod/provider.toml` (полный, как в §4.1).
6. `packages/providers/src/ryotenkai_providers/single_node/provider.toml` (роли training+inference, `is_local=true`, `supports_lifecycle_actions=false`).
6a. `packages/providers/src/ryotenkai_providers/runpod/config.py` — Pydantic-модель `RunPodProviderConfig` (со sub-моделями `connect/training/inference/cleanup`, см. §4.4). Заменяет существующий untyped dict access во всех call sites.
6b. `packages/providers/src/ryotenkai_providers/single_node/config.py` — `SingleNodeProviderConfig`.
6c. Изменения в `packages/shared/src/ryotenkai_shared/config/pipeline/providers.py`: `PipelineProviderMixin.get_provider_config` теперь возвращает не `dict[str, Any]`, а typed `BaseModel` instance, найденный через `registry.get_config_class()`.
7. `RunPodProvider`:
   - Inherits `IPauseResumeProvider, IRecoveryProbeProvider, ICapacityErrorClassifier`.
   - `attempt_recovery(resource_id) -> Result[ProviderStatus, ProviderError]` — извлекаем из `training_monitor._recover_pod_if_needed`.
   - `is_capacity_error(message) -> bool` — переносим логику из `runpod.lifecycle.policy:is_capacity_error_message`.
   - `from_resume_metadata(...)` (уже есть, manifest-declared).
   - Constructor: `__init__(self, ctx: ProviderContext)`.
8. `SingleNodeProvider`: constructor adapt to `ProviderContext`.

#### Modifies (call-site migration)

Все 11 локаций из §7. Каждый — простая замена на `registry.X(...)` или `isinstance(provider, IXProvider)`.

#### Removes

- ❌ `packages/providers/src/ryotenkai_providers/training/factory.py` (целиком).
- ❌ `packages/providers/src/ryotenkai_providers/inference/factory.py` (целиком).
- ❌ `_resolve_required_secrets_for_provider` в `startup_validator.py:37-56`.
- ❌ `is_single_node_provider` helper в `provider_config.py` (или редуцируется в 1-line wrapper, см. open question J.4 ниже).
- ❌ Все `if provider_name == PROVIDER_RUNPOD / PROVIDER_SINGLE_NODE` ветки в названных файлах.
- ❌ `auto_register_providers()` + `__init__.py` import side-effects в RunPod/SingleNode.

#### Pod-side: transitional shim

В PR-1 **сохраняем** `_LIFECYCLE_CLIENT_LOCATORS` в pod registry, но **derive'им** его from manifests at module-import time (literal dict становится derived data). Это делает PR-1 self-contained без необходимости менять pod runtime сейчас.

---

### PR-2: «Pod-side projection + scaffolder + auto-discovery test»

**Touches:** ~12 файлов. Independent from PR-1, can be reviewed/landed separately.

#### Adds

1. `packages/providers/scripts/compile_pod_manifests.py` — build-step для projected pod sub-manifests.
2. `packages/providers/scripts/new_provider.py` — scaffolder dev-script.
3. `packages/providers/scripts/check_manifests.py` — drift validator (§8.7.1) — main UX для разраба.
4. `packages/providers/scripts/generate_config_json_schemas.py` — JSON Schema export (опционально, §8.7.3).
5. `packages/pod/src/ryotenkai_pod/runner/runtime/pod_manifests/runpod.toml` (generated by build-step).
6. `packages/pod/src/ryotenkai_pod/runner/runtime/pod_manifests/single_node.toml` (или skip — see §10).
7. `_BuiltinNoOpClient` в pod registry для providers с `supports_lifecycle_actions=false`.
8. `packages/providers/tests/unit/providers/test_manifest_pod_parity.py` — cross-process invariant test (см. §11).
9. `packages/providers/tests/unit/providers/test_check_manifests.py` — unit-тесты на сам валидатор (positive + negative + boundary).
10. `Makefile` targets: `compile-pod-manifests`, `check-manifests`, `generate-config-schemas`.
11. `scripts/test_scaffolder.sh` — end-to-end smoke (§8.7.4).
12. Pre-commit hook config: `check_manifests.py` runs on changes к `provider.toml` или `<provider>/config.py`.

#### Modifies

- `packages/pod/src/ryotenkai_pod/runner/runtime/provider_registry.py`: реализация switches с transitional shim → real pod-manifest projection reader.
- `packages/providers/tests/unit/providers/training/test_factory_capability_invariant.py`: rewrite на auto-discovery через `ProviderRegistry.from_filesystem().list()` + parametrize.
- `packages/providers/src/ryotenkai_providers/README.md`: новый раздел «Adding a provider» с pointer на scaffolder.

#### Removes

- ❌ Hand-maintained `_LIFECYCLE_CLIENT_LOCATORS` literal в pod registry (заменяется projection reader).
- ❌ Hand-maintained parametrize lists в test_factory_capability_invariant.
- ❌ `packages/providers/src/ryotenkai_providers/single_node/runtime/lifecycle_client.py` (`NoOpPodLifecycleClient` — заменяется built-in `_BuiltinNoOpClient` в pod, см. open Q J.4).

---

## 10. Test strategy — 8 категорий

### 10.1 POSITIVE
- `test_registry_discovers_all_in_tree_manifests` — `from_filesystem()` returns 2 manifests.
- `test_create_training_returns_igpuprovider` — for каждой провайдер с `training` в roles.
- `test_create_inference_returns_iinferenceprovider`.
- `test_required_secrets_returns_manifest_value` — RunPod=`("RUNPOD_API_KEY",)`, SingleNode=`()`.
- `test_capabilities_match_manifest`.
- `test_resume_factory_invokable` — RunPod (declared), SingleNode (no resume_factory → registry returns Err cleanly).
- `test_get_config_class_returns_pydantic_model` — для каждого manifest, `registry.get_config_class(id)` возвращает subclass of `BaseModel`.
- `test_yaml_block_validates_against_provider_config_class` — реальный YAML с RunPod секцией парсится в `RunPodProviderConfig` без ошибок.
- `test_get_config_json_schema_returns_valid_schema` — JSON Schema is valid (`jsonschema.Draft202012Validator.check_schema(...)`).

### 10.2 NEGATIVE
- `test_missing_locator_rejected_at_load` — manifest с `roles=[training]` но без `entry_points.training`.
- `test_malformed_roles_rejected` — `roles=["potato"]`.
- `test_role_mismatch_returns_err` — `create_inference` для training-only.
- `test_unknown_provider_id_returns_err` — `code="PROVIDER_NOT_REGISTERED"`.
- `test_lifecycle_declared_but_no_pod_locator_rejected_at_load`.
- `test_capacity_error_classifier_with_local_provider_rejected_at_load`.
- `test_pause_resume_without_lifecycle_actions_rejected`.
- `test_unknown_field_in_provider_yaml_rejected` — `extra="forbid"` ловит опечатку в `providers.runpod.training.cloud_typo`.
- `test_invalid_type_in_provider_yaml_rejected` — `container_disk_gb: "сто"` ловится как not-int с line:column.
- `test_missing_training_block_when_role_declares_training_rejected` — `roles=["training"]` но в YAML `provider_block.training is None`.
- `test_config_schema_class_not_basemodel_subclass_rejected_at_load` — manifest указывает на класс не-BaseModel.

### 10.3 BOUNDARY
- `test_zero_secret_provider` — SingleNode `required_secrets()=()`.
- `test_unicode_in_provider_id_rejected_by_regex`.
- `test_empty_roles_rejected`.
- `test_schema_version_in_future_rejected_with_upgrade_message`.
- `test_duplicate_provider_id_in_two_manifests_fatal`.
- `test_provider_with_only_training_role_excluded_from_inference_list`.
- `test_max_runtime_hours_null_means_unlimited`.

### 10.4 INVARIANT (parametrized over discovered manifests — auto-extending)
- `test_capability_protocol_parity` — для каждой cap flag в manifest проверить `isinstance(provider, IXProvider)` matches.
- `test_provider_id_consistency` — `manifest.provider.id == training_class.provider_id == inference_class.provider_id == lifecycle_client.provider_id`.
- `test_pod_lifecycle_client_present_when_supports_lifecycle_actions` — Mac↔pod parity (NEW).
- `test_pod_manifest_projection_idempotent` — `project(manifest).load() == project(manifest)` round-trip.
- `test_is_local_implies_no_lifecycle_actions`.
- `test_pause_resume_implies_lifecycle_actions`.
- `test_capacity_error_detection_implies_cloud`.
- `test_explicit_protocol_method_presence` — for each Protocol method, `callable(getattr(cls, name, None))` (R9: not relying on `runtime_checkable`'s shallow check alone).
- `test_config_schema_class_is_basemodel_subclass` — для каждого manifest.
- `test_config_schema_has_extra_forbid` — `model_config.extra == "forbid"`.
- `test_config_schema_has_required_subblocks` — `connect`, `cleanup` обязательны; `training`/`inference` обязательны если role в `roles`.
- `test_required_env_in_class_required_secrets` — `manifest.required_env[].name == provider.required_secrets()` (R28 drift detection).
- `test_check_manifests_script_exits_zero_on_clean_repo` — основная инвариантная команда `python scripts/check_manifests.py` exits 0 — drift catcher.

### 10.5 DEPENDENCY ERROR
- `test_pod_locator_class_not_importable_yields_clear_error` — clear `ProviderRegistryError` с module path + class name.
- `test_pod_locator_class_does_not_implement_protocol`.
- `test_smoke_import_eagerly_resolves_all_entry_points` — CI step (R8).

### 10.6 REGRESSION (one per removed string-dispatch site)
- `test_startup_validator_uses_registry_for_secrets` — replaces old `if provider == RUNPOD` test.
- `test_resume_service_uses_registry_for_resume_provider`.
- `test_resume_service_capacity_error_via_capability_protocol` — provider без `ICapacityErrorClassifier` ⇒ classifier skipped.
- `test_training_monitor_skips_recovery_when_protocol_absent` — SingleNode case.
- `test_training_launcher_uses_caps_is_local`.
- `test_dependency_installer_uses_caps_is_local`.
- `test_config_service_uses_registry_for_secrets`.

### 10.7 LOGIC-SPECIFIC
- `test_list_role_training_excludes_inference_only` — fabricated inference-only manifest в tmp dir.
- `test_list_role_none_returns_all`.
- `test_resolve_pod_lifecycle_client_cls_returns_lazy_imported_class`.
- `test_capability_advertisement_blocks_optional_call` — analogous to MCP capability negotiation: caller bez cap flag не вызовет optional method.

### 10.8 COMBINATORIAL
- 2 providers × 3 roles × 2 lifecycle states (paused/active) — parametrized matrix `test_combinatorial_provider_role_lifecycle.py`.

---

## 11. Risk register — 3 итерации

### Iteration 1 (surface)

| # | Risk | Mitigation |
|---|---|---|
| R1 | Pydantic v2 boot overhead | Lazy-import entry-point classes; eager только manifest TOMLs (~ms-уровень) |
| R2 | Locator string drift при rename класса | CI smoke test: import every entry-point class — fast pre-flight check |
| R3 | Pod image / manifest schema skew | `schema_version` в pod sub-manifest; pod rejects > LATEST_POD_SCHEMA с явным message |
| R4 | Cross-package invariant test поднимается из providers package — нужно импортировать pod | uv-workspace already supports it, `test_manifest_pod_parity` lives in `packages/providers/tests/` |
| R5 | Schema bump → docker rebuild | Policy: additive-only changes don't bump version. Bump только при breaking. Документировать в CHANGELOG |
| R6 | Operator UX при malformed manifest | Catch `ValidationError` at registry-load; rewrap в `ProviderRegistryError` с file path + offending field path |
| R7 | Silent failure: третий провайдер с `supports_lifecycle_actions=true` без pod entry | Schema-level invariant (§4.2 #3) — REJECT manifest at load. Plus `test_manifest_pod_parity`. Defense in depth |

### Iteration 2 (deeper)

| # | Risk | Mitigation |
|---|---|---|
| R8 | importlib lazy hide incompatible API change (RunPod SDK breaks `__init__`) | `compile_pod_manifests.py` имеет `--smoke-import` flag, eagerly imports every entry-point class once. Run в CI |
| R9 | `runtime_checkable` Protocol только attribute presence — false positives | Invariant test проверяет `callable(getattr(cls, name, None))` для каждого Protocol method explicitly |
| R10 | Two sources of truth для capabilities (manifest vs hand-coded `get_capabilities()`) | `IProviderBase.get_capabilities()` derived from `cls._manifest_capabilities` (set by loader). Hand-override invariant-tested |
| R11 | `required_env[].required_when_role` schema rigid | Ignore unknown role-keys with warning (loose mode) per [community.loader pattern](packages/community/src/ryotenkai_community/loader.py) |
| R12 | `ProviderContext` becomes god-object | `frozen=True` + PR-review field set; doc-string locks the contract |
| R13 | Out-of-tree (3rd-party) providers | `from_filesystem(roots=[...])` accepts extra roots from config (`providers.discovery_roots: [path]`) |
| R14 | Auto-discovery test rewrite skips provider если conftest отсутствует | Top-level invariant: «for every manifest, fixture factory registered» — fail names missing conftest |

### Iteration 3 (deepest)

| # | Risk | Mitigation |
|---|---|---|
| R15 | Concurrent `from_filesystem()` race в тестах | Frozen instance, no module-level singleton mutation. Tests instantiate per-fixture. Production — exactly one instance at process start |
| R16 | Schema version invisible to operators | `ryotenkai status` CLI surfaces `loaded N providers, K rejected: ...` (per community catalog pattern) |
| R17 | Three sources of truth: provider.toml + pod_manifests/<id>.toml + pod-side reader | Projection — pure function with idempotence test (R10.4 invariant `test_pod_manifest_projection_idempotent`). Pod-side reader and projector consume same Pydantic model |
| R18 | `ProviderRole.BOTH` later request | Documented decision: roles is multi-valued list, BOTH not enum. Locked in §3.3 |
| R19 | `provider_id` drift class vs manifest | Class's `provider_id` reads `cls._manifest_provider_id` set by loader. Test fakes carry id via fixture. Invariant `test_provider_id_consistency` |
| R20 | `[[required_env]]` runs before secrets manager populates env (Vault/1Password) | Use `Secrets` object as source of truth, not `os.environ`. Schema kind: `secret_present_in_secrets_object`, не «env var present» |
| R21 | Scaffolder generates files that fail mypy | Skeletons emit `raise NotImplementedError("TODO: ...")` — mypy passes; runtime fails fast on missing impl |
| R22 | `max_runtime_hours = 0` sentinel ambiguity | Replace `0` sentinel with `null`/`None` → `Optional[int]` |
| R23 | uv-workspace cross-package import order | Registry lives in `ryotenkai_providers`, consumed by `ryotenkai_control`; never reverse. Sentinel test (already exists) prevents reverse edge |
| R24 | Operator runs `experimental` provider в prod | Registry `create_*` warns when `manifest.stability != "stable"`; CLI `--allow-experimental-providers` flag required для `stability=experimental` |
| R25 | Multiple `ProviderRegistry` instances в тестах wasting importlib cache | Optional `lazy_class_cache` параметр в constructor; tests pass fresh dict, prod uses module-level shared |
| R26 | `compile_pod_manifests.py` запускается вручную → разраб забывает → drift | CI step `make check-pod-manifests` regenerates and `git diff --exit-code`. PR'ы с stale projection не мерджатся |
| R27 | Manifest TOML parse error в CI logs показывается без line/column | Use `tomllib` (stdlib) — error includes line. Wrap для friendly message |
| R28 | Manifest-code drift: разраб переименовал класс / добавил cap flag без Protocol membership / убрал required_env, забыв обновить manifest. Drift молчит до runtime. | `check_manifests.py` (§8.7.1) проверяет каждый инвариант, runs in pre-commit + CI. Дублируется invariant test'ами в pytest (defense in depth). Каждый драйфт даёт human-readable error с file:line + recommendation как починить. |
| R29 | Migration с untyped dict на typed config schemas — существующие YAML конфиги пользователей могут ломаться (extra=forbid отвергает поля которые раньше прокатывали как noise) | Migration script `scripts/validate_user_configs.py` проходит по всем известным `pipeline_config.yaml` в проекте, валидирует через новые schemas. PR-1 release notes — список removed-noise полей с инструкцией. Period beta-release. |
| R30 | Pydantic v2 `model_json_schema()` output не стабилен между minor versions (изменение default ordering) → `config_schema.generated.json` diff в каждом PR | Hash diff с фильтром «structural-only changes». CI step `make check-config-schemas` отвергает только реальные structural изменения, форматирование игнорирует |
| R31 | `RunPodProviderConfig` Pydantic-модель импортирует heavy dependencies на module-load (e.g. `httpx`) — slow boot для single-node-only deployments | Config classes — pure data, никаких side imports. Lint rule в check_manifests: config module's top-level imports — только pydantic + stdlib + ryotenkai_shared.config |
| R32 | YAML errors сегодня показываются глобально (Pydantic собирает все errors), а пользователь хочет stop-on-first-error для quick iteration | Default — collect-all (production). Add `RYOTENKAI_FAIL_FAST=1` env для dev experience |

**Итого: 32 рисков, 29 closed by design, 3 retained as policy** (R5, R12, R16).

---

## 12. Open questions resolved

Восемь дизайн-вопросов из Plan agent:

| # | Question | Decision | Source |
|---|---|---|---|
| J.1 | Manifest location | per-provider colocated (`packages/providers/.../runpod/provider.toml`) | Mirrors community plugin convention |
| J.2 | Pod-shipping | projected sub-manifest (small footprint, schema-bumps don't touch pod) | Plan recommendation |
| J.3 | Scaffolder home | dev-script `packages/providers/scripts/new_provider.py` (НЕ CLI) | **User answer** — providers add only by devs |
| J.4 | NoOpPodLifecycleClient delete | **DELETE** — pod resolver checks cap flag первым, installs `_BuiltinNoOpClient` (10 lines) | KISS, removes dead class |
| J.5 | Registry singleton | Yes, `get_registry()` lazy module-level | Performance + simplicity |
| J.6 | Capability methods placement | **Separate Protocols** (`IPauseResumeProvider`, `IRecoveryProbeProvider`, `ICapacityErrorClassifier`) — symmetric with `ITerminalActionProvider` | **Research-backed**: MCP, LSP, Crossplane v2.2 capability advertisement pattern |
| J.7 | PR phasing | **2 PRs** (registry+migration, then pod parity+scaffolder) | **User answer** |
| J.8 | provider_id vs provider_name | Keep both — id (machine), name (display) | Mirrors community plugin schema |

---

## 13. Verification

После landing PR-1:

1. `uv run pytest packages/providers/tests/ packages/control/tests/sentinel/` — all green.
2. `uv run lint-imports` — все 6 контрактов KEPT (включая «control must not import pod»).
3. `cd /Users/daniil/MyProjects/RyotenkAI && .venv/bin/ryotenkai run start -c <config>` — пайплайн доходит до stage 2-3 (pod creation + training launch).
4. Smoke import: `uv run python -c "from ryotenkai_providers.registry import get_registry; r = get_registry(); print(r.list())"` — `('runpod', 'single_node')`.
5. Search для residual leaks: `grep -rn 'provider_name == \|PROVIDER_RUNPOD ==\|PROVIDER_SINGLE_NODE ==' packages/control/src/` — 0 hits в business-logic коде (только factory/registry seams).
6. Config validation: подмена опечатки в YAML (`cloud_typo: ALL` вместо `cloud_type: ALL`) → `ryotenkai run start` падает на загрузке config'а с clear error `providers.runpod.training: extra fields not permitted [cloud_typo]` + line:column.
7. Typed access works: `uv run mypy packages/control/src/ryotenkai_control/pipeline/stages/managers/deployment/training_launcher.py` — typed access к `provider_block.training.gpu_type` проходит mypy strict mode.

После landing PR-2:

1. `python packages/providers/scripts/new_provider.py test_provider --roles training` → directory created.
2. `mypy packages/providers/src/ryotenkai_providers/test_provider/ packages/providers/tests/unit/providers/test_provider/` — passes.
3. `pytest packages/providers/tests/unit/providers/test_provider/` — passes (NotImplementedError tests).
4. `python packages/providers/scripts/compile_pod_manifests.py && git diff --exit-code packages/pod/src/ryotenkai_pod/runner/runtime/pod_manifests/` — clean (idempotent).
5. End-to-end: `ryotenkai run start ...` снова доходит до stage 3+ — pod registry load работает с projected manifests.
6. Drift detection: искусственное расхождение (например, hand-edit `provider.toml` `has_pause_resume = false` без удаления `IPauseResumeProvider` из bases) → `python packages/providers/scripts/check_manifests.py` exits 1 с описанием drift и рекомендацией fix.
7. Drift detection в CI: pre-commit hook (или CI job) валит PR с drift'ом до merge.
8. JSON Schema export: `python packages/providers/scripts/generate_config_json_schemas.py && git diff --exit-code packages/providers/src/ryotenkai_providers/*/config_schema.generated.json` — clean (idempotent).
9. Scaffolder smoke: `bash scripts/test_scaffolder.sh` — passes без manual cleanup (всё гонится в tmpdir).

---

## 14. Critical files (PR-1)

**Add:**
- `packages/providers/src/ryotenkai_providers/manifest.py` — Pydantic schema + validators
- `packages/providers/src/ryotenkai_providers/registry.py` — ProviderRegistry, ProviderContext
- `packages/providers/src/ryotenkai_providers/runpod/provider.toml`
- `packages/providers/src/ryotenkai_providers/single_node/provider.toml`
- `packages/providers/src/ryotenkai_providers/runpod/config.py` — `RunPodProviderConfig` + sub-models
- `packages/providers/src/ryotenkai_providers/single_node/config.py` — `SingleNodeProviderConfig` + sub-models

**Modify:**
- [packages/providers/src/ryotenkai_providers/training/interfaces.py](packages/providers/src/ryotenkai_providers/training/interfaces.py) — IProviderBase, 3 micro-Protocols, 2 new cap flags
- [packages/providers/src/ryotenkai_providers/inference/interfaces.py](packages/providers/src/ryotenkai_providers/inference/interfaces.py) — extend IProviderBase
- [packages/shared/src/ryotenkai_shared/config/pipeline/providers.py](packages/shared/src/ryotenkai_shared/config/pipeline/providers.py) — `get_provider_config` returns typed BaseModel via `registry.get_config_class()`
- [packages/providers/src/ryotenkai_providers/runpod/training/provider.py](packages/providers/src/ryotenkai_providers/runpod/training/provider.py) — extract `attempt_recovery`, `is_capacity_error`, `from_resume_metadata` (already exists), implement micro-Protocols, ProviderContext init
- [packages/providers/src/ryotenkai_providers/single_node/training/provider.py](packages/providers/src/ryotenkai_providers/single_node/training/provider.py) — ProviderContext init
- [packages/providers/src/ryotenkai_providers/runpod/inference/pods/provider.py](packages/providers/src/ryotenkai_providers/runpod/inference/pods/provider.py) — ProviderContext init
- [packages/providers/src/ryotenkai_providers/single_node/inference/provider.py](packages/providers/src/ryotenkai_providers/single_node/inference/provider.py) — ProviderContext init
- [packages/control/src/ryotenkai_control/pipeline/bootstrap/startup_validator.py](packages/control/src/ryotenkai_control/pipeline/bootstrap/startup_validator.py) — registry-driven secrets
- [packages/control/src/ryotenkai_control/pipeline/launch/resume_service.py](packages/control/src/ryotenkai_control/pipeline/launch/resume_service.py) — registry + Protocol-based
- [packages/control/src/ryotenkai_control/pipeline/stages/training_monitor.py](packages/control/src/ryotenkai_control/pipeline/stages/training_monitor.py) — `IRecoveryProbeProvider` check
- [packages/control/src/ryotenkai_control/pipeline/stages/managers/deployment/provider_config.py](packages/control/src/ryotenkai_control/pipeline/stages/managers/deployment/provider_config.py) — caps.is_local
- [packages/control/src/ryotenkai_control/pipeline/stages/managers/deployment/dependency_installer.py](packages/control/src/ryotenkai_control/pipeline/stages/managers/deployment/dependency_installer.py)
- [packages/control/src/ryotenkai_control/pipeline/stages/managers/deployment/training_launcher.py](packages/control/src/ryotenkai_control/pipeline/stages/managers/deployment/training_launcher.py)
- [packages/control/src/ryotenkai_control/api/services/config_service.py](packages/control/src/ryotenkai_control/api/services/config_service.py)

**Remove:**
- `packages/providers/src/ryotenkai_providers/training/factory.py`
- `packages/providers/src/ryotenkai_providers/inference/factory.py`
- `packages/providers/src/ryotenkai_providers/single_node/runtime/lifecycle_client.py` (PR-2)
- `_resolve_required_secrets_for_provider` function в startup_validator
- `is_single_node_provider` helper
- All `if provider_name == PROVIDER_X` ветки в 11 локациях

## 15. Critical files (PR-2)

**Add:**
- `packages/providers/scripts/new_provider.py`
- `packages/providers/scripts/compile_pod_manifests.py`
- `packages/providers/scripts/check_manifests.py` — drift validator (§8.7.1)
- `packages/providers/scripts/generate_config_json_schemas.py` — JSON Schema export (§8.7.3)
- `scripts/test_scaffolder.sh` — end-to-end smoke (§8.7.4)
- `packages/pod/src/ryotenkai_pod/runner/runtime/pod_manifests/runpod.toml` (generated)
- `packages/pod/src/ryotenkai_pod/runner/runtime/pod_manifests/single_node.toml` (generated)
- `packages/providers/tests/unit/providers/test_manifest_pod_parity.py`
- `packages/providers/tests/unit/providers/test_check_manifests.py` — unit-тесты на сам валидатор
- `.pre-commit-config.yaml` entry: `check_manifests.py` runs on `provider.toml`/`config.py` changes
- `Makefile` targets: `compile-pod-manifests`, `check-manifests`, `generate-config-schemas`

**Modify:**
- [packages/pod/src/ryotenkai_pod/runner/runtime/provider_registry.py](packages/pod/src/ryotenkai_pod/runner/runtime/provider_registry.py) — switch from `_LIFECYCLE_CLIENT_LOCATORS` literal to projection reader
- [packages/providers/tests/unit/providers/training/test_factory_capability_invariant.py](packages/providers/tests/unit/providers/training/test_factory_capability_invariant.py) — auto-discovery rewrite
- [packages/providers/src/ryotenkai_providers/README.md](packages/providers/src/ryotenkai_providers/README.md) — new "Adding a provider" section
- `Makefile` — `compile-pod-manifests` target

**Remove:**
- `_LIFECYCLE_CLIENT_LOCATORS` literal в pod registry
- Hand-maintained parametrize lists в test_factory_capability_invariant
- `packages/providers/src/ryotenkai_providers/single_node/runtime/lifecycle_client.py`

---

## 16. Best-practice compliance check

| Practice | Compliance |
|---|---|
| **SOLID — SRP** | ✅ Каждый Protocol — одна responsibility (training, inference, lifecycle, recovery, capacity-classification). |
| **SOLID — ISP** | ✅ Capability methods — отдельные micro-Protocols, никто не платит за то что не использует. |
| **SOLID — OCP** | ✅ Новый провайдер = manifest + класс; registry/factories/validators не меняются. |
| **SOLID — DIP** | ✅ Control-plane зависит от Protocol'ов, не от concrete classes. |
| **KISS** | ✅ Один registry, один construction signature, один scaffolder dev-script. NoOp-class удаляется (10-line built-in). |
| **DRY** | ✅ Manifest — single source of truth для capabilities, secrets, locators. Two factories коллапсируют в один. |
| **YAGNI** | ✅ `ProviderRole.BOTH` НЕ моделируется (multi-valued list); CLI subcommand НЕ добавляется (dev-only); 3-rd-party провайдеры опциональны. |
| **Boy scout rule** | ✅ Phase 14.D+F миграция доводится до конца, остальные string-checks вычищаются. |
| **Capability advertisement** | ✅ Industry pattern — MCP, LSP, Crossplane v2.2. |
| **Defense in depth** | ✅ Schema-level rejection + invariant test + runtime check для каждого critical invariant. |
| **Observability** | ✅ `ryotenkai status` surfaces loaded/rejected manifests. `LoadFailure` с file path + offending field. |
| **Reliability** | ✅ Defensive plugin loading (один плохой manifest не валит весь registry). Pre-factory secret check fail-fast at startup. Typed YAML schemas — config errors at load, not on stage 5. |
| **Rollback** | ✅ Two PRs — каждый self-contained. PR-2 не требует PR-1 schema bump. |
| **Type safety** | ✅ Provider-specific config — typed Pydantic-модели вместо `dict[str, Any]`. Mypy strict mode passes на typed access (`ctx.provider_block.training.gpu_type`). |
| **Drift detection** | ✅ Three-layer: (1) Pydantic schema validators (load-time), (2) `check_manifests.py` (pre-commit/CI), (3) pytest invariant tests (CI). Каждый уровень даёт human-readable error с file:line. |
| **UI integration** | ✅ JSON Schema из Pydantic (`generate_config_json_schemas.py`) → Web UI рендерит forms динамически. Один источник truth — Python. Убирает текущий drift между `FieldRenderer.tsx` и Python-схемами. |

---

**Plan ready for review.**
