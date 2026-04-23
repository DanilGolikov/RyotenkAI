# Settings → Integrations (Providers / HuggingFace / MLflow)

## Context

Сегодня:
- `/settings` показывает только **Providers** — переиспользуемые compute-конфиги (SingleNode / RunPod) с полным CRUD, историей версий, валидацией. Никаких токенов и «Test connection» у них нет.
- Секреты (`HF_TOKEN`, `RUNPOD_API_KEY`, `MLFLOW_TRACKING_URI`, …) живут **per-project** в `env.json` и редактируются во вкладке проекта `Settings` ([web/src/components/ProjectTabs/SettingsTab.tsx](web/src/components/ProjectTabs/SettingsTab.tsx)). Это смешивает «конфиг проекта» и «мой аккаунт у провайдера».
- MLflow / HuggingFace лежат инлайном в `experiment_tracking.*` ([src/config/integrations/mlflow.py](src/config/integrations/mlflow.py), [huggingface.py](src/config/integrations/huggingface.py)). Чтобы разделить токены между проектами, приходится копировать поля.
- `inference.enabled` — отдельный булев флаг, параллельный `inference.provider`. Два источника истины → грязные состояния (провайдер пустой, но enabled=true и наоборот).

Чего хотим:
1. **Settings → Integrations**: один экран со списками переиспользуемых интеграций — Providers, HuggingFace, MLflow. У каждой — конфиг, токен, кнопка **Test connection**.
2. В проектном YAML: `inference.provider`, `experiment_tracking.mlflow.integration`, `experiment_tracking.huggingface.integration` — dropdown из Settings. Локально в проекте остаются только по-настоящему project-specific поля (`experiment_name`, `repo_id`, `private`).
3. Схема `inference` теряет `enabled`: «провайдер выбран ⇒ этап включён».
4. `HF_TOKEN` и `RUNPOD_API_KEY` уходят из `ProjectTabs/SettingsTab` — они часть глобальной интеграции/провайдера.

## Архитектура

**Ключевой принцип:** параллельный `pipeline/settings/integrations/*` рядом с существующим `pipeline/settings/providers/*` ([registry](src/pipeline/settings/providers/registry.py), [store](src/pipeline/settings/providers/store.py), [models](src/pipeline/settings/providers/models.py)). То же: file-backed registry (`~/.ryotenkai/integrations.json`), per-item workspace (`~/.ryotenkai/integrations/<id>/{integration.json, current.yaml, history/, token.enc}`), атомарная запись, snapshot-per-save. Никакой абстракции-надстройки — это premature generalisation.

### Хранение токенов — Grafana-style envelope encryption

Вдохновлено подходом Grafana ([docs](https://grafana.com/docs/grafana/latest/setup-grafana/configure-security/configure-database-encryption/)): токены никогда не лежат в открытом виде и никогда не возвращаются через API.

- **Master key.** `~/.ryotenkai/.secret.key` (32 байта, mode 0600), генерируется при первом старте backend-а. В `README` / настройках API можно переопределить через env `RYOTENKAI_SECRET_KEY` (Base64) — для CI/prod, где ключ должен храниться в внешнем KMS/Vault.
- **Шифрование.** AES-GCM-256 (`cryptography` — уже транзитивная зависимость через huggingface-hub/requests, иначе добавим). Каждый token-файл — это `nonce || ciphertext || tag`.
- **Storage layout.**
  ```
  ~/.ryotenkai/integrations/<id>/
    integration.json
    current.yaml            # нет токенов здесь
    token.enc               # AES-GCM(Master, token_bytes); 0600
    history/…
  ```
  Провайдер — такой же `token.enc` в `~/.ryotenkai/providers/<id>/`.
- **API-контракт.**
  - `PUT  /integrations/{id}/token` — body `{token: str}`, шифруется и пишется, ответ `204`.
  - `DELETE /integrations/{id}/token` — удаляет файл.
  - `GET  /integrations/{id}` — возвращает `has_token: bool`, **никогда** сам токен.
  - Расшифровка происходит только внутри backend в точке потребления (`load_secret_for`, см. §4).
- **Ротация ключа.** CLI-команда `src/api/cli.py rotate-secret-key` (будущая работа, не в этом PR): читает все `token.enc`, расшифровывает старым ключом, перешифровывает новым.
- **Параллель с Grafana.** Envelope-encryption (DEK на каждый токен) в v1 не делаем — overkill для локального dev-инструмента, но layout специально оставляет место для добавления `data_key` slot позже без breaking change.

### 1. Backend — Integrations registry + API

Новые файлы:
- `src/pipeline/settings/integrations/{__init__,models,store,registry}.py` — зеркалят провайдерские.
- `src/config/integrations/registry.py` — `INTEGRATION_TYPES: dict[str, IntegrationTypeDescriptor]` → `mlflow`, `huggingface`. Шаблон уже есть в [src/config/providers/registry.py](src/config/providers/registry.py).
- `src/api/routers/integrations.py`, `src/api/services/integration_service.py`, `src/api/schemas/integration.py` — зеркалят [routers/providers.py](src/api/routers/providers.py), [services/provider_service.py](src/api/services/provider_service.py), [schemas/provider.py](src/api/schemas/provider.py).

Endpoints:
```
GET/POST/DELETE /integrations[/…]        # CRUD + типы + конфиг + versions — как у /providers
PUT/DELETE      /integrations/{id}/token   # write-only; GET никогда не возвращает значение
POST            /integrations/{id}/test-connection
POST            /providers/{id}/test-connection
PUT/DELETE      /providers/{id}/token      # перенос RUNPOD_API_KEY внутрь провайдера
```

`ProviderSummary` ([schemas/provider.py](src/api/schemas/provider.py)) расширяется полем `has_training: bool` (аналог существующего `has_inference`, см. `_provider_has_inference` в [provider_service.py:104](src/api/services/provider_service.py)) + `has_token: bool` — чтобы `TrainingProviderField` больше не делал N+1 `useQueries` за конфигами ([TrainingProviderField.tsx:34-51](web/src/components/ConfigBuilder/TrainingProviderField.tsx)).

### 2. Backend — Test connection

`src/api/services/connection_test.py`, один dispatcher `TEST_HANDLERS: dict[str, Handler]` с timeout 5 s и всегда-200 ответом (`{ok, latency_ms, detail}`):
- `mlflow` → `GET {tracking_uri}/api/2.0/mlflow/experiments/search?max_results=1` + Bearer, honours `ca_bundle_path`.
- `huggingface` → `GET https://huggingface.co/api/whoami-v2` с Bearer.
- `provider:runpod` → `GET https://api.runpod.io/graphql` (или `rest/v1/pods`).
- `provider:single_node` → **SSH-probe** через [src/utils/ssh_client.py](src/utils/ssh_client.py) (переиспользуем существующий клиент): connect + `echo ok` + проверка `ssh_mount.mount_point` readable/writable. Отдельный `connect_timeout=5s` + `exec_timeout=5s`.

### 3. Backend — Рефакторинг схемы

**`src/config/inference/schema.py` — `enabled` остаётся в схеме.** Решение пользователя: backend-модель не трогаем, логика «провайдер ⇒ включено» реализуется **только на фронте** (см. §6). FE при изменении `inference.provider` синхронно пишет в YAML:
- провайдер пустой/null → `inference.enabled = false`;
- провайдер выбран → `inference.enabled = true`.

Model-validator [src/config/inference/schema.py:44](src/config/inference/schema.py) (проверка «enabled=true ⇒ provider required») остаётся как safety-net для редактирования YAML напрямую. Код runtime (`is_active` / `cfg.inference.enabled`) не меняется — все 27 call-sites сохраняются как есть.

**`src/config/integrations/mlflow.py`** — заменить `MLflowConfig` на project-level `MLflowTrackingRef(integration: str | None, experiment_name: str | None, run_description_file: str | None)`. Все legacy-поля (`tracking_uri`, `local_tracking_uri`, `ca_bundle_path`, system-metrics knobs) переезжают в **новый** `src/config/integrations/mlflow_integration.py::MLflowIntegrationConfig` (уровень Settings).

**`src/config/integrations/huggingface.py`** — заменить `HuggingFaceHubConfig` на project-level `HuggingFaceRef(integration: str | None, repo_id: str | None, private: bool = True)`. Поле `enabled` убирается. Integration-level `HuggingFaceIntegrationConfig` (только token-носитель + будущие shared knobs) в новом `src/config/integrations/huggingface_integration.py`.

**`src/config/integrations/experiment_tracking.py`** — `ExperimentTrackingConfig.mlflow: MLflowTrackingRef | None`, `.huggingface: HuggingFaceRef | None`. **Hard-break для старого YAML:** pydantic `StrictBaseModel` с `extra='forbid'` → старые ключи (`tracking_uri`, `enabled`, `repo_id` на неправильном уровне) дают понятную ошибку валидации с подсказкой «это поле переехало в Settings → Integrations». Никакой автомиграции и silent rewrite.

**Новый `src/config/integrations/resolver.py`** — функции `resolve_mlflow(ref)` и `resolve_huggingface(ref)` читают `IntegrationRegistry` + `IntegrationStore`, сливают уровни и возвращают «плоский» объект для runtime. Ошибка типа `IntegrationNotFound("Open Settings → Integrations…")`. Вызывается в [config_service.py](src/api/services/config_service.py) при validate и в [main.py](src/main.py) / executor на старте прогона.

### 4. Backend — Secrets refactor

- `Secrets.HF_TOKEN` → `str | None = None` ([src/config/secrets/model.py](src/config/secrets/model.py)). Контракт «обязательный top-level HF_TOKEN» снимается.
- Новая функция `load_secret_for(kind, id)` в [loader.py](src/config/secrets/loader.py):
  1. читает `tokens.json` соответствующего ресурса;
  2. fallback — env / `secrets.env`;
  3. `MissingSecretError` с ссылкой на Settings, если нужно и нигде нет.
- Call-sites, читающие `secrets.hf_token` / `secrets.runpod_api_key` напрямую (см. [hf_uploader.py:78](src/pipeline/stages/model_retriever/hf_uploader.py), [providers/runpod/training/provider.py:82](src/providers/runpod/training/provider.py), [providers/runpod/inference/pods/provider.py:159](src/providers/runpod/inference/pods/provider.py)) получают ref из конфига и вызывают `load_secret_for`.

### 5. Frontend — Settings / Integrations UI

Параллельные компоненты, шэринг на уровне примитивов (ConfigBuilder, LabelledRow). Не делать generic `Resource` — названия «Provider» / «Integration» живут в URL и UI-копии, общий абстрактный термин читается хуже.

Новое:
- `web/src/pages/IntegrationsIndex.tsx` — список с табами `HuggingFace | MLflow` (стиль как у [ProviderDetail.tsx:84-101](web/src/pages/ProviderDetail.tsx)).
- `web/src/pages/IntegrationDetail.tsx` — табы `Config | Token | Versions`. Test-connection — кнопка в шапке.
- `web/src/components/IntegrationCard.tsx`, `NewIntegrationModal.tsx` — копии [ProviderCard.tsx](web/src/components/ProviderCard.tsx), [NewProviderModal.tsx](web/src/components/NewProviderModal.tsx).
- `web/src/components/IntegrationTabs/{IntegrationConfigTab,TokenTab,VersionsTab}.tsx` — `ConfigTab` практически копия [ProviderConfigTab.tsx](web/src/components/ProviderTabs/ProviderConfigTab.tsx). `TokenTab` — отдельная PUT/DELETE форма с masked-state чтением (`has_token` флаг).
- `web/src/api/hooks/useIntegrations.ts` — зеркало [useProviders.ts](web/src/api/hooks/useProviders.ts) + `useTestConnection`, `useSetToken`.
- Расширить `useProviders.ts` теми же хуками (test-connection, token).

Маршруты в [App.tsx](web/src/App.tsx):
```
/settings/integrations                    → IntegrationsIndex
/settings/integrations/:type/:id/*        → IntegrationDetail
```
В `SettingsPage` ([Settings.tsx:3-7](web/src/pages/Settings.tsx)) пункт **Integrations** добавляется над Datasets.

### 6. Frontend — ConfigBuilder

**[FieldRenderer.tsx:30-37](web/src/components/ConfigBuilder/FieldRenderer.tsx)** — регистрируем новые custom renderers:
- `experiment_tracking.mlflow.integration` — dropdown `useIntegrations('mlflow')` + inline **Test connection** рядом, и `+ add integration in Settings` row в футере селекта (паттерн из [InferenceProviderField.tsx:57-69](web/src/components/ConfigBuilder/InferenceProviderField.tsx)).
- `experiment_tracking.huggingface.integration` — то же самое без кнопки test-connection (или тоже с кнопкой, по симметрии).

**[schemaUtils.ts REQUIRED_OVERRIDES:173](web/src/components/ConfigBuilder/schemaUtils.ts)**:
- `inference` override: gating по truthiness `provider`. Поле `inference.enabled` прячется из формы (`hidden: ['enabled']`); FE пишет его значение автоматически при изменении `provider` (см. ниже), чтобы backend-валидатор `enabled⇒provider` не ругался.
- `experiment_tracking.mlflow` → только `integration` видим; как выбран — открывается `experiment_name` (+ optional `run_description_file`).
- `experiment_tracking.huggingface` → аналогично; при выбранной интеграции видны `repo_id`, `private`.

**Синхронизация `inference.enabled` с `inference.provider`.** В [InferenceProviderField.tsx](web/src/components/ConfigBuilder/InferenceProviderField.tsx) в `onChange` делаем не просто `onChange(provider)`, а обновляем два поля сразу через доступный родительский `rootValue`/`onRootChange` (паттерн уже использован в [ProviderPickerField.tsx:70-78](web/src/components/ConfigBuilder/ProviderPickerField.tsx)): `{...rootValue, inference: {...inf, provider, enabled: Boolean(provider)}}`. Это делает FE единственным источником истины для пары «provider ↔ enabled», YAML остаётся согласованным.

**[TrainingProviderField.tsx:34-51](web/src/components/ConfigBuilder/TrainingProviderField.tsx)** — заменить inline `useQueries` на фильтр по новому `has_training`.

### 7. Frontend — ProjectTabs/SettingsTab

[SettingsTab.tsx CATALOG:24-51](web/src/components/ProjectTabs/SettingsTab.tsx) — удалить строки `HF_TOKEN` и `RUNPOD_API_KEY`. Остаются `MLFLOW_TRACKING_URI` (устаревающий, отдельный TODO) и `LOG_LEVEL` + custom env vars. Маленькая inline-плашка «Tokens now live in Settings → Integrations / Providers» со ссылкой.

### 8. Миграция / back-compat

- **ET YAML — hard break.** `ExperimentTrackingConfig` (StrictBase, `extra='forbid'`) не принимает старые ключи. Пользователи, у которых в project YAML лежит `experiment_tracking.mlflow.tracking_uri` или `experiment_tracking.huggingface.enabled`, получают понятную pydantic-ошибку:
  > Unknown field `tracking_uri`. It has moved to Settings → Integrations → MLflow. See `docs/migration/integrations.md`.
- **Inference.enabled.** Остаётся в backend-схеме; FE автоматически синхронизирует поле с `inference.provider`. Старые YAML с `enabled: true/false` продолжают парситься без изменений.
- **Secrets fallback.** Один релиз: если токен не найден в `token.enc` интеграции/провайдера, `load_secret_for` пытается прочитать legacy-`secrets.env` и пишет warning «Секреты из secrets.env устарели, добавьте их в Settings → Integrations → <id>». В следующем релизе убираем fallback.
- **Документация.** Добавляем `docs/migration/integrations.md` с пошаговыми инструкциями «как перенести tracking_uri + HF_TOKEN» + ссылкой в README и в тексте pydantic-ошибки.

### 9. Тесты

Новые:
- `src/tests/unit/pipeline/settings/integrations/test_{store,registry}.py` — копии провайдерских.
- `src/tests/unit/api/test_integrations_router.py` — CRUD, token PUT не эхо, config validate.
- `src/tests/unit/api/test_connection_test.py` — httpx-stub на каждый handler + error paths.
- `src/tests/unit/config/integrations/test_experiment_tracking_migration.py` — legacy → new + warning.
- `src/tests/unit/config/inference/test_enabled_removal.py` — old YAML со `enabled: true` парсится; `is_active` from `provider`.
- Обновить существующие 27 `inference.enabled`-сайтов в тестах.

## План выкатки — 4 PR, порядок пересмотрен после итерации 3

1. **PR1 — Backend Integrations API + crypto + test-connection.** Новые модули `pipeline/settings/integrations/*`, `api/routers/integrations.py`, `api/services/{integration_service,connection_test,token_crypto}.py`; расширяем ProviderSummary полями `has_training`, `has_token`; добавляем `POST /providers/{id}/test-connection` + `PUT/DELETE /providers/{id}/token`. Неиспользуемые endpoints безвредны. Тесты (включая `test_no_secret_leaks.py`).
2. **PR2 — Frontend Settings/Integrations UI** + Token tab + Test connection кнопки на Providers. Убираем `HF_TOKEN`/`RUNPOD_API_KEY` из `SettingsTab.tsx`. На этом этапе пользователь может **создавать** интеграции, но они ещё никем не используются.
3. **PR3 — Schema refactor + FE overrides** (мердж-трейн, две коммит-ветки, landed вместе):
   - split ET configs (MLflow/HF project-level ref + integration-level), resolver, `model_validator(mode='before')` с user-friendly error.
   - FE: integration dropdowns в ConfigBuilder, `hidden: ['enabled']` для inference, sync `provider↔enabled`, `TrainingProviderField` на `has_training`.
   - Обязательная связка: schema без FE = ValidationBanner-ад; FE без schema = dropdown без цели.
4. **PR4 — Secrets resolver + runtime wiring.** `Secrets.get_provider_token/get_hf_token`, правки в [runpod/training/provider.py:82](src/providers/runpod/training/provider.py), [runpod/inference/pods/provider.py:159](src/providers/runpod/inference/pods/provider.py), [hf_uploader.py:78](src/pipeline/stages/model_retriever/hf_uploader.py). Secrets из `secrets.env` остаются fallback'ом. Тесты на resolver + тесты что provider без токена ≠ pipeline crash.

## 3 итерации анализа плана

### Итерация 1 — Scope / контракт схем

| # | Вопрос / риск | Ответ (дипсинк) |
|---|---|---|
| 1.1 | Нужен ли `get_integration_registry` в dependencies, как у провайдера? | Да. Добавляется рядом с [get_provider_registry](src/api/dependencies.py) (dependencies.py:51-53) — одна строка, тот же `settings.projects_root_resolved` как корень. |
| 1.2 | `StrictBaseModel` уже `extra='forbid'` ([src/config/base.py:18](src/config/base.py)) — зачем явно писать в плане? | Hard-break срабатывает «из коробки». Но стандартное сообщение pydantic (`Extra inputs are not permitted`) недостаточно дружелюбно. **Решение:** на новом `ExperimentTrackingConfig` — `model_validator(mode='before')`, который ловит известные legacy-ключи (`tracking_uri`, `local_tracking_uri`, `ca_bundle_path`, `system_metrics_*`, верхний `enabled/repo_id/private` у huggingface) и кидает `ValueError` с ссылкой на `docs/migration/integrations.md`. |
| 1.3 | Ломает ли hard-break `configs/presets/*.yaml`? | Проверено grep'ом (`configs/presets/`: 01-small.yaml, 02-medium.yaml, 03-large.yaml — **нет** упоминаний mlflow/huggingface/experiment_tracking/HF_TOKEN). Пресеты не содержат ET/секрет-блоков → миграция пресетов не нужна. |
| 1.4 | Как сегодня устроен контракт Secrets → Provider factory ([providers/training/factory.py:103-153](src/providers/training/factory.py))? | Сигнатура `create(name, provider_config, secrets)`; внутри провайдеров читается `secrets.runpod_api_key`. **Решение:** расширяем `Secrets` двумя методами — `get_provider_token(provider_id) -> str \| None` и `get_hf_token(integration_id) -> str \| None`. Резолвер внутри читает `token.enc` и fall-back на env. Минимальные точечные правки в [providers/runpod/training/provider.py:82](src/providers/runpod/training/provider.py), [providers/runpod/inference/pods/provider.py:159](src/providers/runpod/inference/pods/provider.py), [stages/model_retriever/hf_uploader.py:78](src/pipeline/stages/model_retriever/hf_uploader.py). |
| 1.5 | Конфликт `id` между провайдером и интеграцией? | Нет. Разные namespace (`providers/` vs `integrations/` на диске; разные роутеры). |

### Итерация 2 — Runtime / wiring

| # | Вопрос / риск | Ответ (дипсинк) |
|---|---|---|
| 2.1 | Test connection тестирует текущее (unsaved) или сохранённое значение? | **Сохранённое** (Grafana-style). FE блокирует кнопку пока форма dirty — требует Save. Сервер читает YAML с диска + `token.enc`. Проще код, предсказуемое поведение. |
| 2.2 | 5s timeout для runpod достаточен на плохой сети? | Нет, GraphQL из России/VPN бывает медленным. **Решение:** per-type timeout — mlflow/hf 5s, **runpod 10s, ssh 8s**. Настраивается через env `RYOTENKAI_TEST_CONN_TIMEOUT_{mlflow,hf,runpod,ssh}`. |
| 2.3 | SSH-probe single_node: host-key verification? | Переиспользуем существующий [src/utils/ssh_client.py](src/utils/ssh_client.py) — его политика `StrictHostKeyChecking` уже выбрана проектом. Новые endpoints просто дёргают его. Не изобретаем. |
| 2.4 | Как FE синхронизирует `inference.enabled` с `inference.provider`, если `CUSTOM_FIELD_RENDERERS` имеет сигнатуру только `{value, onChange, onFocus, onBlur}` ([FieldRenderer.tsx:21-29](web/src/components/ConfigBuilder/FieldRenderer.tsx))? | **Решение:** расширяем `CustomFieldProps` опциональными `rootValue?: Record<string, unknown>` и `onRootChange?: (next) => void`; `ObjectFields` уже владеет root-значением (его пробрасывает `ProviderPickerField` через `GroupRendererProps` [ProviderPickerField.tsx:12-22](web/src/components/ConfigBuilder/ProviderPickerField.tsx)). Только FieldRenderer.tsx:284 получает новые props. Существующие renderers игнорируют — zero-risk совместимости. |
| 2.5 | Где прячем `inference.enabled` в UI? | Используем уже существующее `REQUIRED_OVERRIDES['inference'].hidden: ['enabled']` ([schemaUtils.ts:146-162](web/src/components/ConfigBuilder/schemaUtils.ts)). Механизм отработан на `evaluators` в `evaluation`. |
| 2.6 | Пользователи, сохранившие `HF_TOKEN`/`RUNPOD_API_KEY` в `env.json`, — что с мёртвыми ключами после удаления из `SettingsTab CATALOG`? | Мёртвые ключи безвредны — env-слияние на запуске игнорирует неизвестные. Inline-плашка «Tokens moved to Settings → Integrations». Cleanup — задача следующего спринта. |
| 2.7 | `get_report_to()` в `ExperimentTrackingConfig` ([integrations/experiment_tracking.py:23-27](src/config/integrations/experiment_tracking.py)): как решать `["mlflow"]` vs `["none"]` после refactor? | `mlflow is not None and mlflow.integration` (truthy). Не резолвим реестр на каждый вызов — это геттер, он синхронный и часто дёргается. |
| 2.8 | `evaluation.enabled=true requires inference.enabled=true` ([validators/cross.py:286-301](src/config/validators/cross.py), [main.py:794](src/main.py)) — как это работает с новой FE-логикой? | Не трогаем. `inference.enabled` в backend остаётся; FE его синхронизирует. Все 27 сайтов `cfg.inference.enabled` работают как сегодня. |

### Итерация 3 — Security, тесты, roll-out

| # | Вопрос / риск | Ответ (дипсинк) |
|---|---|---|
| 3.1 | Потеря master-key (`~/.ryotenkai/.secret.key`) = все токены мёртвые. | Не критично: токены можно ввести заново в UI. Но сюрприз пользователю → warning в API-health endpoint «master key was regenerated, existing encrypted tokens are invalid» когда mtime ключа старше чем самый старый `token.enc`. CLI-ротация — follow-up PR. |
| 3.2 | `cryptography` зависимость | Транзитивно есть через paramiko, но в `pyproject.toml` не указана явно. **Решение:** добавить в pyproject.toml в `[project].dependencies` — явность важнее. |
| 3.3 | Как проверить, что токен не уходит в OpenAPI-ответы? | Новый тест `src/tests/integration/api/test_no_secret_leaks.py`: генерирует OpenAPI, парсит, проверяет, что ни одна response schema не содержит property `token`. Единственное место где `token` разрешён — `PUT /integrations/{id}/token` request body (не response). |
| 3.4 | Может ли токен случайно попасть в `current.yaml`? | Нет: Integration Pydantic schema (`MLflowIntegrationConfig`, `HuggingFaceIntegrationConfig`) **не декларирует** поле `token`. При валидации `extra='forbid'` отклонит YAML с `token:`. Токен живёт только в `token.enc`, доступ только через отдельные endpoints. |
| 3.5 | PR-порядок — может ли PR3 (schema refactor) landиться до FE? | Опасно: пользователь получит ошибку валидации в старом UI без возможности создать интеграцию. **Пересмотрено:** PR1 (backend API) → PR2 (FE Integrations UI — можно создавать) → PR3 (schema refactor) → PR4 (FE ConfigBuilder overrides). PR3+PR4 мерджим одним merge-train (две коммит-ветки, landed вместе). |
| 3.6 | Existing `inference.enabled` tests (27 сайтов) | Остаются без изменений — поле и логика в backend не трогаются. Только добавляются тесты на FE-синхронизацию `provider→enabled`. |
| 3.7 | Как пользователь поймёт, что надо мигрировать YAML? | Pydantic ошибка на валидации при Save/Load (UI уже показывает их в ValidationBanner). Плюс `docs/migration/integrations.md` — ссылка в тексте ошибки и в README. |
| 3.8 | `MLFLOW_TRACKING_URI` env-var как override — нужен ли приоритет? | Оставляем в `SettingsTab CATALOG` как opt-in override (не токен — просто URL). Приоритет: `MLFLOW_TRACKING_URI env` > `integration.tracking_uri`. Это совместимо с облачными сценариями (CI подменяет URL не трогая интеграцию). Deprecation — в отдельном PR. |

## Риски и митигации

| Риск | Митигация |
|---|---|
| Утечка master-key `~/.ryotenkai/.secret.key` | mode 0600; `RYOTENKAI_SECRET_KEY` env-override для prod/CI с KMS/Vault; документация про ротацию. |
| FE рассинхронизирует `inference.provider` ↔ `inference.enabled` (ручное редактирование YAML) | Backend-валидатор `enabled=true ⇒ provider required` остаётся → ошибка валидации поймает расхождение на Save. |
| `has_training` drift от `has_inference` в `provider_service.list_summaries` | Единый helper `_provider_capabilities(store) -> {training, inference}`. |
| `single_node` test-connection по SSH висит на сетевом таймауте | Отдельный wall-clock timeout (5 s), паттерн уже есть в `src/utils/ssh_client.py` — переиспользовать. |
| Hard-break ET YAML ломает существующие проекты | Чёткий error-message с ссылкой на `docs/migration/integrations.md`; скрипт `scripts/migrate_project_integrations.py` (one-shot, не в PR) конвертит YAML → Settings+новый YAML. |
| Ротация master-key непонятна | CLI `rotate-secret-key` как отдельный follow-up PR; в v1 документируем ручной «удалить все token.enc и ввести заново». |
| `GET /integrations/{id}` случайно засветит токен через `current_config_yaml` | Токен **никогда** не кладётся в `current.yaml`. Integration-schema (Pydantic) не имеет поля `token`. Инвариант покрываем тестом (OpenAPI snapshot). |

## Verification

- `pytest src/tests/unit/pipeline/settings/integrations/` — registry/store SRP.
- `pytest src/tests/unit/api/test_integrations_router.py` — CRUD + 2 точки токенов.
- `pytest src/tests/unit/api/test_connection_test.py` — все handlers.
- `pytest src/tests/unit/config/integrations/test_experiment_tracking_migration.py` — legacy YAML поднимается.
- `pytest src/tests/unit/config/inference/test_enabled_removal.py` — `enabled: true` не ломает парсинг; `is_active` выводится из `provider`.
- `ruff check . && mypy .` — чисто.
- `cd web && npm run build` — TS сборка.
- Manual E2E: создать MLflow-интеграцию → вписать tracking_uri + token → Test connection → выбрать её в проекте → запустить pipeline → убедиться, что runs уезжают в MLflow UI.
- Manual E2E: на RunPod-провайдере — добавить токен через UI → Test connection → запустить training без `RUNPOD_API_KEY` в env.
- Manual UI: в проекте снять `inference.provider` → все inference-поля схлопываются; выбрать — появляются.

## Критические файлы для правки

**Backend**
- Новые: `src/pipeline/settings/integrations/*`, `src/api/routers/integrations.py`, `src/api/services/integration_service.py`, `src/api/schemas/integration.py`, `src/config/integrations/{registry,resolver}.py`, `src/api/services/connection_test.py`
- Правка: [src/config/inference/schema.py](src/config/inference/schema.py), [src/config/integrations/{mlflow,huggingface,experiment_tracking}.py](src/config/integrations/), [src/config/validators/{inference,cross,pipeline}.py](src/config/validators/), [src/config/secrets/{model,loader}.py](src/config/secrets/), [src/api/routers/providers.py](src/api/routers/providers.py), [src/api/services/provider_service.py](src/api/services/provider_service.py), [src/api/schemas/provider.py](src/api/schemas/provider.py), [src/main.py](src/main.py), [src/api/services/config_service.py](src/api/services/config_service.py), [src/pipeline/executor/stage_planner.py](src/pipeline/executor/stage_planner.py), [src/pipeline/stages/model_retriever/hf_uploader.py](src/pipeline/stages/model_retriever/hf_uploader.py), [src/providers/runpod/**/provider.py](src/providers/runpod/)

**Frontend**
- Новые: `web/src/pages/{IntegrationsIndex,IntegrationDetail}.tsx`, `web/src/components/{IntegrationCard,NewIntegrationModal}.tsx`, `web/src/components/IntegrationTabs/*`, `web/src/api/hooks/useIntegrations.ts`
- Правка: [web/src/App.tsx](web/src/App.tsx), [web/src/pages/Settings.tsx](web/src/pages/Settings.tsx), [web/src/components/ConfigBuilder/FieldRenderer.tsx](web/src/components/ConfigBuilder/FieldRenderer.tsx), [web/src/components/ConfigBuilder/schemaUtils.ts](web/src/components/ConfigBuilder/schemaUtils.ts), [web/src/components/ConfigBuilder/TrainingProviderField.tsx](web/src/components/ConfigBuilder/TrainingProviderField.tsx), [web/src/components/ProjectTabs/SettingsTab.tsx](web/src/components/ProjectTabs/SettingsTab.tsx), [web/src/api/hooks/useProviders.ts](web/src/api/hooks/useProviders.ts), [web/src/api/types.ts](web/src/api/types.ts)
