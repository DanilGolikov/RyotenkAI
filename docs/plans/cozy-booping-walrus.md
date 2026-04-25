# Plugin system hardening — production-ready platform + better DX

## Context

Plugin-система RyotenkAI прошла несколько итераций. Базовая инфраструктура работает: `BasePlugin` + `CommunityCatalog` загружают TOML-манифесты, 4 registry раздают плагины (validation / evaluation / reward / reports), фронт даёт DnD-палитру + Configure-modal. Недавно добавлен `[[required_env]]` контракт + `PluginEnvSection` UI.

Однако глубокий аудит (3 explore-агента + 1 plan-агент, 2026-04-25) вскрыл **20 конкретных недостатков** на трёх уровнях:

- **Архитектура**: schema-дублирование (`RequiredEnvSpec` в двух местах), четыре registry с разной сигнатурой (`get_plugin / get / create / build_*`), secrets-injection только в validation pipeline, reports — duck-typed Protocol без общего ABC, `BasePlugin.REQUIRED_ENV` существует но не используется и не cross-check'ится с TOML, legacy `[secrets].required` сосуществует с новым `[[required_env]]` без миграции.
- **Developer Experience**: `params/thresholds: dict[str, Any]` без типов, нет helper'а для чтения env/secrets, manifest-defaults load-bearing (плагин крашит на init если автор забыл `default`), нет dev-документации, нет scaffold-CLI, reward-плагины лезут в TRL trainer внутренности через undocumented batch kwargs.
- **Production gaps**: pipeline стартует без preflight-проверки env (падает посередине вместо fail-fast), loader silently skip'ает битые плагины, нет тестов на компоненты фронта, stale plugin references в YAML не чистятся, reward params broadcast'ятся ко всем strategy-фазам без visible-hint в UI.

Цель плана — закрыть всё это в чёткой последовательности, **не ломая** ни один существующий community-плагин и ни один уже-работающий pipeline run. Ниже — recommended approach с 5 фазами и ~17 PR-блоками.

Решения приняты пользователем: **очерёдность — Foundations first** (refactor + tests без поломок поведения), затем production safety, затем DX, затем UX, затем фронт-тесты. **Schema versioning + deprecation policy включаются сразу** в первую фазу.

---

## Важно: backward-compatibility НЕ нужна

Система пока **не в продакшне**. Никаких deprecation-aliases, grace-period'ов и legacy-bridges. Любое изменение делается «в лоб» — старое API вырезается, конфиги/манифесты обновляются разом, downstream-call-sites правятся в том же PR.

Конкретно это значит:
- Старые registry-методы (`get_plugin / get / create / build_report_plugins`) **удаляются** сразу при переходе на унифицированный `PluginRegistry[T]`. Все вызывающие сайты переписываются в одном PR. Никаких deprecated-aliases.
- `[secrets].required` **удаляется** одним PR — все 4 community-плагина (cerebras_judge + 3 helixql) одновременно переводятся на `[[required_env]]`. Никакой `_migrate_legacy_secrets` validator не нужен.
- Reports `ReportPlugin(BasePlugin, ABC)` обязателен сразу для всех community-loaded плагинов. Все 13 report-плагинов мигрируются в том же PR.
- `PluginRequiredEnv` handwritten interface (`web/src/api/types.ts:119-125`) удаляется в первом же PR — типы только из openapi-ts auto-gen.
- Pipeline preflight-gate включается сразу as-default. Не «опционально под флагом».

Что всё-таки нельзя сломать (это не legacy, это runtime invariants):
- Текущие community-плагины должны продолжать работать **после** их миграции в том же PR (т.е. PR с registry-refactor одновременно правит и регистры, и все 4 reward/eval kinds-callsites, и плагины — атомарно).
- API endpoints `GET /plugins/{kind}` / `GET /plugins/reports/defaults` сохраняют контракт (response shape — wire-compatible с OpenAPI snapshot, но НЕ потому что мы боимся сломать клиентов, а потому что фронт ожидает эту shape).
- Pipeline runs с **валидным** конфигом продолжают давать тот же артефакт. Невалидные начинают fail-fast'иться раньше — это улучшение, не регрессия.

---

## Section A — Architecture (Phase 1-2)

### A1. Single source of truth для `RequiredEnvSpec`
**Defect 1.** Дубликат в `src/community/manifest.py:54-86` (canonical) и `src/api/schemas/plugin.py:17-26` (mirror).

**Решение.** Удалить mirror, импортировать canonical в API schema. `ui_manifest()` (`src/community/manifest.py:303-324`) уже сериализует через `model_dump()` — wire-shape не меняется.

**Файлы.**
- `src/api/schemas/plugin.py:1-58` — `from src.community.manifest import RequiredEnvSpec`, удалить локальный класс.
- CI gate: `make check-openapi` — diff `web/src/api/schema.d.ts` против live OpenAPI dump, fail на drift.
- `web/src/api/types.ts:119-125` — `PluginRequiredEnv` regenerate через openapi-ts.

**Acceptance.** OpenAPI snapshot байт-идентичен текущему. Handwritten TS-интерфейс уходит — sourced from `schema.d.ts`.

### A2. Schema-версионирование манифеста (R3.1)
**Решение.** Ввести `schema_version: int = LATEST_SCHEMA_VERSION` поле в `PluginManifest`. `model_validator(mode="after")` отвергает версии выше supported, силент-лифтит ниже. `LATEST_SCHEMA_VERSION = 3` (текущая) объявляется константой в том же модуле.

**Файлы.**
- `src/community/manifest.py:254-301` — добавить `schema_version: int`, валидатор.
- `src/community/loader.py` — при отсутствии `schema_version` в TOML лог info "treating as v3" и assume default.

**Acceptance.** Старый плагин без `schema_version = ...` грузится. Новый с `schema_version = 99` отвергается с понятной ошибкой.

### A3. Унифицированный `PluginRegistry[T]`
**Defects 2, 7.** Каждый registry со своей сигнатурой. Reports — Protocol, не ABC.

**Решение.** Generic `PluginRegistry[T]` в `src/community/registry_base.py` с **единым публичным интерфейсом**:

```python
class PluginRegistry(Generic[T]):
    def register_from_community(self, loaded: LoadedPlugin) -> None: ...
    def get_class(self, plugin_id: str) -> type[T]: ...
    def instantiate(
        self,
        plugin_id: str,
        *,
        resolver: PluginSecretsResolver | None = None,
        env: Mapping[str, str] | None = None,
        **init_kwargs: Any,
    ) -> T: ...
    def manifest(self, plugin_id: str) -> dict[str, Any]: ...
    def list_ids(self) -> list[str]: ...
    def is_registered(self, plugin_id: str) -> bool: ...
    def clear(self) -> None: ...
```

Каждый из четырёх registry становится thin-subclass'ом с `_make_init_kwargs()` адаптером (validation: `params, thresholds`; reward: `params`; reports: `()`). Старые методы остаются как `@deprecated` aliases на один минорный релиз.

**Reports → ABC.** Вводится `ReportPlugin(BasePlugin, ABC)` в `src/reports/plugins/interfaces.py` рядом с существующим `IReportBlockPlugin` Protocol. ABC обязателен только для community-loaded плагинов; внешние/тестовые моки могут продолжать использовать Protocol.

**Файлы.**
- New: `src/community/registry_base.py`.
- Modify: `src/data/validation/registry.py:14-58`, `src/evaluation/plugins/registry.py:15-59`, `src/training/reward_plugins/registry.py:14-48`, `src/reports/plugins/registry.py:28-57` — каждый ~20 LOC subclass.
- New: `src/reports/plugins/interfaces.py` — `ReportPlugin` ABC.
- Migrate: 13 community/reports/*/plugin.py — наследование от ReportPlugin.

**Reuse.** `LoadedPlugin` dataclass (`src/community/loader.py`), `_attach_community_metadata` (`loader.py:153-161`), `CommunityCatalog._populate_registries` (`catalog.py:138-161`) — ничего не переписываем.

**Acceptance.** Парметризованный тест `tests/community/test_registry_contract.py` проходит для всех 4 kinds; старые методы (`get_plugin`, `get`, `create`, `build_report_plugins`) ещё резолвятся.

### A4. Loader: structured load failures (defect 20)
**Решение.** Двумодовый loader:
- production (default): silent-skip битых плагинов, но фейлы складываются в `CommunityCatalog._failures: dict[str, list[LoadFailure]]`.
- developer (`COMMUNITY_STRICT=1` env / `--strict` CLI flag / `pytest.ini`): re-raise.

API endpoint `GET /plugins/{kind}` начинает возвращать `errors: list[PluginLoadError]` рядом с `plugins: [...]`. UI рендерит amber-banner с trace.

**Файлы.**
- `src/community/loader.py:177-184, 206-213` — capture exceptions в `LoadFailure` dataclass.
- `src/community/catalog.py:51-55` — `_failures` storage.
- `src/api/schemas/plugin.py` — `errors` field в `PluginListResponse`.
- New: `web/src/components/PluginCatalogCard.tsx` — error-banner.

**Acceptance.** Сломанный import показывается в UI вместе с traceback; `pytest.ini` ставит COMMUNITY_STRICT=1 чтобы test-ран ловил это локально.

### A5. Унифицированная secrets injection (defect 3)
**Решение.** Логика инъекции `_secrets` (сейчас живёт в `dataset_validator.py:175-184`) переезжает в `PluginRegistry.instantiate(...)`. Любой kind, объявивший `_required_secrets`, получает их одним и тем же путём.

```python
def _inject_secrets(instance, plugin_cls, resolver):
    keys = getattr(plugin_cls, "_required_secrets", ())
    if keys and resolver is None:
        raise RuntimeError(f"plugin '{plugin_cls.name}' requires {keys} but no resolver was passed")
    if keys:
        object.__setattr__(instance, "_secrets", resolver.resolve(keys))
```

**Resolver factories.** Уже есть для validation (`src/data/validation/secrets.py`) и evaluation (`src/evaluation/plugins/secrets.py`). Добавить `RewardSecretsResolver` (`prefix="RWRD_"`) и `ReportSecretsResolver` (`prefix="RPRT_"`).

**Файлы.**
- New: `src/training/reward_plugins/secrets.py`, `src/reports/plugins/secrets.py`.
- Modify: `src/training/run_training.py` (call site of `RewardPluginRegistry.create`) — pass resolver.
- Modify: `src/reports/plugins/registry.py:99-104` — `build_report_plugins(sections, *, resolver=None)`, проброс из Composer.
- Modify: `src/data/validation/standalone.py` — переключить на новый unified path.

**Acceptance.** Удаляем `os.environ.get(...)` из `helixql_compiler_semantic` и заменяем на `self._secrets["RWRD_..."]` без правок в фреймворке. Тест: `tests/community/test_secret_injection_per_kind.py`.

### A6. Preflight env gate (defect 15)
**Решение.** `src/community/preflight.py: validate_required_env(plugin_ids, project_env) -> list[MissingEnv]` — проверяет non-optional `[[required_env]]` для всех **enabled** плагинов из resolved config.

Вызывается из:
- API: `POST /plugins/preflight` — UI показывает в Launch-modal.
- Pipeline: `src/pipeline/orchestrator.py` перед subprocess fork. Aborts run с `LaunchAbortedError`, никаких 4-минутных стартов на пустых ключах.

**Файлы.**
- New: `src/community/preflight.py`.
- Modify: `src/api/routers/plugins.py`, `src/api/services/plugin_service.py:13-21`.
- Modify: `src/pipeline/orchestrator.py` — ранний gate.

**Acceptance.** Запуск без `EVAL_CEREBRAS_API_KEY` → 422 с listом отсутствующих env; пайплайн не стартует; UI рендерит чип "set up before launch".

### A7. Cross-check `BasePlugin.REQUIRED_ENV` ↔ TOML (defects 4, 5)
**Решение.** Перетипизировать `REQUIRED_ENV: ClassVar[tuple[RequiredEnvSpec, ...]]` (импорт из `src.community.manifest`). В `_attach_community_metadata` (`src/community/loader.py:153-161`): если class объявил non-empty `REQUIRED_ENV`, **assert match** против `manifest.required_env` по `(name, optional, secret, managed_by)` — descriptions могут расходиться. Mismatch → `ValueError` at load.

**Sync CLI.** `ryotenkai community sync-envs <plugin_id>` (новая команда в `src/cli/community.py`) — пишет/обновляет TOML-блок из `REQUIRED_ENV` ClassVar. Используется автором плагина после изменения Python-стороны.

**Файлы.**
- `src/utils/plugin_base.py:47-57` — re-type, import.
- `src/community/loader.py:153-161` — cross-check.
- New: `src/cli/community.py` (sync-envs subcommand).

**Acceptance.** Автор расходится между Python и TOML → loader падает с diff'ом. CLI приводит TOML в соответствие.

---

## Section B — Developer ergonomics (Phase 3)

### B1. Typed `params` / `thresholds` (defect 8)
**Решение.** `_attach_community_metadata` генерирует `TypedDict`-ы из `manifest.params_schema` и кладёт на класс как `Params: ClassVar[type[TypedDict]]` / `Thresholds: ClassVar[type[TypedDict]]`. `BasePlugin._typed_params() -> Params` cast'ит pass-through. Pyright + IDE-autocomplete работают.

**Constraint.** `ParamFieldSchema._check_constraints` ассертит, что field-имена матчат `[a-z_][a-z0-9_]*` (TypedDict требует валидных Python identifier'ов).

**Файлы.**
- `src/community/loader.py:153-161` — `_build_typed_dicts(manifest)`.
- `src/utils/plugin_base.py:24-74` — `_typed_params()` / `_typed_thresholds()` helpers.
- `src/community/manifest.py:127-170` — assert на field names.

### B2. Runtime helpers `_env()` / `_secret()` (defect 9)
**Решение.** Добавить в `BasePlugin`:
- `_env(name, default=None) -> str | None` — читает `self._injected_env` (loader populate'ит).
- `_secret(name) -> str` — читает `self._secrets`, raises clear error если не инжектировано.

Loader thread'ит env-dict через `instantiate(...)` (см. A3).

**Acceptance.** `grep -rn 'os.environ' community/` → 0 hits.

### B3. Instance-level params validation (defect 10)
**Решение.** `src/community/instance_validator.py: validate_instance(manifest, params, thresholds) -> list[str]` использует `params_to_json_schema()` (`src/community/manifest.py:200-209`) + `jsonschema` (уже dep). Вызывается из A6 preflight gate **alongside env preflight**.

UI Configure modal уже требует required-fields; YAML-driven launches теперь не упадут на init.

**Файлы.**
- New: `src/community/instance_validator.py`.
- Modify: orchestrator preflight (см. A6) — добавить вызов.

### B4. Scaffold CLI (defect 13)
**Решение.** `ryotenkai plugin scaffold <kind> <id>` создаёт:
```
community/<kind>/<id>/
  manifest.toml          # rendered from template, schema_version=3
  plugin.py              # inherits the right ABC, REQUIRED_ENV=()
  README.md
  tests/test_plugin.py   # smoke test
```

**Файлы.**
- New: `src/cli/plugin_scaffold.py`, `src/cli/templates/{validation,evaluation,reward,reports}/*`.
- Modify: `src/cli/community.py` — register subcommand.

**Acceptance.** `ryotenkai plugin scaffold validation hello_world` → плагин грузится, регистрируется, виден в `GET /plugins/validation`, проходит `pytest community/validation/hello_world/tests`.

### B5. Plugin authoring guide + deprecation policy (defects 11, 14)
**File.** New: `community/README.md` с разделами:
- **Lifecycle per kind** — call sequence, что инжектируется, что возвращается.
- **`params_schema` → UI form** — как `FieldRenderer` рендерит JSON Schema.
- **Secrets vs envs** — когда какой prefix, кто кого включает.
- **`REQUIRED_ENV` contract** — ссылка на A7.
- **Reward kwargs contract** — задокументировать **undocumented** batch-kwargs (`prompts`, `completions`, `schema_context`, `reference_answer`) которые TRL передаёт в reward callbacks. Типы, источники, lifetime.
- **Testing recipe** — пример с фикстурами из D2.
- **Deprecation policy** — формальное правило: legacy fields живут один минорный релиз после deprecation announcement, удаляются в следующем. Первый кейс — `[secrets].required` (см. C1).

### B6. Cleanup устаревшего `priority` упоминания
**Defect.** Docstrings в `src/data/validation/base.py:75-105`, `src/evaluation/plugins/base.py:74-83` упоминают `priority` (давно мёртвое — порядок задаётся YAML).

**Решение.** Удалить из docstrings + из тестов если есть assertions на priority.

---

## Section C — Migration (Phase 2-4)

### C1. Удалить `[secrets].required`, перевести на `[[required_env]]` (defect 6)
**Решение (no backward compat).** Одним PR:
1. `SecretsSpec` (`src/community/manifest.py`) удаляется из `PluginManifest`. `extra="forbid"` поймает любую остатачную TOML-запись.
2. Все 4 community-плагина (`cerebras_judge` + 3 helixql) обновляют свои `manifest.toml`: `[secrets]` блок удалён, `[[required_env]]` декларирует тот же набор переменных с `secret=true, optional=false, managed_by=""`.
3. Runtime `_required_secrets` (`BasePlugin._required_secrets: ClassVar[tuple[str,...]]`) переименовывается / переезжает на `REQUIRED_ENV` (см. A7) — единая декларация, читаемая из манифеста.
4. Все вызовы `PluginSecretsResolver.resolve(self._required_secrets)` правятся: новый помощник `BasePlugin._secret(name)` (см. B2) знает что брать из `_secrets` инжектированного registry'ем (см. A5).

**Файлы.**
- `src/community/manifest.py` — удалить `SecretsSpec`, поле `secrets`.
- `src/utils/plugin_base.py` — удалить `_required_secrets` ClassVar.
- `src/community/loader.py:153-161` — удалить инжекцию `_required_secrets`, вместо неё писать `REQUIRED_ENV` (см. A7).
- Все `community/*/*/manifest.toml` — конвертировать одним коммитом.
- Все плагины которые делают `self._secrets[...]` — убедиться что `_secret(name)` helper подхватывает значения из той же resolver-цепочки.

### C2. Reward broadcast visibility (defect 19)
**Решение.**
- Backend: `src/training/run_training.py` логирует `reward_param_mirror_event` per strategy получившая params (один раз per instance).
- Frontend: `pluginInstances.writeInstanceDetails` возвращает broadcast-targets list. `PluginConfigModal` рендерит хинт "These params apply to N strategies: grpo, sapo" под Save-кнопкой. Strategy-имена берутся из `manifest.supported_strategies`.

### C3. Унифицированный registry rollout (atomic, no aliases)
**Sequencing.** Один PR делает всё разом, без deprecation-aliases:
- Создаётся `PluginRegistry[T]` (`src/community/registry_base.py`).
- Все 4 registry становятся thin-subclass'ами с `_make_init_kwargs`.
- Все call sites переходят на `instantiate(...)`:
  - `src/data/validation/standalone.py` (was `get_plugin`)
  - `src/evaluation/runner.py:415` (was `get`)
  - `src/training/reward_plugins/factory.py` (was `create`)
  - `src/reports/plugins/registry.py:99-104` (was `build_report_plugins`)
- Старые методы удаляются. Тесты которые pin'ят имена — переписываются в том же PR.

---

## Section D — Testing (Phase 1, 5)

### D1. Frontend component tests (defect 16)
**Stack.** Vitest уже используется (`web/src/components/ProjectTabs/pluginInstances.test.ts`). Добавить `@testing-library/react` + `@testing-library/user-event` + jsdom в `web/vitest.config.ts`.

**Новые тесты** (`web/src/components/__tests__/`):
- `PluginConfigModal.test.tsx` — стаб манифеста, fill required → onSave с правильным payload; missing required → save disabled.
- `PluginEnvSection.test.tsx` — `secret=true` masking, password reveal toggle, `managed_by=integrations` → ссылка present.
- `KindSection.test.tsx` (extracted) — DnD reorder fires callback; remove-кнопка вызывает remove.
- `PluginInfoModal.test.tsx` — описание, params/thresholds, supported_strategies chips.
- `PluginPaletteDrawer.test.tsx` — поиск фильтрует по name + category.

### D2. Backend community-plugin pytest fixtures (defect 12)
**File.** New: `src/tests/community/conftest.py`:

```python
@pytest.fixture
def tmp_community_root(tmp_path) -> Path: ...

@pytest.fixture
def make_plugin_dir(tmp_community_root):
    """Yields a function (kind, id, manifest_dict, plugin_source) → Path"""

@pytest.fixture
def mock_catalog(tmp_community_root) -> CommunityCatalog: ...

@pytest.fixture
def fake_secrets(extras: dict[str, str]) -> Secrets: ...
```

### D3. Список новых тестов (порядок написания)
1. `tests/community/test_registry_contract.py` — A3, парметризован по 4 kinds, использует D2.
2. `tests/community/test_secret_injection_per_kind.py` — A5.
3. `tests/community/test_required_env_crosscheck.py` — A7.
4. `tests/community/test_legacy_secrets_migration.py` — C1.
5. `tests/community/test_preflight_gate.py` — A6.
6. `tests/community/test_strict_mode.py` — A4.
7. `tests/community/test_instance_validation.py` — B3.
8. `tests/community/test_schema_versioning.py` — A2.
9. `tests/cli/test_plugin_scaffold.py` — B4.
10. `web/src/components/__tests__/*` — D1.

---

## Section E — UX hardening (Phase 4)

### E1. Stale plugin reference cleanup (defect 18)
**Решение.** Backend `GET /api/v1/projects/{id}/config` возвращает `stale_plugins: [{kind, id, instance_id}]` (плагин в YAML, но не в catalog). UI добавляет per-row кнопку "Remove from config" → `pluginInstances.remove(...)`.

### E2. Plugin manifest mismatch surfacing (defect 17)
**Решение.** После A1 schema автогенерируется. Добавить CI step `make check-openapi` диффящий `web/src/api/schema.d.ts` против live OpenAPI dump, fail на drift.

### E3. Reward broadcast hint — реализовано в C2.

---

## Phasing & PR-blocks

### Phase 1 — Foundations (atomic refactors, no behaviour change для valid configs)
- **PR1**: A1 (collapse `RequiredEnvSpec` — drop mirror, regen openapi-ts) + E2 (CI gate). ~150 LOC.
- **PR2**: A2 (schema versioning, `LATEST_SCHEMA_VERSION = 3`). ~80 LOC.
- **PR3**: A3 + C3 атомарно — `PluginRegistry[T]` + Reports ABC + миграция всех 4 registry + все call sites + community/reports/* плагины наследуются от ReportPlugin. ~600 LOC. **Это «один-большой-PR»** — backward-compat не нужна, alias'ы не делаем.
- **PR4**: D2 (community pytest fixtures) + `test_registry_contract.py` + `test_schema_versioning.py`. ~250 LOC.
- **PR5**: A4 (strict-mode loader + structured failures + UI errors banner). ~200 LOC.

### Phase 2 — Production safety
- **PR6**: A5 (unified secret injection всех kinds) + C1 atomic — удалить `[secrets]` block из манифестов, перевести 4 плагина на `[[required_env]]`. ~350 LOC.
- **PR7**: A6 (preflight env gate) + UI banner в Launch-modal.
- **PR8**: A7 (cross-check `BasePlugin.REQUIRED_ENV` ↔ TOML + sync-envs CLI).

### Phase 3 — Developer DX
- **PR9**: B2 (`_env` / `_secret` helpers) + B6 (docstring cleanup priority).
- **PR10**: B1 (typed params/thresholds via TypedDict codegen + identifier check в `ParamFieldSchema`).
- **PR11**: B3 (instance-validation в preflight).
- **PR12**: B4 (scaffold CLI) + B5 (community/README.md authoring guide).

### Phase 4 — UX hardening
- **PR13**: C2 (reward broadcast visibility — backend log + UI hint).
- **PR14**: E1 (stale plugin reference cleanup — backend `stale_plugins` + UI button).

### Phase 5 — Frontend test coverage
- **PR15-17**: D1 batches по component-family (Configure / KindSection / Palette+Info).

---

## Risks & open questions (3 итерации)

### Iteration 1 — design risks

| # | Risk | Mitigation |
|---|---|---|
| R1.1 | Reports ABC ломает внешние duck-typed плагины | Сохраняем `IReportBlockPlugin` Protocol; ABC требуется только для community-loaded |
| R1.2 | `PluginRegistry[T].instantiate` over-fits на lookup-by-id; ordering kind-specific | Ordering OUT of registry; каждый call site сохраняет свой ordering pass (validation/evaluation — YAML order; reward — strategy-keyed; reports — sections list) |
| R1.3 | Auto-promote `[secrets]` ставит `secret=True` для public URL по ошибке | Лог-нудж: "auto-promotion assumed secret=true; declare explicit `[[required_env]]` to override" |
| R1.4 | TypedDict codegen ломается на field-имени с дефисами | `ParamFieldSchema._check_constraints` ассертит `[a-z_][a-z0-9_]*` |
| R1.5 | `COMMUNITY_STRICT` дефолт. Strict в CI, loose в проде — диссонанс | `pytest.ini` ставит strict; production остаётся loose |

### Iteration 2 — rollout risks

| # | Risk | Mitigation |
|---|---|---|
| R2.1 | Тесты pin'ят старые имена registry-методов | Grep audit перед PR3, decide per-case alias-vs-update |
| R2.2 | Preflight gate (A6) внезапно ломает projects где env unset | Проверяем только `enabled=true` плагины и только non-optional envs |
| R2.3 | `@testing-library/react` + jsdom инфлейтит install (~6s cold) | Acceptable |
| R2.4 | Removing TOML entry surprising после opt-in REQUIRED_ENV | Документировать в B5 contract что `REQUIRED_ENV` non-empty заставляет TOML match |
| R2.5 | Reward broadcast лог шумный в долгих training runs | Один раз per instance, не per call |
| R2.6 | OpenAPI codegen produces shape отличный от handwritten `PluginRequiredEnv` | Codegen + consumer fix в одном PR (PR1) |

### Iteration 3 — long-term governance

| # | Q/Risk | Decision |
|---|---|---|
| R3.1 | Manifest schema versioning | **Включаем сразу** (A2), constant `LATEST_SCHEMA_VERSION = 3` |
| R3.2 | Deprecation policy | Документируется в B5: legacy один минорный релиз с warning, удаляется в следующем |
| R3.3 | Plugin marketplace / external authors / publish `BasePlugin` на PyPI | Out of scope для этого плана. Surface area стабилизируется через A3+B5 — ABI подготовлен к публикации, но не публикуется |
| R3.4 | Plugin signing / supply-chain | Out of scope, но `LoadFailure` (A4) — правильное место для расширения |
| R3.5 | Schema roundtrip test | Включён в PR1 как `tests/api/test_plugin_manifest_roundtrip.py` |
| R3.6 | Reward batch-kwargs typed contract | Документируется в B5 как stable contract; в follow-up PR — `RewardCallContext` dataclass рядом с `ReportPluginContext` |

---

## Critical files

### Создаются
- `src/community/registry_base.py` — generic `PluginRegistry[T]`
- `src/community/preflight.py` — `validate_required_env`
- `src/community/instance_validator.py` — params/thresholds validation
- `src/training/reward_plugins/secrets.py`, `src/reports/plugins/secrets.py` — kind-specific resolvers
- `src/cli/plugin_scaffold.py` + `src/cli/templates/{validation,evaluation,reward,reports}/`
- `src/tests/community/conftest.py` + 9 новых test-файлов
- `community/README.md` — authoring guide + deprecation policy
- `web/src/components/__tests__/*.tsx` — 5 component test files

### Модифицируются
- `src/community/manifest.py:54-86, 200-209, 254-301` — single source `RequiredEnvSpec`, schema_version, `_migrate_legacy_secrets`
- `src/community/loader.py:153-161, 177-184, 206-213` — typed-dict gen, structured failures, REQUIRED_ENV cross-check
- `src/community/catalog.py:51-78, 138-161` — `_failures` storage, registry-uniform population
- `src/utils/plugin_base.py:24-74` — re-type `REQUIRED_ENV`, `_env/_secret/_typed_params` helpers
- `src/api/schemas/plugin.py:1-58` — drop mirror, re-export from manifest
- `src/data/validation/registry.py`, `src/evaluation/plugins/registry.py`, `src/training/reward_plugins/registry.py`, `src/reports/plugins/registry.py` — thin subclasses
- `src/api/routers/plugins.py`, `src/api/services/plugin_service.py:13-21` — preflight endpoint, errors field
- `src/pipeline/orchestrator.py` — preflight gate перед fork
- `web/src/api/types.ts:119-125` — sourced from openapi-ts (handwriting удаляется)

### Reuse without rewrite
- `LoadedPlugin` dataclass и `_attach_community_metadata` (`src/community/loader.py:153-161`)
- `PluginSecretsResolver` (`src/utils/plugin_secrets.py:17-78`)
- `params_to_json_schema()` (`src/community/manifest.py:200-209`) + `jsonschema` lib (уже dep)
- `field_to_json_schema()` (`src/community/manifest.py:173-209`)
- `IReportBlockPlugin` Protocol (`src/reports/plugins/interfaces.py:69-86`) — рядом с новой ABC
- `KindSection` (`web/src/components/ProjectTabs/PluginsTab.tsx:439-556`) — переиспользуется в Datasets tab
- `useAllPlugins` hook (`web/src/api/hooks/usePlugins.ts:30-53`)

---

## Verification

### Per-PR
1. `tsc --noEmit` (web) — 0 errors.
2. `pytest src/tests/unit/community/ src/tests/unit/api/ src/tests/unit/data/ -q` — green.
3. `make check-openapi` (после PR1) — diff'ит OpenAPI snapshot, fail на drift.

### End-to-end
1. **Schema roundtrip**: `tests/api/test_plugin_manifest_roundtrip.py` — `manifest → ui_manifest() → API → schema.d.ts → web типы` без потерь.
2. **Loader strict mode**: `COMMUNITY_STRICT=1 pytest src/tests/unit/community/test_strict_mode.py` — broken plugin ломает test (не silent skip).
3. **Preflight gate** (PR7+): локально через preview MCP — создать плагин с `EVAL_FOO_KEY` required, не ставить env, нажать Launch в UI → 422 с readable message; pipeline не стартует.
4. **Cross-check** (PR8): расхождение `REQUIRED_ENV` в Python и TOML → `ValueError` at catalog load с понятным diff'ом.
5. **Legacy migration** (PR9): `pytest community/` с одним плагином имеющим только `[secrets].required` → DeprecationWarning, плагин работает, `required_env` populated automatically.
6. **Scaffold CLI** (PR13): `ryotenkai plugin scaffold validation hello_world && pytest community/validation/hello_world/tests` — green.
7. **Visual smoke**: после PR15 (UX) — открыть проект с stale plugin reference → "Remove from config" кнопка работает; reward Configure → broadcast-hint видим.

### Acceptance per phase
- **Phase 1 done**: existing community plugins грузятся без правок; CI gate ловит OpenAPI drift; registry-contract тесты зелёные для всех 4 kinds.
- **Phase 2 done**: secret injection работает в evaluation/reward/reports как в validation; preflight gate отдаёт 422 на missing env; legacy `[secrets]` deprecated с warning.
- **Phase 3 done**: новый автор может писать плагин с typed params, `_env()` / `_secret()` хелперами, `ryotenkai plugin scaffold` создаёт работающую болванку, есть `community/README.md` гайд.
- **Phase 4 done**: Reports — ABC; reward broadcast visible; deprecated registry-методы удалены; stale plugin refs cleanable.
- **Phase 5 done**: 5+ frontend component тестов поверх плагин-UI зелёные.
