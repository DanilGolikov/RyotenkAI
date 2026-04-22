# План: автогенерация и синхронизация manifest.toml

## Context

После введения community-контракта авторы плагинов/пресетов пишут `manifest.toml` вручную. Это трение и источник ошибок:

- Entry-point легко промахнуться (`class = "MyValidator"` vs реальный `MyValidatorPlugin`).
- `params_schema`/`thresholds_schema` дублируют то, что уже есть в коде через `self._param("key", default)` — дубль, который расходится при первом же рефакторинге.
- `version` никто не инкрементирует при правках — поле живёт `1.0.0` годами.
- Поле `kind` легко указать неправильно и получить `kind mismatch: skipped` в логе.
- Для многофайлового плагина — нужно знать, что `entry_point.module` может быть папкой с `__init__.py`.

**Что хотим:** две команды CLI — `community scaffold` (создаёт манифест с нуля) и `community sync` (обновляет существующий, сохраняя ручные правки, авто-бампит версию). Обе дёргают одно ядро инференции, которое знает AST плагина.

Ответы пользователя по развилкам:
1. **Full inference** — сканировать AST на `self._param`/`self._threshold`/`self._secrets["PREFIX_..."]`, 3-way merge с ручными overrides.
2. **Ручной --bump patch|minor|major**, default = `patch`.
3. **Subcommands** `python -m src.main community {scaffold,sync}` (не топ-левел).

---

## Контракт полей (крайне важный раздел)

Трёхуровневая классификация для каждого поля манифеста. Это фиксирует, что делает `sync`, а что нет.

### Plugin manifest

| Поле | Категория | Поведение `scaffold` | Поведение `sync` |
|---|---|---|---|
| `plugin.id` | USER-OWNED | имя папки | не трогает |
| `plugin.name` | USER-OWNED | = `id` | не трогает |
| `plugin.description` | USER-OWNED | первая строка docstring entry-класса | не трогает |
| `plugin.category` | USER-OWNED | `""` с `# TODO` комментарием | не трогает |
| `plugin.stability` | USER-OWNED | `"experimental"` | не трогает |
| `plugin.priority` | USER-OWNED | `50` | не трогает |
| `plugin.kind` | MANAGED | из базового класса | переписывает |
| `plugin.entry_point.module` | MANAGED | `"plugin"` | переписывает |
| `plugin.entry_point.class` | MANAGED | из AST | переписывает |
| `plugin.version` | AUTO-BUMPED | `"0.1.0"` | `+1` к сегменту, заданному `--bump` |
| `params_schema[k].type` / `.default` | MANAGED | инферим | переписывает, **если user не override** |
| `params_schema[k].min` / `.max` / `.options` / `.description` | USER-OWNED | пусто | не трогает |
| `params_schema[k]` (новый ключ) | NEW-DISCOVERED | — | добавляет из inferred |
| `params_schema[k]` (отсутствует в коде) | — | — | удаляет |
| `thresholds_schema[k]` | те же правила, что у `params_schema` | | |
| `suggested_params` / `suggested_thresholds` | USER-OWNED | подставляем inferred defaults | не трогает (кроме удаления ключей, которых нет в schema) |
| `secrets.required` | SUPPLEMENT | инферим из `self._secrets["..."]` | добавляет новые inferred ключи; ничего не удаляет (пользователь мог явно объявить опциональные ключи) |
| `compat.min_core_version` | USER-OWNED | `""` | не трогает |

### Preset manifest

| Поле | Категория | Поведение `scaffold` | Поведение `sync` |
|---|---|---|---|
| `preset.id` | USER-OWNED | имя папки | не трогает |
| `preset.name` | USER-OWNED | = `id` | не трогает |
| `preset.description` | USER-OWNED | `""` с `# TODO` | не трогает |
| `preset.size_tier` | USER-OWNED | `""` с `# TODO` (эвристика по model name на will-not-do) | не трогает |
| `preset.version` | AUTO-BUMPED | `"0.1.0"` | `+1` к `--bump` сегменту |
| `preset.entry_point.file` | MANAGED | ищет `*.yaml` (предпочитает `preset.yaml`) | переписывает |

**Пресеты слабее инферируются по природе** — это чистый YAML, без класса с методами. Почти все поля USER-OWNED.

---

## Целевая структура (новые модули)

```
src/community/
├── inference.py         # AST-анализ плагина: entry class, params, thresholds, secrets
├── toml_writer.py       # минимальный TOML-сериализатор (комментарии, стабильный порядок)
├── scaffold.py          # новый манифест с нуля
└── sync.py              # 3-way merge + version bump

src/cli/
└── community.py         # typer SubTyper `community` с `scaffold` и `sync`
```

Регистрация SubTyper в существующем `src/main.py`:
```python
from src.cli.community import community_app
app.add_typer(community_app, name="community",
              help="Scaffold/sync community plugin and preset manifests")
```

### Почему свой TOML-writer, а не `tomli_w`?

В deps `tomli_w` нет. Наш формат — плоские таблицы и inline-таблицы для schema-полей. Свой сериализатор ~80 строк, контролирует стабильный порядок ключей (иначе `sync` будет генерить шум в диффе) и умеет вставлять `# TODO`-комментарии к пустым полям. Зависимости не тянем.

---

## Детали реализации

### `inference.py`

Чистые функции, принимают `Path` плагинной папки.

- `find_entry_module(plugin_dir: Path) -> Path` — возвращает `plugin.py` или `plugin/__init__.py`. Поднимает `FileNotFoundError` иначе.
- `parse_module(path: Path) -> ast.Module` — через `ast.parse`.
- `find_entry_class(module_ast, base_names) -> ast.ClassDef` — ищет **один** `ClassDef`, чьи `bases` ссылаются на элементы из `base_names` (строковое сравнение имён, не импорт-резолв). Если многофайловый пакет — рекурсивно проходит по импортам внутри `__init__.py` через `importlib.util.spec_from_file_location` только для получения списка классов (но инференцию делаем на AST, чтобы не выполнять код плагина).
- `infer_kind(base_class_name) -> PluginKind` — mapping `ValidationPlugin→validation`, `EvaluatorPlugin→evaluation`, `RewardPlugin→reward`; для reports — специально: класс с атрибутами `plugin_id` + `order` и методом `render` → `reports`.
- `infer_docstring_summary(class_node) -> str` — первая строка docstring'а (`ast.get_docstring(class_node).split("\n")[0].strip()`).
- `infer_config_calls(class_node, attr_name: str) -> dict[str, InferredField]` — обходит тело класса, находит:
  - `self._param("key", default)` / `self._threshold("key", default)` → key + default
  - `self.params.get("key", default)` / `self.thresholds.get("key", default)` → то же
  - Если key — не literal string → warning, пропуск.
  - Если default отсутствует → value = `None`, type = `"string"`.
  - Тип из default: `int→"integer"`, `float→"float"`, `bool→"boolean"`, `str→"string"`, `list→"array"`, `dict→"object"`, `None→"string"`.
- `infer_required_secrets(class_node) -> tuple[str, ...]` — обходит все `ast.Subscript` формы `self._secrets["KEY"]` → собирает literal keys. Если subscript через переменную → warning.

Все инференсы **идемпотентны** и не выполняют пользовательский код.

### `toml_writer.py`

Один публичный helper:
```python
def dump_manifest_toml(
    manifest: dict[str, Any],
    *,
    todo_fields: set[str] = frozenset(),   # пути типа "plugin.category" → получат # TODO комментарий
) -> str: ...
```

Формат:
- Секции в фиксированном порядке: `[plugin]` / `[preset]` → `[plugin.entry_point]` → `[params_schema.*]` → `[thresholds_schema.*]` → `[suggested_params]` → `[suggested_thresholds]` → `[secrets]` → `[compat]`.
- Внутри секции ключи отсортированы по алфавиту (кроме `id`, `kind`, `name`, `version` — эти всегда первыми в `[plugin]`/`[preset]`).
- `todo_fields` → строка `field = ""  # TODO: fill in` вместо обычной пары.

### `scaffold.py`

```python
def scaffold_plugin_manifest(plugin_dir: Path) -> str:
    entry_path = find_entry_module(plugin_dir)
    module_ast = parse_module(entry_path)
    cls = find_entry_class(module_ast, PLUGIN_BASE_NAMES)
    kind = infer_kind(cls)
    entry = {
        "module": "plugin" if entry_path.name == "plugin.py" else "plugin",
        "class": cls.name,
    }
    manifest = build_default_plugin_manifest(
        plugin_id=plugin_dir.name,
        kind=kind,
        description=infer_docstring_summary(cls) or "",
        entry_point=entry,
        params_schema=infer_config_calls(cls, "_param") | infer_config_calls(cls, "params.get"),
        thresholds_schema=infer_config_calls(cls, "_threshold") | infer_config_calls(cls, "thresholds.get"),
        suggested_params={k: v.default for k, v in params if v.default is not None},
        suggested_thresholds={k: v.default for k, v in thresholds if v.default is not None},
        secrets_required=list(infer_required_secrets(cls)),
    )
    todo = {"plugin.category", "plugin.stability", "compat.min_core_version"}
    return dump_manifest_toml(manifest, todo_fields=todo)

def scaffold_preset_manifest(preset_dir: Path) -> str:
    yaml_candidates = [
        preset_dir / "preset.yaml",
        *(p for p in preset_dir.glob("*.yaml") if p.name != "preset.yaml"),
    ]
    yaml_path = next((p for p in yaml_candidates if p.is_file()), None)
    if yaml_path is None:
        raise FileNotFoundError(f"no *.yaml found in {preset_dir}")
    manifest = {
        "preset": {
            "id": preset_dir.name,
            "name": preset_dir.name,
            "description": "",
            "size_tier": "",
            "version": "0.1.0",
            "entry_point": {"file": yaml_path.name},
        }
    }
    todo = {"preset.description", "preset.size_tier"}
    return dump_manifest_toml(manifest, todo_fields=todo)
```

### `sync.py`

Два публичных API:

```python
@dataclass(frozen=True)
class SyncResult:
    new_manifest_text: str     # content to write
    changed: bool              # True if anything differs from existing
    diff: str                  # unified-diff text for logging/CLI


def sync_plugin_manifest(
    plugin_dir: Path,
    *,
    bump: Literal["patch", "minor", "major"] = "patch",
) -> SyncResult:
    existing_path = plugin_dir / "manifest.toml"
    existing = tomllib.loads(existing_path.read_text())
    inferred = _build_inferred_plugin_manifest(plugin_dir)
    merged = _merge_plugin(existing, inferred, bump=bump)
    new_text = dump_manifest_toml(merged, todo_fields=set())
    return SyncResult(new_text, changed=(new_text != existing_path.read_text()), diff=...)
```

**Алгоритм `_merge_plugin`:**

1. **MANAGED поля** — просто берём из `inferred`.
2. **USER-OWNED** (id/name/description/…) — берём из `existing`; если в existing нет или пустое, берём из `inferred`.
3. **`params_schema` / `thresholds_schema`** — per-key merge:
   - Для каждого ключа из `inferred.keys() ∪ existing.keys()`:
     - Если только в `inferred` → добавляем целиком из inferred (NEW).
     - Если только в `existing` → **удаляем** (код больше не использует).
     - Если в обоих → начинаем с `existing[key]`, но если `existing[key].default == _PREVIOUS_INFERRED_DEFAULT[key]` (отследить через стабильные marker'ы невозможно, проще: если `existing[key].default` **отсутствует** → подставляем из inferred).
4. **`suggested_params` / `suggested_thresholds`** — берём целиком из `existing`, но удаляем ключи, которых больше нет в соответствующей `*_schema`. Это убирает мёртвые ссылки.
5. **`secrets.required`** — union `existing ∪ inferred`, порядок по inferred затем добавленные вручную ключи по алфавиту.
6. **`plugin.version`** — парсим semver, инкрементим сегмент по `bump`, сбрасываем младшие в ноль.

**Детектирование `existing[key].default` "было ли override-нуто пользователем"**: проще всего — если существующее значение `default` совпадает с текущим inferred — перезаписываем (user не трогал). Если отличается — оставляем (user явно задал). Это не идеально (если пользователь поставил тот же default случайно), но лучше, чем маркеры в TOML-комментариях.

**Preset sync** — сильно проще, потому что almost everything USER-OWNED:

```python
def sync_preset_manifest(preset_dir: Path, *, bump="patch") -> SyncResult:
    # Merge rules:
    # - entry_point.file: переписать на свежий scan (MANAGED)
    # - version: бампнуть
    # - остальное: не трогать
```

### CLI (`src/cli/community.py`)

```python
community_app = typer.Typer(no_args_is_help=True, help="...")

@community_app.command("scaffold")
def scaffold_cmd(
    path: Annotated[Path, typer.Argument(exists=True, dir_okay=True, file_okay=False)],
    force: Annotated[bool, typer.Option("--force", "-f")] = False,
    kind: Annotated[str | None, typer.Option("--kind")] = None,  # auto | plugin | preset
) -> None:
    """Generate a fresh manifest.toml for a plugin or preset folder."""
    path = path.resolve()
    is_preset = _detect_is_preset(path, override=kind)
    text = scaffold_preset_manifest(path) if is_preset else scaffold_plugin_manifest(path)
    target = path / "manifest.toml"
    if target.exists() and not force:
        typer.echo(f"error: {target} exists; pass --force to overwrite", err=True)
        raise typer.Exit(1)
    target.write_text(text)
    typer.echo(f"wrote {target}")

@community_app.command("sync")
def sync_cmd(
    path: Annotated[Path, typer.Argument(exists=True)],
    bump: Annotated[str, typer.Option("--bump", "-b")] = "patch",
    dry_run: Annotated[bool, typer.Option("--dry-run")] = False,
) -> None:
    """Re-run inference, merge with existing manifest, optionally bump version."""
    if bump not in ("patch", "minor", "major"):
        raise typer.BadParameter("bump must be patch|minor|major")
    is_preset = _detect_is_preset(path)
    result = (sync_preset_manifest if is_preset else sync_plugin_manifest)(path, bump=bump)
    typer.echo(result.diff)
    if dry_run:
        typer.echo("[dry-run] not writing")
        return
    if not result.changed:
        typer.echo("already in sync")
        return
    (path / "manifest.toml").write_text(result.new_manifest_text)
    typer.echo(f"updated {path}/manifest.toml")
```

`_detect_is_preset(path)` — простой: `path.parent.name == "presets"` → preset, иначе plugin. Override через `--kind` можно руками.

---

## Критические файлы

| Файл | Действие |
|---|---|
| `src/community/inference.py` | **создать** |
| `src/community/toml_writer.py` | **создать** |
| `src/community/scaffold.py` | **создать** |
| `src/community/sync.py` | **создать** |
| `src/cli/community.py` | **создать** |
| `src/main.py` | добавить `app.add_typer(community_app, name="community", ...)` |
| `src/community/__init__.py` | (опц.) добавить экспорты `scaffold_*`, `sync_*` для программного использования |
| `src/tests/unit/community/test_inference.py` | **создать** |
| `src/tests/unit/community/test_toml_writer.py` | **создать** |
| `src/tests/unit/community/test_scaffold.py` | **создать** |
| `src/tests/unit/community/test_sync.py` | **создать** |
| `src/tests/unit/community/test_cli_community.py` | **создать** |

### Реиспользуемые утилиты

- `src/community/manifest.py::PluginManifest`, `PresetManifest` — pydantic-модели для финальной валидации (sync всегда валидирует результат перед записью; если не проходит — ошибка).
- `src/community/loader.py` — **не трогаем** (только источник базовых имён для инференции).
- Базовые классы `src/{data/validation,evaluation/plugins,training/reward_plugins}/base.py` + `src/reports/plugins/interfaces.py` — источник имён для `infer_kind`.
- `src/main.py` — уже имеет typer `app`, просто добавляем `add_typer`.

---

## Риски и открытые вопросы

1. **AST-паттерн `self.params.get(...)` vs `self._param(...)`** — существующие плагины используют оба. Inference должен пониматьoth. Тест покрытия: min_samples (использует `_param`/`_threshold`), identical_pairs (использует `params.get`).
2. **Dynamic key access** — `key = self.params.get("mode"); value = self._param(key)` — нельзя заинферить. Решение: warning + пропуск, НЕ падать. Автор зафиксирует в manifest вручную.
3. **3-way merge потеря данных**. При удалении ключа из кода, sync **удаляет** из `params_schema`/`thresholds_schema`. Если это ошибка (забыли вернуть key в коде) — пользователь потеряет ручную валидацию. **Митигатор**: `--dry-run` показывает diff перед записью. Советуем в help: "always run --dry-run first".
4. **secrets.required: добавление, но не удаление**. Пользователь может декларировать опциональные секреты, которые код не требует напрямую (передаст их плагину через config). Поэтому union, не intersection. Покрываем тестом.
5. **Порядок полей в TOML при первой записи `sync`** — если у нас старые manifest'ы имеют нестабильный порядок, после первого `sync` все файлы перегенерятся. Это **ожидаемо**, но стоит смерджить в отдельном коммите, чтобы не смешать с функциональными правками.
6. **Версия пресета** — сейчас 3 существующих пресета имеют `version = "1.0.0"`. После первого `sync` с default-patch они станут `1.0.1`. Это **ожидаемо** — так пользователь явно помечает что manifest'ы были перегенерены.
7. **Preset `entry_point.file` scan** — если в папке два YAML-файла, берём `preset.yaml`, иначе первый по алфавиту. Если ни одного — ошибка.

---

## Verification

1. **Unit-тесты inference**: матрица покрытия
   - `_param` / `_threshold` с literal key + default → правильный type/default.
   - `params.get` / `thresholds.get` → то же.
   - Dynamic key → warning, не падает.
   - `self._secrets["DTST_..."]` → попадает в required.
   - Entry class: single class → OK; zero → error; multiple → выбирает по имени папки или падает с подсказкой.
   - Базовый класс: ValidationPlugin/EvaluatorPlugin/RewardPlugin → правильный `kind`. IReportBlockPlugin-like (plugin_id + order + render) → `reports`.

2. **Unit-тесты scaffold**:
   - Искусственный plugin.py с `_param("a", 10)`, docstring, ValidationPlugin base → сгенерированный manifest проходит `PluginManifest.model_validate`.
   - Preset с `preset.yaml` → корректный preset-manifest.
   - Полученный TOML re-parse'ится в pydantic без ошибок.

3. **Unit-тесты sync (3-way merge)**:
   - User overrode `params_schema.a.min = 1` → sync сохраняет `min`, но обновляет `default`/`type`.
   - Новый ключ `b` появился в коде → добавлен в schema.
   - Ключ `c` исчез из кода → удалён из schema.
   - `secrets.required = ["EVAL_OLD"]` в existing, inferred = `["EVAL_NEW"]` → union `["EVAL_NEW", "EVAL_OLD"]`.
   - `suggested_params` содержит ключ, которого больше нет в schema → удалён.
   - Version bump: `1.2.3` + patch → `1.2.4`; + minor → `1.3.0`; + major → `2.0.0`.

4. **CLI tests** (`typer.testing.CliRunner`):
   - `community scaffold <tmp_path>` создаёт manifest.toml, re-parse'ится pydantic'ом.
   - `community scaffold` на существующей папке без `--force` → exit 1.
   - `community sync <plugin_dir> --dry-run` — не пишет файл, печатает diff.
   - `community sync <plugin_dir> --bump minor` — версия ушла на `.0` младших сегмента.
   - Auto-detection preset vs plugin по имени родителя.

5. **Integration на существующих плагинах**: прогнать `community sync --dry-run` на каждый из 14 существующих плагинов + 3 пресета → убедиться что:
   - diff минимален (только `version`-bump и, возможно, порядок полей);
   - результат всегда валиден `PluginManifest.model_validate` / `PresetManifest.model_validate`;
   - catalog после первой реальной записи (без dry-run) продолжает загружать все плагины (`21/21` community tests остаются green).

6. **Grep-гварды не должны регрессить** (из предыдущих задач — проверяем, что новые модули не возвращают удалённые декораторы).
