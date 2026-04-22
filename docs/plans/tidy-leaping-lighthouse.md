# План: модернизация `ryotenkai` CLI

## Context

Текущий CLI (`src/main.py` + `src/cli/community.py`) работает, но страдает от трёх симптомов — пользователь явно обозначил их в последних итерациях:

1. **Перегруз по информации**: Rich-боксы, длинные многоабзацные docstring-и, ad-hoc `typer.style(...)` в разных файлах, traceback-боксы на user-ошибках. После последнего прохода по community/* это устранено точечно — осталось разнести паттерн на весь CLI.
2. **Hand-rolled вывод**: `src/cli/run_rendering.py` рендерит таблицы через `fmt.format("{:<32}...")` — нет Rich Table, нет JSON-варианта, widths зашиты. Иконки статусов (`◉ ▸ ◈ ◌ ◇ ○`) и цвета тоже раскиданы.
3. **Отсутствие cross-cutting фич из clig.dev / 12-factor-CLI**: нет `--output json`, нет `-V/--version` на корне, нет `--no-color`/`NO_COLOR`, нет глобального `-v/-q`, нет shell-комплишнов (`add_completion=False`), `version` — команда без debug-инфы. `src/main.py` = 996 строк с 17 командами — масштабировать дальше больно.

**Решения по развилкам (от пользователя):**

1. **Scope** = весь ryotenkai CLI (все 17 top-commands + community sub-app). Не точечный патч community.
2. **Framework** = **остаёмся на Typer 0.19**. Аудит показал: структура не костыли, а Cyclopts (хотя объективно лучше спроектирован) — bus-factor=1, ×15 меньше экосистема, Typer 0.19 уже закрыл основной гап (нативные `Literal[...]`). Миграцию держим в уме как опцию, если упрёмся в `int | Literal[...]` Union или в Pydantic-модели как CLI-аргументы.
3. **JSON output** = да, для read-команд (`runs-list`, `inspect-run`, `run-status`, `run-diff`, `config-validate`, `info`). Требование clig.dev и 12-factor-CLI для composability (pipe в `jq`, скрипты CI).

**Что не входит** (чтобы не разбухать):
- TUI (`src/tui/`) — отдельный слой, не трогаем.
- Web API — не меняем.
- Миграция фреймворка — опция на будущее.
- Переписывание бизнес-логики (`src/pipeline/…`) — трогаем только I/O слой.

---

## Принципы, которые применяем (clig.dev + 12-factor-CLI)

Коротко, чтоб на них можно было ссылаться в ревью:

- **Human-first, machine-friendly**. Текстовый вывод по умолчанию, `--output json` для скриптов.
- **Pipe-safe**. `isatty()` → цвета/боксы; пайп → plain text. Уважаем `NO_COLOR`.
- **Tight help**. 1 абзац описание + `\b Examples:` + argument/option таблица. Никаких эссе.
- **Предсказуемые флаги**. `-h/--help`, `-V/--version`, `-v/--verbose`, `-q/--quiet`, `-o/--output`, `-f/--force`, `-n/--dry-run` — одинаковые имена во всех командах.
- **Чистые ошибки**. Одна строка `error: …` + опциональный hint. Никаких Rich-traceback-боксов на предсказуемых ошибках. Неожиданные баги → обычный Python traceback.
- **Did-you-mean** на опечатки (бесплатно через click >= 8.1, оно под капотом Typer).
- **Exit-коды документированы**. `0` успех, `1` бизнес-ошибка, `2` аргументы/usage (это уже делает click).

---

## Целевой UX (эталонные примеры)

```
$ ryotenkai --version
ryotenkai 0.4.2  python 3.13.1  platform darwin-arm64  git 6b78e80

$ ryotenkai runs-list
Run ID                            Status         Att  Duration    Config
────────────────────────────────  ─────────────  ───  ──────────  ─────────────────
run_2026_0422_1732_baking         completed        1  2h 14m 8s   config_qwen.yaml
run_2026_0422_1510_vigorous       failed           3  41m 22s     config_llama.yaml

$ ryotenkai runs-list -o json | jq '.[].status'
"completed"
"failed"

$ ryotenkai inspect-run runs/missing
error: run directory not found: runs/missing
  hint: list available runs with `ryotenkai runs-list`
```

---

## Архитектура слоя CLI

Новая структура:

```
src/cli/
├── __init__.py
├── app.py                 # Typer() instance + root @callback; register sub-apps here
├── context.py             # CLIContext dataclass (output format, color, verbose, quiet, log_level)
├── renderer.py            # Renderer Protocol + TextRenderer + JsonRenderer
├── style.py               # ICONS, COLORS, one Console instance; no more ad-hoc typer.style
├── errors.py              # die(msg, hint=...) + DidYouMean helper
├── completion.py          # --install-completion wiring
├── commands/
│   ├── __init__.py
│   ├── train.py           # train, validate-dataset, list-restart-points, train-local
│   ├── runs.py            # runs-list, inspect-run, run-status, run-diff, logs
│   ├── config.py          # config-validate, info
│   ├── report.py          # report
│   ├── serve.py           # serve, tui
│   ├── version.py         # version + root --version callback
│   └── community.py       # (существующий community_app переезжает сюда)
└── run_rendering.py       # **remove** — поглощается renderer.py + TextRenderer
```

`src/main.py` сжимается до ~20 строк: сигнал-хендлеры + `from src.cli.app import app; app()`.

---

## План по фазам

Шесть фаз, каждая самодостаточна и оставляет CLI рабочим. Делаем отдельными коммитами — bisect-friendly.

### Фаза 1. Shared infrastructure (foundation)

Закладываем общий слой, ничего не ломая.

**Новые модули:**

- `src/cli/context.py` — `CLIContext` dataclass (`output: Literal["text","json"]`, `color: bool`, `verbose: int`, `quiet: bool`, `log_level: str`). Хранится в `typer.Context.obj`.
- `src/cli/style.py` — один `rich.console.Console` (+ `Console(stderr=True)` для ошибок), словарь `ICONS`, константы `COLOR_OK/WARN/ERR`. `style.py` честно честно проверяет `console.is_terminal` и `NO_COLOR` — в пайпе цветов нет.
- `src/cli/renderer.py` — `class Renderer(Protocol)` с методами `table(headers, rows)`, `kv(pairs)`, `tree(root)`, `status_line(...)`. Реализации: `TextRenderer` (через Rich) и `JsonRenderer` (копит структуру, в конце `json.dumps`). Фабрика: `get_renderer(ctx: CLIContext) -> Renderer`.
- `src/cli/errors.py` — `def die(msg, *, hint=None, code=1)`; `suggest(user_input, valid_options)` для did-you-mean.
- `src/cli/completion.py` — обёртка над typer-ом, чтобы включить `add_completion=True` + документировать `ryotenkai --install-completion [bash|zsh|fish]`.

**Root @callback в `src/cli/app.py`:**

```python
@app.callback()
def _root(
    ctx: typer.Context,
    version: bool = typer.Option(False, "-V", "--version", is_eager=True, callback=_print_version),
    output: str = typer.Option("text", "-o", "--output", help="Output format: text | json", envvar="RYOTENKAI_OUTPUT"),
    color: bool = typer.Option(True, "--color/--no-color", envvar="NO_COLOR", help="Colored output (honours NO_COLOR env)"),
    verbose: int = typer.Option(0, "-v", "--verbose", count=True, help="Increase verbosity (-v, -vv)"),
    quiet: bool = typer.Option(False, "-q", "--quiet", help="Suppress non-essential output"),
    log_level: str | None = typer.Option(None, "--log-level", help="Override log level (DEBUG/INFO/WARNING/ERROR)"),
) -> None: ...
```

Контекст передаём в команды через `ctx.obj` — командам не нужно парсить флаги самим.

**Что не трогаем в этой фазе:** команды остаются там же в `src/main.py`. Они просто начинают читать `ctx.obj` вместо собственных флагов.

**Acceptance:** `ryotenkai --version` работает; `ryotenkai runs-list -o json` не падает (даже если пока возвращает `[]`); `NO_COLOR=1 ryotenkai ...` не красит вывод.

---

### Фаза 2. Split `src/main.py`

Разбиваем 996-строчный main на модули в `src/cli/commands/*.py`. Каждый модуль экспортирует `register(app: typer.Typer)` и регистрирует свои команды на переданный app.

`src/main.py` становится:

```python
# src/main.py — 20 строк
from src.cli.app import build_app

def main() -> None:
    app = build_app()
    app()

if __name__ == "__main__":
    main()
```

`src/cli/app.py::build_app()` собирает app + вызывает все `commands.*.register(app)` + привязывает community sub-app.

**Что это даёт:**
- `src/main.py` перестаёт быть местом боли при добавлении команд.
- Команды модульные → можно писать per-command tests в `src/tests/unit/cli/test_runs.py` etc.
- signal-handler остаётся в `src/main.py` (он про pipeline lifecycle, не CLI).

**Acceptance:** старые команды работают идентично; размер `src/main.py` < 200 строк (только signal/cleanup-код + `main()`).

---

### Фаза 3. Renderer для read-команд + Rich-таблицы

Переводим все «вывод данных» команды на `ctx.obj.renderer`. Уходим от `fmt.format("{:<32}...")` → Rich `Table`.

**Команды, которые мигрируют:**

| Команда | Что рендерим |
|---|---|
| `runs-list` | Таблица: Run ID / Status / Attempts / Duration / Config |
| `inspect-run` | Key-value header + вложенные таблицы per attempt |
| `run-status` | Live-snapshot (pipe-safe: plain text при `not isatty`) |
| `run-diff` | Diff-секции (coloured ±) |
| `list-restart-points` | Таблица: # / Stage / Available / Mode / Reason |
| `info` | Key-value + список stages |
| `config-validate` | Список чеков со статус-иконками |

**Правила рендера:**
- Заголовок таблицы: жирный, без панели-бокса (Rich `Table(box=None)` или `box=SIMPLE`).
- Иконки статусов берём из `cli/style.py::ICONS` — один источник правды.
- Если `not console.is_terminal` — автоматически plain text (Rich это умеет, просто конфигурим `Console(force_terminal=None)`).

**Acceptance:** визуальный diff для `runs-list` / `inspect-run` — таблицы ровные, в пайпе получаем plain-text, `-o json` отдаёт корректный JSON.

---

### Фаза 4. JSON output для read-команд

Для каждой read-команды пишем параллельный рендер через `JsonRenderer`.

**Контракт** (стабильная схема, задокументировать в docstring):

```json
// runs-list -o json
[
  {
    "run_id": "run_2026_0422_1732_baking",
    "status": "completed",
    "attempts": 1,
    "started_at": "2026-04-22T17:32:11Z",
    "completed_at": "2026-04-22T19:46:19Z",
    "duration_s": 8048,
    "config_name": "config_qwen.yaml"
  }
]
```

```json
// inspect-run <id> -o json
{
  "run_id": "…",
  "status": "completed",
  "config_path": "…",
  "mlflow_run_id": "…",
  "attempts": [
    {
      "attempt_no": 1,
      "status": "completed",
      "started_at": "…",
      "completed_at": "…",
      "stages": [
        { "name": "Dataset Validator", "status": "completed", "duration_s": 12.3, "mode": "ran", "error": null }
      ]
    }
  ]
}
```

Команды-«действия» (`train`, `report`, `pack`) JSON-вывод **не получают** (у них нет структурированного результата для потребителя — это action, а не query). Допустимо вернуть `{"ok": true}` на успех для consistency — решим по ходу дела, начинаем без этого.

**Acceptance:** `ryotenkai runs-list -o json | jq . | head` — валидный JSON; формат задокументирован в `docs/cli.md`.

---

### Фаза 5. Тесты CLI

Сейчас покрытия top-level команд почти нет. Закладываем базу:

`src/tests/unit/cli/` с файлами per-command-группа:

- `test_help.py` — каждая команда отвечает `exit_code=0` на `--help`, содержит `Usage:` и сводку.
- `test_version.py` — `--version` и `version` команда.
- `test_runs.py` — `runs-list` / `inspect-run` / `run-status` / `run-diff` в text и json режимах, на seeded tmp runs-dir.
- `test_config.py` — `config-validate` / `info` + missing-file.
- `test_global_flags.py` — `NO_COLOR=1`, `--output json`, `-v`, `-q` работают как контракт.

Используем `typer.testing.CliRunner` (как в `test_cli_community.py`). Покрытие — smoke, не полное (полнота — в e2e).

**Acceptance:** новые тесты зелёные, `pytest src/tests/unit/cli/` < 10 с.

---

### Фаза 6. Polish + документация

- **`version` команда** → печатает `ryotenkai X.Y.Z · python A.B.C · platform ··· · git sha ···`. Работает и при `-V/--version`.
- **Did-you-mean на опечатки**: click бросает `UsageError` со списком доступных команд. Нужно перехватить в root и вывести `error: unknown command 'sycn'. Did you mean 'sync'?` — опираясь на `difflib.get_close_matches`.
- **Docstring style guide** в `docs/cli.md`:
  - Первая строка `short_help` < 60 символов.
  - Тело = 1 абзац "что делает" + `\b Examples:` + 2-3 примера.
  - Никакой Rich-разметки (`[bold]…[/bold]`), потому что `rich_markup_mode=None` на app.
- **Пробегаемся sed-ом по всем командам** в `src/cli/commands/*.py` и приводим docstring-и к guide-у.
- **`ryotenkai --install-completion`** — включён, smoke-тест на zsh/bash.
- **CHANGELOG-запись** + короткий `docs/cli.md` с примерами и JSON-схемой.

**Acceptance:** `ryotenkai help` и `ryotenkai <cmd> --help` — компактные, единообразные; `ryotenkai --install-completion zsh` печатает rc-инструкцию.

---

## Критические файлы

| Файл | Действие |
|---|---|
| `src/cli/app.py` | **создать** — Typer-app + root callback + сборка команд |
| `src/cli/context.py` | **создать** — CLIContext |
| `src/cli/renderer.py` | **создать** — Renderer protocol + Text/Json impl |
| `src/cli/style.py` | **создать** — Console, ICONS, цвета |
| `src/cli/errors.py` | **создать** — die(), suggest() |
| `src/cli/completion.py` | **создать** — completion wiring |
| `src/cli/commands/train.py` | **создать** — мигрируют `train`, `validate-dataset`, `list-restart-points`, `train-local` |
| `src/cli/commands/runs.py` | **создать** — мигрируют `runs-list`, `inspect-run`, `run-status`, `run-diff`, `logs` |
| `src/cli/commands/config.py` | **создать** — мигрируют `config-validate`, `info` |
| `src/cli/commands/report.py` | **создать** — мигрирует `report` |
| `src/cli/commands/serve.py` | **создать** — мигрируют `serve`, `tui` |
| `src/cli/commands/version.py` | **создать** — `version` + `-V` callback |
| `src/cli/commands/community.py` | **переезд** — `src/cli/community.py` перемещаем сюда без изменений |
| `src/main.py` | **ужимаем до ~20 строк** (signal-handler + `main()`) |
| `src/cli/run_rendering.py` | **удалить** — логика уходит в `renderer.py` + `TextRenderer` |
| `src/tests/unit/cli/*` | **создать** — 5-6 файлов smoke-тестов |
| `docs/cli.md` | **создать** — стиль-гайд + JSON-схема read-команд |

**Реиспользуем (не трогаем):**
- `src/utils/logger.py` — colorlog-логгер как есть.
- `src/pipeline/run_queries.py` — источник данных для `runs-list` / `inspect-run`, не меняется.
- `src/pipeline/state.py` — state-объекты, рендер читает их.
- `typer.testing.CliRunner` — тест-инфра уже используется в `test_cli_community.py`.

**Новые зависимости:** нет. Rich уже в `pyproject.toml` (`rich>=14.2.0`), просто начинаем его использовать в CLI-слое.

---

## Риски и митигаторы

1. **«Один большой рефактор → регрессы»**. Митигатор: 6 фаз, каждая — отдельный коммит, `pytest` после каждой. Bisect-friendly.
2. **Разная ширина терминалов / CI без TTY**. Rich сам детектит — проверяем в CI, что `CI=true` не ломает вывод.
3. **JSON-схема превращается в API**. Митигатор: документируем в `docs/cli.md`, помечаем как «stable for v1», нарушающие изменения — только на major-bump.
4. **`NO_COLOR` vs `--color/--no-color`** — конфликт приоритетов. Правило: CLI-флаг бьёт env, env бьёт дефолт.
5. **Community sub-app уже свежий**. Не переделываем — только переносим в `commands/community.py` as-is и донаследуем стилем (иконки/рендер из общих модулей).
6. **Миграция Cyclopts в будущем**. План совместим: `cli/app.py` + `cli/commands/*.py` структура 1-в-1 переносится на Cyclopts (см. migration-гайд). Делая этот рефакторинг, мы также снижаем стоимость будущего свича.

---

## Verification

Для каждой фазы отдельно, но плюс финальная прогонка:

**Автоматически:**

```bash
# Весь CLI-набор
pytest src/tests/unit/cli/ -v

# Community не сломан
pytest src/tests/unit/community/ src/tests/integration/api/test_delete_and_config.py -q

# Линт
ruff check src/

# Быстрый smoke
ryotenkai --version
ryotenkai --help
ryotenkai runs-list --help
ryotenkai runs-list -o json
NO_COLOR=1 ryotenkai runs-list
ryotenkai runs-list | cat    # pipe-safe
ryotenkai sycn               # did-you-mean → "sync"?
```

**Ручная проверка:**

- [ ] `ryotenkai --help` — одно-абзацное описание, компактный список команд.
- [ ] `ryotenkai <cmd> --help` для каждой команды — 1 абзац + Examples + Arguments + Options, без `[bold]…[/bold]` артефактов.
- [ ] `ryotenkai runs-list` в TTY — цветная таблица, Rich-ная.
- [ ] `ryotenkai runs-list > out.txt` — plain text без ANSI-кодов.
- [ ] `ryotenkai runs-list -o json | jq '.[0]'` — валидный JSON-объект.
- [ ] `ryotenkai --version` и `ryotenkai version` — одинаковый вывод с debug-инфой.
- [ ] `ryotenkai community sync community/` — старое поведение (batch-mode).
- [ ] Ошибочные вызовы (неизвестный run_dir, битый config) — одна строка `error: …` + hint, без traceback-бокса.

**Успех:** пользователь запускает `ryotenkai runs-list | jq`, `ryotenkai inspect-run <id> -o json`, `ryotenkai train --config ./c.yaml` без сюрпризов; help-экраны короткие и похожие друг на друга; ошибки читабельны с первого взгляда.

---

## Приоритеты / порядок

Если не хотим делать всё сразу — минимальная «приносит пользу сегодня» последовательность:

1. **Фаза 1** (foundation) — без этого остальное костыли.
2. **Фаза 3** (Rich-таблицы) — самый заметный визуальный апгрейд, снимает «перегруз».
3. **Фаза 4** (JSON output) — разблокирует скриптование.
4. **Фаза 6** (polish) — мелкие штрихи, но без них всё не прочитается единообразно.
5. **Фаза 2** (split main.py) — техдолг, можно отложить на неделю после UX-прохода.
6. **Фаза 5** (тесты CLI) — параллельно с фазами 3-4, заводим по ходу.
