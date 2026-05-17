# CLI architecture in this project

Rules for writing, extending, and refactoring the `ryotenkai` CLI.
Born out of the kubectl-style refactor (`docs/plans/b-hazy-phoenix.md`,
merge `e9d5ec3`). Future agents — read before adding any CLI command.

---

## Contents

- [Instructions for LLMs](#instructions-for-llms)
- [Layout](#layout)
- [Mandatory rules](#mandatory-rules)
- [Adding a new command (existing noun)](#adding-a-new-command-existing-noun)
- [Adding a new noun](#adding-a-new-noun)
- [Renderer + output modes](#renderer--output-modes)
- [Global flags](#global-flags)
- [Anti-patterns](#anti-patterns)
- [Tests](#tests)
- [Cheat sheet: noun → service → module](#cheat-sheet-noun--service--module)

---

## Instructions for LLMs

> This document is for developers and LLM agents writing or refactoring
> CLI / API / service-layer code in this repo. When in doubt, follow
> these rules verbatim — they encode why the flat-Typer / TUI / `run.sh`
> shell-soup was burnt down. Diverging without an ADR re-creates that
> mess.

**Quick reference:** kubectl-style `noun verb` only · one noun = one
`src/cli/commands/<noun>.py` · everything through `Renderer`
(no `typer.echo` for data) · CLI and API both call
`src/api/services/*` (no logic duplication) · services stay
pure-python (no FastAPI imports) · no shell wrappers, no
backward-compat shims.

### Mandatory rules (MUST)

1. **Every CLI command MUST be `noun verb`** — no flat top-level
   verbs like `train` / `runs-list`. The root Typer (`src/cli/app.py`)
   only mounts sub-Typers + `version` + hidden `help`.
2. **One noun = one module** at `src/cli/commands/<noun>.py`. The
   module exports `<noun>_app: typer.Typer` and is registered in
   `src/cli/commands/__init__.py::_REGISTRY`.
3. **Data output MUST go through a `Renderer`** from
   `src/cli/renderer.py`. Get it via `get_renderer(state)` where
   `state = ctx.ensure_object(CLIContext)`. `typer.echo` is allowed
   only for one-line acks ("created: X", "wrote merged config: Y").
4. **CLI MUST call `src/api/services/*` for business logic** —
   never duplicate a read or a write. If the service doesn't exist,
   add it there first, then the CLI wrapper. The API router and the
   CLI command end up as two thin shells around the same service
   function.
5. **Services in `src/api/services/` MUST stay pure-python** — no
   `from fastapi import …`, no `HTTPException`, no `Depends`. HTTP
   wiring lives only in `src/api/routers/`. CI guard:
   `grep -rn "fastapi\|HTTPException\|Depends\|Request,\|BackgroundTasks" src/api/services/` ⇒ should hit only Pydantic schema imports.
6. **Top-level imports in `src/cli/` MUST stay lean** — `ryotenkai --help`
   must render under 300 ms. Heavy stuff (orchestrator, mlflow, torch,
   community catalog, `load_config`) goes inside command bodies as
   `import` statements, not at module top.
7. **Long-running commands MUST register their orchestrator** with
   `src.cli._signals.set_active_orchestrator(orch)` and clear it in
   `finally`. The signal handler installed at import time forwards
   SIGINT/SIGTERM to it. Do not roll your own `signal.signal(...)`
   inside a command.
8. **Reusable option/argument shapes MUST come from**
   `src/cli/common_options.py` (`ConfigOpt`, `RunDirArg`,
   `ProjectOpt`, `KindOpt`, `ForceOpt`, `DryRunOpt`, `YesOpt`).
   Don't redeclare the same `Annotated[Path, typer.Option(...)]` in
   every command — help text and validators drift.
9. **Errors MUST flow through `src.cli.errors.die(msg, hint=..., code=...)`**.
   Use `raise die(...)` for control flow. Never `typer.echo(..., err=True);
   raise typer.Exit(1)` by hand — wording drifts and stderr/stdout
   routing breaks `-o json` consumers.
10. **No new shell launchers** (`run.sh`, `setup.sh`, ad-hoc wrappers).
    The `ryotenkai` console-script in `pyproject.toml` is the single
    entry point. If you need a long-running service, add a `<noun> start`
    verb. `web/scripts/*.sh` exists only to babysit dev backend/frontend
    processes — and it MUST call `python -m src.main <noun> <verb>`,
    never a legacy flat verb.
11. **No backward-compat aliases.** When a verb is renamed, delete
    the old one. Document the migration in `README.md` (cheat-sheet
    table). The whole point of avoiding shims is to keep one true
    spelling — every shim is one more grep-target for future agents.

### When you're tempted to skip a rule

Stop, read `docs/plans/b-hazy-phoenix.md` § "Risks & resolutions" —
the failure mode you're about to re-introduce is almost certainly
already documented there (Q-04 deprecation, Q-08 services purity,
NR-02 state-cache, NR-04 dataset-validate, NR-05 coverage).

---

## Layout

```
src/
  cli/
    app.py                      # Typer root + global flags + register_all(app)
    _signals.py                 # SIGINT/SIGTERM handler — install() + set_active_orchestrator()
    common_options.py           # Annotated[...] shared option / argument shapes
    context.py                  # CLIContext dataclass (output, color, project, remote, …)
    renderer.py                 # Renderer protocol + TextRenderer / JsonRenderer / YamlRenderer
    errors.py                   # die() / suggest_hint()
    style.py                    # Rich console singletons, ICONS
    formatters.py               # duration helpers (pure)
    version.py                  # collect_version_info()
    run_rendering.py            # plain-text renderers for run state
    _smoke_runner.py            # batch_smoke runtime (called by commands/smoke.py)
    commands/
      __init__.py               # _REGISTRY: [(sub_app, name), …] + register_all(app)
      run.py        runs.py     config.py    dataset.py
      project.py    plugin.py   preset.py    smoke.py
      server.py     version.py
  cli_state/
    context_store.py            # ~/.ryotenkai/cli-context.json — `project use` pointer
  api/
    services/                   # pure-python business logic (CLI + API both call this)
    routers/                    # FastAPI-specific HTTP wiring (HTTPException lives ONLY here)
  community/
    catalog.py                  # full catalog (plugins + presets + registries)
    loader.py                   # pure load_plugins / load_presets (no registry side-effects)
    install.py                  # ryotenkai plugin install (local / zip / git)
    validate_manifest.py        # standalone manifest validation, no Python import
    scaffold_template.py        # pure render helpers for `plugin scaffold`
  main.py                       # collapsed to `from src.cli.app import app`
```

---

## Mandatory rules

### 1. Noun-verb naming

- ✅ `ryotenkai run start`, `ryotenkai runs ls`, `ryotenkai plugin install`
- ❌ `ryotenkai train`, `ryotenkai runs-list`, `ryotenkai inspect-run`

Hyphens are allowed in **verbs** when the verb is a compound
(`restart-points`, `sync-envs`). Nouns are always singular except
`runs` (plural read-noun, mirrors kubectl's `kubectl get pods`).

### 2. One noun = one module

Each `src/cli/commands/<noun>.py` declares exactly one
`<noun>_app: typer.Typer`, decorates every verb on it, and is
mounted via `_REGISTRY` in `commands/__init__.py`. No verbs leak
out of their noun module.

### 3. Output through `Renderer`

```python
from src.cli.context import CLIContext
from src.cli.renderer import get_renderer

def my_cmd(ctx: typer.Context, ...) -> None:
    state = ctx.ensure_object(CLIContext)
    renderer = get_renderer(state)        # picks Text / Json / Yaml by --output

    if state.is_machine_readable:
        renderer.emit({"key": "value"})    # single document for -o json/yaml
    else:
        renderer.table(headers=[...], rows=[...])
        # or renderer.kv({...}), renderer.text("...")

    renderer.flush()                       # buffered renderers (Json/Yaml) write here
```

`renderer.emit()` is called at most **once per command**. JsonRenderer
raises on double-emit so structured output stays one valid document.

### 4. CLI and API are peers

```
                ┌─────────────────────────────────────┐
                │  src/api/services/<x>_service.py    │  ← pure-python core
                │  (or src/pipeline/* / community/*)  │
                └────────────────┬────────────────────┘
                                 │
            ┌────────────────────┴────────────────────┐
            │                                         │
            ▼                                         ▼
 src/cli/commands/<noun>.py            src/api/routers/<x>.py
   thin Typer wrapper                   thin FastAPI wrapper
   formats result for renderer          formats result for JSON HTTP
```

A CLI command **must not** open `pipeline_state.json`, walk
`community/`, or talk to MLflow directly when a service already
exposes that read. Add the service first, then both peers consume
it. This is enforced by the contract tests in
`src/tests/contract/test_cli_api_parity.py` for read pairs.

### 5. Services pure-python

✅ Allowed in `src/api/services/*.py`:
- `from src.pipeline.* import …`
- `from src.workspace.* import …`
- `from src.community.* import …`
- `from src.api.schemas.* import …` (Pydantic shapes — no FastAPI)

❌ Forbidden in `src/api/services/*.py`:
- `from fastapi import …`
- `from fastapi.responses import …`
- `Request`, `Depends`, `HTTPException`, `BackgroundTasks`

If a service needs to signal an error, raise a domain exception
(`ProjectServiceError`, `InstallError`, …). The router catches it
and translates to `HTTPException(...)`; the CLI catches it and
translates to `die(...)`.

### 6. Lean top-level imports

```python
# src/cli/commands/run.py — top of file
import typer
from src.cli.common_options import ConfigOpt, RunDirArg
from src.cli.context import CLIContext
from src.cli.errors import die
from src.cli.renderer import get_renderer

run_app = typer.Typer(...)

@run_app.command("start")
def start_cmd(...) -> None:
    # Heavy stuff lazy-imported HERE:
    from src.pipeline.orchestrator import PipelineOrchestrator
    from src.utils.config import load_config
    ...
```

Anything that pulls in torch / mlflow / transformers / datasets at
import time blows the 300 ms `--help` budget. Lazy-import inside the
function body.

### 7. Catalog vs loader — pick the lightest

`src.community.catalog.catalog.ensure_loaded()` imports **every**
plugin kind and populates every per-kind registry. That transitively
imports `src.training.reward_plugins.registry` → triggers the
strategies factory at import time (~3 s, plus log noise).

- For **plugin** verbs that genuinely need registries (`plugin ls`,
  `plugin show`, preflight, stale detection) — use `catalog`.
- For **preset** verbs (or anything that only needs manifest text /
  YAML body) — call `src.community.loader.load_presets()` /
  `load_plugins(kind)` directly. They are pure-python, zero side
  effects.

Lesson scar: NR-04 in the plan + commit `2ed2989`.

### 8. Atomic writes for state files

`src/cli_state/context_store.py` writes via
`src.utils.atomic_fs.atomic_write_json` (temp-file + rename). Any new
CLI-owned JSON / TOML file under `~/.ryotenkai/` MUST use the same
helper — Ctrl-C mid-write must never leave a corrupt file.

---

## Adding a new command (existing noun)

Example: add `runs prune` (delete runs older than N days).

1. **Service first.** If `delete_service` doesn't know how to bulk-prune,
   add a `prune_runs(runs_dir: Path, older_than_days: int) -> PruneResult`
   function in `src/api/services/delete_service.py`. Pure-python.
   Unit-test it under `src/tests/unit/api/services/`.
2. **API router** (if Web needs it): thin endpoint in
   `src/api/routers/runs.py` that calls the service.
3. **CLI command** in `src/cli/commands/runs.py`:
   ```python
   @runs_app.command("prune")
   def prune_cmd(
       ctx: typer.Context,
       older_than_days: Annotated[int, typer.Option("--older-than-days")],
       runs_dir: Annotated[Path, typer.Argument(...)] = Path("runs"),
       yes: YesOpt = False,
       dry_run: DryRunOpt = False,
   ) -> None:
       from src.api.services import delete_service

       if not yes and not dry_run and not typer.confirm("Prune?"):
           raise die("aborted by user", code=2)

       result = delete_service.prune_runs(runs_dir, older_than_days)
       state = ctx.ensure_object(CLIContext)
       renderer = get_renderer(state)
       renderer.emit(result.model_dump())
       renderer.flush()
   ```
4. **Smoke test** in `src/tests/smoke/test_cli_help.py` is auto-discovered
   from the registered Typer tree — `--help` smoke fires for free.
5. **Contract test** (only if there's an API pair): add a pair entry
   in `src/tests/contract/test_cli_api_parity.py`.
6. **Docs**: update `docs/cli/runs.md` (if it exists) + the README
   cheat-sheet.

---

## Adding a new noun

Don't, unless the new noun isn't a verb on any existing noun.
Premature noun-proliferation defeats discoverability. Examples of
real new nouns we'd accept: `dataset`, `model`, `experiment`. Examples
that should be verbs on an existing noun: `runs cleanup` (not
`cleanup runs ...`), `plugin uninstall` (not `uninstall plugin`).

When you really need one:

1. Create `src/cli/commands/<noun>.py` with `<noun>_app = typer.Typer(...)`.
2. Add `(<noun>_mod.<noun>_app, "<noun>")` to `_REGISTRY` in
   `src/cli/commands/__init__.py`. Import via the existing pattern
   (one-per-line — ruff isort enforces it).
3. Add `__init__.py` re-exports if the noun exposes shared helpers
   to other modules (rare).
4. Update `docs/cli/<noun>.md` (per-noun page).

---

## Renderer + output modes

`-o text` (default) — human Rich tables / kv / heading.
`-o json` — single JSON document on stdout, indent=2.
`-o yaml` — single YAML document on stdout, `safe_dump`, no flow style.

`CLIContext.is_machine_readable` is the **only** check you should
use to branch — never `state.output == "json"`. It captures both
json and yaml.

Path / datetime / dataclasses serialise via `_json_default` /
`_yaml_normalise` (already wired). New non-trivial types? Add a case
there, not a `default=` in your command.

---

## Global flags

Declared **only** on the root Typer in `src/cli/app.py:_root`:

| Flag | Env | Purpose |
|---|---|---|
| `-o text\|json\|yaml`     | `RYOTENKAI_OUTPUT`  | output format |
| `--color / --no-color`    | `NO_COLOR` honoured | terminal palette |
| `-v / -vv`                | —                   | INFO / DEBUG log level |
| `-q / --quiet`            | —                   | ERROR-only log level |
| `--log-level`             | `LOG_LEVEL`         | explicit override (DEBUG/INFO/WARNING/ERROR) |
| `--project, -p <id>`      | `RYOTENKAI_PROJECT` | project context override |
| `--remote <url>`          | `RYOTENKAI_REMOTE`  | reserved (v1.2); raises clean `die()` |
| `-V / --version`          | —                   | eager — prints version, exits |

Per-command flags (`--dry-run`, `--force`, `--yes`, `--config -c`,
`--run-dir`) come from `src/cli/common_options.py`. Do not redeclare.

---

## Anti-patterns

### ❌ `typer.echo(json.dumps(data))`

```python
# BAD — bypasses renderer, breaks -o yaml, can't be buffered
typer.echo(json.dumps(payload, indent=2))
```
```python
# GOOD
renderer.emit(payload); renderer.flush()
```

### ❌ Reading state files directly from a command

```python
# BAD — duplicates service logic, drift between CLI and API
state = json.loads((run_dir / "pipeline_state.json").read_text())
```
```python
# GOOD
from src.api.services import run_service
detail = run_service.get_run_detail(run_dir, runs_dir)
```

### ❌ Importing heavy modules at top of `commands/*.py`

```python
# BAD — every `ryotenkai --help` pays for these imports
from src.pipeline.orchestrator import PipelineOrchestrator
from src.utils.config import load_config
```
```python
# GOOD
@runs_app.command("foo")
def foo_cmd(...) -> None:
    from src.pipeline.orchestrator import PipelineOrchestrator
    from src.utils.config import load_config
    ...
```

### ❌ Shell wrapper around a Typer command

```bash
# BAD — duplicates help, drifts on rename, hides flags
./scripts/inspect-run.sh runs/foo
```
```bash
# GOOD
ryotenkai runs inspect runs/foo
```

### ❌ `from fastapi import HTTPException` in a service

```python
# BAD — couples the service to FastAPI; CLI can't call it
from fastapi import HTTPException
def get_project_detail(...):
    if not entry: raise HTTPException(404, "not found")
```
```python
# GOOD — raise a domain exception, let each peer translate
class ProjectServiceError(RuntimeError): ...
def get_project_detail(...):
    if not entry: raise ProjectServiceError(f"project not found: {pid}")

# router translates → HTTPException(404, str(exc))
# CLI translates    → raise die(str(exc))
```

### ❌ Custom signal handler

```python
# BAD — fights the global handler, double-cleanup races
signal.signal(signal.SIGINT, my_handler)
```
```python
# GOOD
from src.cli import _signals
_signals.set_active_orchestrator(orch)
try:
    orch.run(...)
finally:
    _signals.set_active_orchestrator(None)
```

### ❌ Deprecation aliases for renamed commands

```python
# BAD — every shim is a future-agent tripwire
@app.command("train", hidden=True)  # alias of `run start`
def _train_alias(...): ...
```
```bash
# GOOD — delete the old name, document in README cheat-sheet
# Anyone scripting the old name fails loudly with `No such command`,
# fixes their script in 30 seconds.
```

### ❌ Calling `catalog.ensure_loaded()` for a preset-only read

```python
# BAD — boots strategies factory + every plugin registry (~3 s)
from src.community.catalog import catalog
catalog.ensure_loaded()
presets = catalog.presets()
```
```python
# GOOD — pure walk over community/presets/, ~200 ms
from src.community.loader import load_presets
presets = load_presets().presets
```

---

## Tests

Every new CLI command MUST come with:

1. **Help-smoke** — automatic in `src/tests/smoke/test_cli_help.py`.
   Just make sure your sub-Typer is registered in `_REGISTRY` — the
   parametrised smoke walks the whole tree and asserts `--help` exits 0.
2. **Unit happy + error** — at minimum one CliRunner test for the
   success path and one for the dominant error path. Coverage gate
   is `fail_under = 83` (`pyproject.toml`); keep new files above
   80 % to avoid dragging it down.
3. **Contract test** (only if there's an API pair) — add the pair
   `(cli_argv, api_method, api_path)` to
   `src/tests/contract/test_cli_api_parity.py::PAIRS`. The harness
   calls both, normalises via `_normalize.py` (ISO → UTC, ms stripped,
   list-of-dict sorted by id), asserts deep equality. `clean_state_cache`
   fixture (autouse in `src/tests/contract/conftest.py`) keeps the
   process-local `src.api.state_cache` honest.

### Anti-test-pattern

- ❌ Don't write a test that invokes `python -m src.main <legacy-name>` —
  legacy names are deleted, not aliased.
- ❌ Don't import `from src.cli.community` or `from src.cli.plugin_scaffold` —
  both modules are gone.
- ❌ Don't shell out to `bash run.sh` from a test. `run.sh` is gone.

---

## Cheat sheet: noun → service → module

| Noun | Verb(s) | Service | Notes |
|---|---|---|---|
| `run` | start / resume / restart / interrupt / restart-points | `launch_service`, `launch.restart_options` | write surface; long-running ⇒ uses `_signals` |
| `runs` | ls / inspect / logs / status / diff / report / rm | `run_service`, `delete_service`, `log_service`, `report_service` | read surface; `logs --follow` uses `LiveLogTail` |
| `config` | validate / show / explain / schema | `config_service` | static checks only — no network |
| `dataset` | validate | orchestrator stage 0 (lazy) | refuses configs with no validation plugins (NR-04) |
| `project` | ls / show / use / current / create / rm / env / run | `project_service` + `cli_state.context_store` | `use` writes `~/.ryotenkai/cli-context.json` |
| `plugin` | ls / show / scaffold / sync / sync-envs / pack / validate / install / preflight / stale | `community.{catalog,loader,scaffold,sync,pack,install,validate_manifest,preflight,stale_plugins}` | `install --git` requires pinned SHA unless `--allow-untrusted` |
| `preset` | ls / show / apply / diff | `community.loader.load_presets` + `preset_apply` | bypasses `catalog` — preset-only path stays fast |
| `smoke` | (single command) | `src.cli._smoke_runner` | port of legacy `batch_smoke.py`; preserves `RYOTENKAI_RUNS_DIR` env contract |
| `server` | start / status / stop | `src.api.cli.run_server` | `status` / `stop` reserved for v1.2 daemon mode |
| `version` | (single command) | `src.cli.version` | matches `-V` eager flag output |

---

## Where to read more

- `docs/plans/b-hazy-phoenix.md` — the full refactor plan (target shape,
  phases, 45 risks with resolutions).
- `docs/codestyle/CODE_ERRORS.md` — error/Result rules; CLI follows them
  via `die()` for user-facing exits and `Err()` for service returns.
- `src/cli/app.py` — the canonical root callback; copy its idioms.
- `src/cli/commands/runs.py` — the densest noun module; good template
  for new commands.
