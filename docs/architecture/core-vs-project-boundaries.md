# Core ↔ Project boundary (Variant 1, Hexagonal)

> **TL;DR.** The pipeline core is **pure**: it accepts
> ``(config, env, metadata)`` and runs. **Project** is one of several
> *adapters* — a UX-level concept that translates a user's project
> directory into that triple. Other adapters (CI, ad-hoc YAML, future
> remote runners) coexist with Project; none of them is privileged.

This document is the canonical reference for the boundary introduced
by the 9-step Variant 1 refactor (commits ``Step 1`` through
``Step 9`` on the ``RESEACRH`` branch). The same picture lives in
the plan at ``docs/plans/robust-rolling-dolphin.md`` (the **why**);
this file is the steady-state contract (the **what**).

---

## 1. The picture

```
┌────────────────────────────────────────────────────────────┐
│  CALLERS                                                    │
│   • Web UI         • CLI human       • Agent (Claude Code)  │
│   • CI / cron      • Tests           • Future remote runners│
└────────────────────────────────────────────────────────────┘
                         │
            ┌────────────┴────────────┐
            ▼                         ▼
   ┌─────────────────┐       ┌────────────────────┐
   │ PROJECT ADAPTER │       │ AD-HOC PATH        │
   │ (one shape)     │       │  -c X.yaml         │
   │                 │       │  No env merge.     │
   │ load_project_   │       │  metadata={}       │
   │   inputs(id) →  │       │                    │
   │   ProjectInputs │       └────────────────────┘
   └────────┬────────┘                 │
            └───────────┬───────────────┘
                        ▼
        ProjectInputs(config, env, metadata)
                        │
                        ▼
   ┌────────────────────────────────────────────┐
   │  CORE  (pure orchestrator)                  │
   │                                              │
   │  PipelineOrchestrator(                       │
   │     config:   PipelineConfig,                │
   │     env:      dict[str,str] | None,          │
   │     metadata: dict[str,Any]  | None,         │
   │     run_directory, settings,                 │
   │  )                                           │
   │                                              │
   │  Knows: config schema, stages, MLflow,       │
   │         integration resolver, run state.     │
   │  Does NOT know: projects, env.json, history. │
   └────────────────────────────────────────────┘
                        │
                        ▼
              ~/.ryotenkai/runs/<id>/
              metadata.project_id stamped if caller gave it
```

### 1.1. Layer responsibilities

| Layer            | Knows                                                            | Does **not** know                                       |
|------------------|------------------------------------------------------------------|---------------------------------------------------------|
| **Core**         | PipelineConfig schema, stages, integration resolver, run state, MLflow lifecycle | Projects, ``env.json``, history snapshots, Web UI       |
| **Project**      | ``~/.ryotenkai/projects/<id>/`` filesystem, ``env.json``, history, datasets manifest, integrations linking | Stages, MLflow runtime, secrets resolution              |
| **Caller**       | How to assemble (or skip) the adapter and feed the core          | Internals of either                                     |

---

## 2. The contracts

### 2.1. Core entry point — `PipelineOrchestrator`

```python
class PipelineOrchestrator:
    def __init__(
        self,
        config_path: Path | None = None,        # legacy back-compat shim
        run_directory: Path | None = None,
        settings: RuntimeSettings | None = None,
        *,
        config: PipelineConfig | None = None,   # Variant 1 keyword shape
        env: dict[str, str] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None: ...
```

Caller MUST supply exactly one of ``{config, config_path}``. The
keyword shape is the canonical Variant 1 path; the positional
``config_path`` shim exists during the migration window and can be
removed once every consumer has migrated (see §6).

### 2.2. Adapter entry point — `load_project_inputs`

```python
@dataclass(frozen=True, slots=True)
class ProjectInputs:
    config: PipelineConfig                    # already resolved
    env: dict[str, str]                       # project env.json
    metadata: dict[str, Any]                  # invariant keys + extras

def load_project_inputs(
    project_id: str,
    *,
    config_override: Path | None = None,
    actor: str | None = None,
    extra_metadata: dict[str, Any] | None = None,
    registry: ProjectRegistry | None = None,
) -> ProjectInputs: ...
```

Lives in ``src/workspace/projects/adapter.py``. Pure: no env
mutation, no orchestrator construction, no side effects beyond
filesystem reads.

### 2.3. Metadata invariants

The adapter always stamps three keys; callers passing
``extra_metadata`` cannot override them:

| Key                       | Source                                                        |
|---------------------------|---------------------------------------------------------------|
| ``project_id``            | the ``project_id`` argument                                   |
| ``actor``                 | explicit > ``RYOTENKAI_ACTOR`` > ``$USER`` > ``"unknown"``    |
| ``config_version_hash``   | sha256 of the YAML file's bytes (literal, not resolved tree)  |

When ``config_override`` is given, ``config_override_path`` is also
stamped as a breadcrumb.

The orchestrator forwards every key in ``metadata`` to MLflow as a
tag under the ``meta.`` prefix (e.g. ``meta.project_id``,
``meta.actor``). Truncation kicks in at MLflow's 5000-char limit.

### 2.4. Integration resolver — `resolve_yaml_integrations(raw)`

```python
# src/workspace/integrations/resolver.py
def resolve_yaml_integrations(
    raw: dict,
    *,
    registry: IntegrationRegistry | None = None,
) -> dict: ...

# src/workspace/integrations/loader.py
def load_pipeline_config(path: Path) -> PipelineConfig: ...
```

Project YAMLs use the convenience shorthand
``integrations.mlflow.integration: <id>`` to pull the
tracking URI / TLS bundle / system-metrics knobs from a saved
Settings integration. That substitution is a **UX-layer concern** —
it happens BEFORE the YAML reaches core's
:class:`PipelineConfig` validation. Core schema knows nothing about
``~/.ryotenkai/integrations/``.

Workflow:

```
yaml.safe_load(path)              ← raw dict
  └→ resolve_yaml_integrations    ← UX layer: inline integration values
       └→ PipelineConfig(**dict)  ← core: pure validation
```

Merge policy: integration provides defaults, project-side keys win
on conflict. The ``integration:`` field is preserved as a
**secrets-tag** in the resulting dict so runtime code can call
``secrets.get_provider_token(cfg.integration)``.

The two errors raised here are caught by the CLI top-level handler
and rendered as clean ``die()`` messages:
- ``IntegrationNotFoundError`` — id not in registry
- ``IntegrationUnresolvedError`` — found but unfit (empty
  ``current.yaml``, schema mismatch, type mismatch)

Use ``load_pipeline_config(path)`` (UX-layer convenience) to do
``yaml.safe_load → resolve → validate`` in one call. CLI's
anonymous and project paths both go through it; legacy core
``load_config(path)`` stays as a pure parse-and-validate helper for
callers that don't use integration shortcuts.

### 2.5. Secrets layer — `load_secrets(env=...)`

```python
def load_secrets(
    env_file: str | Path | None = None,
    *,
    env: Mapping[str, str] | None = None,
) -> Secrets: ...
```

* ``env=None`` (default): historical behaviour — BaseSettings reads
  ``os.environ``; project-root ``secrets.env`` is auto-discovered.
* ``env=<mapping>``: the mapping replaces ``os.environ`` as the env
  layer. Project-root auto-discovery is suppressed (the caller's
  mapping is authoritative).

Variant 1 callers assemble ``process_env ∪ project_env.json`` and
pass it explicitly so per-project overrides take effect without
mutating ``os.environ``.

---

## 3. Caller scenarios

### 3.1. CLI

```bash
ryotenkai run start -c X.yaml          # ad-hoc, anonymous
ryotenkai run start -p A                # project A, project's current.yaml
ryotenkai run start -c Y.yaml -p A      # project A's env+metadata, Y as override
ryotenkai run start                     # falls back to `project use` context
RYOTENKAI_PROJECT=A ryotenkai run start # equivalent to --project A
```

Precedence for the project ID:

```
--project flag  >  RYOTENKAI_PROJECT env  >  cli_state.context_store
```

### 3.2. Web API

* ``POST /projects/{id}/launch`` (existing endpoint) — the launcher
  spawns a subprocess with merged env. The Variant 1 in-process path
  is reserved for the agent / future remote runners; subprocess
  isolation stays the production default.
* ``GET /projects/{id}/runs`` — surfaces the project's append-only
  ledger (Step 6).

### 3.3. Agent / future runners

Agents call ``load_project_inputs`` directly, then either invoke
``PipelineOrchestrator`` in-process (today) or hand the inputs to a
future remote-runner contract (out of scope here).

---

## 4. Filesystem layout

```
~/.ryotenkai/
├── projects.json                     # registry index
├── projects/<id>/
│   ├── project.json
│   ├── configs/
│   │   ├── current.yaml              # config source for adapter
│   │   └── history/<iso>.yaml
│   ├── env.json                      # → ProjectInputs.env
│   └── runs/
│       └── <run_id>/                 # full pipeline run dir
│           ├── pipeline_state.json   # state.metadata.project_id stamped
│           ├── run.lock
│           └── attempts/attempt_N/
└── integrations/<id>/
    ├── integration.json
    ├── current.yaml                  # resolved into core config
    └── token.enc

# Anonymous (no --project) runs land at:
$RYOTENKAI_RUNS_DIR/<run_id>/         # default: ./runs/<run_id>/
```

Project runs live **inside the project's own workspace**. There is no
separate ledger file: the project's run list is just `os.listdir(<project>/runs/)`.
``metadata.project_id`` in each run's ``pipeline_state.json`` is the
audit-trail tag (also propagated to MLflow as ``meta.project_id``).
``project rm`` naturally drops the runs alongside the rest of the
workspace.

Anonymous CLI runs (``ryotenkai run start -c X.yaml`` without
``--project``) still land in ``$RYOTENKAI_RUNS_DIR`` (default
``./runs/``) — they have no project to live inside.

---

## 5. What changed vs. pre-Variant-1

| Before (pre-refactor)                                  | After (Variant 1)                                                   |
|--------------------------------------------------------|---------------------------------------------------------------------|
| Orchestrator took ``config_path: Path``               | Orchestrator takes ``config: PipelineConfig`` (path shim kept)      |
| Web-API mutated subprocess env via ``env_for_run_dir`` | Adapter returns explicit env mapping; caller decides how to use it  |
| Integration refs unresolvable (resolver promised, missing) | ``src/config/integrations/resolver.py`` runs inside ``load_config`` |
| ``PipelineState`` had no project lineage              | ``PipelineState.metadata`` stamps ``project_id`` / ``actor`` / hash |
| ``run start --project`` was a TODO                    | Goes through ``load_project_inputs`` → orchestrator                 |
| Project-launched runs untraceable from the project    | ``runs/index.json`` ledger + ``GET /projects/{id}/runs``            |
| MLflow tags never reflected the project context       | Every ``metadata.*`` key proxied to ``meta.<key>``                  |
| Frontend Runs tab was a placeholder                   | Live list backed by the project ledger                              |

---

## 6. Open follow-ups (deliberately deferred)

1. **Drop the ``config_path`` shim.** After every internal caller has
   migrated to the keyword shape, remove the legacy positional arg
   from ``PipelineOrchestrator.__init__``. Each caller migrates by
   passing ``load_config(path)`` outside, then ``config=cfg``.
2. **Migration tool for legacy YAMLs.** Inline ``tracking_uri:`` blocks
   continue to be rejected with a hint pointing at the integration
   format. A ``ryotenkai config migrate`` command that auto-creates
   integrations and rewrites the YAML is a separate plan.
3. **Audit log per project.** ``audit.log`` JSONL alongside
   ``env.json`` for high-fidelity provenance. Specced but unbuilt.
4. **Concurrency lock for ``runs/index.json``.** Atomic-write is
   sufficient for typical "one launch per process" load. If we see
   real interleaving, add ``fcntl.flock`` around the read-mutate-write
   triple.
5. **Frontend Launch button.** Currently the frontend reads the
   ledger but doesn't trigger launches; that lives behind
   ``POST /projects/{id}/launch`` plus a button on the Runs tab.

---

## 7. Test surface

The contract is pinned by these test files (run them when changing the
boundary):

* ``src/tests/unit/config/integrations/test_resolver.py`` — Step 1
* ``src/tests/unit/pipeline/state/test_metadata_field.py`` — Step 2
* ``src/tests/unit/pipeline/test_orchestrator_boundary.py`` — Step 3
* ``src/tests/unit/config/test_secrets_loader_env_param.py`` — Step 4
* ``src/tests/unit/workspace/projects/test_adapter.py`` — Step 5
* ``src/tests/unit/workspace/projects/test_runs_index.py`` — Step 6
* ``src/tests/integration/api/test_project_runs.py`` — Step 6
* ``src/tests/unit/cli/test_run_project_wiring.py`` — Step 8
* ``web/src/components/ProjectTabs/RunsTab.test.tsx`` — Step 7

Total: 9 files, ~180 tests. Categories covered: positive, negative,
boundary, invariants, dependency-error, regression, logic-specific,
combinatorial.

---

## 8. Quick reference

> "How do I add a new caller?"

```python
from src.workspace.projects.adapter import load_project_inputs
from src.pipeline.orchestrator import PipelineOrchestrator

inputs = load_project_inputs("my-project", actor="my-runner")
orch = PipelineOrchestrator(
    config=inputs.config,
    env=inputs.env,
    metadata=inputs.metadata,
)
result = orch.run()
```

Or skip the adapter for an ad-hoc run:

```python
from src.utils.config import load_config

config = load_config("path/to/pipeline.yaml")
orch = PipelineOrchestrator(config=config)
result = orch.run()
```

That's the contract. Both paths terminate in the same core; the
adapter exists so the project's UX affordances (``env.json``,
metadata, the ledger) don't bleed into core.
