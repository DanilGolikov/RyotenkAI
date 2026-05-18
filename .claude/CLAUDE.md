# CLAUDE.md

## Layout (post Phase B packagization, 2026-05-03)

The codebase has been split from `src/` into 5 uv-workspace members under
`packages/` (per [docs/adrs/2026-05-03-monorepo-uv-workspace-packagization.md](../docs/adrs/2026-05-03-monorepo-uv-workspace-packagization.md)).
Use the `ryotenkai_*` import names — `from src.X` no longer exists.

| Package | Import name | Side | Contents |
|---|---|---|---|
| `packages/shared/` | `ryotenkai_shared` | both | `utils`, `config`, `constants`, `inference`, `infrastructure` (incl. `IMLflowManager`/`IPodLifecycleClient` Protocols) |
| `packages/community/` | `ryotenkai_community` | both | plugin loader / catalog / manifest / sync framework |
| `packages/pod/` | `ryotenkai_pod` | pod | `runner/` (HTTP server) + `trainer/` (subprocess) — one image |
| `packages/providers/` | `ryotenkai_providers` | Mac | `runpod/`, `single_node/`, `inference/vllm/` |
| `packages/control/` | `ryotenkai_control` | Mac | `pipeline`, `api`, `cli`, `data`, `evaluation`, `reports`, `workspace`, `cli_state`, `main.py` |

Allowed dependency directions (importlinter-enforced; `uv run lint-imports`):
shared (leaf) → community → {pod, providers, control}; control may depend on
shared+community+providers but **never** pod; runner ↔ trainer never import
each other (runner spawns trainer via subprocess).

Console scripts: `ryotenkai` (control CLI), `ryotenkai-trainer-run` (pod
trainer entrypoint).

## Agent testing workflow (mandatory)

This codebase is written by agents. The test infrastructure has gates
designed to catch agent-specific failure modes (tautological tests,
over-mocking, "done" claims without coverage). When a subagent (you, or
one you spawn) is about to declare a task **done** after touching
production code, the following MUST happen:

1. **Mutation-testing self-check** for any diff that touches
   `packages/*/src/**/*.py`:
   ```bash
   bash scripts/mutation/validate_agent_output.sh
   ```
   The script auto-detects the integration branch (currently
   `RESEACRH`; falls back to `main`). Override with
   `MUTATION_BASE_REF=<ref>` or pass it explicitly as `$1`.
   Exit 0 = OK. Exit 1 = at least one production file changed in the
   diff has a kill rate below threshold. Strengthen tests until it
   passes, then declare done.

2. **Adding `@pytest.mark.xfail(strict=True, ...)`** REQUIRES adding a
   matching `xfail-debt:<id>` token to the `reason=` text AND a matching
   row in `docs/migration/xfail_debt.md`. The
   `tests/_lint/test_xfail_debt_completeness.py` sentinel will block
   the PR otherwise. Emergency unblock: use token
   `xfail-debt:ad-hoc-<YYYYMMDDTHHMMSS>` (auto-accepted; resolve within
   30 days).

3. **No mocking of Protocols.** Sentinel
   `tests/_lint/test_no_protocol_mocking.py` enforces. Use a canonical
   fake from `tests/_fakes/` (extend or add one if needed).

4. **No new files under `packages/*/src/`** without a corresponding
   test file. Sentinel
   `tests/_lint/test_every_module_has_tests.py` enforces; legitimate
   exemptions go into `tests/_lint/no_test_required.yaml`.

5. **Reading the policy docs** before non-trivial test changes:
   - [docs/testing/mock_policy.md](../docs/testing/mock_policy.md)
   - [docs/testing/mutation_testing.md](../docs/testing/mutation_testing.md)
   - [docs/migration/xfail_debt.md](../docs/migration/xfail_debt.md)

## Event system (typed envelopes + SSOT journal)

The pipeline emits typed events via the `IEventEmitter` Protocol. The
single source of truth is a length-prefixed JSONL journal at
`workspace/runs/<run_id>/events.jsonl`; a sha256-checked copy is
uploaded to MLflow on run finalize. See:

- ADR-0009: [docs/adrs/2026-05-17-unified-event-system.md](../docs/adrs/2026-05-17-unified-event-system.md)
- shared/events README: [packages/shared/src/ryotenkai_shared/events/README.md](../packages/shared/src/ryotenkai_shared/events/README.md)
- control/events README: [packages/control/src/ryotenkai_control/events/README.md](../packages/control/src/ryotenkai_control/events/README.md)

How to emit from a stage:

```python
with self._emitter.stage_scope("training_monitor"):
    self._emitter.emit(TrainingMonitorStartedEvent(
        source="control://orchestrator/training_monitor",
        run_id=self._run_id, offset=0, payload=...,
    ))
```

When you add a new event type:

- Pick a `kind` matching `ryotenkai.<area>.<domain>.<verb>`; pin `kind`
  and `severity` via `Literal` defaults on the event class.
- Register the class in `packages/shared/.../events/discriminator.py`
  (the `Union[...]` block) AND re-export it from `types/__init__.py`.
- Add a test file under `tests/unit/shared/events/types/` covering
  positive round-trip, `extra="forbid"` rejection, and the `kind` /
  `severity` Literal pins (the `test_every_module_has_tests` sentinel
  enforces the file's existence).

Do NOT mock `IEventEmitter` in tests — use the canonical fake at
`tests/_fakes/event_emitter.py` (the `test_no_protocol_mocking`
sentinel enforces this).

<!-- Add your custom instructions below. Repowise will never modify anything outside the REPOWISE markers. -->
<!-- Examples: coding style rules, test commands, workflow preferences, constraints -->

<!-- REPOWISE:START — Do not edit below this line. Auto-generated by Repowise. -->
## IMPORTANT: Codebase Intelligence Instructions for RyotenkAI

> This repository is indexed by [Repowise](https://repowise.dev).
> Use the MCP tools below for orientation, discovery, and enriched context
> (documentation, ownership, history, decisions). **Always verify against
> actual source files before making changes** — the index may be stale.

Last indexed: 2026-05-18 (commit 3ae4ffa). Confidence: 100%.
### Architecture
repo is a robust, Python-centric monorepo designed to manage complex workflows, likely involving machine learning orchestration, evaluation, and system control. With over 378,000 lines of code across 1,846 files, the project is structured into modular packages that handle distinct responsibilities ranging from community-driven evaluations to core system control and pod management. The repository emphasizes shared utilities and standardized event handling to maintain consistency across its distributed components. The project is primarily built with Python, which accounts for ~78% of the codebase, supported by a TypeScript/JavaScript frontend.
### Key Modules
| Module | Purpose | Owner |
|--------|---------|-------|
| `src` | The src module serves as the core engine for the RyotenkAI training pipeline | — |
| `docker` | The docker module serves as a specialized utility package focused on environment | — |
| `scripts` | The scripts module serves as a centralized repository for automation and quality | — |
| `web` | The web module serves as the frontend configuration and tooling root for the pro | — |
| `community` | The community module serves as an extensible integration layer for third-party e | — |
| `packages` | The packages module constitutes the core monorepo structure for the Ryotenkai ec | — |
| `tests` | The tests module serves as the comprehensive testing suite for the Ryotenkai eco | — |
### Entry Points
- `web/src/App.tsx`
- `src/cli/app.py`
- `src/api/main.py`
- `community/evaluation/cerebras_judge/plugin/main.py`
- `src/reports/__main__.py`
- `src/api/schemas/run.py`
- `src/main.py`
- `src/cli/commands/run.py`
- `docker/mlflow/start.sh`
- `src/cli/commands/server.py`
### Tech Stack
**Languages:** Python
**Frameworks:** Pydantic

### Hotspots (High Churn)
| File | Churn | 90d Commits | Owner |
|------|-------|-------------|-------|
| `src/pipeline/orchestrator.py` | 100.0th %ile | 45 | daniil |
| `src/pipeline/stages/training_monitor.py` | 100.0th %ile | 28 | daniil |
| `src/tests/unit/pipeline/test_training_monitor_v2.py` | 99.9th %ile | 16 | daniil |
| `web/src/api/openapi.json` | 99.9th %ile | 14 | daniil |
| `src/pipeline/stages/managers/deployment_manager.py` | 99.9th %ile | 16 | daniil |

### Repowise MCP Tools

This project has a Repowise MCP server configured. These tools provide documentation, ownership, architectural decisions, and risk signals. Use them for orientation and discovery — then read actual source to verify before editing.

**Recommended workflow:**

1. Start with `get_overview()` on a new task to orient yourself.
2. Call `get_context(targets=["path/to/file.py"])` for enriched context on unfamiliar files — but always read the source before editing.
3. Call `get_risk(targets=["path/to/file.py"])` before changing hotspot files.
4. Don't know where something lives? Call `search_codebase(query="authentication flow")`.
5. Need to understand why code is structured a certain way? Call `get_why(query="why JWT over sessions")` before architectural changes.
6. After **architectural changes**, consider calling `update_decision_records(action="create", ...)` to record the rationale.
7. Need to understand how two modules connect? Call `get_dependency_path(source="src/auth", target="src/db")`.
8. Before cleanup tasks, call `get_dead_code()` to find confirmed unused code.
9. For documentation or diagrams, call `get_architecture_diagram(scope="src/auth")`.

**Note:** MCP tool responses reflect the last index run. If the index is stale, verify against source files.

| Tool | When to use |
|------|-------------|
| `get_overview()` | Orient yourself on a new task |
| `get_context(targets=[...])` | Enriched context on unfamiliar files |
| `get_risk(targets=[...])` | Before changing hotspot files |
| `get_why(query="...")` | Before architectural changes |
| `update_decision_records(action=...)` | After architectural changes — record decisions |
| `search_codebase(query="...")` | When locating code |
| `get_dependency_path(source=..., target=...)` | When tracing module connections |
| `get_dead_code()` | Before any cleanup or removal |
| `get_architecture_diagram(scope=...)` | For visual structure or documentation |

### Codebase Conventions
**Commands:**
- Test: `pytest`
- Lint: `ruff check .`
- Format: `ruff format .`
- Typecheck: `mypy .`

<!-- REPOWISE:END -->
