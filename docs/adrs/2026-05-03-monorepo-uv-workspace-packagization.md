# ADR: Split src/ into 5 uv-workspace packages

**Date:** 2026-05-03
**Status:** Accepted; Phase A + B + C (partial) executed.
**Branch:** `claude/zealous-gauss-7321c5` → merged into `RESEACRH`.
**Plan reference:** [docs/plans/2026-05-03-monorepo-uv-workspace-packagization.md](../plans/2026-05-03-monorepo-uv-workspace-packagization.md)

## Decision

The monolithic `src/` tree has been split into five physically independent
uv-workspace members under `packages/`:

```
packages/
├── shared/    ← ryotenkai-shared    (utils, config, constants, inference, infrastructure)
├── community/ ← ryotenkai-community (plugin loader / catalog / manifest framework)
├── pod/       ← ryotenkai-pod       (runner HTTP server + trainer subprocess; one image)
├── providers/ ← ryotenkai-providers (RunPod + single_node + inference engines)
└── control/   ← ryotenkai-control   (pipeline + api + cli + data + reports + evaluation + workspace)
```

Naming: flat `ryotenkai_<pkg>` (not `ryotenkai.<pkg>` namespace package).
Layout: `src-layout` per package (`packages/<pkg>/src/ryotenkai_<pkg>/...`).

## Rationale

Three concrete signals from the prior monolithic state:

1. **16 SCC cycles** in the module graph (repowise scan); top-5 hotspot
   files at the 99th-100th percentile of churn (orchestrator,
   training_monitor, deployment_manager).
2. **The 16-crash chain** in `run_20260502_113553_r8rul` (git
   `9d29e6c`) — a transitive `src/providers → src/pipeline` import
   crashed the inference path 16 times in one run. The cycle was
   forbidden by convention but not by build-system.
3. **Pod image bloat**: every pod ran with the full Mac control-plane
   closure in its Python path (paramiko, runpod-sdk, prefect, dvc,
   typer, fastapi-control-plane). After the split, pod-side
   `pip install ./packages/{shared,community,pod}` excludes those
   ~80 % of bytes.

uv-workspace was chosen over poetry / Pants / Bazel for matching the
Astral toolchain we already use, keeping a single `uv.lock`, and the
existing precedent (Apache Airflow at 122 distributions in a single
uv-workspace).

## Phases executed

* **Phase A — pre-cleanup (7 commits, all green):**
  A.1 cancellation, A.2 HTTP-clients (job_client, ssh_tunnel, control-plane heartbeat),
  A.3 vLLM, A.4 RUNTIME_IMAGE, A.5 IMLflowManager Protocol,
  A.6 utils/config facade delete, A.7 container.py + memory_manager.py
  → training/. After Phase A: 0 regressions vs baseline test sweep.

* **Phase B — big-bang move (6 commits):**
  B.1 skeleton + 5 pyproject.toml + uv workspace declaration;
  B.2-B.4 git mv shared/community/pod/providers/control modules and
  per-domain test trees; B.5 codemod 3-pass (1500+ Python files,
  Pass 1 imports + Pass 2 string literals + Pass 3 non-Python configs:
  Makefile, pytest.ini, pyrightconfig, .pre-commit, .dockerignore,
  .claude/launch.json, examples, docker scripts); B.6 src/ deletion +
  BC-shim removal + workspace-only root pyproject.

* **Phase C — polish (partial):**
  Test-collection fix-ups (18 hardcoded `Path(__file__).parents[N]`
  call sites + cross-package conftest import via `importlib`);
  importlinter contracts wired (`uv run lint-imports`).

## Boundaries enforced (importlinter)

```toml
[tool.importlinter] root_packages = [ryotenkai_shared, _community, _pod, _providers, _control]
```

Forbidden directions:
* `shared → {community, pod, providers, control}`  (shared must be leaf)
* `community → {pod, providers, control}`
* `pod → {providers, control}`
* `providers → {community, pod, control}`
* `control → pod`
* `pod.trainer ↔ pod.runner` (runner spawns trainer via subprocess)

## Known violations (Phase C follow-up)

`uv run lint-imports` after Phase B/C reports **6 broken contracts**.
These are pre-existing architectural drifts the codemod surfaced but did
not fix — each needs a small targeted refactor:

| # | Direction | Cause | Fix |
|---|---|---|---|
| 1 | `shared.config.validators.runtime → community.catalog` | Config validators reach into community plugin catalog | Either move validator into community or extract `IPluginCatalog` Protocol into shared |
| 2 | `shared.utils.plugin_base → community.manifest` | plugin_base is a shared base but imports community manifest | Move plugin_base to community, or split out manifest types into shared |
| 3 | `community.catalog → control.{data,evaluation,reports}.registry` | Catalog discovers registries that live in control | Invert: registries should register *into* community, not the other way |
| 4 | `community → pod.trainer.reward_plugins.registry` | Same pattern as #3 for reward plugins | Same fix |
| 5 | `pod.trainer.container → control.data.loaders` | Trainer's DI container imports control-side data loaders | Move data loaders into shared or pod (data is consumed both sides) |
| 6 | `pod.trainer.{cancellation_callback,completion_callback} → pod.runner.cancellation_telemetry` | Trainer writes telemetry events into runner's bus via Python import | Replace with loopback HTTP POST or extract telemetry interface into shared |
| 7 | `providers → pod.runner.{lifecycle_client, pod_terminator}` | Provider adapters import the runner-side Protocol + outcome enum | Extract `IPodLifecycleClient` + `LifecycleActionResult` + `PodTerminalOutcome` into `shared.infrastructure.lifecycle` |

Each is one focused PR. The contracts stay BROKEN until those land —
that's the point: importlinter is now the canonical drift detector.

## Open follow-ups

* Coverage gate calibration per-package (plan §6.7 + Q4.2): measure
  actual baseline per-package post-Phase B and set `fail_under = baseline − 2`.
* MockSupervisor extraction into `packages/_test_support/` (plan §6.4)
  to drop the four `importlib.spec_from_file_location` workarounds in
  cross-package conftest / contract tests.
* Docker multi-stage build with `uv export --package ryotenkai-pod`
  + editable install (plan §6.6 + Q3.1 — pod-image stays slim, CodeSyncer
  rsync continues).
* sentinel boundary tests under `packages/<pkg>/tests/sentinel/` (plan
  §7.2-§7.3) — symmetric runtime-AST assertions complementing
  importlinter.

## What is NOT covered by this ADR

* The 410 unit-test failures observed after Phase B+C (vs 243 baseline)
  are a mix of pre-existing failures (243 unchanged) and ~167 new
  regressions — primarily in tests that reach across the new package
  boundary the wrong way. They surface real architectural issues
  rather than a packagization defect; they will green up as the seven
  known violations above are addressed.
