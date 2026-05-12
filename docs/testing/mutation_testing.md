# Mutation Testing Policy

> Status: **enforced (advisory mode)** via `.github/workflows/mutation-pr.yml`
> and `.github/workflows/mutation-nightly.yml`
> Last updated: 2026-05-12 (Phase 7 of the agent-driven testing strategy)
> Related: [mutation_testing_report.md](../migration/mutation_testing_report.md),
> [mock_policy.md](mock_policy.md)

## TL;DR

1. **Mutation testing is mandatory** for every PR that touches
   production code (`packages/*/src/*.py`).
2. **Hotspot files** listed in [`.mutation-hotspots.yml`](../../.mutation-hotspots.yml)
   have **hard kill-rate thresholds** (0.65-0.70). Non-hotspot files
   get an **advisory warning** below `default_kill_rate_warning` (0.50).
3. The per-PR job runs Cosmic Ray on the **diff only** — typically
   1-5 files, <25 min wall clock.
4. The nightly job runs **all hotspots** and enforces a **5-percentage-
   point ratchet** (no regression > 5pp on any baselined module).
5. **PEP 604 noise is filtered** at init time via
   `cr-filter-operators` (drops every `BitOr_*` operator). Real-signal
   kill rates are reported in `effective_kill_rate`.
6. Mode is currently **advisory** — failures do NOT block merges yet.
   Flip `mode: blocking` in `.mutation-hotspots.yml` once the
   thresholds are stable.

## Why mutation testing matters in agent-driven dev

100 % of the production code in this repository is written by AI
agents (Claude). Conventional CI catches the easy failure modes —
syntax, types, line coverage — but it cannot detect:

- **Tautological tests** (`assert x == x`)
- **Over-mocking** that bypasses real production logic
- **Cargo-culted assertions** that always pass regardless of input
- **"Done" claims without actual work** (we've seen this multiple
  times in the migration)

Mutation testing is the **external validator** that breaks the closed
loop of "agent writes code + agent writes tests + agent self-verifies".
Without it, test quality silently decays.

The mathematical statement of the gate:

> For every committed production file, the unit test suite must catch
> at least `min_kill_rate` of the small, semantically-meaningful edits
> Cosmic Ray makes to that file — otherwise the tests are not
> actually exercising the file's behaviour.

## The three tiers

| Tier | Trigger | Scope | Action |
|---|---|---|---|
| **1. Per-PR incremental** | every PR to `main` | changed files in `packages/*/src/` only | **BLOCK merge** (when `mode: blocking`) if hotspot kill rate < threshold; **WARN** for non-hotspots |
| **2. Hotspot hard gate** | every PR that touches a hotspot file | listed in `.mutation-hotspots.yml` | **HARD BLOCK** if kill rate < threshold (subset of tier 1) |
| **3. Nightly full ratchet** | cron daily 03:00 UTC | top-N hotspots | **ADVISORY** + ratchet (kill rate must not drop > 5pp vs baseline) |
| **4. Agent output gate** | subagent self-validates BEFORE declaring "done" | files the agent touched | manual script call; not CI-enforced |

### Tier 1 + 2 (`.github/workflows/mutation-pr.yml`)

- Triggers on every `pull_request` to `main`.
- Computes the diff vs `origin/main`.
- For each changed `packages/*/src/*.py`:
  - If listed as a hotspot: enforce the file's `min_kill_rate`.
  - Otherwise: warn-only against `default_kill_rate_warning`.
- A check annotation is left on the PR with per-file numbers.
- **Time budget**: 30 minutes hard cap. Files that would push the run
  past 25 minutes are skipped with a helpful message.

### Tier 3 (`.github/workflows/mutation-nightly.yml`)

- Cron `0 3 * * *` (daily 03:00 UTC).
- Runs every entry in `.mutation-hotspots.yml`.
- Runs `scripts/mutation/check_ratchet.py` to enforce the 5pp
  no-regression rule against
  `scripts/mutation/ratchet_baseline.json`.
- Time budget: 180 minutes.

### Tier 4 (`scripts/mutation/validate_agent_output.sh`)

Subagents call this **before** declaring a task done:

```bash
bash scripts/mutation/validate_agent_output.sh main
```

Exit 0 means every changed production file met its threshold. Exit 1
means something is below — the agent should write additional tests
that kill the surviving mutations before pushing the PR.

The script forces `--strict` so advisory-mode warnings still fail
locally; CI is the authoritative gate but this catches problems
earlier.

## How kill rate is computed

Cosmic Ray classifies each mutation:

- **KILLED** — at least one test fails when the mutation is applied.
  This is the desired outcome.
- **SURVIVED** — every test still passes. The test suite did not
  catch the bug.
- **INCOMPETENT** — the mutation produced invalid Python (e.g., an
  AST that wouldn't compile). Excluded from the denominator.
- **TIMEOUT** — the mutated code hung. Counted as not-killed.
- **SKIPPED** — filtered out by `cr-filter-operators` (PEP 604 noise)
  or `cr-filter-pragma` (`# pragma: no mutate`). Excluded entirely.

The reported metric is:

```
effective_kill_rate = KILLED / (KILLED + SURVIVED + TIMEOUT)
```

`mutations_total` is the count Cosmic Ray queued *before* filtering;
`mutations_executed` is what actually ran.

### How to interpret the numbers

| effective_kill_rate | meaning | action |
|---|---|---|
| < 30 % | low signal — tests barely exercise the file | usually means heavy over-mocking; rewrite tests with fakes |
| 30-50 % | tests run but big gaps | add targeted tests for each surviving mutation |
| 50-70 % | acceptable for non-hotspots | improve incrementally |
| 70-80 % | strong | maintain; tighten threshold next cycle |
| > 80 % | excellent | ratchet up the floor |

## The PEP 604 noise problem (and how we solve it)

Cosmic Ray's core operators include `ReplaceBinaryOperator_BitOr_*`
(replace `|` with `+`, `-`, `*`, etc.). PEP 604 type-union
annotations also use `|` — `int | None`, `Foo | Bar | None`. Mutating
those is a no-op at runtime because Python does not evaluate
annotations for behaviour. Pre-filter, ~30-50 % of "survivors" on
this codebase were pure annotation noise.

**Solution**: the Cosmic Ray TOML written by
`scripts/mutation/orchestrate.py` includes:

```toml
[cosmic-ray.filters.operators-filter]
exclude-operators = [
    "core/ReplaceBinaryOperator_BitOr_.*",
    "core/ReplaceBinaryOperator_.*_BitOr",
]
```

After `cosmic-ray init`, the orchestrator runs `cr-filter-operators`
which marks every matching mutation as `SKIPPED`. They never execute
and never count against the kill rate.

If a future change uses real bitwise-or (`flags = A | B`), the filter
will incorrectly skip a real mutation. Until that becomes common, we
accept the trade-off. The fallback is a manual review of the
`mutations_skipped` field in the JSON report; if it's >50 % of
`mutations_total` for a non-flag-heavy module, something is wrong.

## How to handle false positives

Three options, in order of preference:

1. **`# pragma: no mutate`** on the specific line. Honoured by
   `cr-filter-pragma`.
2. **Add the operator to the per-target exclude list** in the rewritten
   TOML (see `scripts/mutation/orchestrate.py::_write_config`). Only
   do this if the operator generates noise on every module.
3. **Add a test that kills the mutation.** Always the best answer
   when the mutation actually represents a behaviour change.

## How to add a hotspot

Append to `.mutation-hotspots.yml`:

```yaml
hotspots:
  - path: packages/<pkg>/src/<...>.py
    min_kill_rate: 0.65  # start conservative
    tests: tests/unit/<pkg>/<...>
    rationale: "why this is critical"
```

The next nightly run will baseline the file. From the run after,
the 5pp ratchet applies.

## How to raise a threshold over time

1. Run `scripts/mutation/check_ratchet.py` against the latest report.
2. If it reports improvements above the `--improvement` threshold,
   bump `min_kill_rate` for the relevant hotspot.
3. Commit the change in a PR with the updated baseline.

## Agent workflow

Subagents touching production code MUST:

1. Write tests that kill the obvious mutations (boundary conditions,
   constants, branch directions).
2. Run `bash scripts/mutation/validate_agent_output.sh` before
   declaring "done".
3. If the gate fails, read the JSON report under
   `scripts/mutation/reports/agent_*.json`, identify surviving
   mutations, add or strengthen tests, and re-run.

This is not optional. The CI gate (tier 1) will catch violations
either way, but running locally is faster and avoids burning a CI
iteration on a known-bad PR.

## Scripts reference

| Script | Purpose |
|---|---|
| `scripts/mutation/orchestrate.py` | Core orchestrator. `--diff`, `--files`, `--all-hotspots`. Writes JSON report. |
| `scripts/mutation/run_on_diff.sh` | Thin wrapper for the per-PR job (tier 1). |
| `scripts/mutation/run_full.sh` | Thin wrapper for the nightly job (tier 3). |
| `scripts/mutation/check_ratchet.py` | Compare current report to `ratchet_baseline.json`. |
| `scripts/mutation/validate_agent_output.sh` | Agent self-validation (tier 4). Forces `--strict`. |
| `scripts/run_mutation_testing.sh` | Original Phase 6 driver. Kept for ad-hoc full runs. |

## Configuration files

| File | Purpose |
|---|---|
| `cosmic-ray.toml` | Reference template; overwritten per-target by the orchestrator. |
| `.mutation-hotspots.yml` | Hotspot list with thresholds, test targets, mode (advisory/blocking), and budgets. |
| `scripts/mutation/ratchet_baseline.json` | Committed baseline kill rates; updated explicitly via `check_ratchet.py --update-baseline`. |

## Artifacts

Per-run artifacts (uploaded by both workflows):

- `scripts/mutation/sessions/<slug>.sqlite` — raw Cosmic Ray session DB
- `scripts/mutation/reports/<id>.json` — structured kill-rate report

Both are retained 14 days (PR) / 90 days (nightly).

## When to break glass

If a PR is blocked by mutation testing and you genuinely cannot kill
the surviving mutations (e.g., the mutation reflects a
deliberately-untested error path), the escape hatch is:

1. Add `# pragma: no mutate` to the specific line, with a comment
   explaining why.
2. Open a follow-up issue documenting the gap.

Do NOT lower the hotspot threshold to make a single PR pass.
