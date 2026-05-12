# Lighthouse CI budgets — Phase 6 rationale

Lighthouse runs against the static Storybook bundle, not the SPA.
Reasoning: the SPA needs the control-plane API + auth to render; the
Storybook static build is a self-contained, deterministic site we can
serve from CI with zero infra. Component-level perf is a reasonable
proxy for app-level perf during phase 6.

## Budgets in [lighthouserc.json](./lighthouserc.json)

| Category / Metric | Budget | Severity | Rationale |
|---|---|---|---|
| `performance` | ≥ 0.80 | warn | Storybook iframe adds ~200ms overhead vs a bare SPA route; 0.80 is realistic, 0.90 punishes the test rig. |
| `accessibility` | ≥ 0.95 | **error** | Hard gate — a11y is a Phase 6 deliverable. We aim for 100; 0.95 leaves headroom for one minor lint issue per story. |
| `best-practices` | ≥ 0.90 | warn | Catches console errors / mixed content / outdated APIs without being noisy about Storybook's iframe gymnastics. |
| FCP | ≤ 2000ms | warn | Storybook bootstraps slowly cold; 2s is the upper bound after caches warm. |
| LCP | ≤ 3000ms | warn | Same reasoning as FCP, scaled. |

## Why most are `warn`, not `error`

Phase 6 is **infrastructure-first**, not "every metric green on day
one". Accessibility is the one hard gate because the project policy
treats a11y violations as bugs. Performance regressions surface in
the PR comment; the engineer decides whether to fix or defer.

## Future tightening

After 3 months of clean baselines we move performance to `error` and
introduce per-story budgets (e.g. RunRow under 100ms FCP). Tracked in
the Phase 7 doc.
