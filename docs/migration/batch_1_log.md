# Batch 1 — Legacy test decommissioning log

Date: 2026-05-11
Plan: [docs/plans/structured-hopping-starfish.md](../plans/structured-hopping-starfish.md) Phase 7
ADR: [docs/adrs/2026-05-11-legacy-test-decommissioning.md](../adrs/2026-05-11-legacy-test-decommissioning.md)
Worktree: `.claude/worktrees/ecstatic-khorana-1baa6b`

This batch is the **process-proof** of the legacy decommissioning workflow.
Small, hand-picked, low-risk set of architecture sentinels + contract tests.

## Summary

- 14 legacy files removed (`packages/<pkg>/tests/`) — 79 tests
- 3 colocated support files removed (sidecar allowlist, `_normalize`, `conftest`)
- 9 greenfield files added (`tests/_lint/`, `tests/contract/`) — 79 tests + 3
  synthetic-violation self-checks added beyond what legacy had
- Legacy pytest collection: 7122 → 7043 tests (−79, matches)
- Greenfield pytest collection: 404/413 → 483/492 tests (+79)
- All five DUP cases proven via synthetic-violation injection (see notes
  per row below)
- One stale entry removed from the control→pod baseline that legacy had
  carried since pre-Phase-D extraction; greenfield baseline reflects
  current source-tree reality.

## Per-file table

| # | Legacy file | Category | Action | Greenfield equivalent | Tests in legacy | Tests after |
|---|---|---|---|---|---|---|
| 1 | `packages/community/tests/sentinel/test_no_downstream_imports.py` | DUP | Deleted | importlinter rule `community depends only on shared` | 1 | 0 |
| 2 | `packages/control/tests/sentinel/test_no_pod_imports.py` | UNIQUE (allowlist baseline) | Migrated | `tests/_lint/test_control_no_new_pod_imports.py` | 1 | 3 |
| 3 | `packages/control/tests/sentinel/test_no_runtime_ssh_exec_command.py` | UNIQUE | Migrated | `tests/_lint/test_no_runtime_ssh_exec_command.py` + `tests/_lint/_ssh_bootstrap_allowlist.py` | 2 | 3 |
| 4 | `packages/control/tests/contract/test_cli_api_parity.py` | PARTIAL | Migrated (payload-equality leg) | `tests/contract/cli_api/test_payload_parity.py` + `_normalize.py` + `conftest.py` (existing `tests/contract/cli_api/test_parity.py` keeps the reachability gate) | 3 | 3 |
| 5 | `packages/engines/tests/sentinel/test_discriminator_uniformity.py` | UNIQUE | Migrated | `tests/_lint/test_discriminator_uniformity.py` | 1 | 2 |
| 6 | `packages/engines/tests/sentinel/test_no_io_in_engine_prepare.py` | UNIQUE | Migrated | `tests/_lint/test_no_io_in_engine_prepare.py` | 2 | 3 |
| 7 | `packages/engines/tests/sentinel/test_no_provider_imports.py` | DUP | Deleted | importlinter rule `engines is a leaf (depends only on shared)` | 2 | 0 |
| 8 | `packages/engines/tests/contract/test_engine_protocol_parity.py` | UNIQUE | Migrated | `tests/contract/engines/test_engine_protocol_parity.py` | 7 | 7 |
| 9 | `packages/engines/tests/contract/test_prepare_model_signature_uniformity.py` | UNIQUE | Migrated | `tests/contract/engines/test_prepare_model_signature_uniformity.py` | 3 | 3 |
| 10 | `packages/pod/tests/sentinel/test_no_control_imports.py` | DUP | Deleted | importlinter rule `pod depends only on shared and community` | 1 | 0 |
| 11 | `packages/providers/tests/sentinel/test_no_pod_imports.py` | DUP | Deleted | importlinter rule `providers depend only on shared (no community, no pod, no control)` | 1 | 0 |
| 12 | `packages/shared/tests/sentinel/test_runner_api_dto_location.py` | UNIQUE | Migrated | `tests/_lint/test_runner_api_dto_location.py` | 3 | 4 |
| 13 | `packages/shared/tests/sentinel/test_shared_is_leaf.py` | DUP | Deleted | importlinter rule `shared has no internal deps (must be leaf)` | 1 | 0 |
| 14 | `packages/shared/tests/contract/test_dto_round_trip.py` | UNIQUE | Migrated | `tests/contract/runner_api/test_dto_round_trip.py` | 51 | 51 |
| | | | **Totals** | | **79** | **79 + 3 self-violation** |

## DUP equivalence proofs

Each DUP was proven equivalent to its importlinter rule by injecting a
synthetic forbidden import into the production source-tree and observing
both the AST sentinel AND `lint-imports` fail; then removing the inject
and observing both pass.

| # | Inject location | Inject content | AST sentinel | importlinter contract |
|---|---|---|---|---|
| 1 | `packages/community/src/ryotenkai_community/_tmp_inject_test.py` | `from ryotenkai_control.cli import app` | FAIL → cleanup → PASS | `community depends only on shared` BROKEN → cleanup → KEPT |
| 7 | `packages/engines/src/ryotenkai_engines/_tmp_inject_test.py` | `import ryotenkai_providers` | FAIL → cleanup → PASS | `engines is a leaf (depends only on shared)` BROKEN → cleanup → KEPT |
| 10 | `packages/pod/src/ryotenkai_pod/_tmp_inject_test.py` | `import ryotenkai_control` | FAIL → cleanup → PASS | `pod depends only on shared and community` BROKEN → cleanup → KEPT |
| 11 | `packages/providers/src/ryotenkai_providers/_tmp_inject_test.py` | `import ryotenkai_pod` | FAIL → cleanup → PASS | `providers depend only on shared` BROKEN → cleanup → KEPT |
| 13 | `packages/shared/src/ryotenkai_shared/_tmp_inject_test.py` | `import ryotenkai_community` | FAIL → cleanup → PASS | `shared has no internal deps` BROKEN → cleanup → KEPT |

All five DUP rows: AST and importlinter fail on the same boundary; both
recover after cleanup. Equivalence proven; legacy AST sentinel is
redundant.

## Special notes

### Row 2 — control → pod has known violations

importlinter `control must not import pod` is currently BROKEN by five
legacy import sites awaiting Phase D extraction PRs. importlinter's
BROKEN signal is binary — it cannot distinguish "no regression" from
"new regression added". The migrated sentinel
`tests/_lint/test_control_no_new_pod_imports.py` carries the `_EXPECTED_KNOWN`
baseline forward and passes only when the violation set matches that
baseline exactly. It also adds a NEW `test_expected_known_baseline_does_not_grow_silently`
that fires when a previously-known violation disappears from source —
forcing the baseline to tighten as Phase D PRs land.

Discovered during migration: the legacy `_EXPECTED_KNOWN` included
`ryotenkai_control/data/__init__.py: from ryotenkai_pod.trainer.data_loaders`,
but that import has already been removed from source — only the docstring
mentions the path. The greenfield baseline drops this stale entry. The
legacy test had been silently passing despite the violation no longer
existing because the legacy assertion was one-directional (it asked
"are new violations absent" but never "are old ones still there").

### Row 4 — CLI ↔ API parity is PARTIAL

The greenfield `tests/contract/cli_api/test_parity.py` already enforces:

- CLI surface contains expected commands
- Every documented GET endpoint responds
- Known pinned pairings (`runs ls` ↔ `/api/v1/runs`, `preset ls` ↔
  `/api/v1/config/presets`) reach both surfaces and respond

This is the **weaker reachability** property. The legacy test
additionally asserts **payload-equality** — the CLI and API emit the
same data, modulo cosmetic normalisations. That stronger property
moves into the new `tests/contract/cli_api/test_payload_parity.py`
with a colocated `conftest.py` (renamed fixtures: `parity_runs_dir`,
`parity_api_client`, etc., to avoid colliding with broader greenfield
fixtures) and a colocated `_normalize.py` copied verbatim.

The companion `_normalize.py` and `conftest.py` from the legacy contract
directory were deleted along with the test — they had no other consumers.

### Row 8 — engines protocol parity

Legacy uses class-scoped `@pytest.mark.parametrize` for four
sub-tests. Greenfield keeps the same shape — 4 parametrized + 2
top-level (registered, no load failures) = 6 effective unit tests
under one engine (only `vllm` ships today), but pytest counts each
parametrize × method combination as a distinct test → 7 tests. Same as
legacy's count.

### Row 14 — DTO round-trip

51 tests in legacy is the parametrized expansion of `_FIXTURES.items()`
(23 DTO fixtures) + 27 `ErrorCode` enum members + 1 coverage check.
Greenfield preserves the full fixture set verbatim — same count.

## Files created in greenfield

- `tests/_lint/test_control_no_new_pod_imports.py` — control → pod
  drift sentinel with baseline tracking
- `tests/_lint/_ssh_bootstrap_allowlist.py` — SSH allowlist (sidecar)
- `tests/_lint/test_no_runtime_ssh_exec_command.py` — Mac runtime
  `exec_command` sentinel
- `tests/_lint/test_discriminator_uniformity.py` — pydantic
  `Discriminator("kind")` sentinel
- `tests/_lint/test_no_io_in_engine_prepare.py` — IO-free engines
  sentinel
- `tests/_lint/test_runner_api_dto_location.py` — runner-API DTO
  location sentinel
- `tests/contract/engines/__init__.py`
- `tests/contract/engines/test_engine_protocol_parity.py` —
  IInferenceEngine compliance per engine
- `tests/contract/engines/test_prepare_model_signature_uniformity.py` —
  `prepare_model` kwargs pinning
- `tests/contract/runner_api/__init__.py`
- `tests/contract/runner_api/test_dto_round_trip.py` — DTO JSON
  round-trip per wire model
- `tests/contract/cli_api/_normalize.py` — cosmetic normalisation
  helpers
- `tests/contract/cli_api/conftest.py` — payload-parity fixtures
- `tests/contract/cli_api/test_payload_parity.py` — CLI ↔ API
  payload-equality

## Files deleted from legacy

- `packages/community/tests/sentinel/test_no_downstream_imports.py`
- `packages/control/tests/sentinel/test_no_pod_imports.py`
- `packages/control/tests/sentinel/test_no_runtime_ssh_exec_command.py`
- `packages/control/tests/sentinel/bootstrap_allowlist.py` (sidecar)
- `packages/control/tests/contract/test_cli_api_parity.py`
- `packages/control/tests/contract/_normalize.py` (sidecar)
- `packages/control/tests/contract/conftest.py` (sidecar)
- `packages/engines/tests/sentinel/test_discriminator_uniformity.py`
- `packages/engines/tests/sentinel/test_no_io_in_engine_prepare.py`
- `packages/engines/tests/sentinel/test_no_provider_imports.py`
- `packages/engines/tests/contract/test_engine_protocol_parity.py`
- `packages/engines/tests/contract/test_prepare_model_signature_uniformity.py`
- `packages/pod/tests/sentinel/test_no_control_imports.py`
- `packages/providers/tests/sentinel/test_no_pod_imports.py`
- `packages/shared/tests/sentinel/test_runner_api_dto_location.py`
- `packages/shared/tests/sentinel/test_shared_is_leaf.py`
- `packages/shared/tests/contract/test_dto_round_trip.py`
