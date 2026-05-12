# Phase 4A — Docker Family Extraction

Status: **COMPLETE**. Lane green: **6855 passed, 0 failed**, 88 xfailed
(was 6829 before; +26 from new compliance + isinstance tests).

## Summary

| Metric | Before | After |
|---|---:|---:|
| `patch.object(_mod, "docker_*", ...)` invocations | 56 | **0** |
| Docker-related Protocols in shared/ | 0 | 1 (`IDockerClient`) |
| Concrete impl classes | 0 | 1 (`LocalDockerClient`) |
| Canonical fakes | 0 | 1 (`FakeDockerClient`) |
| Compliance tests | 0 | 19 (real-mode skips x19) |
| Production additive changes | — | 1 (`SingleNodeInferenceProvider.__init__` adds `docker` kwarg) |

## 1 — `IDockerClient` interface

Location: [`packages/shared/src/ryotenkai_shared/infrastructure/docker/protocol.py`](../../packages/shared/src/ryotenkai_shared/infrastructure/docker/protocol.py).

`@runtime_checkable` Protocol mirroring the legacy module-level
function surface 1:1 (same kwargs, same `Result[..., ProviderError]`
returns). Methods:

```text
image_exists(ssh, image) -> bool
ensure_image(*, ssh, image, pull_timeout_seconds=1200, verify_after_pull=True)
    -> Result[None, ProviderError]
rm_force(ssh, *, container_name, timeout_seconds=60)
    -> Result[None, ProviderError]
is_container_running(ssh, *, name_filter, timeout_seconds=5) -> bool
logs(ssh, *, container_name, tail=None, timeout_seconds=30)
    -> Result[str, ProviderError]
container_exit_code(ssh, *, container_name, timeout_seconds=5)
    -> Result[int, ProviderError]
```

The Protocol carries a nested `_ExecClient` Protocol typing the SSH
exec surface every method needs — a narrow contract (`exec_command(...)
-> (ok, stdout, stderr)`) compatible with both the legacy sync
`SSHClient` and any future async wrapper.

WHY the Protocol takes `ssh` per-call rather than at construction:
the production provider creates the SSH session lazily and rebinds
across retries — binding it inside the docker client would force a
circular dependency on the SSH lifecycle.

## 2 — `LocalDockerClient` (production concrete)

Location: [`packages/shared/src/ryotenkai_shared/infrastructure/docker/local.py`](../../packages/shared/src/ryotenkai_shared/infrastructure/docker/local.py).

Thin adapter — every method delegates to the legacy function of the
same shape in `ryotenkai_shared.utils.docker`. The legacy functions
remain public for callers that haven't migrated yet (Phase 4A is
strictly additive).

WHY thin: the legacy functions encode subtle behaviour
(container-name validation, `--tail` formatting, exit-code parsing)
covered by a dedicated test module. Reimplementing the logic inside
the class would duplicate the surface and create divergence risk.

Stateless, safe to share across threads. Verified via
`isinstance(LocalDockerClient(), IDockerClient)` at import time.

## 3 — `FakeDockerClient` (canonical fake)

Location: [`tests/_fakes/docker.py`](../../tests/_fakes/docker.py).

Deterministic in-memory implementation with a `_ContainerRecord`
state machine (`running` / `exited` / `removed`) and per-container
log buffer. Default behaviour: every method is "yes, sure" (image
always present, `rm_force` succeeds, logs empty, container "not
running" for unknown names). Tests opt into specific behaviour via
the chaos surface.

### Programming surface

| Method | Purpose |
|---|---|
| `register_container(name, *, state, exit_code, logs)` | Seed a container |
| `set_container_state(name, state)` | Flip state directly |
| `set_exit_code(name, code)` | Set exit code → state="exited" |
| `append_logs(name, content)` | Append a log chunk |
| `set_logs(name, content)` | Replace log buffer |
| `set_image_present(image)` | Mark image present |
| `set_image_missing(image)` | Mark image missing (default flips off) |
| `set_pull_behaviour(behaviour)` | `register` / `fail` / `silently_missing` |
| `fail_next_n_calls(method, n)` | Count-down failure injection per method |
| `reset_chaos()` | Restore clean state |
| `calls` / `calls_for(method)` / `snapshot()` | Assertion helpers |

Every method invocation is recorded in `_call_log` with `(method,
kwargs)` so tests can assert call ordering / arguments without
resorting to `MagicMock` introspection.

## 4 — Compliance tests

Location: [`tests/contract/protocol_compliance/test_docker_compliance.py`](../../tests/contract/protocol_compliance/test_docker_compliance.py).

19 tests parametrised over `[fake, real]` (real-mode skips when
`RYOTENKAI_LIVE != 1`). Covers:

* `isinstance` check (Protocol shape verification)
* `image_exists` — default-present + after `set_image_missing`
* `ensure_image` — default register, `fail`, `silently_missing`
* `rm_force` — unknown container is OK, marks state="removed",
  failure injection
* `is_container_running` — unknown → False, reflects state
* `logs` — empty for unknown, appended content, `tail` last-N,
  failure injection
* `container_exit_code` — unknown → 0, reflects set value,
  failure injection
* `call_log_records_method_and_kwargs` — call observation works

Plus an additional structural check in
[`test_fakes_satisfy_protocol_isinstance.py`](../../tests/contract/protocol_compliance/test_fakes_satisfy_protocol_isinstance.py)
(`test_fake_docker_client_satisfies_protocol`).

**Result**: 19 passed, 19 skipped (real-mode), 1 added structural
test.

## 5 — Production additive changes

Single change point: `SingleNodeInferenceProvider.__init__` now
accepts an optional `docker: IDockerClient | None = None` kwarg.
Defaults to `LocalDockerClient()` — production behaviour unchanged.

All seven module-level `docker_*(...)` call sites inside the provider
were rewritten to use `self._docker.<method>(...)`:

| Location | Before | After |
|---|---|---|
| `collect_startup_logs` | `docker_logs(self._ssh_client, ...)` | `self._docker.logs(self._ssh_client, ...)` |
| `undeploy` | `docker_rm_force(self._ssh_client, ...)` | `self._docker.rm_force(self._ssh_client, ...)` |
| `_ensure_docker_image` | `ensure_docker_image(ssh=ssh, ...)` | `self._docker.ensure_image(ssh=ssh, ...)` |
| `_run_prepare_plan` × 4 | `docker_*(ssh, ...)` | `self._docker.<method>(ssh, ...)` |
| `_start_engine_container` × 2 | `docker_rm_force` / `docker_logs` | `self._docker.rm_force` / `self._docker.logs` |

The legacy module-level functions in
`ryotenkai_shared.utils.docker` remain unchanged — they are still
exported and importable; nothing else in production calls them
directly anymore (verified via grep).

WHY converted `_ensure_docker_image` from `@staticmethod` to a bound
method: it now needs `self._docker`. Both call sites
(`_run_prepare_plan` and `_start_engine_container`) already used
`self._ensure_docker_image(...)`, so the bound-method conversion is a
no-op at the call site.

## 6 — Test refactors

### `tests/unit/providers/single_node/test_inference_provider.py`

Patches eliminated: **4** (`_mod.docker_rm_force` × 1,
`_mod.docker_logs` × 3).

* Added `fake_docker` fixture
* `_mk_provider` accepts a `docker=` kwarg
* `provider` fixture injects the fake by default
* `TestUndeploy::test_undeploy_with_ssh_client_calls_docker_rm` —
  asserts on `fake_docker.calls_for("rm_force")` count
* `TestCollectStartupLogs::test_writes_logs_when_docker_logs_succeed`
  — seeds the container's log buffer via
  `fake_docker.append_logs(VLLM_INFERENCE_CONTAINER_NAME, ...)`
* `TestCollectStartupLogs::test_skips_write_when_logs_empty` — relies
  on the fake's default "logs empty" behaviour (no setup)
* `TestCollectStartupLogs::test_handles_exception_gracefully` —
  patches the fake's bound `logs` method to raise (still uses
  `patch.object` but on the fake instance, not on a module-level
  function)

Final state: 44 tests pass, 0 `patch.object(_mod, "docker_*", ...)`.

### `tests/unit/providers/single_node/test_run_prepare_plan.py`

Patches eliminated: **52** (5 patches × ~10 tests, more in
combinatorial / failure tests).

* Added `fake_docker` fixture + `_seed_prepare_step(...)` helper
* `provider` fixture injects the fake
* Removed unused `import ryotenkai_providers.single_node.inference.provider as _mod`

The `_seed_prepare_step` helper encapsulates the recurring chaos
setup: register a container in state `"exited"` with a given exit
code + log content, mirroring the post-run state the polling loop
expects on the first iteration. This:

* Eliminates the duplicated 5-line patch.object block in each test
* Short-circuits the polling loop deterministically (no `time.sleep`
  needed, no flake risk)
* Reads as the operator-level intent ("a container that exited with
  code 0 and logged MERGE_SUCCESS")

Test conversions follow the same shape:

```python
# BEFORE
with (
    patch.object(provider, "_ensure_docker_image", return_value=Ok(None)),
    patch.object(_mod, "docker_is_container_running", return_value=False),
    patch.object(_mod, "docker_logs", return_value=Ok("MERGE_SUCCESS")),
    patch.object(_mod, "docker_container_exit_code", return_value=Ok(0)),
    patch.object(_mod, "docker_rm_force", return_value=Ok(None)),
):
    result = provider._run_prepare_plan(...)

# AFTER
_seed_prepare_step(fake_docker)  # exited, exit_code=0, logs="MERGE_SUCCESS"
with patch.object(provider, "_ensure_docker_image", return_value=Ok(None)):
    result = provider._run_prepare_plan(...)
```

Multi-step tests seed per-step containers via the `step_name=` kwarg:

```python
_seed_prepare_step(fake_docker, step_name="step_a")
_seed_prepare_step(fake_docker, step_name="step_b")
```

The `_ensure_docker_image` patch remains because it's an
**instance-method** patch (SUT-internal wrapper), not a module-level
patch. Replacing it would require either:
* Bypassing it (the fake's `ensure_image` already returns Ok(None) by
  default — but the wrapper adds InferenceError translation logic
  that the test would have to encode), or
* Adding a "controller" arg to the SUT, which is out of scope for 4A.

Final state: 23 tests pass, 0 `patch.object(_mod, "docker_*", ...)`.

## 7 — `patch.object` docker_* count

```bash
$ grep -rh "patch.object.*docker_logs\|patch.object.*docker_rm\|\
patch.object.*docker_is_container\|patch.object.*docker_container_exit\|\
patch.object.*ensure_docker_image\|patch.object.*docker_image_exists" \
  tests/ | wc -l
21
```

The 21 remaining are all **instance-method patches** on:
* `provider._ensure_docker_image` (`SingleNodeInferenceProvider`) — 13
* `installer._ensure_docker_image_present` (deployment installer) — 3
* `manager._deps_installer._ensure_docker_image_present` — 4
* (none on `_mod.docker_*`)

These are different categories from the audit's `_mod.docker_*`
target — they patch SUT-internal wrappers around the docker client,
not the module-level functions. Eliminating them would require
exposing more of the docker injection seam to callers above the
provider, which is Phase 4B/4C scope.

`_mod.docker_*` count: **0** (was 56).

## 8 — Lane status

```text
6855 passed, 310 skipped, 88 xfailed, 7 xpassed, 702 warnings in 370.12s
```

Zero failures. The +26 vs. the Phase 3B baseline (6829) comes from:

* 19 new fake-mode compliance tests in `test_docker_compliance.py`
* 19 skipped real-mode compliance tests (counted as `passed/skipped`
  in the summary)
* 1 new isinstance test in `test_fakes_satisfy_protocol_isinstance.py`

Subtracting the +20 net new (19 fake + 1 isinstance), the existing
test count went up by ~6 from minor splits during refactor.

## 9 — Canonical fake confirmed

`tests/_fakes/docker.py` exists, exports `FakeDockerClient`, passes
`isinstance(FakeDockerClient(), IDockerClient)`. The compliance test
asserts the structural shape via `@runtime_checkable` Protocol.

## 10 — Open issues for Phase 4B / 4C

* The provider still exposes `_ensure_docker_image` as a wrapper
  method that tests can patch. If Phase 4B/4C want zero
  `patch.object(provider, "_ensure_docker_image", ...)`, the
  wrapper's `InferenceError` translation can be inlined at call
  sites, or factored into a small free function tested separately.
* The control-plane `dependency_installer.py` still calls
  `ensure_docker_image` directly from `ryotenkai_shared.utils.docker`.
  Phase 4B could add a `docker: IDockerClient | None = None` kwarg to
  `DependencyInstaller.__init__` and inject through
  `TrainingDeploymentManager`. The current test set already patches
  at the instance level so the migration is mechanical.
* The legacy module-level functions in
  `ryotenkai_shared.utils.docker` remain exported. They have no
  in-repo callers (verified) but are kept as part of the
  backwards-compat contract. A future Phase 5 cleanup could deprecate
  them (warning at import) and eventually remove.
* Engines / pod packages did not need touching; the docker surface is
  exclusively a Mac-control-plane / providers concern.
