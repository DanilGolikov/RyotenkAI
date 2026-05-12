"""Allowlist of legitimate ``unittest.mock`` usages in greenfield tests.

This list is MONOTONICALLY SHRINKING: entries may be removed (when
mocks are eliminated or refactored), but new entries require code-owner
review. Adding to this list is a smell -- prefer refactoring the test.

Entry format
------------
``AllowlistEntry`` carries the file path, line number, semantic pattern,
human-readable justification, the date the entry was added and last renewed,
and the owner who last reviewed it.

* ``line > 0``: a **pinned** entry. The sentinel only excuses a mock at
  exactly that (path, line) tuple. Pin entries are preferred -- they cannot
  silently absorb new violations.

* ``line == 0``: a **pattern** entry. ``path`` is interpreted as a glob
  matched against the test file, and ``pattern`` carries the patch-target
  the entry covers (e.g. ``patch_torch_cuda``). Pattern entries exist to
  avoid one row per ``@patch("torch.cuda.is_available")`` site; each
  external library has at most one pattern entry.

Lifecycle
---------
* ``added``: ISO date when the entry was first added.
* ``renewed``: ISO date of the last quarterly review confirming the entry
  is still legitimate.

Entries older than 365 days without renewal are flagged by
``tests/_lint/test_no_protocol_mocking.py::test_allowlist_entries_renewed_within_365_days``.

The allowlist is enforced by
``tests/_lint/test_no_protocol_mocking.py::test_no_unallowlisted_mocks``;
if you grow this list without a real justification, that test will be the
last gate -- but code review is the gate that matters.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class AllowlistEntry:
    """One documented exemption from the mock-elimination policy.

    Attributes
    ----------
    path:
        Pinned file path (e.g. ``tests/unit/foo.py``) or glob pattern
        (e.g. ``tests/**/test_*.py``) when ``line == 0``.
    line:
        ``> 0`` for a pinned entry; ``0`` selects pattern-matching mode.
    pattern:
        Short identifier for the *kind* of mock. See ``Patterns`` below.
    reason:
        One-sentence explanation of WHY this mock is legitimate.
    added:
        ISO date when the entry was first added.
    renewed:
        ISO date of the last review confirming the entry is still needed.
    owner:
        Username of the last reviewer.
    """

    path: str
    line: int
    pattern: str
    reason: str
    added: str
    renewed: str
    owner: str = "daniil"


# ---------------------------------------------------------------------------
# Patterns vocabulary
# ---------------------------------------------------------------------------
# AsyncMock_interaction        -- per-method async stub on a SimpleNamespace,
#                                 used to verify await_args / assert_awaited_*
# AsyncMock_module_free_async  -- patching a module-level free async function
# AsyncMock_lifecycle_observer -- AsyncMock used to assert close()/aclose()
#                                 lifecycle hooks fire exactly once
# AsyncMock_retry_sleep        -- AsyncMock used to count retry-loop sleeps
# AsyncMock_factory_default    -- AsyncMock() as a placeholder default in a
#                                 _make_*() test-factory parameter
# MagicMock_spec_threading_Timer -- one-off documented in Phase 3B
# patch_torch_cuda             -- @patch on torch.cuda.* probes
# patch_time                   -- @patch on time.sleep/time.time/etc.
# patch_mlflow_external        -- @patch on mlflow.set_tracking_uri etc.
# patch_huggingface_hub        -- @patch on huggingface_hub.* APIs
# patch_subprocess             -- @patch on subprocess.run etc.
# patch_datasets               -- @patch on datasets.load_dataset etc.
# patch_peft                   -- @patch on peft.PeftModel etc.
# patch_concurrent_futures     -- @patch on concurrent.futures.Future.result
# patch_httpx_external         -- @patch on httpx.* network primitives


# ---------------------------------------------------------------------------
# The allowlist itself
# ---------------------------------------------------------------------------

ALLOWLIST: list[AllowlistEntry] = [
    # =====================================================================
    # AsyncMock pinned interaction tests (from Phase 3A log -- 73 KEEPS)
    # =====================================================================
    # tests/unit/control/api/routers/test_jobs_router.py -- jobs router proxy
    # forwards CLI args (grace_seconds, stream params) to the runner.
    # The behaviour IS the call shape; await_args.kwargs is the assertion.
    AllowlistEntry(
        path="tests/unit/control/api/routers/test_jobs_router.py",
        line=132,
        pattern="AsyncMock_interaction",
        reason="Verifies runner.get_status proxy forwards request args -- behaviour IS the call.",
        added="2026-05-12",
        renewed="2026-05-12",
    ),
    AllowlistEntry(
        path="tests/unit/control/api/routers/test_jobs_router.py",
        line=153,
        pattern="AsyncMock_interaction",
        reason="Verifies runner.get_status proxy forwards request args -- behaviour IS the call.",
        added="2026-05-12",
        renewed="2026-05-12",
    ),
    AllowlistEntry(
        path="tests/unit/control/api/routers/test_jobs_router.py",
        line=170,
        pattern="AsyncMock_interaction",
        reason="Verifies runner.get_status proxy forwards request args -- behaviour IS the call.",
        added="2026-05-12",
        renewed="2026-05-12",
    ),
    AllowlistEntry(
        path="tests/unit/control/api/routers/test_jobs_router.py",
        line=218,
        pattern="AsyncMock_interaction",
        reason="Verifies runner.get_status proxy forwards request args -- behaviour IS the call.",
        added="2026-05-12",
        renewed="2026-05-12",
    ),
    AllowlistEntry(
        path="tests/unit/control/api/routers/test_jobs_router.py",
        line=319,
        pattern="AsyncMock_interaction",
        reason="Verifies runner.request_stop proxy forwards grace_seconds=15.0 -- behaviour IS the call.",
        added="2026-05-12",
        renewed="2026-05-12",
    ),
    AllowlistEntry(
        path="tests/unit/control/api/routers/test_jobs_router.py",
        line=339,
        pattern="AsyncMock_interaction",
        reason="Verifies runner.request_stop proxy forwards grace_seconds -- behaviour IS the call.",
        added="2026-05-12",
        renewed="2026-05-12",
    ),
    AllowlistEntry(
        path="tests/unit/control/api/routers/test_jobs_router.py",
        line=356,
        pattern="AsyncMock_interaction",
        reason="Verifies runner.request_stop ConnectionError surfaces as HTTP 503.",
        added="2026-05-12",
        renewed="2026-05-12",
    ),
    AllowlistEntry(
        path="tests/unit/control/api/routers/test_jobs_router.py",
        line=377,
        pattern="AsyncMock_interaction",
        reason="Verifies runner.get_status proxy forwards stream params -- behaviour IS the call.",
        added="2026-05-12",
        renewed="2026-05-12",
    ),
    AllowlistEntry(
        path="tests/unit/control/api/routers/test_jobs_router.py",
        line=424,
        pattern="AsyncMock_interaction",
        reason="Verifies runner.request_stop response shape propagates to HTTP 202.",
        added="2026-05-12",
        renewed="2026-05-12",
    ),
    AllowlistEntry(
        path="tests/unit/control/api/routers/test_jobs_router.py",
        line=460,
        pattern="AsyncMock_interaction",
        reason="Verifies runner.get_status proxy forwards events filter args.",
        added="2026-05-12",
        renewed="2026-05-12",
    ),
    AllowlistEntry(
        path="tests/unit/control/api/routers/test_jobs_router.py",
        line=519,
        pattern="AsyncMock_interaction",
        reason="Verifies runner.get_status proxy forwards events filter args.",
        added="2026-05-12",
        renewed="2026-05-12",
    ),
    AllowlistEntry(
        path="tests/unit/control/api/routers/test_jobs_router.py",
        line=567,
        pattern="AsyncMock_interaction",
        reason="Verifies router maps runner.get_status RuntimeError to HTTP 500.",
        added="2026-05-12",
        renewed="2026-05-12",
    ),
    # tests/unit/control/cli/test_job_command.py -- CLI -> client proxy
    # forwards args (grace_seconds, follow flags). Behaviour IS the call.
    AllowlistEntry(
        path="tests/unit/control/cli/test_job_command.py",
        line=110,
        pattern="AsyncMock_interaction",
        reason="Verifies CLI->client.get_status forwards args -- behaviour IS the call.",
        added="2026-05-12",
        renewed="2026-05-12",
    ),
    AllowlistEntry(
        path="tests/unit/control/cli/test_job_command.py",
        line=127,
        pattern="AsyncMock_interaction",
        reason="Verifies CLI->client.get_status forwards args -- behaviour IS the call.",
        added="2026-05-12",
        renewed="2026-05-12",
    ),
    AllowlistEntry(
        path="tests/unit/control/cli/test_job_command.py",
        line=184,
        pattern="AsyncMock_interaction",
        reason="Verifies CLI->client.request_stop forwards grace_seconds.",
        added="2026-05-12",
        renewed="2026-05-12",
    ),
    AllowlistEntry(
        path="tests/unit/control/cli/test_job_command.py",
        line=206,
        pattern="AsyncMock_interaction",
        reason="Verifies CLI->client.request_stop forwards grace_seconds.",
        added="2026-05-12",
        renewed="2026-05-12",
    ),
    AllowlistEntry(
        path="tests/unit/control/cli/test_job_command.py",
        line=326,
        pattern="AsyncMock_interaction",
        reason="Verifies CLI->client.get_status forwards job_id positional arg.",
        added="2026-05-12",
        renewed="2026-05-12",
    ),
    AllowlistEntry(
        path="tests/unit/control/cli/test_job_command.py",
        line=340,
        pattern="AsyncMock_interaction",
        reason="Verifies CLI->client.get_status forwards job_id positional arg.",
        added="2026-05-12",
        renewed="2026-05-12",
    ),
    AllowlistEntry(
        path="tests/unit/control/cli/test_job_command.py",
        line=402,
        pattern="AsyncMock_interaction",
        reason="Verifies CLI->client.get_status forwards args (status-then-stop chain).",
        added="2026-05-12",
        renewed="2026-05-12",
    ),
    AllowlistEntry(
        path="tests/unit/control/cli/test_job_command.py",
        line=415,
        pattern="AsyncMock_interaction",
        reason="Verifies CLI->client.request_stop is the next call after get_status.",
        added="2026-05-12",
        renewed="2026-05-12",
    ),
    AllowlistEntry(
        path="tests/unit/control/cli/test_job_command.py",
        line=429,
        pattern="AsyncMock_interaction",
        reason="Verifies CLI->client request_stop is invoked after get_status confirms running.",
        added="2026-05-12",
        renewed="2026-05-12",
    ),
    AllowlistEntry(
        path="tests/unit/control/cli/test_job_command.py",
        line=481,
        pattern="AsyncMock_interaction",
        reason="Verifies CLI maps client.get_status ConnectionError to user-facing exit.",
        added="2026-05-12",
        renewed="2026-05-12",
    ),
    AllowlistEntry(
        path="tests/unit/control/cli/test_job_command.py",
        line=578,
        pattern="AsyncMock_interaction",
        reason="Verifies CLI->client.get_status forwards args.",
        added="2026-05-12",
        renewed="2026-05-12",
    ),
    # tests/unit/control/pipeline/test_training_monitor_v2.py --
    # cleanup() lifecycle: SUT must call client.aclose() / tunnel.close()
    # exactly once via SimpleNamespace facade. Interaction-test on lifecycle.
    AllowlistEntry(
        path="tests/unit/control/pipeline/test_training_monitor_v2.py",
        line=133,
        pattern="AsyncMock_lifecycle_observer",
        reason="Verifies cleanup() does NOT call aclose() before lifecycle end.",
        added="2026-05-12",
        renewed="2026-05-12",
    ),
    AllowlistEntry(
        path="tests/unit/control/pipeline/test_training_monitor_v2.py",
        line=134,
        pattern="AsyncMock_interaction",
        reason="Pins get_status return value to drive the monitor loop transition.",
        added="2026-05-12",
        renewed="2026-05-12",
    ),
    AllowlistEntry(
        path="tests/unit/control/pipeline/test_training_monitor_v2.py",
        line=271,
        pattern="AsyncMock_lifecycle_observer",
        reason="Verifies aclose is awaited exactly once when subscribe_events raises.",
        added="2026-05-12",
        renewed="2026-05-12",
    ),
    AllowlistEntry(
        path="tests/unit/control/pipeline/test_training_monitor_v2.py",
        line=284,
        pattern="AsyncMock_lifecycle_observer",
        reason="Verifies aclose is awaited exactly once on subscribe_events error.",
        added="2026-05-12",
        renewed="2026-05-12",
    ),
    AllowlistEntry(
        path="tests/unit/control/pipeline/test_training_monitor_v2.py",
        line=306,
        pattern="AsyncMock_interaction",
        reason="Pins get_status return value for monitor-loop transition.",
        added="2026-05-12",
        renewed="2026-05-12",
    ),
    AllowlistEntry(
        path="tests/unit/control/pipeline/test_training_monitor_v2.py",
        line=307,
        pattern="AsyncMock_lifecycle_observer",
        reason="Verifies aclose lifecycle invariant on terminal state.",
        added="2026-05-12",
        renewed="2026-05-12",
    ),
    AllowlistEntry(
        path="tests/unit/control/pipeline/test_training_monitor_v2.py",
        line=322,
        pattern="AsyncMock_interaction",
        reason="Pins get_status return value for monitor-loop transition.",
        added="2026-05-12",
        renewed="2026-05-12",
    ),
    AllowlistEntry(
        path="tests/unit/control/pipeline/test_training_monitor_v2.py",
        line=323,
        pattern="AsyncMock_lifecycle_observer",
        reason="Verifies aclose is awaited exactly once on monitor cleanup.",
        added="2026-05-12",
        renewed="2026-05-12",
    ),
    AllowlistEntry(
        path="tests/unit/control/pipeline/test_training_monitor_v2.py",
        line=347,
        pattern="AsyncMock_lifecycle_observer",
        reason="Verifies tunnel.close is awaited exactly once on cleanup.",
        added="2026-05-12",
        renewed="2026-05-12",
    ),
    AllowlistEntry(
        path="tests/unit/control/pipeline/test_training_monitor_v2.py",
        line=366,
        pattern="AsyncMock_lifecycle_observer",
        reason="Verifies aclose is awaited even when raising on subscribe_events.",
        added="2026-05-12",
        renewed="2026-05-12",
    ),
    AllowlistEntry(
        path="tests/unit/control/pipeline/test_training_monitor_v2.py",
        line=368,
        pattern="AsyncMock_lifecycle_observer",
        reason="Verifies tunnel.close raised RuntimeError doesn't suppress cleanup chain.",
        added="2026-05-12",
        renewed="2026-05-12",
    ),
    AllowlistEntry(
        path="tests/unit/control/pipeline/test_training_monitor_v2.py",
        line=841,
        pattern="AsyncMock_interaction",
        reason="Pins JobClient.get_diagnostics for postmortem HTTP probe test (nonzero exit triggers full probe set).",
        added="2026-05-12",
        renewed="2026-05-12",
    ),
    AllowlistEntry(
        path="tests/unit/control/pipeline/test_training_monitor_v2.py",
        line=859,
        pattern="AsyncMock_interaction",
        reason="Pins JobClient.get_diagnostics for postmortem HTTP probe test (SIGTERM-kill with exit-0 still triggers postmortem).",
        added="2026-05-12",
        renewed="2026-05-12",
    ),
    AllowlistEntry(
        path="tests/unit/control/pipeline/test_training_monitor_v2.py",
        line=905,
        pattern="AsyncMock_interaction",
        reason="Pins JobClient.get_diagnostics for postmortem HTTP probe test (per-block error sentinel rendering).",
        added="2026-05-12",
        renewed="2026-05-12",
    ),
    AllowlistEntry(
        path="tests/unit/control/pipeline/test_training_monitor_v2.py",
        line=968,
        pattern="AsyncMock_interaction",
        reason="Pins JobClient.get_diagnostics for postmortem HTTP probe test (log prefix + content rendering).",
        added="2026-05-12",
        renewed="2026-05-12",
    ),
    # tests/unit/pod/runner/test_pod_terminator_retry.py -- pure interaction
    # tests for the retry-loop sleep semantics: assert_awaited_with(2.5),
    # await_count == 5, etc.
    AllowlistEntry(
        path="tests/unit/pod/runner/test_pod_terminator_retry.py",
        line=60,
        pattern="AsyncMock_factory_default",
        reason="Factory default for sleep param so the type signature stays AsyncMock | None.",
        added="2026-05-12",
        renewed="2026-05-12",
    ),
    AllowlistEntry(
        path="tests/unit/pod/runner/test_pod_terminator_retry.py",
        line=84,
        pattern="AsyncMock_retry_sleep",
        reason="Verifies retry loop sleeps the correct ticks (await_count, await_args).",
        added="2026-05-12",
        renewed="2026-05-12",
    ),
    AllowlistEntry(
        path="tests/unit/pod/runner/test_pod_terminator_retry.py",
        line=112,
        pattern="AsyncMock_retry_sleep",
        reason="Verifies retry loop sleeps the correct ticks (await_count, await_args).",
        added="2026-05-12",
        renewed="2026-05-12",
    ),
    AllowlistEntry(
        path="tests/unit/pod/runner/test_pod_terminator_retry.py",
        line=138,
        pattern="AsyncMock_retry_sleep",
        reason="Verifies retry loop sleeps the correct ticks (await_count, await_args).",
        added="2026-05-12",
        renewed="2026-05-12",
    ),
    AllowlistEntry(
        path="tests/unit/pod/runner/test_pod_terminator_retry.py",
        line=159,
        pattern="AsyncMock_retry_sleep",
        reason="Verifies retry loop sleeps the correct ticks (await_count, await_args).",
        added="2026-05-12",
        renewed="2026-05-12",
    ),
    AllowlistEntry(
        path="tests/unit/pod/runner/test_pod_terminator_retry.py",
        line=188,
        pattern="AsyncMock_retry_sleep",
        reason="Verifies retry loop sleeps the correct ticks (await_count, await_args).",
        added="2026-05-12",
        renewed="2026-05-12",
    ),
    AllowlistEntry(
        path="tests/unit/pod/runner/test_pod_terminator_retry.py",
        line=204,
        pattern="AsyncMock_retry_sleep",
        reason="Verifies retry loop sleeps the correct ticks (await_count, await_args).",
        added="2026-05-12",
        renewed="2026-05-12",
    ),
    AllowlistEntry(
        path="tests/unit/pod/runner/test_pod_terminator_retry.py",
        line=226,
        pattern="AsyncMock_retry_sleep",
        reason="Verifies retry loop sleeps the correct ticks (await_count, await_args).",
        added="2026-05-12",
        renewed="2026-05-12",
    ),
    AllowlistEntry(
        path="tests/unit/pod/runner/test_pod_terminator_retry.py",
        line=243,
        pattern="AsyncMock_retry_sleep",
        reason="Verifies retry loop sleeps the correct ticks (await_count, await_args).",
        added="2026-05-12",
        renewed="2026-05-12",
    ),
    # tests/unit/pod/runner/api/test_resources.py -- patches the
    # *module-level free async function* default_health_snapshot.
    # No Protocol or class involved; this IS the canonical pattern.
    AllowlistEntry(
        path="tests/unit/pod/runner/api/test_resources.py",
        line=48,
        pattern="AsyncMock_module_free_async",
        reason="Patches free async function default_health_snapshot at module scope.",
        added="2026-05-12",
        renewed="2026-05-12",
    ),
    AllowlistEntry(
        path="tests/unit/pod/runner/api/test_resources.py",
        line=65,
        pattern="AsyncMock_module_free_async",
        reason="Patches free async function default_health_snapshot at module scope.",
        added="2026-05-12",
        renewed="2026-05-12",
    ),
    AllowlistEntry(
        path="tests/unit/pod/runner/api/test_resources.py",
        line=83,
        pattern="AsyncMock_module_free_async",
        reason="Patches free async function default_health_snapshot at module scope.",
        added="2026-05-12",
        renewed="2026-05-12",
    ),
    AllowlistEntry(
        path="tests/unit/pod/runner/api/test_resources.py",
        line=109,
        pattern="AsyncMock_module_free_async",
        reason="Patches free async function default_health_snapshot at module scope.",
        added="2026-05-12",
        renewed="2026-05-12",
    ),
    AllowlistEntry(
        path="tests/unit/pod/runner/api/test_resources.py",
        line=124,
        pattern="AsyncMock_module_free_async",
        reason="Patches free async function default_health_snapshot at module scope.",
        added="2026-05-12",
        renewed="2026-05-12",
    ),
    AllowlistEntry(
        path="tests/unit/pod/runner/api/test_resources.py",
        line=139,
        pattern="AsyncMock_module_free_async",
        reason="Patches free async function default_health_snapshot at module scope.",
        added="2026-05-12",
        renewed="2026-05-12",
    ),
    AllowlistEntry(
        path="tests/unit/pod/runner/api/test_resources.py",
        line=151,
        pattern="AsyncMock_module_free_async",
        reason="Patches free async function default_health_snapshot at module scope.",
        added="2026-05-12",
        renewed="2026-05-12",
    ),
    AllowlistEntry(
        path="tests/unit/pod/runner/api/test_resources.py",
        line=180,
        pattern="AsyncMock_module_free_async",
        reason="Patches free async function default_health_snapshot at module scope.",
        added="2026-05-12",
        renewed="2026-05-12",
    ),
    # tests/unit/control/pipeline/stages/managers/deployment/test_training_launcher_v2.py
    # -- xfail class; AsyncMock chain is needed just to satisfy constructor
    # surface during xfail re-validation. Documented in Phase 3A log.
    AllowlistEntry(
        path="tests/unit/control/pipeline/stages/managers/deployment/test_training_launcher_v2.py",
        line=212,
        pattern="AsyncMock_interaction",
        reason="xfail class -- constructor surface needs AsyncMock until fixture refactor.",
        added="2026-05-12",
        renewed="2026-05-12",
    ),
    AllowlistEntry(
        path="tests/unit/control/pipeline/stages/managers/deployment/test_training_launcher_v2.py",
        line=216,
        pattern="AsyncMock_interaction",
        reason="xfail class -- constructor surface needs AsyncMock until fixture refactor.",
        added="2026-05-12",
        renewed="2026-05-12",
    ),
    AllowlistEntry(
        path="tests/unit/control/pipeline/stages/managers/deployment/test_training_launcher_v2.py",
        line=218,
        pattern="AsyncMock_interaction",
        reason="xfail class -- constructor surface needs AsyncMock until fixture refactor.",
        added="2026-05-12",
        renewed="2026-05-12",
    ),
    AllowlistEntry(
        path="tests/unit/control/pipeline/stages/managers/deployment/test_training_launcher_v2.py",
        line=281,
        pattern="AsyncMock_interaction",
        reason="xfail class -- tunnel open/close stubs for runner-side launcher path.",
        added="2026-05-12",
        renewed="2026-05-12",
    ),
    AllowlistEntry(
        path="tests/unit/control/pipeline/stages/managers/deployment/test_training_launcher_v2.py",
        line=282,
        pattern="AsyncMock_interaction",
        reason="xfail class -- tunnel open/close stubs for runner-side launcher path.",
        added="2026-05-12",
        renewed="2026-05-12",
    ),
    AllowlistEntry(
        path="tests/unit/control/pipeline/stages/managers/deployment/test_training_launcher_v2.py",
        line=287,
        pattern="AsyncMock_interaction",
        reason="xfail class -- runner client constructor needs health_check / submit_job stubs.",
        added="2026-05-12",
        renewed="2026-05-12",
    ),
    AllowlistEntry(
        path="tests/unit/control/pipeline/stages/managers/deployment/test_training_launcher_v2.py",
        line=289,
        pattern="AsyncMock_interaction",
        reason="xfail class -- runner client constructor needs aclose stub for cleanup.",
        added="2026-05-12",
        renewed="2026-05-12",
    ),
    AllowlistEntry(
        path="tests/unit/control/pipeline/stages/managers/deployment/test_training_launcher_v2.py",
        line=318,
        pattern="AsyncMock_interaction",
        reason="xfail class -- tunnel open stub for runner-side launcher path.",
        added="2026-05-12",
        renewed="2026-05-12",
    ),
    AllowlistEntry(
        path="tests/unit/control/pipeline/stages/managers/deployment/test_training_launcher_v2.py",
        line=319,
        pattern="AsyncMock_interaction",
        reason="xfail class -- tunnel close stub for runner-side launcher path.",
        added="2026-05-12",
        renewed="2026-05-12",
    ),
    AllowlistEntry(
        path="tests/unit/control/pipeline/stages/managers/deployment/test_training_launcher_v2.py",
        line=324,
        pattern="AsyncMock_interaction",
        reason="xfail class -- failure-path returns False from health_check.",
        added="2026-05-12",
        renewed="2026-05-12",
    ),
    # tests/unit/control/pipeline/stages/managers/deployment/test_training_launcher_runner.py
    # -- mixed xfail and non-xfail; AsyncMock satisfies the runner-side
    # constructor surface (per Phase 3A log).
    AllowlistEntry(
        path="tests/unit/control/pipeline/stages/managers/deployment/test_training_launcher_runner.py",
        line=94,
        pattern="AsyncMock_interaction",
        reason="Tunnel facade stubs for runner-side launcher constructor.",
        added="2026-05-12",
        renewed="2026-05-12",
    ),
    AllowlistEntry(
        path="tests/unit/control/pipeline/stages/managers/deployment/test_training_launcher_runner.py",
        line=99,
        pattern="AsyncMock_interaction",
        reason="Runner client health_check / submit_job stubs.",
        added="2026-05-12",
        renewed="2026-05-12",
    ),
    AllowlistEntry(
        path="tests/unit/control/pipeline/stages/managers/deployment/test_training_launcher_runner.py",
        line=101,
        pattern="AsyncMock_interaction",
        reason="Runner client aclose stub for cleanup chain.",
        added="2026-05-12",
        renewed="2026-05-12",
    ),
    # tests/unit/control/pipeline/stages/managers/deployment/test_file_uploader.py
    # -- await_args.args[0] pins the upload-target enum (interaction test).
    AllowlistEntry(
        path="tests/unit/control/pipeline/stages/managers/deployment/test_file_uploader.py",
        line=73,
        pattern="AsyncMock_interaction",
        reason="await_args.args[0] pins the upload-target enum -- interaction contract.",
        added="2026-05-12",
        renewed="2026-05-12",
    ),
    # =====================================================================
    # MagicMock(spec=...) KEEPS from Phase 3B
    # =====================================================================
    AllowlistEntry(
        path="tests/unit/shared/utils/test_cancellation.py",
        line=256,
        pattern="MagicMock_spec_threading_Timer",
        reason="Real threading.Timer spawns a background thread -> flaky timing for zero coverage gain.",
        added="2026-05-12",
        renewed="2026-05-12",
    ),
    # =====================================================================
    # External-library @patch pattern entries (line=0, glob in path)
    #
    # These cover the high-volume legitimate @patch usages on third-party
    # APIs that cannot reasonably be tested otherwise. They group many sites
    # under one entry because per-line accounting offers no marginal benefit
    # for "patch torch.cuda.is_available" -- the pattern itself is the policy.
    # =====================================================================
    AllowlistEntry(
        path="tests/**/test_*.py",
        line=0,
        pattern="patch_torch_cuda",
        reason="torch.cuda.* probes are external system calls; cannot be tested without patching.",
        added="2026-05-12",
        renewed="2026-05-12",
    ),
    AllowlistEntry(
        path="tests/**/test_*.py",
        line=0,
        pattern="patch_time",
        reason="time.sleep/time.time patches keep unit tests deterministic and fast.",
        added="2026-05-12",
        renewed="2026-05-12",
    ),
    AllowlistEntry(
        path="tests/**/test_*.py",
        line=0,
        pattern="patch_mlflow_external",
        reason="mlflow.set_tracking_uri / mlflow.genai.load_prompt are SDK entrypoints to a remote service.",
        added="2026-05-12",
        renewed="2026-05-12",
    ),
    AllowlistEntry(
        path="tests/**/test_*.py",
        line=0,
        pattern="patch_huggingface_hub",
        reason="huggingface_hub.HfApi / dataset_info hit the Hugging Face Hub network.",
        added="2026-05-12",
        renewed="2026-05-12",
    ),
    AllowlistEntry(
        path="tests/**/test_*.py",
        line=0,
        pattern="patch_subprocess",
        reason="subprocess.run / Popen are OS-level system calls.",
        added="2026-05-12",
        renewed="2026-05-12",
    ),
    AllowlistEntry(
        path="tests/**/test_*.py",
        line=0,
        pattern="patch_datasets",
        reason="datasets.load_dataset / get_dataset_split_names access HF Datasets cache or network.",
        added="2026-05-12",
        renewed="2026-05-12",
    ),
    AllowlistEntry(
        path="tests/**/test_*.py",
        line=0,
        pattern="patch_peft",
        reason="peft.PeftModel materializes heavy weights; mocking it preserves test isolation.",
        added="2026-05-12",
        renewed="2026-05-12",
    ),
    AllowlistEntry(
        path="tests/**/test_*.py",
        line=0,
        pattern="patch_concurrent_futures",
        reason="concurrent.futures.Future.result is a stdlib synchronisation primitive.",
        added="2026-05-12",
        renewed="2026-05-12",
    ),
    AllowlistEntry(
        path="tests/**/test_*.py",
        line=0,
        pattern="patch_httpx_external",
        reason="httpx network primitives must be patched to avoid real HTTP traffic in unit tests.",
        added="2026-05-12",
        renewed="2026-05-12",
    ),
]


# ---------------------------------------------------------------------------
# Pattern -> external-target prefixes used by the sentinel
# ---------------------------------------------------------------------------
# When the sentinel sees a @patch("module.path") or patch("module.path") call,
# it checks the patch target against these prefixes. If the prefix matches AND
# the corresponding pattern is present in ALLOWLIST, the call is excused.

EXTERNAL_PATCH_PREFIXES: dict[str, tuple[str, ...]] = {
    "patch_torch_cuda": ("torch.cuda.",),
    "patch_time": ("time.sleep", "time.time", "time.monotonic", "time.perf_counter"),
    "patch_mlflow_external": (
        "mlflow.set_tracking_uri",
        "mlflow.genai.",
        "mlflow.tracking.",
        "mlflow.entities.",
    ),
    "patch_huggingface_hub": ("huggingface_hub.",),
    "patch_subprocess": ("subprocess.run", "subprocess.Popen", "subprocess.check_"),
    "patch_datasets": ("datasets.load_dataset", "datasets.get_dataset_"),
    "patch_peft": ("peft.",),
    "patch_concurrent_futures": ("concurrent.futures.",),
    "patch_httpx_external": ("httpx.",),
}


def allowed_external_patterns() -> frozenset[str]:
    """The set of pattern keys currently active for external @patch calls."""
    active = {e.pattern for e in ALLOWLIST if e.line == 0}
    return frozenset(p for p in EXTERNAL_PATCH_PREFIXES if p in active)


def pinned_entries() -> dict[tuple[str, int], AllowlistEntry]:
    """All (path, line) -> entry pairs for pinned exemptions."""
    return {(e.path, e.line): e for e in ALLOWLIST if e.line > 0}
