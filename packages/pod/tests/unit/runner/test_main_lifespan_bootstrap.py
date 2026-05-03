"""Phase 14.B § 8.6 — :func:`_lifespan` bootstrap contract.

End-to-end wiring tests using FastAPI :class:`TestClient`. Verify
that the lifespan reads env via
:func:`~src.runner.runtime.provider_registry.resolve_lifecycle_client_from_env`
and surfaces :class:`BootstrapConfigError` when env config is wrong.

Slim-venv compatible — uses ``sys.modules`` stub for ``runpod`` SDK
in case any RunPod path imports it transitively.

7-cat coverage applied across the bootstrap matrix (Phase 14.B
§ 8.6 § 1 → § 7).
"""

from __future__ import annotations

import sys
import types
from pathlib import Path
from typing import TYPE_CHECKING
from unittest.mock import MagicMock

import pytest

# Stub `runpod` SDK at module load — same pattern used by
# Phase 14.A test files. Single-node default never imports it, but
# the RunPod-bootstrap tests below force the import path.
if "runpod" not in sys.modules:
    _stub = types.ModuleType("runpod")
    _stub.api_key = ""
    _stub.create_pod = MagicMock()
    _stub.get_pod = MagicMock()
    _stub.stop_pod = MagicMock()
    _stub.resume_pod = MagicMock()
    _stub.terminate_pod = MagicMock()
    sys.modules["runpod"] = _stub


from fastapi.testclient import TestClient  # noqa: E402

from ryotenkai_shared.constants import (  # noqa: E402
    PROVIDER_RUNPOD,
    PROVIDER_SINGLE_NODE,
    RUNTIME_PROVIDER_ENV_VAR,
)
from ryotenkai_pod.runner.main import create_app  # noqa: E402
from ryotenkai_pod.runner.runtime.lifecycle_client import IPodLifecycleClient  # noqa: E402
from ryotenkai_pod.runner.runtime.provider_registry import BootstrapConfigError  # noqa: E402
from tests.unit.runner.conftest import MockSupervisor  # noqa: E402

if TYPE_CHECKING:
    pass


# ---------------------------------------------------------------------------
# 1. Positive — env=runpod (full creds) boots with RunPod client
# ---------------------------------------------------------------------------


class TestPositive:
    def test_runpod_full_env_boots_with_runpod_client(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setenv("RYOTENKAI_WORKSPACE", str(tmp_path))
        monkeypatch.setenv(RUNTIME_PROVIDER_ENV_VAR, "runpod")
        monkeypatch.setenv("RUNPOD_API_KEY", "rk-test")
        monkeypatch.setenv("RUNPOD_POD_ID", "pod-zzz")

        with TestClient(create_app(supervisor_factory=MockSupervisor)) as client:
            r = client.get("/healthz")
            assert r.status_code == 200
            terminator = client.app.state.pod_terminator
            # Lifespan wired the right client.
            assert isinstance(terminator._client, IPodLifecycleClient)
            assert terminator._client.provider_name == PROVIDER_RUNPOD
            # Lifespan-static config snapshot.
            assert terminator._resource_id == "pod-zzz"

    def test_single_node_minimal_env_boots_with_noop_client(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setenv("RYOTENKAI_WORKSPACE", str(tmp_path))
        monkeypatch.setenv(RUNTIME_PROVIDER_ENV_VAR, "single_node")
        # No RunPod env — single_node doesn't need it.

        with TestClient(create_app(supervisor_factory=MockSupervisor)) as client:
            r = client.get("/healthz")
            assert r.status_code == 200
            terminator = client.app.state.pod_terminator
            assert terminator._client.provider_name == PROVIDER_SINGLE_NODE
            # Single-node: empty resource_id (no RunPod pod concept).
            assert terminator._resource_id == ""


# ---------------------------------------------------------------------------
# 2. Negative — bootstrap failures cause uvicorn to exit non-zero
# ---------------------------------------------------------------------------


class TestNegativeBootstrapFailures:
    def test_unset_provider_env_fails_lifespan(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        # Phase 14.B § 1.7: missing RYOTENKAI_RUNTIME_PROVIDER ⇒
        # BootstrapConfigError ⇒ lifespan re-raises ⇒ uvicorn exits.
        # TestClient surfaces the exception when entering the lifespan.
        monkeypatch.setenv("RYOTENKAI_WORKSPACE", str(tmp_path))
        monkeypatch.delenv(RUNTIME_PROVIDER_ENV_VAR, raising=False)

        with pytest.raises(BootstrapConfigError) as exc_info:
            with TestClient(create_app(supervisor_factory=MockSupervisor)):
                pass
        # Operator-friendly: lists known providers in the error message.
        assert RUNTIME_PROVIDER_ENV_VAR in str(exc_info.value)
        assert "single_node" in str(exc_info.value)

    def test_runpod_without_api_key_fails_lifespan(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setenv("RYOTENKAI_WORKSPACE", str(tmp_path))
        monkeypatch.setenv(RUNTIME_PROVIDER_ENV_VAR, "runpod")
        monkeypatch.setenv("RUNPOD_POD_ID", "pod-zzz")
        monkeypatch.delenv("RUNPOD_API_KEY", raising=False)

        with pytest.raises(BootstrapConfigError) as exc_info:
            with TestClient(create_app(supervisor_factory=MockSupervisor)):
                pass
        assert "RUNPOD_API_KEY" in str(exc_info.value)

    def test_runpod_without_pod_id_fails_lifespan(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setenv("RYOTENKAI_WORKSPACE", str(tmp_path))
        monkeypatch.setenv(RUNTIME_PROVIDER_ENV_VAR, "runpod")
        monkeypatch.setenv("RUNPOD_API_KEY", "rk-test")
        monkeypatch.delenv("RUNPOD_POD_ID", raising=False)

        with pytest.raises(BootstrapConfigError) as exc_info:
            with TestClient(create_app(supervisor_factory=MockSupervisor)):
                pass
        assert "RUNPOD_POD_ID" in str(exc_info.value)

    def test_unregistered_provider_fails_lifespan(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setenv("RYOTENKAI_WORKSPACE", str(tmp_path))
        monkeypatch.setenv(RUNTIME_PROVIDER_ENV_VAR, "lambda")

        with pytest.raises(BootstrapConfigError) as exc_info:
            with TestClient(create_app(supervisor_factory=MockSupervisor)):
                pass
        assert "lambda" in str(exc_info.value)


# ---------------------------------------------------------------------------
# 3. Boundary — invalid volume_kind clamped to persistent
# ---------------------------------------------------------------------------


class TestBoundary:
    def test_invalid_volume_kind_clamped_to_persistent(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setenv("RYOTENKAI_WORKSPACE", str(tmp_path))
        monkeypatch.setenv(RUNTIME_PROVIDER_ENV_VAR, "runpod")
        monkeypatch.setenv("RUNPOD_API_KEY", "rk-test")
        monkeypatch.setenv("RUNPOD_POD_ID", "pod-zzz")
        monkeypatch.setenv("RUNPOD_VOLUME_KIND", "garbage_value")

        with TestClient(create_app(supervisor_factory=MockSupervisor)) as client:
            terminator = client.app.state.pod_terminator
            # Clamped at lifespan boot.
            assert terminator._volume_kind == "persistent"

    def test_keep_on_error_unset_is_false(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setenv("RYOTENKAI_WORKSPACE", str(tmp_path))
        monkeypatch.setenv(RUNTIME_PROVIDER_ENV_VAR, "single_node")
        monkeypatch.delenv("RUNPOD_KEEP_ON_ERROR", raising=False)

        with TestClient(create_app(supervisor_factory=MockSupervisor)) as client:
            terminator = client.app.state.pod_terminator
            assert terminator._keep_on_error is False

    def test_keep_on_error_true_is_propagated(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setenv("RYOTENKAI_WORKSPACE", str(tmp_path))
        monkeypatch.setenv(RUNTIME_PROVIDER_ENV_VAR, "runpod")
        monkeypatch.setenv("RUNPOD_API_KEY", "rk-test")
        monkeypatch.setenv("RUNPOD_POD_ID", "pod-zzz")
        monkeypatch.setenv("RUNPOD_KEEP_ON_ERROR", "true")

        with TestClient(create_app(supervisor_factory=MockSupervisor)) as client:
            terminator = client.app.state.pod_terminator
            assert terminator._keep_on_error is True


# ---------------------------------------------------------------------------
# 4. Invariants — terminator + heartbeat both reachable on app.state
# ---------------------------------------------------------------------------


class TestInvariants:
    def test_app_state_carries_terminator_and_heartbeat(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setenv("RYOTENKAI_WORKSPACE", str(tmp_path))
        monkeypatch.setenv(RUNTIME_PROVIDER_ENV_VAR, "single_node")

        with TestClient(create_app(supervisor_factory=MockSupervisor)) as client:
            assert hasattr(client.app.state, "pod_terminator")
            assert hasattr(client.app.state, "heartbeat")

    def test_health_reporter_started_in_lifespan(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        # The Mac-side Training Monitor stage relies on
        # ``health_snapshot`` events to render its [MONITOR] ALIVE
        # status line; if the reporter never starts, the bus has no
        # source and the monitor stays silent between trainer events.
        monkeypatch.setenv("RYOTENKAI_WORKSPACE", str(tmp_path))
        monkeypatch.setenv(RUNTIME_PROVIDER_ENV_VAR, "single_node")

        with TestClient(create_app(supervisor_factory=MockSupervisor)) as client:
            reporter = client.app.state.health_reporter
            assert reporter is not None
            assert reporter.is_running
        # After lifespan shutdown the task should be torn down.
        assert not reporter.is_running

    def test_terminator_no_longer_reads_env_directly(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        # Phase 14.B § 1.4 invariant: ``decide_and_act`` no longer
        # accepts ``env=...``. Calling it with the old kwarg raises
        # TypeError — pin the new signature.
        import inspect
        from ryotenkai_pod.runner.pod_terminator import PodTerminator
        sig = inspect.signature(PodTerminator.decide_and_act)
        assert "env" not in sig.parameters
        # And the constructor takes the new lifespan-static params.
        ctor = inspect.signature(PodTerminator.__init__)
        for required in ("client", "resource_id", "volume_kind", "keep_on_error"):
            assert required in ctor.parameters
        # Old graphql params are gone.
        for removed in ("graphql_url", "request_timeout", "max_attempts", "http_client_factory"):
            assert removed not in ctor.parameters


# ---------------------------------------------------------------------------
# 5. Dependency errors — N/A (lifespan failure modes covered above)
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# 6. Regressions — pre-14.B import surface still importable
# ---------------------------------------------------------------------------


class TestRegressions:
    def test_pod_terminator_module_no_longer_exports_graphql_url(self) -> None:
        # Pin: ``DEFAULT_RUNPOD_GRAPHQL_URL`` was removed from
        # :mod:`src.runner.pod_terminator` in Phase 14.B (lives in
        # :mod:`src.providers.runpod.runtime.lifecycle_client` now).
        from ryotenkai_pod.runner import pod_terminator as pt
        assert not hasattr(pt, "DEFAULT_RUNPOD_GRAPHQL_URL")
        assert "DEFAULT_RUNPOD_GRAPHQL_URL" not in pt.__all__

    def test_pod_terminator_no_httpx_import(self) -> None:
        # Pin: pod_terminator no longer imports httpx.
        import inspect
        from ryotenkai_pod.runner import pod_terminator as pt
        source = Path(inspect.getsourcefile(pt)).read_text(  # type: ignore[arg-type]
            encoding="utf-8",
        )
        assert "import httpx" not in source


# ---------------------------------------------------------------------------
# 7. Logic-specific — lifespan reads env exactly once, terminator inherits
# ---------------------------------------------------------------------------


class TestLogicSpecific:
    def test_terminator_volume_kind_snapshot_independent_of_post_boot_env(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        # Phase 14.B § 1.4: lifespan reads env ONCE. Mutating env
        # after the lifespan boots must NOT change the terminator's
        # snapshot.
        monkeypatch.setenv("RYOTENKAI_WORKSPACE", str(tmp_path))
        monkeypatch.setenv(RUNTIME_PROVIDER_ENV_VAR, "single_node")
        monkeypatch.setenv("RUNPOD_VOLUME_KIND", "persistent")

        with TestClient(create_app(supervisor_factory=MockSupervisor)) as client:
            terminator = client.app.state.pod_terminator
            assert terminator._volume_kind == "persistent"

            # Mutate env post-boot — should NOT propagate.
            monkeypatch.setenv("RUNPOD_VOLUME_KIND", "network")
            assert terminator._volume_kind == "persistent"

    def test_runpod_resource_id_pulled_from_pod_id(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        # Phase 14.B § 8.6 § 7: ``resource_id`` snapshot equals
        # ``RUNPOD_POD_ID`` at boot time.
        monkeypatch.setenv("RYOTENKAI_WORKSPACE", str(tmp_path))
        monkeypatch.setenv(RUNTIME_PROVIDER_ENV_VAR, "runpod")
        monkeypatch.setenv("RUNPOD_API_KEY", "rk-test")
        monkeypatch.setenv("RUNPOD_POD_ID", "pod-snapshot-test")

        with TestClient(create_app(supervisor_factory=MockSupervisor)) as client:
            terminator = client.app.state.pod_terminator
            assert terminator._resource_id == "pod-snapshot-test"
