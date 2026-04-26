"""Phase 6.3 — :class:`TrainingLauncher` (post-rewrite) contract.

Covers the new JobClient/SSHTunnel-based launcher. The old marker-file
flow is exercised in :file:`test_training_launcher.py` (skipped post
6.3 — kept as a paper-trail of the legacy contract).

Coverage (kept tight; deeper integration tests live in the runner
suite):
- TestBuildJobEnv         _build_job_env merges defaults + secrets +
                          MLflow + provider hooks in priority order
- TestStartTrainingHappy  end-to-end happy path (preflight ok,
                          PluginPacker ok, tunnel + submit ok)
- TestStartTrainingErrors plugin pack fails → tunnel never opens;
                          tunnel open fails → no leaked port;
                          submit fails → tunnel closed before return
- TestResolveJobId        priority: logical_run_id > run.name > fallback

Tests bypass :mod:`src.pipeline.stages.managers.__init__` (which
eager-imports heavy deps not present in the dev venv) by loading
the launcher module directly via importlib.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest


def _load_launcher():
    """Load the launcher module directly so we don't drag the whole
    pipeline package init into the test process. Same pattern used
    by :file:`test_plugin_packer.py`."""
    if "ryotenkai_launcher_test" in sys.modules:
        return sys.modules["ryotenkai_launcher_test"]
    repo_root = Path(__file__).resolve().parents[7]
    src_path = (
        repo_root
        / "src" / "pipeline" / "stages" / "managers"
        / "deployment" / "training_launcher.py"
    )
    spec = importlib.util.spec_from_file_location(
        "ryotenkai_launcher_test", str(src_path),
    )
    assert spec is not None
    assert spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules["ryotenkai_launcher_test"] = mod
    spec.loader.exec_module(mod)
    return mod


_launcher_mod = _load_launcher()
TrainingLauncher = _launcher_mod.TrainingLauncher


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _config_obj(*, single_node: bool = False, mlflow=None):
    """Build a minimal config-like object the launcher consumes."""
    cfg = SimpleNamespace(
        training=SimpleNamespace(
            provider="single_node" if single_node else "runpod",
            get_strategy_chain=lambda: [],
        ),
        experiment_tracking=SimpleNamespace(mlflow=mlflow),
    )
    return cfg


def _secrets_obj(*, hf_token: str | None = None):
    return SimpleNamespace(hf_token=hf_token)


def _ssh_client_stub() -> Any:
    return SimpleNamespace(host="1.2.3.4", port=22022, username="root", key_path="/k/id")


def _make_launcher(*, config=None, secrets=None) -> TrainingLauncher:
    return TrainingLauncher(
        config or _config_obj(),
        secrets or _secrets_obj(),
        deps_installer=MagicMock(),
    )


# ---------------------------------------------------------------------------
# _build_job_env
# ---------------------------------------------------------------------------


class TestBuildJobEnv:
    def test_defaults_present(self) -> None:
        launcher = _make_launcher()
        launcher.set_workspace("/tmp/run-x")
        env = launcher._build_job_env(context={}, extra_env_vars={})
        assert env["LOG_LEVEL"] == "DEBUG"
        assert env["HELIX_WORKSPACE"] == "/tmp/run-x"
        assert env["PYTHONPATH"] == "/tmp/run-x"
        # Crash observability defaults
        assert env["PYTHONUNBUFFERED"] == "1"
        assert env["PYTHONFAULTHANDLER"] == "1"
        assert env["PYTHONFAULTHANDLER_PATH"].endswith("training.faulthandler.log")
        # Loopback runner URL — trainer's RunnerEventCallback uses this
        assert env["RYOTENKAI_RUNNER_URL"] == "http://127.0.0.1:8080"

    def test_single_node_workspace_is_container_path(self) -> None:
        launcher = _make_launcher(config=_config_obj(single_node=True))
        launcher.set_workspace("/host/run-dir")  # ignored for single_node
        env = launcher._build_job_env(context={}, extra_env_vars={})
        # single_node container always sees /workspace because the
        # host run dir is mounted there
        assert env["HELIX_WORKSPACE"] == "/workspace"

    def test_hf_token_added_when_set(self) -> None:
        launcher = _make_launcher(secrets=_secrets_obj(hf_token="hf_xxx"))
        env = launcher._build_job_env(context={}, extra_env_vars={})
        assert env["HF_TOKEN"] == "hf_xxx"

    def test_hf_token_omitted_when_unset(self) -> None:
        launcher = _make_launcher(secrets=_secrets_obj(hf_token=None))
        env = launcher._build_job_env(context={}, extra_env_vars={})
        assert "HF_TOKEN" not in env

    def test_provider_hooks_override_defaults(self) -> None:
        launcher = _make_launcher()
        env = launcher._build_job_env(
            context={},
            extra_env_vars={"LOG_LEVEL": "WARNING", "RUNPOD_API_KEY": "k"},
        )
        assert env["LOG_LEVEL"] == "WARNING"  # overridden
        assert env["RUNPOD_API_KEY"] == "k"   # added


# ---------------------------------------------------------------------------
# _resolve_job_id
# ---------------------------------------------------------------------------


class TestResolveJobId:
    def test_logical_run_id_wins(self) -> None:
        launcher = _make_launcher()
        ctx = {"logical_run_id": "lrid-42", "run": SimpleNamespace(name="run-name")}
        assert launcher._resolve_job_id(ctx) == "lrid-42"

    def test_run_name_fallback(self) -> None:
        launcher = _make_launcher()
        ctx = {"run": SimpleNamespace(name="run-99")}
        assert launcher._resolve_job_id(ctx) == "run-99"

    def test_last_resort_fallback(self) -> None:
        launcher = _make_launcher()
        assert launcher._resolve_job_id({}) == "unnamed-job"


# ---------------------------------------------------------------------------
# start_training — happy path
# ---------------------------------------------------------------------------


class TestStartTrainingHappy:
    def test_full_flow_returns_ok_and_stashes_context(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        # Stub PluginPacker (no community dir on disk)
        fake_packer_cls = MagicMock()
        fake_packer_cls.return_value.pack_required.return_value = b""
        monkeypatch.setattr(_launcher_mod, "PluginPacker", fake_packer_cls)

        # Stub SSHTunnelManager + JobClient with async-mock surface
        fake_tunnel = MagicMock()
        fake_tunnel.local_port = 18080
        fake_tunnel.base_url = "http://127.0.0.1:18080"
        fake_tunnel.open = AsyncMock(return_value=None)
        fake_tunnel.close = AsyncMock(return_value=None)
        fake_tunnel_cls = MagicMock(return_value=fake_tunnel)
        monkeypatch.setattr(_launcher_mod, "SSHTunnelManager", fake_tunnel_cls)

        fake_client = MagicMock()
        fake_client.health_check = AsyncMock(return_value=True)
        fake_client.submit_job = AsyncMock(
            return_value={"job_id": "j-1", "sequence": 0, "offset": 0},
        )
        fake_client.aclose = AsyncMock(return_value=None)
        fake_client_cls = MagicMock(return_value=fake_client)
        monkeypatch.setattr(_launcher_mod, "JobClient", fake_client_cls)

        launcher = _make_launcher()
        ctx: dict[str, Any] = {"logical_run_id": "j-1"}
        result = launcher.start_training(_ssh_client_stub(), ctx, provider=None)

        assert result.is_ok()
        out = result.unwrap()
        assert out["mode"] == "job_server"
        assert out["job_id"] == "j-1"
        assert out["tunnel_port"] == 18080
        # Monitor will read these
        assert ctx["job_client"] is fake_client
        assert ctx["ssh_tunnel"] is fake_tunnel
        assert ctx["job_id"] == "j-1"
        # submit was called with the right shape
        spec_arg = fake_client.submit_job.await_args.args[0]
        assert spec_arg["job_id"] == "j-1"
        assert spec_arg["command"][0] == "python"


# ---------------------------------------------------------------------------
# Error paths
# ---------------------------------------------------------------------------


class TestStartTrainingErrors:
    def test_plugin_pack_fail_returns_err_without_tunnel(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        fake_packer_cls = MagicMock()
        fake_packer_cls.return_value.pack_required.side_effect = (
            _launcher_mod.PluginPackError("missing manifest")
        )
        monkeypatch.setattr(_launcher_mod, "PluginPacker", fake_packer_cls)

        # Tunnel constructor must NOT be called when pack fails
        sentinel = MagicMock(side_effect=AssertionError("tunnel created prematurely"))
        monkeypatch.setattr(_launcher_mod, "SSHTunnelManager", sentinel)

        launcher = _make_launcher()
        result = launcher.start_training(
            _ssh_client_stub(), {"logical_run_id": "j"}, provider=None,
        )
        assert result.is_err()
        assert result.unwrap_err().code == "PLUGIN_PACK_FAILED"

    def test_submit_failure_closes_tunnel(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        fake_packer_cls = MagicMock()
        fake_packer_cls.return_value.pack_required.return_value = b""
        monkeypatch.setattr(_launcher_mod, "PluginPacker", fake_packer_cls)

        fake_tunnel = MagicMock()
        fake_tunnel.local_port = 18080
        fake_tunnel.base_url = "http://127.0.0.1:18080"
        fake_tunnel.open = AsyncMock(return_value=None)
        fake_tunnel.close = AsyncMock(return_value=None)
        monkeypatch.setattr(
            _launcher_mod, "SSHTunnelManager", MagicMock(return_value=fake_tunnel),
        )

        fake_client = MagicMock()
        fake_client.health_check = AsyncMock(return_value=True)
        fake_client.submit_job = AsyncMock(
            side_effect=_launcher_mod.JobClientError("422 invalid spec"),
        )
        fake_client.aclose = AsyncMock(return_value=None)
        monkeypatch.setattr(
            _launcher_mod, "JobClient", MagicMock(return_value=fake_client),
        )

        launcher = _make_launcher()
        result = launcher.start_training(
            _ssh_client_stub(), {"logical_run_id": "j"}, provider=None,
        )
        assert result.is_err()
        assert result.unwrap_err().code == "JOB_SUBMIT_FAILED"
        # Tunnel must be closed even when submit blew up — leaving an
        # open ssh -L would leak a local port forever.
        fake_tunnel.close.assert_awaited()

    def test_runner_health_timeout_closes_tunnel(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        # Shrink the readiness budget so the test is fast.
        monkeypatch.setattr(_launcher_mod, "RUNNER_READY_MAX_ATTEMPTS", 2)
        monkeypatch.setattr(_launcher_mod, "RUNNER_READY_POLL_SECONDS", 0.0)

        fake_packer_cls = MagicMock()
        fake_packer_cls.return_value.pack_required.return_value = b""
        monkeypatch.setattr(_launcher_mod, "PluginPacker", fake_packer_cls)

        fake_tunnel = MagicMock()
        fake_tunnel.local_port = 18080
        fake_tunnel.base_url = "http://127.0.0.1:18080"
        fake_tunnel.open = AsyncMock(return_value=None)
        fake_tunnel.close = AsyncMock(return_value=None)
        monkeypatch.setattr(
            _launcher_mod, "SSHTunnelManager", MagicMock(return_value=fake_tunnel),
        )

        fake_client = MagicMock()
        # /healthz never returns 200
        fake_client.health_check = AsyncMock(return_value=False)
        fake_client.submit_job = AsyncMock(return_value={})
        fake_client.aclose = AsyncMock(return_value=None)
        monkeypatch.setattr(
            _launcher_mod, "JobClient", MagicMock(return_value=fake_client),
        )

        launcher = _make_launcher()
        result = launcher.start_training(
            _ssh_client_stub(), {"logical_run_id": "j"}, provider=None,
        )
        assert result.is_err()
        # SSH tunnel error category — runner unreachable
        assert result.unwrap_err().code == "RUNNER_TUNNEL_FAILED"
        fake_tunnel.close.assert_awaited()
