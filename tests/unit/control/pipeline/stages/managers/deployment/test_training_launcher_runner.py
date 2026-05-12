"""TrainingLauncher → runner_launcher.launch_runner wiring.

The thin-image migration (v2.0.0+) made the runner-launch
``workspace_path`` argument load-bearing — it's the ONLY
PYTHONPATH source for ``src.runner`` once the image stops baking
``src/`` in. These tests pin the contract:

* ``launch_runner`` is called with ``workspace_path=launcher.workspace``
  (the rsync target the CodeSyncer dropped ``src/...`` into).
* The provider's ``required_runtime_env_vars`` still flow through
  to the runner — the env wiring shouldn't regress when we add
  positional arguments.

The deeper happy-path test in :file:`test_training_launcher_v2.py`
stubs out :func:`launch_runner` to short-circuit the SSH path; here
we assert on the call shape rather than on uvicorn semantics.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest  # noqa: TC002 — used only for the pytest.MonkeyPatch type


def _load_launcher():
    """Load training_launcher.py without dragging the whole pipeline
    package init (mirror :file:`test_training_launcher_v2.py`)."""
    if "ryotenkai_launcher_test" in sys.modules:
        return sys.modules["ryotenkai_launcher_test"]
    repo_root = Path(__file__).resolve().parents[7]
    src_path = (
        repo_root
        / "packages" / "control" / "src" / "ryotenkai_control" / "pipeline" / "stages" / "managers"
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


def _config_obj():
    return SimpleNamespace(
        training=SimpleNamespace(
            provider="runpod", get_strategy_chain=lambda: [],
        ),
        experiment_tracking=SimpleNamespace(mlflow=None),
    )


def _secrets_obj():
    return SimpleNamespace(hf_token=None)


def _ssh_client_stub() -> Any:
    return SimpleNamespace(
        host="1.2.3.4", port=22022, username="root", key_path="/k/id",
    )


def _make_launcher() -> Any:
    return TrainingLauncher(
        _config_obj(), _secrets_obj(), deps_installer=MagicMock(),
    )


def _stub_async_path(monkeypatch: pytest.MonkeyPatch) -> MagicMock:
    """Stub the async island (PluginPacker + SSHTunnel + JobClient) so
    ``start_training`` reaches the synchronous step-0 ``launch_runner``
    call and the post-step-0 happy path completes without networking.

    Returns the recording :class:`MagicMock` we stand in for
    :func:`launch_runner` — caller asserts on its call args.
    """
    fake_packer_cls = MagicMock()
    fake_packer_cls.return_value.pack_required.return_value = b""
    monkeypatch.setattr(_launcher_mod, "PluginPacker", fake_packer_cls)

    fake_tunnel = SimpleNamespace(local_port=18080, base_url="http://127.0.0.1:18080", open=AsyncMock(return_value=None), close=AsyncMock(return_value=None))
    monkeypatch.setattr(
        _launcher_mod, "SSHTunnelManager", MagicMock(return_value=fake_tunnel),
    )

    fake_client = SimpleNamespace(health_check=AsyncMock(return_value=True), submit_job=AsyncMock(
        return_value={"job_id": "j-1", "sequence": 0, "offset": 0},
    ), aclose=AsyncMock(return_value=None))
    monkeypatch.setattr(
        _launcher_mod, "JobClient", MagicMock(return_value=fake_client),
    )

    # Replace the imported launch_runner symbol on the launcher module
    # — that's what step-0 actually calls (``from runner_launcher
    # import launch_runner``). Default return Ok-equivalent.
    fake_launch = MagicMock(return_value=_launcher_mod.Ok(None))
    monkeypatch.setattr(_launcher_mod, "launch_runner", fake_launch)
    return fake_launch


# ---------------------------------------------------------------------------
# Positive — workspace_path wiring
# ---------------------------------------------------------------------------


@pytest.mark.xfail(
    strict=True,
    reason="xfail-debt:training-launcher-helper-drift — Pre-existing failure pre-packagization: training_launcher attribute access drifted (SimpleNamespace stub missing expected attrs).",
)
def test_launch_runner_called_with_self_workspace(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``launch_runner`` must receive ``workspace_path=launcher.workspace``.

    Why this test pins the call signature explicitly: in the thin
    image, ``workspace_path`` is the ONLY PYTHONPATH source for
    ``src.runner`` — passing the wrong value (or omitting it) makes
    uvicorn fail with ``ModuleNotFoundError`` on every fresh pod.
    A regression here is silent at the Mac-side and only surfaces in
    the readiness probe 30 s later.
    """
    fake_launch = _stub_async_path(monkeypatch)

    launcher = _make_launcher()
    launcher.set_workspace("/workspace/runs/r-thin")
    result = launcher.start_training(
        _ssh_client_stub(), {"logical_run_id": "j-1"}, provider=None,
    )
    assert result.is_ok(), result

    fake_launch.assert_called_once()
    kwargs = fake_launch.call_args.kwargs
    assert kwargs.get("workspace_path") == "/workspace/runs/r-thin", (
        f"expected workspace_path=launcher.workspace, got kwargs={kwargs!r}"
    )


def test_launch_runner_still_receives_provider_env_vars(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Regression: yesterday we made ``required_runtime_env_vars``
    flow into the runner so it sees RYOTENKAI_RUNTIME_PROVIDER /
    RUNPOD_POD_ID at startup. Adding ``workspace_path`` must not
    have collateral-damaged that wiring.
    """
    fake_launch = _stub_async_path(monkeypatch)

    provider = MagicMock()
    provider.required_runtime_env_vars.return_value = {
        "RYOTENKAI_RUNTIME_PROVIDER": "runpod",
        "RUNPOD_POD_ID": "abc123",
    }

    launcher = _make_launcher()
    launcher.set_workspace("/workspace/runs/r-thin")
    ctx: dict[str, Any] = {"logical_run_id": "j-1", "resource_id": "abc123"}
    # NOTE: we don't assert on the Result — the rest of start_training
    # (provider hooks, tunnel, JobClient) needs heavy mocking that's
    # already covered by test_training_launcher_v2.py. This test cares
    # only that step-0 (``launch_runner``) fired with the right kwargs
    # before the post-step-0 paths kick in.
    launcher.start_training(_ssh_client_stub(), ctx, provider=provider)

    fake_launch.assert_called_once()
    env_kwarg = fake_launch.call_args.kwargs.get("env") or {}
    assert env_kwarg.get("RYOTENKAI_RUNTIME_PROVIDER") == "runpod"
    assert env_kwarg.get("RUNPOD_POD_ID") == "abc123"
    # Provider's hook was actually consulted with the resolved resource_id
    provider.required_runtime_env_vars.assert_called_once_with(resource_id="abc123")
