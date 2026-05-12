"""Integration: PluginPacker (Mac) ↔ PluginUnpacker (pod) round-trip.

The R-2 plugin-delivery gap (Phase 6.1 / 6.2) is covered by unit
tests on each side, but the wire-level handshake — the multipart
upload + the unpacker's writeback to the workspace — only proves
correct end-to-end if both halves share the same ZIP layout.

This test:

1. Builds a minimal reward plugin tree under a temp ``community/``
   root (manifest.toml + plugin.py).
2. Packs it via :class:`PluginPacker` from ``src/pipeline``.
3. Submits the resulting ZIP to the runner via ``POST /api/v1/jobs``.
4. Asserts the runner's :class:`PluginUnpacker` extracted the
   plugin to ``<workspace>/community/reward/<id>/`` with files
   intact.

Failure here indicates either side regressed the layout convention
that lets the trainer ``import`` the reward plugin at run time.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest


def _build_minimal_reward_plugin(
    community_root: Path, plugin_id: str = "echo_reward",
) -> Path:
    """Lay out a minimal reward plugin under ``community/reward/<id>/``."""
    plugin_dir = community_root / "reward" / plugin_id
    plugin_dir.mkdir(parents=True)
    (plugin_dir / "manifest.toml").write_text(
        f"""
[plugin]
id = "{plugin_id}"
kind = "reward"
name = "Echo reward"
version = "1.0.0"
category = "semantic"
stability = "stable"
description = "Round-trip fixture for plugin payload integration tests."
supported_strategies = ["grpo"]

[plugin.entry_point]
module = "plugin"
class = "EchoRewardPlugin"
""".strip(),
        encoding="utf-8",
    )
    (plugin_dir / "plugin.py").write_text(
        "class EchoRewardPlugin:\n"
        "    def __init__(self, params=None):\n"
        "        self.params = params or {}\n"
        "    def __call__(self, x):\n"
        "        return 0.0\n",
        encoding="utf-8",
    )
    return plugin_dir


def test_pack_then_submit_unpacks_under_workspace_community(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Build a plugin, pack it via the Mac-side packer, submit it
    through the runner, and assert it lands in ``community/reward/``.
    """
    # PluginPacker is dragged in only when needed — keeps the rest of
    # the integration suite free of the heavy pipeline import.
    from ryotenkai_control.pipeline.stages.managers.deployment.plugin_packer import (
        PluginPacker,
        PluginRef,
    )

    community_root = tmp_path / "community-src"
    plugin_dir = _build_minimal_reward_plugin(community_root)

    # PluginPacker's public API takes a config + uses ``community_root``
    # to locate plugins. Our minimal flow side-steps the config walk
    # by calling ``pack`` with explicit refs — same code path the
    # production walk lands on.
    packer = PluginPacker(config=None, community_root=community_root)
    refs = [PluginRef(kind="reward", plugin_id="echo_reward",
                      source_path=plugin_dir)]
    payload = packer.pack(refs)
    assert payload, "expected non-empty plugin payload"

    # Now submit through the runner. We fix the runner's workspace
    # to a known tmp dir and assert the unpacker writes under it.
    workspace = tmp_path / "runner-workspace"
    workspace.mkdir()
    monkeypatch.setenv("RYOTENKAI_WORKSPACE", str(workspace))

    from fastapi.testclient import TestClient

    from ryotenkai_pod.runner.main import create_app
    # Post-Phase-B: ``tests/unit/runner/conftest.py`` no longer exists at
    # the legacy path. ``MockSupervisor`` is re-exported by the sibling
    # runner conftest via the ``importlib.util.spec_from_file_location``
    # trick — pull it from there.
    monkeypatch.setenv("RYOTENKAI_RUNTIME_PROVIDER", "single_node")
    import pathlib as _p
    import importlib.util as _ilu
    _path = (_p.Path(__file__).resolve().parents[3] / "unit" / "pod" / "runner" / "conftest.py")
    _spec = _ilu.spec_from_file_location("_pod_conftest_local", str(_path))
    assert _spec is not None and _spec.loader is not None
    _mod = _ilu.module_from_spec(_spec)
    _spec.loader.exec_module(_mod)
    MockSupervisor = _mod.MockSupervisor

    app = create_app(supervisor_factory=MockSupervisor)
    with TestClient(app) as client:
        response = client.post(
            "/api/v1/jobs",
            data={"job_spec": json.dumps({
                "job_id": "j-payload",
                "command": ["python", "-c", "pass"],
            })},
            files={"plugins_payload": ("plugins.zip", payload, "application/zip")},
        )
        assert response.status_code in (200, 202), response.text

    # The unpacker writes to ``<workspace>/community/reward/<id>/``.
    unpacked = workspace / "community" / "reward" / "echo_reward"
    assert unpacked.is_dir(), f"expected unpacked dir at {unpacked}"
    assert (unpacked / "manifest.toml").is_file()
    assert (unpacked / "plugin.py").is_file()
    # Sanity: contents survived the round-trip.
    assert "Echo reward" in (unpacked / "manifest.toml").read_text(encoding="utf-8")


def test_empty_payload_is_accepted_for_sft_only_runs(
    runner_testclient,  # type: ignore[no-untyped-def]
) -> None:
    """SFT-only runs have no reward plugins — empty payload is the
    documented no-op path. The unpacker emits the
    ``plugins_unpacked`` event with zero installed and the runner
    proceeds straight to spawning the trainer.
    """
    _, client = runner_testclient
    response = client.post(
        "/api/v1/jobs",
        data={"job_spec": json.dumps({
            "job_id": "j-sft-only",
            "command": ["python", "-c", "pass"],
        })},
        files={"plugins_payload": ("plugins.zip", b"", "application/zip")},
    )
    assert response.status_code in (200, 202), response.text
