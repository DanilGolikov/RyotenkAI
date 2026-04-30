from __future__ import annotations

import ast

import pytest

from src.providers.runpod.inference.pods.artifacts import CHAT_SCRIPT, render_readme

pytestmark = pytest.mark.unit


def test_render_readme_includes_manifest_and_endpoint() -> None:
    readme = render_readme(
        manifest_filename="custom_manifest.json",
        endpoint_url="http://127.0.0.1:9000/v1",
    )

    assert "custom_manifest.json" in readme
    assert "http://127.0.0.1:9000/v1" in readme
    assert "RUNPOD_API_KEY" in readme
    assert "chat_inference.py" in readme


def test_chat_script_contains_critical_runtime_markers() -> None:
    required_markers = [
        "import runpod",
        "inference_manifest.json",
        "def _runpod_sdk_call(",
        "def _wait_for_pod_ssh_ready(",
        "def _wait_http_ok(",
        "def _runpod_start_pod(",
        "def _runpod_stop_pod(",
        "def _runpod_delete_pod(",
        "def _start_chat(",
        "/v1/models",
        "def _ssh_base_args(",
        "def _ensure_ssh_tunnel(",
        "/clear",
        "/stop",
        "/exit",
    ]

    for marker in required_markers:
        assert marker in CHAT_SCRIPT


def test_chat_script_is_valid_python() -> None:
    module = ast.parse(CHAT_SCRIPT)

    assert isinstance(module, ast.Module)
    assert module.body


def test_render_readme_handles_none_endpoint_url() -> None:
    """RunPod-pods deploy() now returns endpoint_url=None until activate_for_eval
    opens the tunnel. The README must render a hint instead of "None"."""
    readme = render_readme(
        manifest_filename="m.json",
        endpoint_url=None,
    )
    assert "None" not in readme
    # The placeholder hints at the chat-script which is what eventually
    # opens the tunnel.
    assert "chat_inference.py" in readme.lower() or "ssh tunnel" in readme.lower()
