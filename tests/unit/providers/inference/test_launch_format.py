"""Unit tests for ``format_docker_run`` (LaunchSpec → docker shell command).

Boundary the engine ↔ provider contract: the engine returns a structured
:class:`LaunchSpec`; the provider must format it into an unambiguous,
shell-safe ``docker run …`` invocation. Bugs here surface as broken or
unsafe container starts on every deploy, so we assert tight invariants.
"""

from __future__ import annotations

import pytest
from ryotenkai_engines.interfaces import LaunchSpec

from ryotenkai_providers.inference.launch import format_docker_run


def _spec(**overrides) -> LaunchSpec:
    base = {
        "image": "ryotenkai/inference-vllm:1.0.0",
        "container_name": "infer_run123",
        "args": ("serve", "/workspace/model", "--port", "8000"),
        "env": {"HF_HOME": "/workspace/hf_cache"},
        "port": 8000,
        "volumes": (("/workspace", "/workspace"),),
    }
    base.update(overrides)
    return LaunchSpec(**base)


class TestPositive:
    def test_full_command_contains_all_pieces(self) -> None:
        spec = _spec()
        cmd = format_docker_run(spec, host_bind="0.0.0.0")

        assert cmd.startswith("docker run")
        assert "--detach" in cmd
        assert "--gpus all" in cmd
        assert "--name infer_run123" in cmd
        assert "-p 0.0.0.0:8000:8000" in cmd
        assert "-v /workspace:/workspace" in cmd
        assert "-e HF_HOME=/workspace/hf_cache" in cmd
        assert "ryotenkai/inference-vllm:1.0.0" in cmd
        assert "serve /workspace/model --port 8000" in cmd

    def test_host_bind_127_locks_down_to_loopback(self) -> None:
        cmd = format_docker_run(_spec(), host_bind="127.0.0.1")
        assert "-p 127.0.0.1:8000:8000" in cmd

    def test_detach_false_emits_foreground_run(self) -> None:
        cmd = format_docker_run(_spec(), detach=False)
        assert "--detach" not in cmd
        assert cmd.startswith("docker run --name")

    def test_gpus_all_can_be_disabled(self) -> None:
        cmd = format_docker_run(_spec(), gpus_all=False)
        assert "--gpus all" not in cmd


class TestShellSafety:
    def test_image_with_spaces_is_quoted(self) -> None:
        spec = _spec(image="my registry/img:tag")
        cmd = format_docker_run(spec)
        # shlex quotes a string containing whitespace with single quotes.
        assert "'my registry/img:tag'" in cmd

    def test_args_with_special_chars_are_quoted(self) -> None:
        spec = _spec(args=("serve", "path with space", "--prompt", "hi$you"))
        cmd = format_docker_run(spec)
        assert "'path with space'" in cmd
        assert "'hi$you'" in cmd

    def test_env_value_with_metacharacter_is_quoted(self) -> None:
        spec = _spec(env={"TOKEN": "a;rm -rf /"})
        cmd = format_docker_run(spec)
        # The whole KEY=VALUE pair lands in a single -e token, fully quoted.
        assert "-e 'TOKEN=a;rm -rf /'" in cmd
        # And it must NOT appear as an unquoted shell command.
        assert " rm -rf / " not in cmd

    def test_volume_paths_with_spaces_are_quoted(self) -> None:
        spec = _spec(volumes=(("/host with space", "/inside"),))
        cmd = format_docker_run(spec)
        assert "-v '/host with space':/inside" in cmd


class TestBoundary:
    def test_empty_env_emits_no_env_flags(self) -> None:
        spec = _spec(env={})
        cmd = format_docker_run(spec)
        assert " -e " not in cmd

    def test_empty_volumes_emits_no_volume_flags(self) -> None:
        spec = _spec(volumes=())
        cmd = format_docker_run(spec)
        assert " -v " not in cmd

    def test_empty_args_still_emits_image_at_end(self) -> None:
        spec = _spec(args=())
        cmd = format_docker_run(spec)
        assert cmd.endswith("ryotenkai/inference-vllm:1.0.0")


class TestInvariant:
    def test_arg_order_preserved(self) -> None:
        # Engines often rely on flag order (e.g. positional then flags).
        spec = _spec(args=("serve", "model", "--port", "8000", "--quantization", "awq"))
        cmd = format_docker_run(spec)
        idx = cmd.index("serve")
        # Following tokens must appear in the same order.
        ordered = [
            "serve",
            "model",
            "--port",
            "8000",
            "--quantization",
            "awq",
        ]
        positions = [cmd.find(tok, idx) for tok in ordered]
        assert positions == sorted(positions)
        assert -1 not in positions

    @pytest.mark.parametrize("port", [1, 8080, 65535])
    def test_port_publish_uses_spec_port(self, port: int) -> None:
        cmd = format_docker_run(_spec(port=port))
        assert f"-p 0.0.0.0:{port}:{port}" in cmd
