"""Unit tests for ``format_prepare_step`` (PrepareStep → docker shell command).

Sibling to :mod:`test_launch_format` — same engine ↔ provider contract,
applied to ephemeral preparation containers (LoRA merge, GGUF conversion,
…). Bugs here corrupt model artifacts on every deploy, so we assert
tight invariants.

Categories: positive, shell-safety (injection), boundary (empty fields,
None entrypoint), invariant (image always after flags), logic-specific
(extra_env wins on collision; --entrypoint multi-token unwrapping).
"""

from __future__ import annotations

from ryotenkai_engines.interfaces import PrepareStep

from ryotenkai_providers.inference.launch import format_prepare_step


def _step(**overrides) -> PrepareStep:  # type: ignore[no-untyped-def]
    base: dict = {
        "name": "merge_lora",
        "image": None,
        "entrypoint": ("python3",),
        "args": (
            "/opt/helix/merge_lora.py",
            "--base-model",
            "org/base",
            "--adapter",
            "/workspace/ad",
            "--output",
            "/workspace/runs/r1/model",
            "--cache-dir",
            "/workspace/hf_cache",
        ),
        "env": {"HF_HOME": "/workspace/hf_cache"},
        "volumes": (("/host/ws", "/workspace"),),
        "inputs": ("/workspace/ad",),
        "outputs": ("/workspace/runs/r1/model",),
        "success_marker": "MERGE_SUCCESS",
        "success_artifact": "/workspace/runs/r1/model/config.json",
        "timeout_seconds": 3600,
    }
    base.update(overrides)
    return PrepareStep(**base)


# ===========================================================================
# Positive
# ===========================================================================


class TestPositive:
    def test_full_command_contains_all_pieces(self) -> None:
        s = _step()
        cmd = format_prepare_step(
            s,
            image="ryotenkai/inference-vllm:1.0.0",
            container_name="helix-prepare-r1-merge_lora",
        )
        assert cmd.startswith("docker run --detach")
        assert "--gpus all" in cmd
        assert "--name helix-prepare-r1-merge_lora" in cmd
        assert "-v /host/ws:/workspace" in cmd
        assert "-e HF_HOME=/workspace/hf_cache" in cmd
        assert "--entrypoint python3" in cmd
        assert "ryotenkai/inference-vllm:1.0.0" in cmd
        assert "/opt/helix/merge_lora.py" in cmd

    def test_no_port_publishing(self) -> None:
        """Prep steps don't expose ports — sibling difference vs
        ``format_docker_run``."""
        cmd = format_prepare_step(
            _step(), image="img:1", container_name="c"
        )
        assert "-p " not in cmd

    def test_no_entrypoint_uses_image_default(self) -> None:
        """``entrypoint=None`` ⇒ no ``--entrypoint`` flag emitted."""
        s = _step(entrypoint=None)
        cmd = format_prepare_step(s, image="img:1", container_name="c")
        assert "--entrypoint" not in cmd


# ===========================================================================
# Shell-injection safety
# ===========================================================================


class TestShellSafety:
    def test_image_with_spaces_quoted(self) -> None:
        cmd = format_prepare_step(
            _step(), image="my registry/img:1", container_name="c"
        )
        assert "'my registry/img:1'" in cmd

    def test_container_name_with_spaces_quoted(self) -> None:
        cmd = format_prepare_step(
            _step(), image="img:1", container_name="my container"
        )
        assert "'my container'" in cmd

    def test_arg_with_semicolon_quoted(self) -> None:
        """Defense-in-depth: even though engine shouldn't emit shell-meta
        in args, the formatter MUST quote anyway."""
        s = _step(args=("/opt/x.py", "; rm -rf /"))
        cmd = format_prepare_step(s, image="img:1", container_name="c")
        assert "'; rm -rf /'" in cmd
        # And the literal evil string must NOT appear unquoted.
        assert " ; rm -rf /" not in cmd

    def test_volume_path_with_spaces_quoted(self) -> None:
        s = _step(volumes=(("/host with space", "/workspace"),))
        cmd = format_prepare_step(s, image="img:1", container_name="c")
        assert "'/host with space':/workspace" in cmd

    def test_env_value_with_dollar_quoted(self) -> None:
        """env values can contain $ — must be single-quoted to suppress
        shell expansion."""
        s = _step(env={"K": "value-with-$VAR"})
        cmd = format_prepare_step(s, image="img:1", container_name="c")
        assert "'K=value-with-$VAR'" in cmd

    def test_env_value_with_space_quoted(self) -> None:
        s = _step(env={"K": "value with space"})
        cmd = format_prepare_step(s, image="img:1", container_name="c")
        assert "'K=value with space'" in cmd


# ===========================================================================
# Boundary
# ===========================================================================


class TestBoundary:
    def test_empty_env(self) -> None:
        s = _step(env={})
        cmd = format_prepare_step(s, image="img:1", container_name="c")
        assert "-e " not in cmd

    def test_empty_volumes(self) -> None:
        s = _step(volumes=())
        cmd = format_prepare_step(s, image="img:1", container_name="c")
        assert "-v " not in cmd

    def test_empty_args(self) -> None:
        """Some prep steps have no args — image's entrypoint runs alone."""
        s = _step(args=(), entrypoint=None)
        cmd = format_prepare_step(s, image="img:1", container_name="c")
        # Image is the last token, no trailing args.
        assert cmd.endswith("img:1")

    def test_extra_env_empty(self) -> None:
        cmd = format_prepare_step(
            _step(), image="img:1", container_name="c", extra_env={}
        )
        # Step env still emitted.
        assert "-e HF_HOME=/workspace/hf_cache" in cmd

    def test_extra_env_none(self) -> None:
        cmd = format_prepare_step(
            _step(), image="img:1", container_name="c", extra_env=None
        )
        assert "-e HF_HOME=/workspace/hf_cache" in cmd


# ===========================================================================
# Invariants
# ===========================================================================


class TestInvariants:
    def test_image_always_appears_after_flags(self) -> None:
        """The image token must be after all -e/-v/--name/--gpus/--entrypoint
        flags, and before the args (when entrypoint is not set)."""
        s = _step(entrypoint=None)
        cmd = format_prepare_step(s, image="img:1", container_name="c")
        # Image must precede the script path.
        i_image = cmd.index("img:1")
        i_script = cmd.index("/opt/helix/merge_lora.py")
        assert i_image < i_script

    def test_arg_order_preserved(self) -> None:
        """Engines emit args in canonical order; formatter must not reorder."""
        s = _step()
        cmd = format_prepare_step(s, image="img:1", container_name="c")
        i_base = cmd.index("--base-model")
        i_adapter = cmd.index("--adapter")
        i_output = cmd.index("--output")
        i_cache = cmd.index("--cache-dir")
        assert i_base < i_adapter < i_output < i_cache

    def test_no_trailing_whitespace(self) -> None:
        cmd = format_prepare_step(_step(), image="img:1", container_name="c")
        assert cmd == cmd.strip()

    def test_single_line_output(self) -> None:
        """Legacy bug regression — multi-line backslash continuation
        broke stdin handling. Result MUST be a single line."""
        cmd = format_prepare_step(_step(), image="img:1", container_name="c")
        assert "\n" not in cmd
        assert "\\\n" not in cmd


# ===========================================================================
# Logic-specific
# ===========================================================================


class TestLogic:
    def test_gpus_all_disabled(self) -> None:
        cmd = format_prepare_step(
            _step(), image="img:1", container_name="c", gpus_all=False
        )
        assert "--gpus all" not in cmd

    def test_extra_env_overrides_step_env(self) -> None:
        """The HF_TOKEN injection contract: provider's ``extra_env``
        overrides the engine's step.env on key collision (secrets boundary)."""
        s = _step(env={"HF_TOKEN": "WILL_BE_OVERWRITTEN"})
        cmd = format_prepare_step(
            s,
            image="img:1",
            container_name="c",
            extra_env={"HF_TOKEN": "REAL_SECRET"},
        )
        assert "REAL_SECRET" in cmd
        assert "WILL_BE_OVERWRITTEN" not in cmd

    def test_extra_env_added_when_no_collision(self) -> None:
        cmd = format_prepare_step(
            _step(),
            image="img:1",
            container_name="c",
            extra_env={"HF_TOKEN": "secret123"},
        )
        # Both step env and extra env present.
        assert "-e HF_HOME=/workspace/hf_cache" in cmd
        assert "-e HF_TOKEN=secret123" in cmd

    def test_entrypoint_single_token_uses_flag(self) -> None:
        s = _step(entrypoint=("python3",))
        cmd = format_prepare_step(s, image="img:1", container_name="c")
        assert "--entrypoint python3" in cmd
        # Image immediately after --entrypoint flag.
        i_ep = cmd.index("--entrypoint python3")
        i_image = cmd.index("img:1")
        assert i_ep < i_image

    def test_entrypoint_multi_token_prepends_to_args(self) -> None:
        """Docker's ``--entrypoint`` flag accepts only a single token. If
        the engine emits a multi-token entrypoint (forward-compat), the tail
        is prepended to args — provider documents this in the docstring."""
        s = _step(
            entrypoint=("bash", "-c"),
            args=("/opt/script.sh",),
        )
        cmd = format_prepare_step(s, image="img:1", container_name="c")
        # Single-token portion goes to --entrypoint.
        assert "--entrypoint bash" in cmd
        # Tail (-c) appears before the script path.
        i_dash_c = cmd.index(" -c ")
        i_script = cmd.index("/opt/script.sh")
        assert i_dash_c < i_script
