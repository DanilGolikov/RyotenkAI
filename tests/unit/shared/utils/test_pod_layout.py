"""Tests for :class:`src.utils.pod_layout.PodLayout`.

Coverage matrix (per project policy):
- positive / negative / boundary
- invariants (per-run isolation, provider parity, Mac↔Pod symmetry)
- regression
"""

from __future__ import annotations

from pathlib import PurePosixPath

import pytest

from ryotenkai_shared.utils.logs_layout import (
    REMOTE_RUNNER_LOG_NAME,
    REMOTE_TRAINER_STDIO_LOG_NAME,
)
from ryotenkai_shared.utils.pod_layout import (
    COMMUNITY_DIR_NAME,
    CONFIG_DIR_NAME,
    CONFIG_FILE_NAME,
    DATA_DIR_NAME,
    EVENTS_DIR_NAME,
    FSM_STATE_JSON_NAME,
    FSM_STATE_JSONL_NAME,
    LOGS_DIR_NAME,
    OUTPUT_DIR_NAME,
    RUNNER_LOG_NAME,
    SRC_DIR_NAME,
    STATE_DIR_NAME,
    TRAINER_STDIO_LOG_NAME,
    PodLayout,
)


class TestFactory:
    """``PodLayout.from_root`` validates and normalises input."""

    def test_from_root_string_absolute(self) -> None:
        layout = PodLayout.from_root("/workspace/runs/abc")
        assert layout.root == PurePosixPath("/workspace/runs/abc")

    def test_from_root_purepath_absolute(self) -> None:
        layout = PodLayout.from_root(PurePosixPath("/data/runs/xyz"))
        assert layout.root == PurePosixPath("/data/runs/xyz")

    def test_from_root_rejects_relative_path(self) -> None:
        with pytest.raises(ValueError, match="must be absolute"):
            PodLayout.from_root("relative/path")

    def test_from_root_rejects_empty(self) -> None:
        with pytest.raises(ValueError, match="must be absolute"):
            PodLayout.from_root("")


class TestProperties:
    """Each property concatenates root with the canonical sub-path."""

    @pytest.fixture
    def layout(self) -> PodLayout:
        return PodLayout.from_root("/workspace/runs/r1")

    def test_subdirectories(self, layout: PodLayout) -> None:
        assert layout.src_dir == PurePosixPath("/workspace/runs/r1/src")
        assert layout.config_dir == PurePosixPath("/workspace/runs/r1/config")
        assert layout.data_dir == PurePosixPath("/workspace/runs/r1/data")
        assert layout.community_dir == PurePosixPath("/workspace/runs/r1/community")
        assert layout.output_dir == PurePosixPath("/workspace/runs/r1/output")
        assert layout.logs_dir == PurePosixPath("/workspace/runs/r1/logs")
        assert layout.events_dir == PurePosixPath("/workspace/runs/r1/events")
        assert layout.state_dir == PurePosixPath("/workspace/runs/r1/state")

    def test_files(self, layout: PodLayout) -> None:
        assert layout.config_file == PurePosixPath(
            "/workspace/runs/r1/config/pipeline_config.yaml",
        )
        assert layout.runner_log == PurePosixPath(
            "/workspace/runs/r1/logs/runner.log",
        )
        assert layout.trainer_stdio_log == PurePosixPath(
            "/workspace/runs/r1/logs/trainer.stdio.log",
        )
        assert layout.fsm_state_json == PurePosixPath(
            "/workspace/runs/r1/state/job.json",
        )
        assert layout.fsm_state_jsonl == PurePosixPath(
            "/workspace/runs/r1/state/job.jsonl",
        )

    def test_constants_match_filenames(self, layout: PodLayout) -> None:
        """No filename should drift from its public constant."""
        assert layout.config_file.name == CONFIG_FILE_NAME
        assert layout.runner_log.name == RUNNER_LOG_NAME
        assert layout.trainer_stdio_log.name == TRAINER_STDIO_LOG_NAME
        assert layout.fsm_state_json.name == FSM_STATE_JSON_NAME
        assert layout.fsm_state_jsonl.name == FSM_STATE_JSONL_NAME

    def test_all_dirs_includes_every_subdirectory(self, layout: PodLayout) -> None:
        names = {d.name for d in layout.all_dirs()}
        assert names == {
            SRC_DIR_NAME,
            CONFIG_DIR_NAME,
            DATA_DIR_NAME,
            COMMUNITY_DIR_NAME,
            OUTPUT_DIR_NAME,
            LOGS_DIR_NAME,
            EVENTS_DIR_NAME,
            STATE_DIR_NAME,
        }


class TestImmutability:
    """``PodLayout`` is a frozen dataclass — must reject mutation."""

    def test_root_is_immutable(self) -> None:
        layout = PodLayout.from_root("/workspace/runs/r1")
        with pytest.raises((AttributeError, Exception)):
            layout.root = PurePosixPath("/other")  # type: ignore[misc]

    def test_layout_is_hashable(self) -> None:
        a = PodLayout.from_root("/workspace/runs/r1")
        b = PodLayout.from_root("/workspace/runs/r1")
        assert hash(a) == hash(b)
        assert {a, b} == {a}


class TestPerRunIsolation:
    """Two different run_ids produce DISJOINT path trees.

    Critical: this is the property that prevents the resume-collision
    bug (where /workspace/runner.log was overwritten by sequential runs
    on the same pod).
    """

    def test_disjoint_subtrees(self) -> None:
        a = PodLayout.from_root("/workspace/runs/run_alpha")
        b = PodLayout.from_root("/workspace/runs/run_beta")

        # No path of A is a prefix of any path of B and vice versa
        # (besides root parents which are not owned by either layout).
        for path_a in (
            a.runner_log,
            a.trainer_stdio_log,
            a.config_file,
            a.events_dir,
            a.state_dir,
        ):
            for path_b in (
                b.runner_log,
                b.trainer_stdio_log,
                b.config_file,
                b.events_dir,
                b.state_dir,
            ):
                assert not str(path_a).startswith(str(path_b) + "/")
                assert not str(path_b).startswith(str(path_a) + "/")
                assert path_a != path_b

    def test_runs_share_only_the_grandparent(self) -> None:
        a = PodLayout.from_root("/workspace/runs/run_alpha")
        b = PodLayout.from_root("/workspace/runs/run_beta")
        # Their roots' parents are the same (the namespace), but the
        # roots themselves are distinct — that's the whole point of
        # per-run isolation.
        assert a.root.parent == b.root.parent
        assert a.root != b.root


class TestProviderParity:
    """Same layout under different ``root`` values (RunPod vs single_node).

    Each provider supplies its own ``root``; the structure under it must
    be identical bit-for-bit. This is the SOLID-L invariant.
    """

    def test_runpod_vs_single_node_identical_substructure(self) -> None:
        runpod = PodLayout.from_root("/workspace/runs/r1")
        single_node = PodLayout.from_root("/home/user/ryotenkai/runs/r1")

        runpod_rels = sorted(d.relative_to(runpod.root) for d in runpod.all_dirs())
        single_node_rels = sorted(
            d.relative_to(single_node.root) for d in single_node.all_dirs()
        )
        assert runpod_rels == single_node_rels


class TestMacPodSymmetry:
    """LogManager scp does 1:1 mapping by filename. Names must match."""

    def test_runner_log_name_matches_mac_side(self) -> None:
        layout = PodLayout.from_root("/workspace/runs/r1")
        assert layout.runner_log.name == REMOTE_RUNNER_LOG_NAME

    def test_trainer_stdio_log_name_matches_mac_side(self) -> None:
        layout = PodLayout.from_root("/workspace/runs/r1")
        assert layout.trainer_stdio_log.name == TRAINER_STDIO_LOG_NAME
        # Mac↔Pod parity: same filename on both sides → 1:1 scp mapping.
        assert layout.trainer_stdio_log.name == REMOTE_TRAINER_STDIO_LOG_NAME
        assert REMOTE_TRAINER_STDIO_LOG_NAME == "trainer.stdio.log"


class TestEnsureDirsCommand:
    """``ensure_dirs_command`` produces a single idempotent ``mkdir -p``."""

    def test_command_starts_with_mkdir_p(self) -> None:
        layout = PodLayout.from_root("/workspace/runs/r1")
        cmd = layout.ensure_dirs_command()
        assert cmd.startswith("mkdir -p ")

    def test_command_includes_every_subdirectory(self) -> None:
        layout = PodLayout.from_root("/workspace/runs/r1")
        cmd = layout.ensure_dirs_command()
        for d in layout.all_dirs():
            assert str(d) in cmd

    def test_command_quotes_paths_with_spaces(self) -> None:
        """Operator could legitimately use a path with spaces (especially
        on single_node where workspace is user-configured)."""
        layout = PodLayout.from_root("/home/user/ML Runs/r1")
        cmd = layout.ensure_dirs_command()
        # shlex.quote uses single quotes for paths containing spaces
        assert "'/home/user/ML Runs/r1/logs'" in cmd

    def test_command_quotes_paths_with_metacharacters(self) -> None:
        """Defensive: shell metachars in root must not break."""
        layout = PodLayout.from_root("/home/user/runs/r1$X;rm")
        cmd = layout.ensure_dirs_command()
        # Single-quoted, no expansion possible
        assert "$X" not in cmd or "'" in cmd


class TestRegression:
    """Anchor scenarios that motivated this design."""

    def test_resume_does_not_collide_logs(self) -> None:
        """Pre-fix bug: two sequential runs on the same pod overwrote
        /workspace/runner.log. Post-fix: each run lives under its own
        runs/<run_id>/ tree."""
        run1 = PodLayout.from_root("/workspace/runs/run_2026_05_01_v1")
        run2 = PodLayout.from_root("/workspace/runs/run_2026_05_01_v2")
        assert run1.runner_log != run2.runner_log
        assert run1.trainer_stdio_log != run2.trainer_stdio_log
        # And neither path is the legacy global path
        assert str(run1.runner_log) != "/workspace/runner.log"
        assert str(run2.runner_log) != "/workspace/runner.log"
