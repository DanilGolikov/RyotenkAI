"""Tests for src.utils.logs_layout.LogLayout."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.utils.logs_layout import (
    LOGS_DIR_NAME,
    PIPELINE_LOG_NAME,
    REMOTE_TRAINING_LOG_NAME,
    REMOTE_TRAINING_LOG_PATHS_KEY,
    STAGE_LOG_PATHS_KEY,
    STAGE_LOG_SUFFIX,
    LogLayout,
    _slugify,
)

# ---------------------------------------------------------------------------
# Positive — path construction
# ---------------------------------------------------------------------------

def test_logs_dir_is_attempt_dir_slash_logs(tmp_path: Path) -> None:
    layout = LogLayout(tmp_path / "attempt_1")
    assert layout.logs_dir == tmp_path / "attempt_1" / LOGS_DIR_NAME


def test_pipeline_log_path(tmp_path: Path) -> None:
    layout = LogLayout(tmp_path / "attempt_1")
    assert layout.pipeline_log == tmp_path / "attempt_1" / "logs" / PIPELINE_LOG_NAME


def test_stage_log_uses_stage_name_and_suffix(tmp_path: Path) -> None:
    layout = LogLayout(tmp_path / "attempt_1")
    assert layout.stage_log("dataset_validator") == (
        tmp_path / "attempt_1" / "logs" / f"dataset_validator{STAGE_LOG_SUFFIX}"
    )


def test_remote_training_log_path(tmp_path: Path) -> None:
    layout = LogLayout(tmp_path / "attempt_1")
    assert layout.remote_training_log == tmp_path / "attempt_1" / "logs" / REMOTE_TRAINING_LOG_NAME


def test_attempt_dir_property_preserves_input(tmp_path: Path) -> None:
    layout = LogLayout(tmp_path / "attempt_1")
    assert layout.attempt_dir == tmp_path / "attempt_1"


# ---------------------------------------------------------------------------
# Positive — side effects
# ---------------------------------------------------------------------------

def test_ensure_logs_dir_creates_missing_directory(tmp_path: Path) -> None:
    attempt_dir = tmp_path / "attempt_1"
    attempt_dir.mkdir()
    layout = LogLayout(attempt_dir)

    assert not (attempt_dir / LOGS_DIR_NAME).exists()
    result = layout.ensure_logs_dir()
    assert result == attempt_dir / LOGS_DIR_NAME
    assert result.is_dir()


def test_ensure_logs_dir_idempotent(tmp_path: Path) -> None:
    layout = LogLayout(tmp_path / "attempt_1")
    layout.ensure_logs_dir()
    layout.ensure_logs_dir()  # must not raise


def test_ensure_logs_dir_creates_parent_chain(tmp_path: Path) -> None:
    """attempt_dir itself may not exist — ensure_logs_dir creates the chain."""
    deep = tmp_path / "run" / "attempts" / "attempt_1"
    assert not deep.exists()
    LogLayout(deep).ensure_logs_dir()
    assert (deep / LOGS_DIR_NAME).is_dir()


# ---------------------------------------------------------------------------
# Negative
# ---------------------------------------------------------------------------

def test_stage_log_empty_name_rejected(tmp_path: Path) -> None:
    layout = LogLayout(tmp_path / "attempt_1")
    with pytest.raises(ValueError, match="stage_name must be non-empty"):
        layout.stage_log("")


# ---------------------------------------------------------------------------
# Boundary — input types / unusual paths
# ---------------------------------------------------------------------------

def test_accepts_path_argument_as_posix_str(tmp_path: Path) -> None:
    """LogLayout coerces str into Path via Path() — used by consumers that pass both."""
    layout = LogLayout(Path(str(tmp_path / "attempt_1")))
    assert layout.pipeline_log.is_absolute()


def test_stage_name_with_hyphen_and_numbers(tmp_path: Path) -> None:
    """Slug collapses any non-[a-z0-9] run into a single underscore."""
    layout = LogLayout(tmp_path / "attempt_1")
    assert layout.stage_log("inference-deployer-v2").name == "inference_deployer_v2.log"


def test_relative_returns_path_relative_to_attempt_dir(tmp_path: Path) -> None:
    layout = LogLayout(tmp_path / "attempt_1")
    rel = layout.relative(layout.pipeline_log)
    assert rel == f"{LOGS_DIR_NAME}/{PIPELINE_LOG_NAME}"


def test_relative_returns_absolute_for_outside_path(tmp_path: Path) -> None:
    """Paths outside attempt_dir must fall back to absolute — registry stays meaningful."""
    layout = LogLayout(tmp_path / "attempt_1")
    outside = tmp_path / "somewhere_else.log"
    assert layout.relative(outside) == str(outside)


# ---------------------------------------------------------------------------
# Invariants
# ---------------------------------------------------------------------------

def test_invariant_logs_dir_is_always_child_of_attempt_dir(tmp_path: Path) -> None:
    layout = LogLayout(tmp_path / "attempt_1")
    assert layout.logs_dir.parent == layout.attempt_dir


def test_invariant_all_log_paths_live_under_logs_dir(tmp_path: Path) -> None:
    layout = LogLayout(tmp_path / "attempt_1")
    for path in (layout.pipeline_log, layout.remote_training_log, layout.stage_log("x")):
        assert path.parent == layout.logs_dir


def test_invariant_stage_log_name_ends_with_suffix(tmp_path: Path) -> None:
    layout = LogLayout(tmp_path / "attempt_1")
    assert layout.stage_log("whatever").name.endswith(STAGE_LOG_SUFFIX)


# ---------------------------------------------------------------------------
# Registry (stage_log_registry) — positive + combinatorial
# ---------------------------------------------------------------------------

def test_registry_without_remote_training_has_only_stage_key(tmp_path: Path) -> None:
    layout = LogLayout(tmp_path / "attempt_1")
    registry = layout.stage_log_registry("gpu_deployer", include_remote_training=False)
    assert registry == {STAGE_LOG_PATHS_KEY: f"{LOGS_DIR_NAME}/gpu_deployer.log"}


def test_registry_with_remote_training_adds_remote_key(tmp_path: Path) -> None:
    layout = LogLayout(tmp_path / "attempt_1")
    registry = layout.stage_log_registry("training_monitor", include_remote_training=True)
    assert registry == {
        STAGE_LOG_PATHS_KEY: f"{LOGS_DIR_NAME}/training_monitor.log",
        REMOTE_TRAINING_LOG_PATHS_KEY: f"{LOGS_DIR_NAME}/{REMOTE_TRAINING_LOG_NAME}",
    }


@pytest.mark.parametrize("stage_name", ["dataset_validator", "gpu_deployer", "model_retriever"])
@pytest.mark.parametrize("include_remote_training", [True, False])
def test_registry_combinations(tmp_path: Path, stage_name: str, include_remote_training: bool) -> None:
    layout = LogLayout(tmp_path / "attempt_1")
    registry = layout.stage_log_registry(stage_name, include_remote_training=include_remote_training)

    assert STAGE_LOG_PATHS_KEY in registry
    assert registry[STAGE_LOG_PATHS_KEY].endswith(f"{stage_name}.log")
    assert (REMOTE_TRAINING_LOG_PATHS_KEY in registry) is include_remote_training


def test_registry_returns_relative_paths_only(tmp_path: Path) -> None:
    """Paths in registry must be relative (portable across machines)."""
    layout = LogLayout(tmp_path / "attempt_1")
    registry = layout.stage_log_registry("x", include_remote_training=True)
    for value in registry.values():
        assert not value.startswith("/"), value


# ---------------------------------------------------------------------------
# _slugify — the core normalization function
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "raw,expected",
    [
        ("Dataset Validator", "dataset_validator"),
        ("GPU Deployer", "gpu_deployer"),
        ("Training Monitor", "training_monitor"),
        ("Model Retriever", "model_retriever"),
        ("Model Evaluator", "model_evaluator"),
        ("Inference Deployer", "inference_deployer"),
        ("already_slug", "already_slug"),
        ("MiXeD CaSe", "mixed_case"),
        ("multi   space", "multi_space"),
        ("  leading trailing  ", "leading_trailing"),
        ("dashes-and_unders", "dashes_and_unders"),
        ("with.dots.and/slashes", "with_dots_and_slashes"),
        ("unicode-кириллица", "unicode"),
        ("numbers 42 inline", "numbers_42_inline"),
    ],
)
def test_slugify_normalizes_all_canonical_stage_names(raw: str, expected: str) -> None:
    assert _slugify(raw) == expected


def test_slugify_empty_falls_back_to_stage() -> None:
    """Empty or purely non-alphanumeric input must not yield an empty filename."""
    assert _slugify("") == "stage"
    assert _slugify("   ") == "stage"
    assert _slugify("---") == "stage"
    assert _slugify("!!!") == "stage"


def test_stage_log_uses_slug_for_file_name(tmp_path: Path) -> None:
    """StageNames with spaces must produce filesystem-friendly slug filenames."""
    layout = LogLayout(tmp_path / "attempt_1")
    assert layout.stage_log("Dataset Validator").name == "dataset_validator.log"
    assert layout.stage_log("GPU Deployer").name == "gpu_deployer.log"


def test_stage_log_registry_stores_slug_path(tmp_path: Path) -> None:
    """Registry value (persisted to state) must use the slug, not the raw name."""
    layout = LogLayout(tmp_path / "attempt_1")
    registry = layout.stage_log_registry("Dataset Validator")
    assert registry[STAGE_LOG_PATHS_KEY] == "logs/dataset_validator.log"


def test_stage_log_is_deterministic_across_calls(tmp_path: Path) -> None:
    layout = LogLayout(tmp_path / "attempt_1")
    a = layout.stage_log("Dataset Validator")
    b = layout.stage_log("Dataset Validator")
    assert a == b


def test_different_inputs_slugifying_to_same_name_collide_intentionally(tmp_path: Path) -> None:
    """Invariant: any name that slugifies the same must target the same file.

    Documented behavior — collisions are acceptable for the current StageNames
    set (they are all distinct when slugified) but must be explicit."""
    layout = LogLayout(tmp_path / "attempt_1")
    assert layout.stage_log("Dataset Validator") == layout.stage_log("dataset_validator")
    assert layout.stage_log("Dataset Validator") == layout.stage_log("DATASET-VALIDATOR")
