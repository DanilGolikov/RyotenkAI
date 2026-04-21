from __future__ import annotations

import importlib
import logging
import sys
from pathlib import Path

import pytest

from src.utils.logger import (
    _current_stage,
    get_run_log_dir,
    get_run_log_layout,
    init_run_logging,
    setup_logger,
    stage_logging_context,
)
from src.utils.logs_layout import LogLayout

# Re-bind the actual module (sys.modules bypasses the name shadowing done by
# src/utils/__init__.py which exposes `logger` as a Logger instance under the
# same attribute name as the submodule).
logger_module = importlib.import_module("src.utils.logger")


# ---------------------------------------------------------------------------
# Pre-existing — console handler wiring
# ---------------------------------------------------------------------------

def test_setup_logger_uses_stderr_for_console_handler() -> None:
    logger = setup_logger("ryotenkai.test_logger_stream", level=logging.INFO, log_file=None, use_color=False)

    stream_handlers = [
        handler
        for handler in logger.handlers
        if isinstance(handler, logging.StreamHandler) and not isinstance(handler, logging.FileHandler)
    ]

    assert len(stream_handlers) == 1
    assert stream_handlers[0].stream is sys.__stderr__


# ---------------------------------------------------------------------------
# Shared fixture — isolates global module state between tests
# ---------------------------------------------------------------------------

@pytest.fixture
def _fresh_run_logging(monkeypatch: pytest.MonkeyPatch):
    """Isolate module-level run/logging state and restore handlers around the test."""
    monkeypatch.setattr(logger_module, "_run_name", None)
    monkeypatch.setattr(logger_module, "_run_log_dir", None)
    monkeypatch.setattr(logger_module, "_run_log_layout", None)
    monkeypatch.setattr(logger_module, "_enable_file_logs", True)
    base = logging.getLogger("ryotenkai")
    saved_handlers = list(base.handlers)
    base.handlers.clear()
    try:
        yield
    finally:
        for handler in list(base.handlers):
            base.removeHandler(handler)
        for handler in saved_handlers:
            base.addHandler(handler)


# ---------------------------------------------------------------------------
# init_run_logging / get_run_log_* — positive, negative, regression
# ---------------------------------------------------------------------------

def test_init_run_logging_creates_pipeline_log_under_logs_dir(
    tmp_path: Path, _fresh_run_logging
) -> None:
    attempt_dir = tmp_path / "run_a" / "attempts" / "attempt_1"
    returned = init_run_logging("run_a", log_dir=attempt_dir)

    # Returns the attempt dir (not logs/), so callers like log_manager keep working.
    assert returned == attempt_dir
    # pipeline.log is now under logs/, not at attempt root.
    assert (attempt_dir / "logs" / "pipeline.log").exists()
    assert not (attempt_dir / "pipeline.log").exists()


def test_get_run_log_layout_returns_layout_after_init(
    tmp_path: Path, _fresh_run_logging
) -> None:
    init_run_logging("run_b", log_dir=tmp_path / "attempt_1")
    layout = get_run_log_layout()
    assert isinstance(layout, LogLayout)
    assert layout.attempt_dir == tmp_path / "attempt_1"


def test_get_run_log_layout_before_init_raises(_fresh_run_logging) -> None:
    with pytest.raises(RuntimeError, match="Run logging not initialized"):
        get_run_log_layout()


def test_get_run_log_dir_before_init_raises(_fresh_run_logging) -> None:
    with pytest.raises(RuntimeError, match="Run logging not initialized"):
        get_run_log_dir()


def test_init_run_logging_rejects_empty_name(_fresh_run_logging) -> None:
    with pytest.raises(ValueError, match="run_name must be a non-empty string"):
        init_run_logging("")


def test_init_run_logging_disabled_skips_file_handler(
    tmp_path: Path, _fresh_run_logging, monkeypatch: pytest.MonkeyPatch
) -> None:
    """HELIX_NO_FILE_LOGS=1 must skip FileHandler but still expose layout."""
    monkeypatch.setattr(logger_module, "_enable_file_logs", False)
    init_run_logging("run_c", log_dir=tmp_path / "attempt_1")
    base = logging.getLogger("ryotenkai")
    file_handlers = [h for h in base.handlers if isinstance(h, logging.FileHandler)]
    assert file_handlers == []
    # Layout is still available for downstream consumers (log_manager).
    assert get_run_log_layout().attempt_dir == tmp_path / "attempt_1"


# ---------------------------------------------------------------------------
# stage_logging_context — positive, negative, invariants
# ---------------------------------------------------------------------------

def test_stage_logging_context_creates_stage_file_and_routes_records(
    tmp_path: Path, _fresh_run_logging
) -> None:
    init_run_logging("run_d", log_dir=tmp_path / "attempt_1")
    layout = get_run_log_layout()
    base = logging.getLogger("ryotenkai")

    with stage_logging_context("dataset_validator", layout):
        base.info("inside dataset_validator")
    base.info("outside any stage")

    stage_log = layout.stage_log("dataset_validator")
    assert stage_log.exists()
    stage_content = stage_log.read_text(encoding="utf-8")
    assert "inside dataset_validator" in stage_content
    assert "outside any stage" not in stage_content

    # The aggregated pipeline.log still captures BOTH messages.
    aggregate = layout.pipeline_log.read_text(encoding="utf-8")
    assert "inside dataset_validator" in aggregate
    assert "outside any stage" in aggregate


def test_stage_logging_context_removes_handler_on_exit(
    tmp_path: Path, _fresh_run_logging
) -> None:
    init_run_logging("run_e", log_dir=tmp_path / "attempt_1")
    layout = get_run_log_layout()
    base = logging.getLogger("ryotenkai")
    before = list(base.handlers)

    with stage_logging_context("x", layout):
        assert len(base.handlers) == len(before) + 1

    assert list(base.handlers) == before


def test_stage_logging_context_removes_handler_on_exception(
    tmp_path: Path, _fresh_run_logging
) -> None:
    init_run_logging("run_f", log_dir=tmp_path / "attempt_1")
    layout = get_run_log_layout()
    base = logging.getLogger("ryotenkai")
    before_count = len(base.handlers)

    with pytest.raises(RuntimeError), stage_logging_context("x", layout):
        raise RuntimeError("boom")

    assert len(base.handlers) == before_count


def test_stage_logging_context_restores_previous_stage(
    tmp_path: Path, _fresh_run_logging
) -> None:
    """Nested contexts: inner sets stage, exit restores outer stage (ContextVar reset)."""
    init_run_logging("run_g", log_dir=tmp_path / "attempt_1")
    layout = get_run_log_layout()

    with stage_logging_context("outer", layout):
        assert _current_stage.get() == "outer"
        with stage_logging_context("inner", layout):
            assert _current_stage.get() == "inner"
        assert _current_stage.get() == "outer"
    assert _current_stage.get() is None


def test_stage_logging_context_empty_stage_name_raises(
    tmp_path: Path, _fresh_run_logging
) -> None:
    init_run_logging("run_h", log_dir=tmp_path / "attempt_1")
    with pytest.raises(ValueError, match="stage_name must be non-empty"):
        with stage_logging_context("", get_run_log_layout()):
            pass


def test_stage_logging_context_respects_file_logs_disabled(
    tmp_path: Path, _fresh_run_logging, monkeypatch: pytest.MonkeyPatch
) -> None:
    """With file logs disabled, no per-stage FileHandler is attached."""
    monkeypatch.setattr(logger_module, "_enable_file_logs", False)
    init_run_logging("run_i", log_dir=tmp_path / "attempt_1")
    base = logging.getLogger("ryotenkai")
    before = list(base.handlers)

    with stage_logging_context("x", get_run_log_layout()):
        assert list(base.handlers) == before


# ---------------------------------------------------------------------------
# Combinatorial — record emitted inside/outside a stage, with/without filter match
# ---------------------------------------------------------------------------

def test_stage_filter_admits_only_matching_stage(
    tmp_path: Path, _fresh_run_logging
) -> None:
    """Record in one stage must not bleed into another stage's handler/file."""
    init_run_logging("run_j", log_dir=tmp_path / "attempt_1")
    layout = get_run_log_layout()
    base = logging.getLogger("ryotenkai")

    with stage_logging_context("stage_a", layout):
        base.info("msg_a")
    with stage_logging_context("stage_b", layout):
        base.info("msg_b")

    a_content = layout.stage_log("stage_a").read_text(encoding="utf-8")
    b_content = layout.stage_log("stage_b").read_text(encoding="utf-8")

    assert "msg_a" in a_content
    assert "msg_b" not in a_content
    assert "msg_b" in b_content
    assert "msg_a" not in b_content


# ---------------------------------------------------------------------------
# Iteration-2 scope: third-party logger capture + pipeline.log on root
# ---------------------------------------------------------------------------


def _clear_extra_root_handlers(saved_root_handlers: list[logging.Handler]) -> None:
    root = logging.getLogger()
    for h in list(root.handlers):
        if h not in saved_root_handlers:
            root.removeHandler(h)


@pytest.fixture
def _fresh_root_logger(_fresh_run_logging, monkeypatch: pytest.MonkeyPatch):
    """Isolate root logger + pipeline_file_handler singleton + third-party
    logger state between tests. Other tests in the suite may import libs
    (mlflow, transformers, paramiko) that install their own handlers or
    change levels; we snapshot/restore them so order doesn't matter."""
    monkeypatch.setattr(logger_module, "_pipeline_file_handler", None)
    root = logging.getLogger()
    saved_root_handlers = list(root.handlers)
    saved_root_level = root.level

    tracked_names = set(logger_module._PROPAGATED_THIRD_PARTY) | set(logger_module._NOISY_LIBRARIES)
    tracked_snapshot: dict[str, tuple[int, bool]] = {
        name: (logging.getLogger(name).level, logging.getLogger(name).propagate)
        for name in tracked_names
    }

    try:
        yield
    finally:
        _clear_extra_root_handlers(saved_root_handlers)
        root.setLevel(saved_root_level)
        for name, (level, propagate) in tracked_snapshot.items():
            lg = logging.getLogger(name)
            lg.setLevel(level)
            lg.propagate = propagate


def _isolated_third_party_logger(name: str) -> logging.Logger:
    """Make a fresh third-party-style logger (not ryotenkai.*) with a
    guaranteed INFO level and propagation enabled. Isolates the test from
    state that other imports in the suite may have set."""
    lg = logging.getLogger(name)
    lg.setLevel(logging.INFO)
    lg.propagate = True
    return lg


def test_pipeline_log_captures_third_party_logger(tmp_path: Path, _fresh_root_logger) -> None:
    """Any non-ryotenkai logger must land in pipeline.log (via root FileHandler)."""
    init_run_logging("run_tp", log_dir=tmp_path / "attempt_1")
    layout = get_run_log_layout()

    _isolated_third_party_logger("third_party_marker_aaa").info("mlflow says hello")

    pipeline_content = layout.pipeline_log.read_text(encoding="utf-8")
    assert "mlflow says hello" in pipeline_content


def test_stage_log_captures_third_party_logger(tmp_path: Path, _fresh_root_logger) -> None:
    """Inside a stage context, third-party loggers land in <stage>.log."""
    init_run_logging("run_tp2", log_dir=tmp_path / "attempt_1")
    layout = get_run_log_layout()

    with stage_logging_context("Dataset Validator", layout):
        _isolated_third_party_logger("third_party_marker_bbb").info("ssh connect")
        logging.getLogger("ryotenkai.x").info("our own msg")

    stage_content = layout.stage_log("Dataset Validator").read_text(encoding="utf-8")
    assert "ssh connect" in stage_content, "third-party must be captured in stage.log"
    assert "our own msg" in stage_content, "ryotenkai.* must also be captured in stage.log"

    # Filename must be slugified.
    assert layout.stage_log("Dataset Validator").name == "dataset_validator.log"


def test_stage_log_ryotenkai_record_appears_only_once(tmp_path: Path, _fresh_root_logger) -> None:
    """ryotenkai handler writes the record; root handler's ExcludeRyotenkaiFilter
    must keep it from being written a second time to the same file."""
    init_run_logging("run_dup", log_dir=tmp_path / "attempt_1")
    layout = get_run_log_layout()

    with stage_logging_context("stage_x", layout):
        logging.getLogger("ryotenkai.uniqmod").info("UNIQUE_MARKER_X")

    content = layout.stage_log("stage_x").read_text(encoding="utf-8")
    assert content.count("UNIQUE_MARKER_X") == 1


def test_third_party_not_in_console_stream(tmp_path: Path, _fresh_root_logger) -> None:
    """init_run_logging installs the console StreamHandler on ``ryotenkai``
    only, never on root. (pytest caplog adds its own LogCaptureHandler to root
    — that's fine; we only check that WE didn't.)"""
    init_run_logging("run_con", log_dir=tmp_path / "attempt_1")

    ryotenkai = logging.getLogger("ryotenkai")
    console_handlers = [
        h for h in ryotenkai.handlers
        if isinstance(h, logging.StreamHandler)
        and not isinstance(h, logging.FileHandler)
        and getattr(h, "stream", None) is sys.__stderr__
    ]
    assert console_handlers, "ryotenkai must keep its console StreamHandler on sys.__stderr__"

    root = logging.getLogger()
    root_console = [
        h for h in root.handlers
        if isinstance(h, logging.StreamHandler)
        and not isinstance(h, logging.FileHandler)
        and getattr(h, "stream", None) is sys.__stderr__
    ]
    assert not root_console, "init_run_logging must not install a console handler on root"


def test_noisy_libraries_are_quieted(tmp_path: Path, _fresh_root_logger) -> None:
    """After init, httpx/urllib3/filelock/botocore must not emit INFO anymore."""
    # Pre-set INFO to prove the init actually raises the level.
    for name in logger_module._NOISY_LIBRARIES:
        logging.getLogger(name).setLevel(logging.INFO)

    init_run_logging("run_quiet", log_dir=tmp_path / "attempt_1")

    for name in logger_module._NOISY_LIBRARIES:
        assert logging.getLogger(name).level == logging.WARNING, name


def test_third_party_propagation_is_forced_on(tmp_path: Path, _fresh_root_logger) -> None:
    """Libraries like mlflow set propagate=False on import — init must re-enable
    propagation so their records reach our root-attached FileHandler."""
    # Simulate mlflow disabling propagation (as it does in reality).
    for name in logger_module._PROPAGATED_THIRD_PARTY:
        logging.getLogger(name).propagate = False

    init_run_logging("run_prop", log_dir=tmp_path / "attempt_1")

    for name in logger_module._PROPAGATED_THIRD_PARTY:
        assert logging.getLogger(name).propagate is True, name


def test_root_level_raised_to_enable_info_capture(tmp_path: Path, _fresh_root_logger) -> None:
    """Root logger defaults to WARNING — init must lower the gate to INFO (default)
    so third-party INFO actually reaches pipeline.log / stage.log handlers."""
    logging.getLogger().setLevel(logging.WARNING)

    init_run_logging("run_root_lvl", log_dir=tmp_path / "attempt_1")

    assert logging.getLogger().level <= logging.INFO


def test_pipeline_log_path_moved_to_logs_subdir(tmp_path: Path, _fresh_root_logger) -> None:
    """Regression for iteration-1: pipeline.log must live under logs/."""
    init_run_logging("run_loc", log_dir=tmp_path / "attempt_1")
    layout = get_run_log_layout()

    # After init, one INFO record from init_run_logging itself must already
    # be in the file (file logging init message).
    assert layout.pipeline_log.exists()
    assert layout.pipeline_log.parent.name == "logs"


def test_set_log_level_reattaches_pipeline_handler_at_new_level(
    tmp_path: Path, _fresh_root_logger
) -> None:
    """Changing log level at runtime re-wires the pipeline handler
    and the new DEBUG records reach pipeline.log."""
    init_run_logging("run_setlevel", log_dir=tmp_path / "attempt_1")

    logger_module.set_log_level("DEBUG")

    layout = get_run_log_layout()
    test_logger = _isolated_third_party_logger("third_party_marker_ccc")
    test_logger.setLevel(logging.DEBUG)
    test_logger.debug("dbg_marker_XYZ")
    assert "dbg_marker_XYZ" in layout.pipeline_log.read_text(encoding="utf-8")
