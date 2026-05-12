"""
Trainer crash-observability install — FileHandler attach to ``training.log``.

PR1 of the "trainer log file + Mac event mirror + frontend live-tail"
plan: trainer needs to write its own ``training.log`` so that
``log_manager`` can scp it onto the Mac as a post-mortem artefact.
The supervisor's PIPE-pump → ``trainer_log`` event-bus stream
remains as a parallel realtime channel; this test suite only
exercises the file-write side.

Categories covered (per project test policy):
* Positive — env var present → FileHandler attached + actually writes
* Negative — env var absent / path unwritable → no crash, training
  continues
* Boundary — existing file appended (logging.FileHandler default)
* Invariant — repeat install does not pile up duplicate handlers
* Dependency-error — ``setup_logger`` raises → install swallows it
* Regression — faulthandler still attaches when training-log path
  fails or is absent
* Logic-specific — env-var precedence, path passed through verbatim
"""

from __future__ import annotations

import contextlib
import logging
import os
from pathlib import Path
from typing import TYPE_CHECKING
from unittest.mock import patch

import pytest

from ryotenkai_pod.trainer.run_training import _install_crash_observability
from ryotenkai_shared.utils.logger import logger as ryotenkai_logger

if TYPE_CHECKING:
    from collections.abc import Iterator


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _isolate_logger_handlers() -> Iterator[None]:
    """Snapshot the ``ryotenkai`` logger's handler list before each test
    and restore it after. ``_install_crash_observability`` re-attaches
    via ``setup_logger`` which clears + rebuilds the handler chain;
    without this snapshot test order would matter.
    """
    saved = list(ryotenkai_logger.handlers)
    saved_level = ryotenkai_logger.level
    try:
        yield
    finally:
        for h in list(ryotenkai_logger.handlers):
            with contextlib.suppress(Exception):
                h.close()
        ryotenkai_logger.handlers = saved
        ryotenkai_logger.level = saved_level


@pytest.fixture
def isolated_env(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Strip the two env vars install reads so each test has a clean
    starting state."""
    monkeypatch.delenv("RYOTENKAI_TRAINING_LOG_PATH", raising=False)
    monkeypatch.delenv("PYTHONFAULTHANDLER_PATH", raising=False)
    yield


# ---------------------------------------------------------------------------
# Positive
# ---------------------------------------------------------------------------


# DEAD (xfail audit 2026-05-12): removed 5 FileHandler tests — production
# _install_crash_observability no longer reads RYOTENKAI_TRAINING_LOG_PATH or
# attaches a FileHandler. Supervisor captures trainer stdio into
# trainer.stdio.log instead. The remaining tests (no-env-var/faulthandler/
# setup-logger paths) cover the observability install surface that still exists.


# ---------------------------------------------------------------------------
# Negative
# ---------------------------------------------------------------------------


def test_install_no_env_var_no_file_handler(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    isolated_env: None,
) -> None:
    """Without env var, install does not attach any FileHandler.

    Trainer remains usable (stdout still works through Supervisor PIPE),
    we just don't get a file artefact. This is the standalone /
    unit-test path.
    """
    # Sanity: env var really is missing
    assert "RYOTENKAI_TRAINING_LOG_PATH" not in os.environ
    _install_crash_observability()
    file_handlers = [
        h for h in ryotenkai_logger.handlers
        if isinstance(h, logging.FileHandler)
    ]
    assert file_handlers == [], (
        f"unexpected FileHandlers attached: {file_handlers!r}"
    )


def test_install_unwritable_path_warns_not_raises(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    isolated_env: None,
) -> None:
    """Path inside non-existent + unwriteable directory → install
    must NOT raise. Trainer continues. Warning is logged but a
    test-time check on the warning is brittle — we only assert
    no exception escapes.
    """
    bad = tmp_path / "definitely" / "does" / "not" / "exist" / "training.log"
    # Make parent unwriteable to force the OSError path even if mkdir
    # would create it. We mock setup_logger to bypass the parent.mkdir
    # behaviour and force OSError on FileHandler open.
    monkeypatch.setenv("RYOTENKAI_TRAINING_LOG_PATH", str(bad))

    real_setup = __import__("ryotenkai_shared.utils.logger", fromlist=["setup_logger"]).setup_logger

    def _raising_setup(*args, **kwargs):
        if "log_file" in kwargs and kwargs["log_file"] is not None:
            raise OSError("simulated read-only fs")
        return real_setup(*args, **kwargs)

    with patch("ryotenkai_shared.utils.logger.setup_logger", side_effect=_raising_setup):
        # Must not raise — install is best-effort.
        _install_crash_observability()


# ---------------------------------------------------------------------------
# Dependency-error
# ---------------------------------------------------------------------------


def test_install_when_setup_logger_raises_does_not_kill_training(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    isolated_env: None,
) -> None:
    """If ``setup_logger`` raises an unexpected (non-OS) exception,
    install must swallow it. Observability infra is best-effort —
    a broken FileHandler must not prevent training from starting.
    """
    target = tmp_path / "training.log"
    monkeypatch.setenv("RYOTENKAI_TRAINING_LOG_PATH", str(target))

    def _bad(*args, **kwargs):
        raise RuntimeError("simulated unexpected error")

    with patch("ryotenkai_shared.utils.logger.setup_logger", side_effect=_bad):
        _install_crash_observability()  # MUST NOT raise


# ---------------------------------------------------------------------------
# Regression
# ---------------------------------------------------------------------------


def test_install_does_not_break_faulthandler(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    isolated_env: None,
) -> None:
    """Adding the FileHandler block must not regress the existing
    faulthandler.enable invocation. PYTHONFAULTHANDLER_PATH is still
    read and ``faulthandler.is_enabled()`` is true after install.
    """
    import faulthandler

    fault_target = tmp_path / "training.faulthandler.log"
    monkeypatch.setenv("PYTHONFAULTHANDLER_PATH", str(fault_target))
    _install_crash_observability()
    assert faulthandler.is_enabled()


def test_install_works_when_only_faulthandler_env_set(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    isolated_env: None,
) -> None:
    """Trainer must function with only PYTHONFAULTHANDLER_PATH set
    (no training-log var). The two env vars are independent — failing
    either path must not poison the other.
    """
    monkeypatch.setenv(
        "PYTHONFAULTHANDLER_PATH", str(tmp_path / "fh.log"),
    )
    _install_crash_observability()  # MUST NOT raise


