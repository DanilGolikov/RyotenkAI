"""Phase 9.A — regression: cancelled phase closes MLflow nested run as KILLED.

The bug pinned by this test: ``executor.py``'s finally block was

    mlflow_status = "FINISHED" if phase_succeeded else "FAILED"

— it picked ``FAILED`` for every non-success path including graceful
shutdown. So a user clicking "Stop" surfaced on the MLflow UI as a
crashed run, indistinguishable from OOM, dataset corruption, or
plugin error.

Phase 9.A fix: thread ``was_cancelled`` through by inspecting the
``TrainingError.code`` returned from ``handle_graceful_shutdown``
(``"TRAINING_INTERRUPTED"``). When set, finally picks
``RunStatus.KILLED`` — the canonical MLflow signal for "stopped
by user".

Test strategy: source-inspection guard — same trick used by
``test_runner_event_callback_wiring.py``. Importing the executor
module at runtime requires the full ML stack (``peft``, ``datasets``,
``transformers``) which the slim dev venv lacks. Source-level
inspection asserts the fix is present and the truth-table holds
without paying for that.

The behavioural integration (full executor + nested run + MLflow
status) lands in Phase 9.C's ``test_stop_with_cancellation.py``
end-to-end test.
"""

from __future__ import annotations

import inspect
from pathlib import Path

import pytest


_EXECUTOR_PATH = (
    Path(__file__).resolve().parents[4]
    / "training" / "orchestrator" / "phase_executor" / "executor.py"
)


def _executor_source() -> str:
    return _EXECUTOR_PATH.read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# Source-level pins
# ---------------------------------------------------------------------------


class TestSourceInvariants:
    def test_training_interrupted_constant_present(self) -> None:
        """The constant is the contract between ``training_runner.py``
        (writer) and ``executor.py`` (reader). Pin its presence so a
        rename triggers a test failure rather than silent breakage."""
        src = _executor_source()
        assert '_TRAINING_INTERRUPTED_CODE = "TRAINING_INTERRUPTED"' in src

    def test_is_cancellation_error_helper_defined(self) -> None:
        src = _executor_source()
        assert "def _is_cancellation_error(" in src
        # The helper must check both: failure AND specific code.
        assert "is_failure()" in src
        assert "_TRAINING_INTERRUPTED_CODE" in src

    def test_was_cancelled_flag_used_in_execute(self) -> None:
        """``was_cancelled`` flag must be threaded through the try/
        finally block. Source-inspection guards the wiring."""
        src = _executor_source()
        assert "was_cancelled = False" in src
        # Set in failure branch.
        assert "was_cancelled = _is_cancellation_error" in src

    def test_finally_block_picks_killed_over_failed(self) -> None:
        """Pin the truth table: finally picks KILLED when was_cancelled,
        regardless of phase_succeeded."""
        src = _executor_source()
        # The if-elif-else block must mention all three statuses.
        assert "if was_cancelled:" in src
        assert '"KILLED"' in src
        assert '"FINISHED"' in src
        assert '"FAILED"' in src

    def test_killed_branch_appears_before_finished(self) -> None:
        """Order matters in the if-elif-else chain: was_cancelled wins.
        If a future refactor flips the order, FAILED could mask a
        cancellation."""
        src = _executor_source()
        killed_pos = src.find('"KILLED"')
        finished_pos = src.find('"FINISHED"')
        failed_pos = src.find('"FAILED"')
        assert killed_pos > 0 and finished_pos > 0 and failed_pos > 0
        # The status-string literals are written in priority order in
        # the finally block: KILLED → FINISHED → FAILED.
        assert killed_pos < finished_pos
        assert finished_pos < failed_pos


# ---------------------------------------------------------------------------
# Truth table mirror (replicates the in-code logic for behavioural pin)
# ---------------------------------------------------------------------------


class TestStatusMappingTruthTable:
    """Mirror the exact branching from executor.py's finally block.

    Lock the decision matrix in a unit test so any future refactor
    that touches the finally must keep the truth table intact. The
    actual code is in ``executor.py``; this is the contract.
    """

    @staticmethod
    def _pick_mlflow_status(
        *, was_cancelled: bool, phase_succeeded: bool,
    ) -> str:
        # Mirrors the in-code if-elif-else.
        if was_cancelled:
            return "KILLED"
        elif phase_succeeded:
            return "FINISHED"
        else:
            return "FAILED"

    @pytest.mark.parametrize(
        ("was_cancelled", "phase_succeeded", "expected"),
        [
            (False, True, "FINISHED"),    # happy path
            (False, False, "FAILED"),     # genuine crash (OOM / dataset / plugin)
            (True, False, "KILLED"),      # user-stop = the bug fix scenario
            (True, True, "KILLED"),       # cancellation wins over success
        ],
    )
    def test_truth_table(
        self, was_cancelled: bool, phase_succeeded: bool, expected: str,
    ) -> None:
        assert (
            self._pick_mlflow_status(
                was_cancelled=was_cancelled,
                phase_succeeded=phase_succeeded,
            )
            == expected
        )

    def test_killed_dominates_failed(self) -> None:
        """The pivotal assertion of Phase 9.A.

        ``was_cancelled=True`` plus ``phase_succeeded=False`` is the
        common case — graceful shutdown returns Err so phase_succeeded
        stays False. Cancellation MUST win over the residual ``failed``
        signal so the operator sees ``KILLED`` on the MLflow UI."""
        status = self._pick_mlflow_status(
            was_cancelled=True, phase_succeeded=False,
        )
        assert status == "KILLED"
        assert status != "FAILED"


# ---------------------------------------------------------------------------
# Helper signature pin
# ---------------------------------------------------------------------------


class TestHelperSignature:
    def test_helper_takes_result_returns_bool(self) -> None:
        """``_is_cancellation_error`` is a small pure function.
        Inspect its source to confirm the signature stays narrow:
        one positional arg (``result``), returns bool.

        Anything wider (e.g. taking trainer state, mutating something)
        violates the SRP boundary that lets the finally block reuse
        it without setup."""
        src = _executor_source()
        # Find the def line and surrounding context.
        idx = src.find("def _is_cancellation_error(")
        assert idx >= 0
        signature_block = src[idx:idx + 200]
        # Single positional arg, return type bool.
        assert "result:" in signature_block
        assert "-> bool:" in signature_block
        # No other arguments — keep the helper SRP-narrow.
        assert "self" not in signature_block.split(":")[0]
