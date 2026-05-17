"""Tests for the Phase H1 ``_log_pipeline_outcome`` outcome-block helper.

Coverage split by 7-class policy:

1. **Positive**          — success/failure formats are well-formed.
2. **Negative**          — missing fields are gracefully omitted.
3. **Boundary**          — empty context, no detail, no stage info.
4. **Invariants**        — emits via logger.error on failure, logger.info on success.
5. **Regression**        — context private keys (``_xxx``) are stripped.
6. **Logic-specific**    — _extract_attempt_fields pulls H1-stamped keys only.
7. **Combinatorial**     — header lines for success/failure attempt-numbering matrix.
"""

from __future__ import annotations

import io
import json
import logging
from typing import Any

import pytest

from ryotenkai_control.pipeline.worker import (
    _extract_attempt_fields,
    _format_attempt_summary,
    _log_pipeline_outcome,
)
from ryotenkai_shared.errors import InternalError, PipelineStageFailedError


# ---------------------------------------------------------------------------
# 1. Positive — happy-path format
# ---------------------------------------------------------------------------


class TestPositive:
    def test_success_header_format(self) -> None:
        text = _format_attempt_summary(
            None,
            attempt_no=2,
            stage_name=None,
            stage_idx=None,
            stage_total=None,
            trace_id=None,
            request_id=None,
            detail=None,
            context=None,
            duration_seconds=12.34,
            code=None,
            title=None,
            total_stages=8,
        )
        lines = text.splitlines()
        assert lines[0] == "=" * 80
        assert lines[1].startswith("Pipeline COMPLETED at attempt 2 (wall=12.3s, stages=8/8)")
        assert lines[2] == "=" * 80

    def test_failure_full_block(self) -> None:
        exc = PipelineStageFailedError(detail="GPU not found", context={})
        text = _format_attempt_summary(
            exc,
            attempt_no=3,
            stage_name="gpu_deployer",
            stage_idx=2,
            stage_total=8,
            trace_id="abc123",
            request_id="req456",
            detail="GPU not found",
            context={"reason": "no_quota"},
            duration_seconds=42.5,
            code="PIPELINE_STAGE_FAILED",
            title="Pipeline stage failed",
        )
        assert "Pipeline FAILED at attempt 3" in text
        assert "  error:    Pipeline stage failed" in text
        assert "  code:     PIPELINE_STAGE_FAILED" in text
        assert "  stage:    gpu_deployer (Stage 3/8)" in text
        assert "  trace:    abc123" in text
        assert "  request:  req456" in text
        assert "  detail:   GPU not found" in text
        assert '"reason": "no_quota"' in text
        assert "  duration: 42.5s" in text
        assert "  attempts: 3" in text
        assert "  outcome:  FAILED" in text


# ---------------------------------------------------------------------------
# 2. Negative — missing fields omitted gracefully
# ---------------------------------------------------------------------------


class TestNegative:
    def test_pre_stage_failure_omits_stage_line(self) -> None:
        text = _format_attempt_summary(
            InternalError(detail="config missing"),
            attempt_no=None,
            stage_name=None,
            stage_idx=None,
            stage_total=None,
            trace_id=None,
            request_id="r-x",
            detail="config missing",
            context=None,
            duration_seconds=None,
            code="INTERNAL_ERROR",
            title="Internal error",
        )
        assert "  stage:" not in text
        # trace defaults to "-"
        assert "  trace:    -" in text
        # attempts: line omitted when attempt_no is None
        assert "  attempts:" not in text
        # duration: line omitted when seconds are None
        assert "  duration:" not in text

    def test_no_detail_omits_detail_line(self) -> None:
        text = _format_attempt_summary(
            InternalError(),
            attempt_no=1,
            stage_name=None,
            stage_idx=None,
            stage_total=None,
            trace_id=None,
            request_id=None,
            detail=None,
            context=None,
            duration_seconds=None,
            code="INTERNAL_ERROR",
            title="Internal error",
        )
        assert "  detail:" not in text

    def test_no_context_omits_context_line(self) -> None:
        text = _format_attempt_summary(
            InternalError(detail="x"),
            attempt_no=1,
            stage_name=None,
            stage_idx=None,
            stage_total=None,
            trace_id=None,
            request_id=None,
            detail="x",
            context=None,
            duration_seconds=None,
            code="INTERNAL_ERROR",
            title="Internal error",
        )
        assert "  context:" not in text


# ---------------------------------------------------------------------------
# 3. Boundary — empty / unusual inputs
# ---------------------------------------------------------------------------


class TestBoundary:
    def test_empty_context_dict_omits_context_line(self) -> None:
        text = _format_attempt_summary(
            InternalError(),
            attempt_no=None,
            stage_name=None,
            stage_idx=None,
            stage_total=None,
            trace_id=None,
            request_id=None,
            detail=None,
            context={},
            duration_seconds=None,
            code="INTERNAL_ERROR",
            title="Internal error",
        )
        assert "  context:" not in text

    def test_only_private_context_keys_omit_block(self) -> None:
        # ``_internal`` keys are stripped — when only private keys exist,
        # nothing is rendered.
        text = _format_attempt_summary(
            InternalError(),
            attempt_no=None,
            stage_name=None,
            stage_idx=None,
            stage_total=None,
            trace_id=None,
            request_id=None,
            detail=None,
            context={"_internal": "x", "_more": "y"},
            duration_seconds=None,
            code="INTERNAL_ERROR",
            title="Internal error",
        )
        assert "  context:" not in text

    def test_success_with_no_duration_no_stages(self) -> None:
        text = _format_attempt_summary(
            None,
            attempt_no=1,
            stage_name=None,
            stage_idx=None,
            stage_total=None,
            trace_id=None,
            request_id=None,
            detail=None,
            context=None,
            duration_seconds=None,
            code=None,
            title=None,
            total_stages=None,
        )
        assert "Pipeline COMPLETED at attempt 1" in text


# ---------------------------------------------------------------------------
# 4. Invariants — emits via ryotenkai logger
# ---------------------------------------------------------------------------


class TestInvariants:
    def test_log_outcome_failure_emits_at_error_level(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        from ryotenkai_shared.utils.logger import logger as base_logger

        with caplog.at_level(logging.DEBUG, logger=base_logger.name):
            _log_pipeline_outcome(
                InternalError(detail="boom"),
                attempt_no=1,
                detail="boom",
                code="INTERNAL_ERROR",
                title="Internal error",
            )
        # Look for a record carrying our outcome block.
        error_records = [
            r for r in caplog.records
            if r.levelno == logging.ERROR and "Pipeline FAILED" in r.getMessage()
        ]
        assert error_records, "expected an ERROR record with the failure block"

    def test_log_outcome_success_emits_at_info_level(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        from ryotenkai_shared.utils.logger import logger as base_logger

        with caplog.at_level(logging.DEBUG, logger=base_logger.name):
            _log_pipeline_outcome(
                None,
                attempt_no=1,
                duration_seconds=10.0,
                total_stages=4,
            )
        info_records = [
            r for r in caplog.records
            if r.levelno == logging.INFO and "Pipeline COMPLETED" in r.getMessage()
        ]
        assert info_records, "expected an INFO record with the success block"


# ---------------------------------------------------------------------------
# 5. Regression — private context keys stripped
# ---------------------------------------------------------------------------


class TestRegression:
    def test_underscore_prefixed_context_keys_stripped(self) -> None:
        text = _format_attempt_summary(
            InternalError(),
            attempt_no=None,
            stage_name=None,
            stage_idx=None,
            stage_total=None,
            trace_id=None,
            request_id=None,
            detail=None,
            context={"_internal": "secret", "user_visible": "ok"},
            duration_seconds=None,
            code="INTERNAL_ERROR",
            title="Internal error",
        )
        assert "_internal" not in text
        assert "user_visible" in text


# ---------------------------------------------------------------------------
# 6. Logic-specific — _extract_attempt_fields semantics
# ---------------------------------------------------------------------------


class TestExtractAttemptFields:
    def test_extracts_h1_stamped_keys_only(self) -> None:
        exc = PipelineStageFailedError(
            detail="x",
            context={
                "stage_name": "trainer",
                "stage_idx": 3,
                "stage_total": 9,
                "attempt_no": 2,
                "irrelevant": "drop me",
            },
        )
        fields = _extract_attempt_fields(exc)
        assert fields == {
            "stage_name": "trainer",
            "stage_idx": 3,
            "stage_total": 9,
            "attempt_no": 2,
        }

    def test_handles_none_exc(self) -> None:
        assert _extract_attempt_fields(None) == {}

    def test_handles_non_dict_context(self) -> None:
        exc = PipelineStageFailedError(detail="x")
        exc.context = "not a dict"  # type: ignore[assignment]
        assert _extract_attempt_fields(exc) == {}


# ---------------------------------------------------------------------------
# 7. Combinatorial — attempt numbering matrix
# ---------------------------------------------------------------------------


class TestCombinatorial:
    @pytest.mark.parametrize(
        "attempt_no, expected_header_fragment",
        [
            (None, "Pipeline COMPLETED"),
            (1, "Pipeline COMPLETED at attempt 1"),
            (5, "Pipeline COMPLETED at attempt 5"),
        ],
    )
    def test_success_attempt_numbering(
        self, attempt_no: int | None, expected_header_fragment: str
    ) -> None:
        text = _format_attempt_summary(
            None,
            attempt_no=attempt_no,
            stage_name=None,
            stage_idx=None,
            stage_total=None,
            trace_id=None,
            request_id=None,
            detail=None,
            context=None,
            duration_seconds=1.0,
            code=None,
            title=None,
            total_stages=1,
        )
        assert expected_header_fragment in text

    @pytest.mark.parametrize(
        "attempt_no, expected_header_fragment",
        [
            (None, "Pipeline FAILED"),
            (1, "Pipeline FAILED at attempt 1"),
            (3, "Pipeline FAILED at attempt 3"),
        ],
    )
    def test_failure_attempt_numbering(
        self, attempt_no: int | None, expected_header_fragment: str
    ) -> None:
        text = _format_attempt_summary(
            InternalError(detail="x"),
            attempt_no=attempt_no,
            stage_name=None,
            stage_idx=None,
            stage_total=None,
            trace_id=None,
            request_id=None,
            detail="x",
            context=None,
            duration_seconds=None,
            code="INTERNAL_ERROR",
            title="Internal error",
        )
        assert expected_header_fragment in text
