"""
Unit tests for save_plugin_report() in evaluation/plugins/utils.py.

Tests verify the answers.md-style Markdown output (sections, not table)
without any RunPod or real inference calls.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from src.evaluation.plugins.base import EvalResult
from src.evaluation.plugins.utils import PluginReportRow, save_plugin_report

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_result(passed: bool = True, extra_metrics: dict | None = None) -> EvalResult:
    metrics = {"valid_count": 2, "total_count": 3, "valid_ratio": 0.667}
    if extra_metrics:
        metrics.update(extra_metrics)
    return EvalResult(
        plugin_name="test_plugin",
        passed=passed,
        metrics=metrics,
        errors=[] if passed else ["valid_ratio=0.667 is below threshold 0.800 (2/3 samples passed)"],
        sample_count=3,
        failed_samples=[2] if not passed else [],
    )


def _make_rows(with_expected: bool = True, with_raw_score: bool = False) -> list[PluginReportRow]:
    rows = [
        PluginReportRow(
            idx=0,
            question="QUERY GetAll () =>\n    items <- N<User>\n    RETURN items",
            model_answer="QUERY GetAll () =>\n    items <- N<User>\n    RETURN items",
            expected_answer="QUERY GetAll () =>\n    items <- N<User>\n    RETURN items" if with_expected else None,
            score=1.0 if not with_raw_score else 0.75,
            raw_score=4 if with_raw_score else None,
            note="" if not with_raw_score else "Mostly correct but missing edge case",
        ),
        PluginReportRow(
            idx=1,
            question="Need GetUser endpoint — fetch by id.",
            model_answer="QUERY GetUser (id: ID) =>\n    user <- N<User>(id)\n    RETURN user",
            expected_answer="QUERY GetUser (id: ID) =>\n    user <- N<User>(id)\n    RETURN user" if with_expected else None,
            score=1.0 if not with_raw_score else 1.0,
            raw_score=5 if with_raw_score else None,
            note="",
        ),
        PluginReportRow(
            idx=2,
            question="Need AddSensor endpoint — create record.",
            model_answer="some invalid schema response",
            expected_answer="QUERY AddSensor () =>\n    item <- AddN<Sensor>({})\n    RETURN item" if with_expected else None,
            score=0.0,
            raw_score=1 if with_raw_score else None,
            note="missing QUERY keyword" if not with_raw_score else "Completely wrong structure",
        ),
    ]
    return rows


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def patched_log_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Patch get_run_log_dir() to return a tmp_path so no run context is needed."""
    monkeypatch.setattr(
        "src.evaluation.plugins.utils.get_run_log_dir",
        lambda: tmp_path,
    )
    return tmp_path


# ---------------------------------------------------------------------------
# Tests: file creation
# ---------------------------------------------------------------------------

class TestSavePluginReportFileCreation:
    def test_creates_evaluation_subdir(self, patched_log_dir: Path) -> None:
        save_plugin_report("test_plugin", _make_rows(), _make_result())
        assert (patched_log_dir / "evaluation").is_dir()

    def test_creates_report_file(self, patched_log_dir: Path) -> None:
        save_plugin_report("test_plugin", _make_rows(), _make_result())
        assert (patched_log_dir / "evaluation" / "test_plugin_report.md").exists()

    def test_filename_uses_plugin_name(self, patched_log_dir: Path) -> None:
        save_plugin_report("helixql_syntax", _make_rows(), _make_result())
        assert (patched_log_dir / "evaluation" / "helixql_syntax_report.md").exists()

    def test_does_not_raise_on_empty_rows(self, patched_log_dir: Path) -> None:
        result = EvalResult(
            plugin_name="test_plugin", passed=True,
            metrics={"valid_count": 0, "total_count": 0, "valid_ratio": 1.0},
            errors=["No samples to evaluate"], sample_count=0,
        )
        save_plugin_report("test_plugin", [], result)
        assert (patched_log_dir / "evaluation" / "test_plugin_report.md").exists()


# ---------------------------------------------------------------------------
# Tests: header
# ---------------------------------------------------------------------------

class TestSavePluginReportHeader:
    def test_title_contains_plugin_name(self, patched_log_dir: Path) -> None:
        save_plugin_report("helixql_syntax", _make_rows(), _make_result())
        content = (patched_log_dir / "evaluation" / "helixql_syntax_report.md").read_text()
        assert "# helixql_syntax Report" in content

    def test_passed_verdict(self, patched_log_dir: Path) -> None:
        save_plugin_report("test_plugin", _make_rows(), _make_result(passed=True))
        content = (patched_log_dir / "evaluation" / "test_plugin_report.md").read_text()
        assert "PASSED" in content
        assert "FAILED" not in content

    def test_failed_verdict(self, patched_log_dir: Path) -> None:
        save_plugin_report("test_plugin", _make_rows(), _make_result(passed=False))
        content = (patched_log_dir / "evaluation" / "test_plugin_report.md").read_text()
        assert "FAILED" in content

    def test_metrics_in_header(self, patched_log_dir: Path) -> None:
        save_plugin_report("test_plugin", _make_rows(), _make_result())
        content = (patched_log_dir / "evaluation" / "test_plugin_report.md").read_text()
        assert "valid_count" in content
        assert "valid_ratio" in content

    def test_error_message_present_when_failed(self, patched_log_dir: Path) -> None:
        save_plugin_report("test_plugin", _make_rows(), _make_result(passed=False))
        content = (patched_log_dir / "evaluation" / "test_plugin_report.md").read_text()
        assert "valid_ratio=0.667 is below threshold" in content


# ---------------------------------------------------------------------------
# Tests: answers.md-style section format (no table)
# ---------------------------------------------------------------------------

class TestSavePluginReportSectionFormat:
    def test_no_markdown_table(self, patched_log_dir: Path) -> None:
        """Report must NOT contain pipe-separated table rows."""
        save_plugin_report("test_plugin", _make_rows(), _make_result())
        content = (patched_log_dir / "evaluation" / "test_plugin_report.md").read_text()
        # A markdown table row looks like "| col | col |" — detect 3+ pipes in one line
        table_lines = [
            ln for ln in content.splitlines()
            if ln.count("|") >= 3
        ]
        assert table_lines == [], f"Found table-style lines: {table_lines}"

    def test_task_sections_present(self, patched_log_dir: Path) -> None:
        rows = _make_rows()
        save_plugin_report("test_plugin", rows, _make_result())
        content = (patched_log_dir / "evaluation" / "test_plugin_report.md").read_text()
        for i in range(1, len(rows) + 1):
            assert f"### Task {i}" in content

    def test_model_answer_section_present(self, patched_log_dir: Path) -> None:
        save_plugin_report("test_plugin", _make_rows(), _make_result())
        content = (patched_log_dir / "evaluation" / "test_plugin_report.md").read_text()
        assert "### Model answer" in content

    def test_expected_answer_section_present_when_provided(self, patched_log_dir: Path) -> None:
        save_plugin_report("test_plugin", _make_rows(with_expected=True), _make_result())
        content = (patched_log_dir / "evaluation" / "test_plugin_report.md").read_text()
        assert "### Expected answer" in content

    def test_expected_answer_section_absent_when_none(self, patched_log_dir: Path) -> None:
        save_plugin_report("test_plugin", _make_rows(with_expected=False), _make_result())
        content = (patched_log_dir / "evaluation" / "test_plugin_report.md").read_text()
        assert "### Expected answer" not in content

    def test_content_wrapped_in_tildes(self, patched_log_dir: Path) -> None:
        save_plugin_report("test_plugin", _make_rows(), _make_result())
        content = (patched_log_dir / "evaluation" / "test_plugin_report.md").read_text()
        tilde_blocks = re.findall(r"~~~", content)
        # 3 rows × (question block open+close + model_answer open+close + expected open+close) = 18
        assert len(tilde_blocks) >= 6, f"Expected at least 6 tilde fences, got {len(tilde_blocks)}"

    def test_multiline_query_not_truncated(self, patched_log_dir: Path) -> None:
        """Long multi-line answers must appear in full — no truncation, no strip."""
        long_answer = "QUERY GetAll () =>\n    items <- N<User>\n    RETURN items"
        rows = [
            PluginReportRow(
                idx=0, question="q", model_answer=long_answer,
                expected_answer=None, score=1.0, raw_score=None, note="",
            )
        ]
        result = EvalResult(plugin_name="p", passed=True, metrics={}, errors=[], sample_count=1)
        save_plugin_report("p", rows, result)
        content = (patched_log_dir / "evaluation" / "p_report.md").read_text()
        assert "items <- N<User>" in content
        assert "RETURN items" in content

    def test_raw_answer_with_code_fence_preserved(self, patched_log_dir: Path) -> None:
        """Model may return ```helixschema ...``` blocks — must be stored as-is, no manipulation."""
        raw_answer = "```helixschema\nN::User {\n    name: String,\n}\n```"
        rows = [
            PluginReportRow(
                idx=0, question="q", model_answer=raw_answer,
                expected_answer=None, score=0.0, raw_score=None, note="missing QUERY keyword",
            )
        ]
        result = EvalResult(plugin_name="p", passed=False, metrics={}, errors=[], sample_count=1)
        save_plugin_report("p", rows, result)
        content = (patched_log_dir / "evaluation" / "p_report.md").read_text()
        assert "```helixschema" in content
        assert "N::User" in content

    def test_leading_trailing_whitespace_not_stripped(self, patched_log_dir: Path) -> None:
        """Content with leading/trailing newlines must NOT be stripped."""
        answer_with_newlines = "\nQUERY A () =>\n    x <- N<X>\n    RETURN x\n"
        rows = [
            PluginReportRow(
                idx=0, question="q", model_answer=answer_with_newlines,
                expected_answer=None, score=1.0, raw_score=None, note="",
            )
        ]
        result = EvalResult(plugin_name="p", passed=True, metrics={}, errors=[], sample_count=1)
        save_plugin_report("p", rows, result)
        content = (patched_log_dir / "evaluation" / "p_report.md").read_text()
        assert answer_with_newlines in content


# ---------------------------------------------------------------------------
# Tests: score display
# ---------------------------------------------------------------------------

class TestSavePluginReportScoreDisplay:
    def test_syntax_valid_score(self, patched_log_dir: Path) -> None:
        """helixql_syntax style: score=1.0, raw_score=None → shows ✓ valid"""
        rows = [
            PluginReportRow(
                idx=0, question="q", model_answer="QUERY A() =>\n  x <- N<X>\n  RETURN x",
                expected_answer=None, score=1.0, raw_score=None, note="",
            )
        ]
        result = EvalResult(plugin_name="p", passed=True, metrics={}, errors=[], sample_count=1)
        save_plugin_report("p", rows, result)
        content = (patched_log_dir / "evaluation" / "p_report.md").read_text()
        assert "✓ valid" in content

    def test_syntax_invalid_score(self, patched_log_dir: Path) -> None:
        """helixql_syntax style: score=0.0, raw_score=None → shows ✗ invalid"""
        rows = [
            PluginReportRow(
                idx=0, question="q", model_answer="bad answer",
                expected_answer=None, score=0.0, raw_score=None, note="missing QUERY keyword",
            )
        ]
        result = EvalResult(plugin_name="p", passed=False, metrics={}, errors=[], sample_count=1)
        save_plugin_report("p", rows, result)
        content = (patched_log_dir / "evaluation" / "p_report.md").read_text()
        assert "✗ invalid" in content

    def test_llm_judge_raw_score_displayed(self, patched_log_dir: Path) -> None:
        """cerebras_judge style: score=0.75, raw_score=4 → shows normalized + raw"""
        rows = _make_rows(with_raw_score=True)
        result = EvalResult(plugin_name="p", passed=True, metrics={}, errors=[], sample_count=3)
        save_plugin_report("p", rows, result)
        content = (patched_log_dir / "evaluation" / "p_report.md").read_text()
        assert "raw: 4/5" in content

    def test_failed_score_displays_failed(self, patched_log_dir: Path) -> None:
        rows = [
            PluginReportRow(
                idx=0, question="q", model_answer="a",
                expected_answer=None, score=None, raw_score=None, note="API error: timeout",
            )
        ]
        result = EvalResult(plugin_name="p", passed=False, metrics={}, errors=[], sample_count=1)
        save_plugin_report("p", rows, result)
        content = (patched_log_dir / "evaluation" / "p_report.md").read_text()
        assert "FAILED" in content

    def test_note_present_when_nonempty(self, patched_log_dir: Path) -> None:
        rows = [
            PluginReportRow(
                idx=0, question="q", model_answer="bad",
                expected_answer=None, score=0.0, raw_score=None, note="missing QUERY keyword",
            )
        ]
        result = EvalResult(plugin_name="p", passed=False, metrics={}, errors=[], sample_count=1)
        save_plugin_report("p", rows, result)
        content = (patched_log_dir / "evaluation" / "p_report.md").read_text()
        assert "missing QUERY keyword" in content
        assert "Comment" in content

    def test_note_absent_when_empty(self, patched_log_dir: Path) -> None:
        rows = [
            PluginReportRow(
                idx=0, question="q",
                model_answer="QUERY A() =>\n  x <- N<X>\n  RETURN x",
                expected_answer=None, score=1.0, raw_score=None, note="",
            )
        ]
        result = EvalResult(plugin_name="p", passed=True, metrics={}, errors=[], sample_count=1)
        save_plugin_report("p", rows, result)
        content = (patched_log_dir / "evaluation" / "p_report.md").read_text()
        assert "Comment" not in content


# ---------------------------------------------------------------------------
# Tests: resilience
# ---------------------------------------------------------------------------

class TestSavePluginReportResilience:
    def test_does_not_raise_on_bad_log_dir(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """If get_run_log_dir raises, save_plugin_report must not propagate the error."""
        monkeypatch.setattr(
            "src.evaluation.plugins.utils.get_run_log_dir",
            lambda: (_ for _ in ()).throw(RuntimeError("Run logging not initialized")),
        )
        save_plugin_report("p", _make_rows(), _make_result())  # must not raise
