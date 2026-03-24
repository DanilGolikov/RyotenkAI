"""
Unit tests for EvalSample.metadata passthrough and EvaluationRunner._extract_row_fields.

Coverage matrix:
─────────────────────────────────────────────────────────────────────────────
Category              | Test name
──────────────────────┼──────────────────────────────────────────────────────
Positives             | test_metadata_single_extra_field
                      | test_metadata_multiple_extra_fields
                      | test_metadata_nested_value
                      | test_metadata_numeric_value
                      | test_metadata_boolean_value
                      | test_metadata_list_value
──────────────────────┼──────────────────────────────────────────────────────
Negatives             | test_metadata_empty_when_no_extra_fields
                      | test_metadata_empty_for_messages_format
                      | test_reserved_field_context_not_in_metadata
──────────────────────┼──────────────────────────────────────────────────────
Boundary / edge       | test_metadata_all_reserved_fields_excluded
                      | test_metadata_only_reserved_plus_one_extra
                      | test_metadata_unicode_key_and_value
                      | test_metadata_null_value_passes_through
                      | test_metadata_key_with_spaces
──────────────────────┼──────────────────────────────────────────────────────
Invariants            | test_reserved_fields_never_appear_in_metadata
                      | test_metadata_is_always_dict_not_none
                      | test_evalsample_default_metadata_is_empty_dict
                      | test_metadata_does_not_mutate_original_row
──────────────────────┼──────────────────────────────────────────────────────
Plugin contract       | test_plugin_can_read_metadata_field
                      | test_plugin_receives_empty_metadata_for_legacy_dataset
                      | test_plugin_metadata_get_returns_none_for_missing_key
──────────────────────┼──────────────────────────────────────────────────────
Regressions           | test_question_still_extracted_correctly
                      | test_expected_answer_still_extracted_correctly
                      | test_answer_alias_still_works
                      | test_messages_format_question_extraction_unchanged
                      | test_context_field_silently_dropped_not_in_metadata
──────────────────────┼──────────────────────────────────────────────────────
Dependency errors     | test_collect_model_answers_inference_failure_preserves_metadata
──────────────────────┼──────────────────────────────────────────────────────
Logic-specific        | test_extract_row_fields_returns_correct_tuple_structure
                      | test_reserved_fields_constant_contains_expected_keys
──────────────────────┼──────────────────────────────────────────────────────
Combinatorial         | test_flat_format_with_all_reserved_and_extra_fields
                      | test_messages_format_with_extra_fields
                      | test_multiple_reserved_aliases_with_multiple_extras
─────────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

from dataclasses import field, fields
from typing import Any
from unittest.mock import MagicMock

import pytest

from src.evaluation.plugins.base import EvalSample
from src.evaluation.runner import EvaluationRunner


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _extract(row: dict[str, Any]) -> tuple[str, str | None, dict[str, Any]]:
    """Thin wrapper so tests call the static method without instantiating EvaluationRunner."""
    return EvaluationRunner._extract_row_fields(row)


def _make_sample(
    question: str = "Q",
    model_answer: str = "A",
    expected_answer: str | None = "E",
    metadata: dict[str, Any] | None = None,
) -> EvalSample:
    return EvalSample(
        question=question,
        model_answer=model_answer,
        expected_answer=expected_answer,
        metadata=metadata or {},
    )


# ===========================================================================
# Positive tests — metadata is populated correctly
# ===========================================================================


@pytest.mark.unit
class TestMetadataPositive:
    def test_metadata_single_extra_field(self):
        row = {"question": "Q", "expected_answer": "E", "docs": "API docs"}
        _, _, meta = _extract(row)
        assert meta == {"docs": "API docs"}

    def test_metadata_multiple_extra_fields(self):
        row = {"question": "Q", "expected_answer": "E", "docs": "d", "difficulty": "easy", "topic": "sql"}
        _, _, meta = _extract(row)
        assert meta == {"docs": "d", "difficulty": "easy", "topic": "sql"}

    def test_metadata_nested_value(self):
        row = {"question": "Q", "schema": {"nodes": ["User"], "edges": []}}
        _, _, meta = _extract(row)
        assert meta == {"schema": {"nodes": ["User"], "edges": []}}

    def test_metadata_numeric_value(self):
        row = {"question": "Q", "max_score": 10}
        _, _, meta = _extract(row)
        assert meta == {"max_score": 10}

    def test_metadata_boolean_value(self):
        row = {"question": "Q", "is_holdout": True}
        _, _, meta = _extract(row)
        assert meta == {"is_holdout": True}

    def test_metadata_list_value(self):
        row = {"question": "Q", "tags": ["rag", "nl2sql", "graph"]}
        _, _, meta = _extract(row)
        assert meta == {"tags": ["rag", "nl2sql", "graph"]}


# ===========================================================================
# Negative tests — metadata is empty when there are no extra fields
# ===========================================================================


@pytest.mark.unit
class TestMetadataNegative:
    def test_metadata_empty_when_no_extra_fields(self):
        row = {"question": "Q", "expected_answer": "E"}
        _, _, meta = _extract(row)
        assert meta == {}

    def test_metadata_empty_for_messages_format(self):
        row = {"messages": [{"role": "user", "content": "Q"}, {"role": "assistant", "content": "A"}]}
        _, _, meta = _extract(row)
        assert meta == {}

    def test_reserved_field_context_not_in_metadata(self):
        """context is reserved: must not appear in metadata even if present in the row."""
        row = {"question": "Q", "context": "some context string"}
        _, _, meta = _extract(row)
        assert "context" not in meta


# ===========================================================================
# Boundary / edge-case tests
# ===========================================================================


@pytest.mark.unit
class TestMetadataBoundary:
    def test_metadata_all_reserved_fields_excluded(self):
        """A row containing only reserved fields → metadata is empty."""
        row = {
            "question": "Q",
            "expected_answer": "E",
            "answer": "E",
            "context": "ctx",
            "messages": [],
        }
        _, _, meta = _extract(row)
        assert meta == {}

    def test_metadata_only_reserved_plus_one_extra(self):
        row = {"question": "Q", "expected_answer": "E", "context": "ctx", "extra": "val"}
        _, _, meta = _extract(row)
        assert meta == {"extra": "val"}

    def test_metadata_unicode_key_and_value(self):
        row = {"question": "Q", "documentation": "sample non-ASCII text: café"}
        _, _, meta = _extract(row)
        assert meta == {"documentation": "sample non-ASCII text: café"}

    def test_metadata_null_value_passes_through(self):
        row = {"question": "Q", "optional_hint": None}
        _, _, meta = _extract(row)
        assert "optional_hint" in meta
        assert meta["optional_hint"] is None

    def test_metadata_key_with_spaces(self):
        row = {"question": "Q", "extra key with spaces": "val"}
        _, _, meta = _extract(row)
        assert meta == {"extra key with spaces": "val"}


# ===========================================================================
# Invariant tests
# ===========================================================================


@pytest.mark.unit
class TestMetadataInvariants:
    def test_reserved_fields_never_appear_in_metadata(self):
        """Parameterized invariant: each reserved key must always be absent from metadata."""
        reserved = {"question", "expected_answer", "answer", "context", "messages"}
        for key in reserved:
            row = {"question": "Q", key: "some-value", "extra": "x"}
            _, _, meta = _extract(row)
            assert key not in meta, f"Reserved key '{key}' leaked into metadata"

    def test_metadata_is_always_dict_not_none(self):
        """metadata must be a dict in every case, never None."""
        rows = [
            {"question": "Q"},
            {"question": "Q", "expected_answer": "E"},
            {"messages": [{"role": "user", "content": "Q"}]},
            {},
        ]
        for row in rows:
            _, _, meta = _extract(row)
            assert isinstance(meta, dict), f"metadata is not dict for row={row}"

    def test_evalsample_default_metadata_is_empty_dict(self):
        """EvalSample created without metadata= must default to {}."""
        sample = EvalSample(question="Q", model_answer="A")
        assert sample.metadata == {}
        assert isinstance(sample.metadata, dict)

    def test_metadata_does_not_mutate_original_row(self):
        """Extracting metadata must not modify the source row dict."""
        row = {"question": "Q", "docs": "d"}
        original = dict(row)
        _extract(row)
        assert row == original


# ===========================================================================
# Plugin contract tests
# ===========================================================================


@pytest.mark.unit
class TestPluginContract:
    def test_plugin_can_read_metadata_field(self):
        """Simulate a plugin accessing sample.metadata — must find the expected key."""
        sample = _make_sample(metadata={"docs": "API reference"})
        assert sample.metadata.get("docs") == "API reference"

    def test_plugin_receives_empty_metadata_for_legacy_dataset(self):
        """Legacy samples (created before metadata existed) must still work."""
        sample = _make_sample()
        assert sample.metadata.get("docs") is None
        assert sample.metadata == {}

    def test_plugin_metadata_get_returns_none_for_missing_key(self):
        """Plugin's safe access pattern: .get() returns None, not KeyError."""
        sample = _make_sample(metadata={"topic": "graph"})
        result = sample.metadata.get("nonexistent_key")
        assert result is None


# ===========================================================================
# Regression tests — existing behaviour must be unchanged
# ===========================================================================


@pytest.mark.unit
class TestRegressions:
    def test_question_still_extracted_correctly(self):
        row = {"question": "What is X?", "expected_answer": "X is Y"}
        question, _, _ = _extract(row)
        assert question == "What is X?"

    def test_expected_answer_still_extracted_correctly(self):
        row = {"question": "Q", "expected_answer": "correct answer"}
        _, expected, _ = _extract(row)
        assert expected == "correct answer"

    def test_answer_alias_still_works(self):
        """The 'answer' key must still be treated as expected_answer fallback."""
        row = {"question": "Q", "answer": "aliased answer"}
        _, expected, _ = _extract(row)
        assert expected == "aliased answer"

    def test_messages_format_question_extraction_unchanged(self):
        row = {"messages": [{"role": "user", "content": "user question"}, {"role": "assistant", "content": "ans"}]}
        question, expected, _ = _extract(row)
        assert question == "user question"
        assert expected == "ans"

    def test_context_field_silently_dropped_not_in_metadata(self):
        """context was removed from EvalSample — must not sneak into metadata either."""
        row = {"question": "Q", "context": "old context"}
        _, _, meta = _extract(row)
        assert "context" not in meta
        # EvalSample must not have a context attribute at all
        field_names = {f.name for f in fields(EvalSample)}
        assert "context" not in field_names


# ===========================================================================
# Dependency error tests
# ===========================================================================


@pytest.mark.unit
class TestDependencyErrors:
    def test_collect_model_answers_inference_failure_preserves_metadata(self):
        """
        When inference raises RuntimeError for a sample, the sample is still added
        with model_answer="" and metadata must be preserved (not lost).
        """
        mock_client = MagicMock()
        mock_client.generate.side_effect = RuntimeError("timeout")

        # Build a minimal EvaluationRunner (we only need the instance method)
        runner = _make_runner()
        raw = [{"question": "Q?", "expected_answer": "E", "docs": "doc content"}]
        samples = runner._collect_model_answers(raw, mock_client)

        assert len(samples) == 1
        sample = samples[0]
        assert sample.model_answer == ""
        assert sample.metadata == {"docs": "doc content"}


# ===========================================================================
# Logic-specific tests
# ===========================================================================


@pytest.mark.unit
class TestLogicSpecific:
    def test_extract_row_fields_returns_correct_tuple_structure(self):
        """Return type must always be exactly (str, str|None, dict)."""
        row = {"question": "Q", "expected_answer": "E", "extra": "x"}
        result = _extract(row)
        assert len(result) == 3
        question, expected, meta = result
        assert isinstance(question, str)
        assert expected is None or isinstance(expected, str)
        assert isinstance(meta, dict)

    def test_reserved_fields_constant_contains_expected_keys(self):
        """_RESERVED_FIELDS must contain all five expected reserved keys."""
        expected = {"question", "expected_answer", "answer", "context", "messages"}
        assert EvaluationRunner._RESERVED_FIELDS == expected


# ===========================================================================
# Combinatorial tests
# ===========================================================================


@pytest.mark.unit
class TestCombinatorial:
    def test_flat_format_with_all_reserved_and_extra_fields(self):
        """All reserved fields present + two extra → only extras in metadata."""
        row = {
            "question": "Q",
            "expected_answer": "E",
            "answer": "E2",
            "context": "ctx",
            "messages": [],
            "docs": "d",
            "difficulty": "hard",
        }
        question, expected, meta = _extract(row)
        assert question == "Q"
        assert expected == "E"
        assert set(meta.keys()) == {"docs", "difficulty"}

    def test_messages_format_with_extra_fields(self):
        """messages format + extra fields → extras go to metadata, messages does not."""
        row = {
            "messages": [{"role": "user", "content": "user q"}, {"role": "assistant", "content": "ans"}],
            "topic": "rag",
            "source": "benchmark-v2",
        }
        question, expected, meta = _extract(row)
        assert question == "user q"
        assert expected == "ans"
        assert meta == {"topic": "rag", "source": "benchmark-v2"}
        assert "messages" not in meta

    def test_multiple_reserved_aliases_with_multiple_extras(self):
        """
        When both expected_answer and answer present, expected_answer wins.
        Extra fields still all appear in metadata.
        """
        row = {
            "question": "Q",
            "expected_answer": "primary",
            "answer": "alias",
            "tag1": "a",
            "tag2": "b",
            "tag3": "c",
        }
        _, expected, meta = _extract(row)
        assert expected == "primary"
        assert set(meta.keys()) == {"tag1", "tag2", "tag3"}


# ===========================================================================
# Private helpers for tests that need a runner instance
# ===========================================================================


def _make_runner() -> EvaluationRunner:
    """Create a minimal EvaluationRunner without any real config."""
    cfg = MagicMock()
    return EvaluationRunner(eval_config=cfg)
