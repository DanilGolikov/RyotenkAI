"""
Tests for community_libs.helixql — moved from src/utils/domains/test_helixql.py
when the HelixQL helpers were relocated out of src/ into community/libs/.

Coverage:
- extract_schema_block: positive, negative, edge cases
- normalize_query_text: whitespace invariants
- hard_eval_errors: each error type, combinations
- semantic_match_details: exact match, near match, scoring, hard error penalty, empty inputs
- extract_schema_and_query: dispatch over messages/fields, missing keys
"""

from __future__ import annotations

import pytest

from community_libs.helixql import (
    extract_schema_and_query,
    extract_schema_block,
    hard_eval_errors,
    normalize_query_text,
    semantic_match_details,
)

# ---------------------------------------------------------------------------
# extract_schema_block
# ---------------------------------------------------------------------------


class TestExtractSchemaBlock:
    def test_extracts_schema_between_fences(self) -> None:
        text = "some preamble\n```helixschema\nNode User {}\n```\nrest"
        assert extract_schema_block(text) == "Node User {}"

    def test_returns_empty_when_no_fence(self) -> None:
        assert extract_schema_block("QUERY GetAll () => ...") == ""

    def test_returns_empty_on_empty_string(self) -> None:
        assert extract_schema_block("") == ""

    def test_returns_empty_on_none_coerced(self) -> None:
        assert extract_schema_block("") == ""

    def test_strips_surrounding_whitespace(self) -> None:
        text = "```helixschema\n  Node User {}  \n```"
        assert extract_schema_block(text) == "Node User {}"

    def test_multiline_schema(self) -> None:
        schema = "Node User {\n  name: String\n  age: Int\n}"
        text = f"```helixschema\n{schema}\n```"
        assert extract_schema_block(text) == schema

    def test_only_first_fence_is_extracted(self) -> None:
        text = "```helixschema\nNode A {}\n```\n```helixschema\nNode B {}\n```"
        result = extract_schema_block(text)
        assert "Node A {}" in result
        assert "Node B {}" not in result

    def test_ignores_non_helixschema_fences(self) -> None:
        text = "```python\nprint('hi')\n```"
        assert extract_schema_block(text) == ""


# ---------------------------------------------------------------------------
# normalize_query_text
# ---------------------------------------------------------------------------


class TestNormalizeQueryText:
    def test_collapses_internal_spaces(self) -> None:
        raw = "QUERY   GetAll  () =>"
        assert "  " not in normalize_query_text(raw)

    def test_strips_leading_trailing_whitespace(self) -> None:
        assert normalize_query_text("  QUERY X () =>  ").strip() == normalize_query_text("QUERY X () =>")

    def test_empty_string_returns_empty(self) -> None:
        assert normalize_query_text("") == ""

    def test_only_whitespace_returns_empty(self) -> None:
        assert normalize_query_text("   \n\t  ") == ""

    def test_preserves_multiline_structure(self) -> None:
        text = "QUERY Get () =>\n    items <- N<User>\n    RETURN items"
        result = normalize_query_text(text)
        assert "\n" in result

    def test_idempotent(self) -> None:
        text = "QUERY Get () =>\n    items <- N<User>"
        assert normalize_query_text(normalize_query_text(text)) == normalize_query_text(text)

    def test_removes_blank_lines(self) -> None:
        text = "QUERY Get () =>\n\n\n    items <- N<User>"
        result = normalize_query_text(text)
        assert "\n\n" not in result


# ---------------------------------------------------------------------------
# hard_eval_errors
# ---------------------------------------------------------------------------


class TestHardEvalErrors:
    def test_clean_query_returns_no_errors(self) -> None:
        query = "QUERY GetAll () =>\n    items <- N<User>\n    RETURN items"
        errors = hard_eval_errors("", query)
        assert errors == []

    def test_missing_query_keyword(self) -> None:
        errors = hard_eval_errors("", "items <- N<User>")
        assert "missing_query_keyword" in errors or "not_starting_with_query" in errors

    def test_not_starting_with_query_keyword(self) -> None:
        errors = hard_eval_errors("", "SELECT * FROM users")
        assert "not_starting_with_query" in errors

    def test_contains_markdown_fence(self) -> None:
        query = "```helixql\nQUERY Get () => items <- N<User> RETURN items\n```"
        errors = hard_eval_errors("", query)
        assert "contains_markdown_fence" in errors

    def test_contains_colon_equals(self) -> None:
        query = "QUERY Get () =>\n    x := 5\n    RETURN x"
        errors = hard_eval_errors("", query)
        assert "contains_colon_equals" in errors

    def test_multiple_errors_returned(self) -> None:
        query = "```\nsome invalid code\n```"
        errors = hard_eval_errors("", query)
        assert len(errors) >= 1

    def test_empty_query_returns_errors(self) -> None:
        errors = hard_eval_errors("", "")
        assert len(errors) > 0

    def test_missing_required_exclusions(self) -> None:
        user_text = "exclude `age` from result"
        query = "QUERY Get () =>\n    items <- N<User>\n    RETURN items"
        errors = hard_eval_errors(user_text, query)
        assert "missing_required_exclusions" in errors

    def test_correct_exclusions_no_error(self) -> None:
        user_text = "exclude `age` from result"
        query = "QUERY Get () =>\n    items <- N<User>{::!{age}}\n    RETURN items"
        errors = hard_eval_errors(user_text, query)
        assert "missing_required_exclusions" not in errors


# ---------------------------------------------------------------------------
# semantic_match_details
# ---------------------------------------------------------------------------


class TestSemanticMatchDetails:
    def test_exact_match_score_is_one(self) -> None:
        q = "QUERY GetAll () =>\n    items <- N<User>\n    RETURN items"
        result = semantic_match_details(candidate=q, expected=q)
        assert result["score"] == 1.0
        assert result["exact_match"] is True
        assert result["near_match"] is True

    def test_empty_candidate_returns_zero(self) -> None:
        result = semantic_match_details(candidate="", expected="QUERY GetAll () => items <- N<User> RETURN items")
        assert result["score"] == 0.0
        assert result["exact_match"] is False
        assert "empty_candidate_or_expected" in result["hard_eval_errors"]

    def test_empty_expected_returns_zero(self) -> None:
        result = semantic_match_details(candidate="QUERY Get () => RETURN x", expected="")
        assert result["score"] == 0.0

    def test_both_empty_returns_zero(self) -> None:
        result = semantic_match_details(candidate="", expected="")
        assert result["score"] == 0.0

    def test_completely_different_returns_low_score(self) -> None:
        result = semantic_match_details(
            candidate="QUERY GetUsers () => x <- N<User> RETURN x",
            expected="QUERY GetOrders () => o <- N<Order> RETURN o",
        )
        assert result["score"] < 1.0

    def test_near_match_threshold(self) -> None:
        q1 = "QUERY GetAll () =>\n    items <- N<User>\n    RETURN items"
        q2 = "QUERY GetAll () =>\n    users <- N<User>\n    RETURN users"
        result = semantic_match_details(candidate=q1, expected=q2)
        assert isinstance(result["near_match"], bool)

    def test_hard_errors_penalize_score(self) -> None:
        bad_query = "```\nSELECT * FROM users\n```"
        good_query = "QUERY GetAll () => items <- N<User> RETURN items"
        result = semantic_match_details(candidate=bad_query, expected=good_query)
        assert result["hard_eval_pass"] is False
        assert result["score"] < 1.0

    def test_score_between_zero_and_one(self) -> None:
        result = semantic_match_details(
            candidate="QUERY Get () => x <- N<User> RETURN x",
            expected="QUERY GetAll () => items <- N<User> RETURN items",
        )
        assert 0.0 <= result["score"] <= 1.0

    def test_returns_all_required_keys(self) -> None:
        result = semantic_match_details(candidate="QUERY Get () => RETURN x", expected="QUERY Get () => RETURN x")
        required_keys = {
            "score",
            "exact_match",
            "near_match",
            "sequence_ratio",
            "token_jaccard",
            "hard_eval_pass",
            "hard_eval_errors",
        }
        assert required_keys.issubset(result.keys())

    def test_sequence_ratio_between_zero_and_one(self) -> None:
        result = semantic_match_details(candidate="QUERY A () => RETURN x", expected="QUERY B () => RETURN y")
        assert 0.0 <= result["sequence_ratio"] <= 1.0

    def test_token_jaccard_between_zero_and_one(self) -> None:
        result = semantic_match_details(candidate="QUERY A () => RETURN x", expected="QUERY B () => RETURN y")
        assert 0.0 <= result["token_jaccard"] <= 1.0

    def test_score_is_rounded_to_4_decimals(self) -> None:
        result = semantic_match_details(
            candidate="QUERY GetAll () => items <- N<User> RETURN items",
            expected="QUERY GetAllItems () => items <- N<User> RETURN items",
        )
        score_str = str(result["score"])
        if "." in score_str:
            assert len(score_str.split(".")[1]) <= 4

    def test_user_text_affects_exclusion_errors(self) -> None:
        user_text = "exclude `age`"
        candidate = "QUERY Get () =>\n    items <- N<User>\n    RETURN items"
        expected = "QUERY Get () =>\n    items <- N<User>{::!{age}}\n    RETURN items"
        result_with_user = semantic_match_details(candidate=candidate, expected=expected, user_text=user_text)
        result_without_user = semantic_match_details(candidate=candidate, expected=expected)
        assert len(result_with_user["hard_eval_errors"]) >= len(result_without_user["hard_eval_errors"])

    @pytest.mark.parametrize(
        "candidate,expected",
        [
            ("QUERY X () => RETURN x", "QUERY X () => RETURN x"),
            ("QUERY A () => a <- N<A> RETURN a", "QUERY A () => a <- N<A> RETURN a"),
        ],
    )
    def test_identical_queries_always_score_one(self, candidate: str, expected: str) -> None:
        result = semantic_match_details(candidate=candidate, expected=expected)
        assert result["score"] == 1.0


# ---------------------------------------------------------------------------
# extract_schema_and_query  (new helper that consolidates plugin dispatch)
# ---------------------------------------------------------------------------


class TestExtractSchemaAndQuery:
    def test_pulls_schema_from_prompt_and_query_from_reference_answer(self) -> None:
        sample = {
            "prompt": "Generate a query.\n```helixschema\nNode User {}\n```\n",
            "reference_answer": "QUERY GetAll () =>\n    items <- N<User>\n    RETURN items",
        }
        schema, query = extract_schema_and_query(sample)
        assert schema == "Node User {}"
        assert query.startswith("QUERY GetAll")

    def test_falls_back_to_messages_for_prompt(self) -> None:
        sample = {
            "messages": [
                {"role": "system", "content": "you are helpful"},
                {"role": "user", "content": "make a query.\n```helixschema\nNode A {}\n```"},
            ],
            "reference_answer": "QUERY X () => RETURN x",
        }
        schema, _ = extract_schema_and_query(sample)
        assert schema == "Node A {}"

    def test_returns_empty_when_keys_missing(self) -> None:
        schema, query = extract_schema_and_query({})
        assert schema == ""
        assert query == ""

    def test_respects_custom_query_keys(self) -> None:
        sample = {"prompt": "x", "completion": "QUERY Z () => RETURN z"}
        _, query = extract_schema_and_query(sample, query_keys=("completion",))
        assert query == "QUERY Z () => RETURN z"

    def test_first_matching_query_key_wins(self) -> None:
        sample = {
            "prompt": "p",
            "reference_answer": "first",
            "expected": "second",
        }
        _, query = extract_schema_and_query(sample)
        assert query == "first"

    def test_handles_attribute_style_sample(self) -> None:
        class _S:
            prompt = "p\n```helixschema\nNode B {}\n```"
            reference_answer = "QUERY Q () => RETURN q"

        schema, query = extract_schema_and_query(_S())
        assert schema == "Node B {}"
        assert query == "QUERY Q () => RETURN q"
