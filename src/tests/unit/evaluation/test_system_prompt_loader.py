"""
Unit tests for SystemPromptLoader and SystemPromptResult.

Coverage matrix:
─────────────────────────────────────────────────────────────────────────────
Category              | Test name
──────────────────────┼──────────────────────────────────────────────────────
Positives             | test_load_from_file_returns_text_and_source
                      | test_load_from_file_strips_whitespace
                      | test_load_from_file_source_type_is_file
                      | test_load_from_mlflow_returns_text_and_source
                      | test_load_from_mlflow_source_type_is_mlflow
                      | test_load_no_source_returns_none
──────────────────────┼──────────────────────────────────────────────────────
Negatives             | test_load_file_not_found_returns_none
                      | test_load_file_empty_returns_none
                      | test_load_file_whitespace_only_returns_none
                      | test_load_mlflow_cfg_none_raises
                      | test_load_mlflow_cfg_disabled_raises
                      | test_load_mlflow_network_error_returns_none
                      | test_load_mlflow_empty_template_returns_none
                      | test_load_mlflow_whitespace_template_returns_none
──────────────────────┼──────────────────────────────────────────────────────
Boundary / edge       | test_load_file_single_char_content
                      | test_load_file_tilde_path_expanded
                      | test_load_file_os_error_returns_none
                      | test_load_mlflow_template_strips_whitespace
                      | test_load_mlflow_version_is_string_in_source
                      | test_load_mlflow_name_plain_format
                      | test_load_mlflow_name_version_uri_format
                      | test_load_mlflow_name_alias_uri_format
──────────────────────┼──────────────────────────────────────────────────────
Invariants            | test_result_text_is_always_str
                      | test_result_source_is_always_dict
                      | test_file_source_always_has_type_and_path_keys
                      | test_mlflow_source_always_has_type_name_version_keys
                      | test_load_never_raises_on_file_errors
                      | test_load_result_text_not_empty_when_returned
──────────────────────┼──────────────────────────────────────────────────────
Dependency errors     | test_mlflow_import_error_returns_none
                      | test_mlflow_connection_refused_returns_none
                      | test_mlflow_prompt_not_found_returns_none
──────────────────────┼──────────────────────────────────────────────────────
Regressions           | test_llm_cfg_both_sources_raises_at_validation
                      | test_llm_cfg_only_path_valid
                      | test_llm_cfg_only_mlflow_name_valid
                      | test_llm_cfg_neither_source_valid
                      | test_file_source_does_not_call_mlflow
                      | test_mlflow_source_does_not_read_files
──────────────────────┼──────────────────────────────────────────────────────
Logic-specific        | test_mlflow_takes_priority_over_file_when_both_set_internal
                      | test_from_file_called_when_only_path_set
                      | test_from_mlflow_called_when_only_mlflow_name_set
                      | test_tracking_uri_passed_to_mlflow
                      | test_source_path_is_expanded_absolute
──────────────────────┼──────────────────────────────────────────────────────
Combinatorial         | test_file_path_none_mlflow_name_none_both_none
                      | test_enabled_false_mlflow_name_set_raises
                      | test_mlflow_cfg_none_mlflow_name_set_raises_not_returns_none
                      | test_file_exists_and_enabled_mlflow_cfg_uses_file
─────────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

import os
from dataclasses import fields as dataclass_fields
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from pydantic import ValidationError

from src.config.inference.common import InferenceLLMConfig
from src.evaluation.system_prompt import SystemPromptLoader, SystemPromptResult


# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------


def _llm_cfg(
    path: str | None = None,
    mlflow_name: str | None = None,
) -> InferenceLLMConfig:
    """Build InferenceLLMConfig bypassing mutual-exclusion validator when needed."""
    return InferenceLLMConfig.model_construct(
        system_prompt_path=path,
        system_prompt_mlflow_name=mlflow_name,
    )


def _mlflow_cfg(enabled: bool = True, tracking_uri: str = "http://mlflow:5000") -> MagicMock:
    cfg = MagicMock()
    cfg.enabled = enabled
    cfg.tracking_uri = tracking_uri
    return cfg


def _mock_prompt(name: str = "my-prompt", version: int = 1, template: str = "You are helpful.") -> MagicMock:
    p = MagicMock()
    p.name = name
    p.version = version
    p.template = template
    return p


# ---------------------------------------------------------------------------
# Positives
# ---------------------------------------------------------------------------


class TestPositives:
    def test_load_from_file_returns_text_and_source(self, tmp_path: Path) -> None:
        f = tmp_path / "sys.txt"
        f.write_text("You are an assistant.", encoding="utf-8")
        result = SystemPromptLoader.load(_llm_cfg(path=str(f)), None)
        assert result is not None
        assert result.text == "You are an assistant."

    def test_load_from_file_strips_whitespace(self, tmp_path: Path) -> None:
        f = tmp_path / "sys.txt"
        f.write_text("  Hello world  \n", encoding="utf-8")
        result = SystemPromptLoader.load(_llm_cfg(path=str(f)), None)
        assert result is not None
        assert result.text == "Hello world"

    def test_load_from_file_source_type_is_file(self, tmp_path: Path) -> None:
        f = tmp_path / "sys.txt"
        f.write_text("Prompt text.", encoding="utf-8")
        result = SystemPromptLoader.load(_llm_cfg(path=str(f)), None)
        assert result is not None
        assert result.source["type"] == "file"
        assert "path" in result.source

    def test_load_from_mlflow_returns_text_and_source(self) -> None:
        prompt = _mock_prompt(template="MLflow system prompt.")
        with patch("mlflow.set_tracking_uri"), patch("mlflow.genai.load_prompt", return_value=prompt):
            result = SystemPromptLoader.load(_llm_cfg(mlflow_name="my-prompt"), _mlflow_cfg())
        assert result is not None
        assert result.text == "MLflow system prompt."

    def test_load_from_mlflow_source_type_is_mlflow(self) -> None:
        prompt = _mock_prompt(name="my-prompt", version=3, template="Hello.")
        with patch("mlflow.set_tracking_uri"), patch("mlflow.genai.load_prompt", return_value=prompt):
            result = SystemPromptLoader.load(_llm_cfg(mlflow_name="my-prompt"), _mlflow_cfg())
        assert result is not None
        assert result.source["type"] == "mlflow"
        assert result.source["name"] == "my-prompt"
        assert result.source["version"] == "3"

    def test_load_no_source_returns_none(self) -> None:
        result = SystemPromptLoader.load(_llm_cfg(), None)
        assert result is None


# ---------------------------------------------------------------------------
# Negatives
# ---------------------------------------------------------------------------


class TestNegatives:
    def test_load_file_not_found_returns_none(self, tmp_path: Path) -> None:
        result = SystemPromptLoader.load(_llm_cfg(path=str(tmp_path / "missing.txt")), None)
        assert result is None

    def test_load_file_empty_returns_none(self, tmp_path: Path) -> None:
        f = tmp_path / "empty.txt"
        f.write_text("", encoding="utf-8")
        result = SystemPromptLoader.load(_llm_cfg(path=str(f)), None)
        assert result is None

    def test_load_file_whitespace_only_returns_none(self, tmp_path: Path) -> None:
        f = tmp_path / "ws.txt"
        f.write_text("   \n\t  \n", encoding="utf-8")
        result = SystemPromptLoader.load(_llm_cfg(path=str(f)), None)
        assert result is None

    def test_load_mlflow_cfg_none_raises(self) -> None:
        with pytest.raises(ValueError, match="experiment_tracking.mlflow is not"):
            SystemPromptLoader.load(_llm_cfg(mlflow_name="my-prompt"), None)

    def test_load_mlflow_cfg_ignores_legacy_enabled_flag(self) -> None:
        prompt = _mock_prompt(template="MLflow prompt.")
        with patch("mlflow.set_tracking_uri"), patch("mlflow.genai.load_prompt", return_value=prompt):
            result = SystemPromptLoader.load(_llm_cfg(mlflow_name="my-prompt"), _mlflow_cfg(enabled=False))
        assert result is not None
        assert result.text == "MLflow prompt."

    def test_load_mlflow_network_error_returns_none(self) -> None:
        with (
            patch("mlflow.set_tracking_uri"),
            patch("mlflow.genai.load_prompt", side_effect=ConnectionError("refused")),
        ):
            result = SystemPromptLoader.load(_llm_cfg(mlflow_name="my-prompt"), _mlflow_cfg())
        assert result is None

    def test_load_mlflow_empty_template_returns_none(self) -> None:
        prompt = _mock_prompt(template="")
        with patch("mlflow.set_tracking_uri"), patch("mlflow.genai.load_prompt", return_value=prompt):
            result = SystemPromptLoader.load(_llm_cfg(mlflow_name="my-prompt"), _mlflow_cfg())
        assert result is None

    def test_load_mlflow_whitespace_template_returns_none(self) -> None:
        prompt = _mock_prompt(template="   \n  ")
        with patch("mlflow.set_tracking_uri"), patch("mlflow.genai.load_prompt", return_value=prompt):
            result = SystemPromptLoader.load(_llm_cfg(mlflow_name="my-prompt"), _mlflow_cfg())
        assert result is None


# ---------------------------------------------------------------------------
# Boundary / edge cases
# ---------------------------------------------------------------------------


class TestBoundary:
    def test_load_file_single_char_content(self, tmp_path: Path) -> None:
        f = tmp_path / "one.txt"
        f.write_text("X", encoding="utf-8")
        result = SystemPromptLoader.load(_llm_cfg(path=str(f)), None)
        assert result is not None
        assert result.text == "X"

    def test_load_file_tilde_path_expanded(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        f = tmp_path / "prompt.txt"
        f.write_text("Tilde prompt.", encoding="utf-8")
        monkeypatch.setenv("HOME", str(tmp_path))
        tilde_path = "~/prompt.txt"
        result = SystemPromptLoader._from_file(tilde_path)
        assert result is not None
        assert result.text == "Tilde prompt."

    def test_load_file_os_error_returns_none(self, tmp_path: Path) -> None:
        f = tmp_path / "sys.txt"
        f.write_text("content", encoding="utf-8")
        with patch.object(Path, "read_text", side_effect=OSError("permission denied")):
            result = SystemPromptLoader._from_file(str(f))
        assert result is None

    def test_load_mlflow_template_strips_whitespace(self) -> None:
        prompt = _mock_prompt(template="  System message.  \n")
        with patch("mlflow.set_tracking_uri"), patch("mlflow.genai.load_prompt", return_value=prompt):
            result = SystemPromptLoader.load(_llm_cfg(mlflow_name="p"), _mlflow_cfg())
        assert result is not None
        assert result.text == "System message."

    def test_load_mlflow_version_is_string_in_source(self) -> None:
        prompt = _mock_prompt(version=42, template="Hello.")
        with patch("mlflow.set_tracking_uri"), patch("mlflow.genai.load_prompt", return_value=prompt):
            result = SystemPromptLoader.load(_llm_cfg(mlflow_name="p"), _mlflow_cfg())
        assert result is not None
        assert isinstance(result.source["version"], str)
        assert result.source["version"] == "42"

    def test_load_mlflow_name_plain_format(self) -> None:
        prompt = _mock_prompt(template="Plain.")
        with patch("mlflow.set_tracking_uri") as mock_uri, patch(
            "mlflow.genai.load_prompt", return_value=prompt
        ) as mock_load:
            SystemPromptLoader.load(_llm_cfg(mlflow_name="helixql-prompt"), _mlflow_cfg())
        mock_load.assert_called_once_with("helixql-prompt")

    def test_load_mlflow_name_version_uri_format(self) -> None:
        prompt = _mock_prompt(template="Versioned.")
        with patch("mlflow.set_tracking_uri"), patch(
            "mlflow.genai.load_prompt", return_value=prompt
        ) as mock_load:
            SystemPromptLoader.load(_llm_cfg(mlflow_name="prompts:/helixql-prompt/3"), _mlflow_cfg())
        mock_load.assert_called_once_with("prompts:/helixql-prompt/3")

    def test_load_mlflow_name_alias_uri_format(self) -> None:
        prompt = _mock_prompt(template="Aliased.")
        with patch("mlflow.set_tracking_uri"), patch(
            "mlflow.genai.load_prompt", return_value=prompt
        ) as mock_load:
            SystemPromptLoader.load(_llm_cfg(mlflow_name="prompts:/helixql-prompt@production"), _mlflow_cfg())
        mock_load.assert_called_once_with("prompts:/helixql-prompt@production")


# ---------------------------------------------------------------------------
# Invariants
# ---------------------------------------------------------------------------


class TestInvariants:
    def test_result_text_is_always_str(self, tmp_path: Path) -> None:
        f = tmp_path / "sys.txt"
        f.write_text("Hello.", encoding="utf-8")
        result = SystemPromptLoader.load(_llm_cfg(path=str(f)), None)
        assert result is not None
        assert isinstance(result.text, str)

    def test_result_source_is_always_dict(self, tmp_path: Path) -> None:
        f = tmp_path / "sys.txt"
        f.write_text("Hello.", encoding="utf-8")
        result = SystemPromptLoader.load(_llm_cfg(path=str(f)), None)
        assert result is not None
        assert isinstance(result.source, dict)

    def test_file_source_always_has_type_and_path_keys(self, tmp_path: Path) -> None:
        f = tmp_path / "sys.txt"
        f.write_text("Hello.", encoding="utf-8")
        result = SystemPromptLoader._from_file(str(f))
        assert result is not None
        assert "type" in result.source
        assert "path" in result.source

    def test_mlflow_source_always_has_type_name_version_keys(self) -> None:
        prompt = _mock_prompt(name="p", version=1, template="Hi.")
        with patch("mlflow.set_tracking_uri"), patch("mlflow.genai.load_prompt", return_value=prompt):
            result = SystemPromptLoader._from_mlflow("p", _mlflow_cfg())
        assert result is not None
        for key in ("type", "name", "version"):
            assert key in result.source

    def test_load_never_raises_on_file_errors(self, tmp_path: Path) -> None:
        missing = str(tmp_path / "no.txt")
        try:
            result = SystemPromptLoader.load(_llm_cfg(path=missing), None)
        except Exception as exc:  # noqa: BLE001
            pytest.fail(f"load() raised unexpectedly: {exc}")
        assert result is None

    def test_load_result_text_not_empty_when_returned(self, tmp_path: Path) -> None:
        f = tmp_path / "sys.txt"
        f.write_text("Non-empty.", encoding="utf-8")
        result = SystemPromptLoader.load(_llm_cfg(path=str(f)), None)
        assert result is not None
        assert result.text.strip() != ""


# ---------------------------------------------------------------------------
# Dependency errors
# ---------------------------------------------------------------------------


class TestDependencyErrors:
    def test_mlflow_import_error_returns_none(self) -> None:
        with (
            patch("mlflow.set_tracking_uri"),
            patch("mlflow.genai.load_prompt", side_effect=ImportError("mlflow not installed")),
        ):
            result = SystemPromptLoader.load(_llm_cfg(mlflow_name="p"), _mlflow_cfg())
        assert result is None

    def test_mlflow_connection_refused_returns_none(self) -> None:
        with (
            patch("mlflow.set_tracking_uri"),
            patch("mlflow.genai.load_prompt", side_effect=Exception("Connection refused")),
        ):
            result = SystemPromptLoader.load(_llm_cfg(mlflow_name="p"), _mlflow_cfg())
        assert result is None

    def test_mlflow_prompt_not_found_returns_none(self) -> None:
        with (
            patch("mlflow.set_tracking_uri"),
            patch("mlflow.genai.load_prompt", side_effect=Exception("Prompt not found")),
        ):
            result = SystemPromptLoader.load(_llm_cfg(mlflow_name="missing-prompt"), _mlflow_cfg())
        assert result is None


# ---------------------------------------------------------------------------
# Regressions
# ---------------------------------------------------------------------------


class TestRegressions:
    def test_llm_cfg_both_sources_raises_at_validation(self) -> None:
        """Config validator must reject both sources at once."""
        with pytest.raises(ValidationError, match="system_prompt_mlflow_name, not both"):
            InferenceLLMConfig(
                system_prompt_path="/some/path.txt",
                system_prompt_mlflow_name="my-prompt",
            )

    def test_llm_cfg_only_path_valid(self) -> None:
        cfg = InferenceLLMConfig(system_prompt_path="/some/path.txt")
        assert cfg.system_prompt_path == "/some/path.txt"
        assert cfg.system_prompt_mlflow_name is None

    def test_llm_cfg_only_mlflow_name_valid(self) -> None:
        cfg = InferenceLLMConfig(system_prompt_mlflow_name="my-prompt")
        assert cfg.system_prompt_mlflow_name == "my-prompt"
        assert cfg.system_prompt_path is None

    def test_llm_cfg_neither_source_valid(self) -> None:
        cfg = InferenceLLMConfig()
        assert cfg.system_prompt_path is None
        assert cfg.system_prompt_mlflow_name is None

    def test_file_source_does_not_call_mlflow(self, tmp_path: Path) -> None:
        """File source must never touch MLflow."""
        f = tmp_path / "sys.txt"
        f.write_text("Hello.", encoding="utf-8")
        with patch("mlflow.set_tracking_uri") as mock_uri, patch("mlflow.genai.load_prompt") as mock_load:
            SystemPromptLoader.load(_llm_cfg(path=str(f)), _mlflow_cfg())
        mock_uri.assert_not_called()
        mock_load.assert_not_called()

    def test_mlflow_source_does_not_read_files(self) -> None:
        """MLflow source must not open files."""
        prompt = _mock_prompt(template="From MLflow.")
        with (
            patch("mlflow.set_tracking_uri"),
            patch("mlflow.genai.load_prompt", return_value=prompt),
            patch.object(Path, "read_text") as mock_read,
        ):
            SystemPromptLoader.load(_llm_cfg(mlflow_name="p"), _mlflow_cfg())
        mock_read.assert_not_called()


# ---------------------------------------------------------------------------
# Logic-specific
# ---------------------------------------------------------------------------


class TestLogicSpecific:
    def test_mlflow_takes_priority_over_file_when_both_set_internal(self) -> None:
        """
        load() checks mlflow_name first.
        Uses model_construct to bypass config validator.
        """
        prompt = _mock_prompt(template="From MLflow.")
        with (
            patch("mlflow.set_tracking_uri"),
            patch("mlflow.genai.load_prompt", return_value=prompt) as mock_load,
        ):
            result = SystemPromptLoader.load(
                _llm_cfg(path="/should/not/be/read.txt", mlflow_name="my-prompt"),
                _mlflow_cfg(),
            )
        mock_load.assert_called_once()
        assert result is not None
        assert result.text == "From MLflow."

    def test_from_file_called_when_only_path_set(self, tmp_path: Path) -> None:
        f = tmp_path / "sys.txt"
        f.write_text("File content.", encoding="utf-8")
        with patch.object(SystemPromptLoader, "_from_file", wraps=SystemPromptLoader._from_file) as spy:
            SystemPromptLoader.load(_llm_cfg(path=str(f)), None)
        spy.assert_called_once_with(str(f))

    def test_from_mlflow_called_when_only_mlflow_name_set(self) -> None:
        prompt = _mock_prompt(template="MLflow content.")
        with (
            patch("mlflow.set_tracking_uri"),
            patch("mlflow.genai.load_prompt", return_value=prompt),
            patch.object(
                SystemPromptLoader, "_from_mlflow", wraps=SystemPromptLoader._from_mlflow
            ) as spy,
        ):
            SystemPromptLoader.load(_llm_cfg(mlflow_name="p"), _mlflow_cfg())
        spy.assert_called_once()

    def test_tracking_uri_passed_to_mlflow(self) -> None:
        prompt = _mock_prompt(template="Hi.")
        uri = "http://custom-mlflow:9000"
        with patch("mlflow.set_tracking_uri") as mock_uri, patch(
            "mlflow.genai.load_prompt", return_value=prompt
        ):
            SystemPromptLoader.load(_llm_cfg(mlflow_name="p"), _mlflow_cfg(tracking_uri=uri))
        mock_uri.assert_called_once_with(uri)

    def test_source_path_is_expanded_absolute(self, tmp_path: Path) -> None:
        f = tmp_path / "sys.txt"
        f.write_text("Hello.", encoding="utf-8")
        result = SystemPromptLoader._from_file(str(f))
        assert result is not None
        assert os.path.isabs(result.source["path"])


# ---------------------------------------------------------------------------
# Combinatorial
# ---------------------------------------------------------------------------


class TestCombinatorial:
    def test_file_path_none_mlflow_name_none_both_none(self) -> None:
        """Both fields None → load() returns None without raising."""
        result = SystemPromptLoader.load(_llm_cfg(path=None, mlflow_name=None), None)
        assert result is None

    def test_legacy_enabled_false_does_not_block_mlflow_prompt_loading(self) -> None:
        """Legacy enabled flag is ignored once MLflow config is present."""
        prompt = _mock_prompt(template="Prompt via MLflow.")
        with patch("mlflow.set_tracking_uri"), patch("mlflow.genai.load_prompt", return_value=prompt):
            result = SystemPromptLoader.load(_llm_cfg(mlflow_name="my-prompt"), _mlflow_cfg(enabled=False))
        assert result is not None
        assert result.text == "Prompt via MLflow."

    def test_mlflow_cfg_none_mlflow_name_set_raises_not_returns_none(self) -> None:
        """MLflow cfg missing + name set → ValueError, not None."""
        with pytest.raises(ValueError):
            SystemPromptLoader.load(_llm_cfg(mlflow_name="my-prompt"), None)

    def test_file_exists_and_enabled_mlflow_cfg_uses_file(self, tmp_path: Path) -> None:
        """If only path is set, MLflow cfg presence does not affect result."""
        f = tmp_path / "sys.txt"
        f.write_text("File wins.", encoding="utf-8")
        result = SystemPromptLoader.load(_llm_cfg(path=str(f)), _mlflow_cfg(enabled=True))
        assert result is not None
        assert result.text == "File wins."
        assert result.source["type"] == "file"

    def test_mlflow_error_and_file_not_configured_returns_none(self) -> None:
        """MLflow fails + file unset → None (non-fatal)."""
        with (
            patch("mlflow.set_tracking_uri"),
            patch("mlflow.genai.load_prompt", side_effect=RuntimeError("boom")),
        ):
            result = SystemPromptLoader.load(_llm_cfg(mlflow_name="p"), _mlflow_cfg())
        assert result is None

    def test_result_dataclass_has_text_and_source_fields(self) -> None:
        """SystemPromptResult dataclass has text and source."""
        field_names = {f.name for f in dataclass_fields(SystemPromptResult)}
        assert "text" in field_names
        assert "source" in field_names

    def test_result_default_source_is_empty_dict(self) -> None:
        r = SystemPromptResult(text="Hello.")
        assert r.source == {}

    def test_multiple_errors_each_return_none_independently(self, tmp_path: Path) -> None:
        """Each error path independently returns None."""
        cases = [
            _llm_cfg(path=str(tmp_path / "missing.txt")),  # file not found
        ]
        for llm in cases:
            assert SystemPromptLoader.load(llm, None) is None
