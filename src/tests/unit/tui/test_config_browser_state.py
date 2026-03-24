from __future__ import annotations

from pathlib import Path

from src.tui.config_browser_state import ConfigBrowserState


def _write_config(path: Path) -> Path:
    path.write_text(
        """
model:
  name: "Qwen/Test"
training:
  provider: runpod
  strategies:
    - strategy_type: "sft"
datasets:
  default:
    source_type: local
    source_local:
      local_paths:
        train: "/tmp/train.jsonl"
        eval: "/tmp/eval.jsonl"
    validations:
      plugins:
        - id: min_samples_train
          plugin: min_samples
          apply_to: ["train"]
providers:
  runpod:
    connect:
      ssh:
        key_path: "~/.ssh/id_ed25519_runpod"
    training:
      gpu_type: "NVIDIA RTX A4000"
evaluation:
  enabled: true
  evaluators:
    plugins:
      - id: syntax_main
        plugin: helixql_syntax
        enabled: true
""".strip(),
        encoding="utf-8",
    )
    return path


def test_config_browser_state_exposes_root_sections_with_human_summaries(tmp_path: Path) -> None:
    state = ConfigBrowserState.load(_write_config(tmp_path / "config.yaml"))

    sections = {item.label: item for item in state.section_items()}

    assert sections["Model"].subtitle == ""
    assert sections["Providers"].subtitle == ""
    assert sections["Datasets"].subtitle == ""
    assert sections["Training"].subtitle == ""
    assert sections["Model"].is_section is True


def test_config_browser_state_formats_provider_and_dataset_entries(tmp_path: Path) -> None:
    state = ConfigBrowserState.load(_write_config(tmp_path / "config.yaml"))

    provider_items = state.list_children(("providers",))
    dataset_items = state.list_children(("datasets",))

    assert provider_items[0].label == "runpod"
    assert provider_items[0].field_count == 2
    assert provider_items[0].value_text is None
    assert dataset_items[0].label == "default"
    assert dataset_items[0].field_count == 3
    assert dataset_items[0].value_text is None


def test_config_browser_state_formats_plugin_entries_and_details(tmp_path: Path) -> None:
    state = ConfigBrowserState.load(_write_config(tmp_path / "config.yaml"))

    dataset_plugins = state.list_children(("datasets", "default", "validations", "plugins"))
    evaluation_plugins = state.list_children(("evaluation", "evaluators", "plugins"))

    assert dataset_plugins[0].label == "min_samples_train (min_samples)"
    assert dataset_plugins[0].field_count == 3
    assert dataset_plugins[0].value_text is None
    assert evaluation_plugins[0].label == "syntax_main (helixql_syntax)"
    assert evaluation_plugins[0].field_count == 3
    assert evaluation_plugins[0].value_text is None

    detail = state.describe(("datasets", "default"))
    assert "Source type: local" in detail
    assert "Train: /tmp/train.jsonl" in detail
    assert "Validation plugins: 1" in detail


def test_config_browser_state_formats_scalar_items_as_colored_value_candidates(tmp_path: Path) -> None:
    state = ConfigBrowserState.load(_write_config(tmp_path / "config.yaml"))

    model_items = {item.label: item for item in state.list_children(("model",))}

    assert model_items["Name"].value_text == "Qwen/Test"
    assert model_items["Name"].item_count is None
    assert model_items["Name"].field_count is None
