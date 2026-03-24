from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import yaml

if TYPE_CHECKING:
    from pathlib import Path

type ConfigPathPart = str | int
type ConfigPath = tuple[ConfigPathPart, ...]

_ROOT_SECTION_LABELS = {
    "model": "Model",
    "training": "Training",
    "datasets": "Datasets",
    "providers": "Providers",
    "evaluation": "Evaluation",
    "inference": "Inference",
    "experiment_tracking": "Experiment Tracking",
}
_PROVIDER_BLOCK_ORDER = ("connect", "cleanup", "training", "inference")


@dataclass(frozen=True, slots=True)
class ConfigBrowserItem:
    label: str
    path: ConfigPath
    has_children: bool
    subtitle: str = ""
    value_text: str | None = None
    item_count: int | None = None
    field_count: int | None = None
    is_section: bool = False


class ConfigBrowserState:
    """Structured, read-only navigation model for pipeline YAML files."""

    def __init__(self, *, path: Path, data: dict[str, Any]) -> None:
        self.path = path.expanduser().resolve()
        self._data = data

    @classmethod
    def load(cls, path: Path) -> ConfigBrowserState:
        resolved_path = path.expanduser().resolve()
        raw = yaml.safe_load(resolved_path.read_text(encoding="utf-8", errors="replace"))
        if not isinstance(raw, dict):
            raise ValueError("Config root must be a mapping")
        return cls(path=resolved_path, data=raw)

    def section_items(self) -> tuple[ConfigBrowserItem, ...]:
        items = [
            ConfigBrowserItem(
                label=_ROOT_SECTION_LABELS.get(key, _humanize_key(key)),
                path=(key,),
                has_children=_has_children(value),
                is_section=True,
            )
            for key, value in self._data.items()
        ]
        return tuple(items)

    def list_children(self, path: ConfigPath) -> tuple[ConfigBrowserItem, ...]:
        value = self.resolve(path)
        if isinstance(value, dict):
            return tuple(
                ConfigBrowserItem(
                    label=_label_for_mapping_item(path, key, child_value),
                    path=(*path, key),
                    has_children=_has_children(child_value),
                    value_text=_item_value_text(child_value),
                    item_count=_item_count(child_value),
                    field_count=_field_count(child_value),
                )
                for key, child_value in value.items()
            )
        if isinstance(value, list):
            return tuple(
                ConfigBrowserItem(
                    label=_label_for_list_item(path, index, child_value),
                    path=(*path, index),
                    has_children=_has_children(child_value),
                    value_text=_item_value_text(child_value),
                    item_count=_item_count(child_value),
                    field_count=_field_count(child_value),
                )
                for index, child_value in enumerate(value)
            )
        return ()

    def describe(self, path: ConfigPath) -> tuple[str, ...]:
        value = self.resolve(path)
        lines = [
            f"Path: {_format_path(path)}",
            f"Type: {_describe_type(value)}",
        ]
        summary = _summary_lines(path, value)
        if summary:
            lines.extend(summary)
        dump = yaml.safe_dump(value, sort_keys=False, allow_unicode=False, default_flow_style=False).rstrip()
        if dump:
            lines.append("")
            lines.append("YAML:")
            lines.extend(f"  {line}" for line in dump.splitlines())
        return tuple(lines)

    def resolve(self, path: ConfigPath) -> Any:
        value: Any = self._data
        for part in path:
            if isinstance(part, int):
                if not isinstance(value, list):
                    raise KeyError(f"Path {_format_path(path)} does not resolve to a list")
                value = value[part]
                continue
            if not isinstance(value, dict):
                raise KeyError(f"Path {_format_path(path)} does not resolve to a mapping")
            value = value[part]
        return value


def _label_for_mapping_item(parent_path: ConfigPath, key: str, value: Any) -> str:
    if parent_path == ("providers",):
        return key
    if parent_path == ("datasets",):
        return key
    if parent_path == ("training", "strategies"):
        strategy_type = value.get("strategy_type") if isinstance(value, dict) else None
        return f"{key} ({strategy_type})" if strategy_type else key
    return _humanize_key(key)


def _label_for_list_item(parent_path: ConfigPath, index: int, value: Any) -> str:
    if _is_dataset_plugin_list(parent_path) or _is_evaluation_plugin_list(parent_path):
        plugin_id = value.get("id") if isinstance(value, dict) else None
        plugin_name = value.get("plugin") if isinstance(value, dict) else None
        if plugin_id and plugin_name:
            return f"{plugin_id} ({plugin_name})"
        if plugin_id:
            return str(plugin_id)
        if plugin_name:
            return str(plugin_name)
    if isinstance(value, dict):
        for key in ("name", "id", "plugin", "strategy_type", "provider"):
            candidate = value.get(key)
            if candidate:
                return str(candidate)
    return f"[{index}]"


def _summarize_value(path: ConfigPath, value: Any) -> str:
    if path == ("providers",):
        return _pluralize(_mapping_size(value), "provider")
    if path == ("datasets",):
        return _pluralize(_mapping_size(value), "dataset")
    if path == ("training",):
        provider = value.get("provider") if isinstance(value, dict) else None
        strategies = len(value.get("strategies", [])) if isinstance(value, dict) else 0
        parts = []
        if provider:
            parts.append(f"provider: {provider}")
        if strategies:
            parts.append(_pluralize(strategies, "strategy"))
        return ", ".join(parts) or _describe_type(value)
    if path == ("evaluation",):
        plugins = _plugin_count(_nested_get(value, "evaluators", "plugins"))
        enabled = value.get("enabled") if isinstance(value, dict) else None
        parts = []
        if enabled is not None:
            parts.append("enabled" if enabled else "disabled")
        if plugins:
            parts.append(_pluralize(plugins, "plugin"))
        return ", ".join(parts) or _describe_type(value)
    if path == ("inference",) and isinstance(value, dict):
        provider = value.get("provider")
        engine = value.get("engine")
        parts = [part for part in (provider, engine) if part]
        return " / ".join(str(part) for part in parts) or _describe_type(value)
    if path == ("model",) and isinstance(value, dict) and value.get("name"):
        return str(value["name"])
    if len(path) == 2 and path[0] == "providers" and isinstance(value, dict):
        blocks = [key for key in _PROVIDER_BLOCK_ORDER if value.get(key)]
        return ", ".join(blocks) or "no blocks"
    if len(path) == 2 and path[0] == "datasets" and isinstance(value, dict):
        source_type = value.get("source_type", "unknown")
        plugin_count = _plugin_count(_nested_get(value, "validations", "plugins"))
        parts = [f"source: {source_type}"]
        if plugin_count:
            parts.append(_pluralize(plugin_count, "plugin"))
        return ", ".join(parts)
    if _is_dataset_plugin(path) and isinstance(value, dict):
        parts = [str(value.get("plugin", "plugin"))]
        apply_to = value.get("apply_to")
        if isinstance(apply_to, list) and apply_to:
            parts.append("/".join(str(part) for part in apply_to))
        return " | ".join(parts)
    if _is_evaluation_plugin(path) and isinstance(value, dict):
        parts = [str(value.get("plugin", "plugin"))]
        enabled = value.get("enabled")
        if enabled is not None:
            parts.append("enabled" if enabled else "disabled")
        return " | ".join(parts)
    if isinstance(value, dict):
        return _pluralize(len(value), "field")
    if isinstance(value, list):
        return _pluralize(len(value), "item")
    return _format_scalar(value)


def _item_value_text(value: Any) -> str | None:
    if isinstance(value, dict | list):
        return None
    return _format_scalar(value)


def _item_count(value: Any) -> int | None:
    return len(value) if isinstance(value, list) else None


def _field_count(value: Any) -> int | None:
    return len(value) if isinstance(value, dict) else None


def _summary_lines(path: ConfigPath, value: Any) -> list[str]:
    lines: list[str] = []
    if len(path) == 2 and path[0] == "providers" and isinstance(value, dict):
        blocks = [key for key in _PROVIDER_BLOCK_ORDER if key in value]
        if blocks:
            lines.append(f"Blocks: {', '.join(blocks)}")
        return lines
    if len(path) == 2 and path[0] == "datasets" and isinstance(value, dict):
        lines.append(f"Source type: {value.get('source_type', 'unknown')}")
        train_ref = _nested_get(value, "source_local", "local_paths", "train") or _nested_get(
            value, "source_hf", "train_id"
        )
        eval_ref = _nested_get(value, "source_local", "local_paths", "eval") or _nested_get(
            value, "source_hf", "eval_id"
        )
        if train_ref:
            lines.append(f"Train: {train_ref}")
        if eval_ref:
            lines.append(f"Eval: {eval_ref}")
        plugin_count = _plugin_count(_nested_get(value, "validations", "plugins"))
        if plugin_count:
            lines.append(f"Validation plugins: {plugin_count}")
        return lines
    if _is_dataset_plugin(path) and isinstance(value, dict):
        lines.append(f"Plugin: {value.get('plugin', 'unknown')}")
        apply_to = value.get("apply_to")
        if isinstance(apply_to, list) and apply_to:
            lines.append(f"Apply to: {', '.join(str(part) for part in apply_to)}")
        thresholds = value.get("thresholds")
        if isinstance(thresholds, dict) and thresholds:
            lines.append(f"Threshold fields: {', '.join(str(key) for key in thresholds)}")
        return lines
    if _is_evaluation_plugin(path) and isinstance(value, dict):
        lines.append(f"Plugin: {value.get('plugin', 'unknown')}")
        enabled = value.get("enabled")
        if enabled is not None:
            lines.append(f"Enabled: {enabled}")
        thresholds = value.get("thresholds")
        if isinstance(thresholds, dict) and thresholds:
            lines.append(f"Threshold fields: {', '.join(str(key) for key in thresholds)}")
        return lines
    if isinstance(value, dict):
        lines.append(f"Fields: {len(value)}")
    elif isinstance(value, list):
        lines.append(f"Items: {len(value)}")
    return lines


def _nested_get(value: Any, *parts: str) -> Any:
    current = value
    for part in parts:
        if not isinstance(current, dict):
            return None
        current = current.get(part)
    return current


def _plugin_count(value: Any) -> int:
    return len(value) if isinstance(value, list) else 0


def _mapping_size(value: Any) -> int:
    return len(value) if isinstance(value, dict) else 0


def _pluralize(value: int, noun: str) -> str:
    return f"{value} {noun}" + ("" if value == 1 else "s")


def _humanize_key(key: str) -> str:
    return str(key).replace("_", " ").title()


def _has_children(value: Any) -> bool:
    return isinstance(value, dict | list) and bool(value)


def _describe_type(value: Any) -> str:
    if isinstance(value, dict):
        return "mapping"
    if isinstance(value, list):
        return "list"
    if isinstance(value, bool):
        return "boolean"
    if value is None:
        return "null"
    return type(value).__name__


def _format_scalar(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


def _format_path(path: ConfigPath) -> str:
    parts: list[str] = []
    for part in path:
        if isinstance(part, int):
            if parts:
                parts[-1] = f"{parts[-1]}[{part}]"
            else:
                parts.append(f"[{part}]")
        else:
            parts.append(str(part))
    return ".".join(parts)


def _is_dataset_plugin_list(path: ConfigPath) -> bool:
    return len(path) >= 2 and path[-2:] == ("validations", "plugins")


def _is_dataset_plugin(path: ConfigPath) -> bool:
    return len(path) >= 3 and isinstance(path[-1], int) and path[-3:-1] == ("validations", "plugins")


def _is_evaluation_plugin_list(path: ConfigPath) -> bool:
    return len(path) >= 2 and path[-2:] == ("evaluators", "plugins")


def _is_evaluation_plugin(path: ConfigPath) -> bool:
    return len(path) >= 3 and isinstance(path[-1], int) and path[-3:-1] == ("evaluators", "plugins")


__all__ = ["ConfigBrowserItem", "ConfigBrowserState", "ConfigPath", "ConfigPathPart"]
