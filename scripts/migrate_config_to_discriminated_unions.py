#!/usr/bin/env python3
"""YAML migration: pre-discriminated-union shape → discriminated-union shape.

Rewrites legacy YAML config files to the new shape across four concerns:

  1. INFERENCE ENGINE
     before:  inference: { engine: vllm, engines: { vllm: {...} } }
     after:   inference: { engine: { kind: vllm, ... } }

  2. TRAINING ADAPTER
     before:  training: { type: qlora, qlora: { r: 16, ... } }
     after:   training: { adapter: { kind: qlora, r: 16, ... } }

  3. DATASET SOURCE
     before:  datasets.<name>: { source_type: local, source_local: { ... } }
     after:   datasets.<name>: { source: { kind: local, ... } }

  4. PLUGIN AUTO-IDS (optional)
     Strips ``id:`` lines from evaluator/validator plugin entries IFF the
     value matches the auto-generated hash. Preserves explicit human-
     readable ids unchanged.

Usage:
    # Preview
    python scripts/migrate_config_to_discriminated_unions.py --dry-run path/to/config.yaml ...

    # Apply
    python scripts/migrate_config_to_discriminated_unions.py path/to/config.yaml ...

Idempotent: running twice on a migrated config is a no-op.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

try:
    import yaml
except ImportError:  # pragma: no cover
    print("PyYAML required: pip install pyyaml", file=sys.stderr)
    sys.exit(2)


# ---------------------------------------------------------------------------
# Migrations
# ---------------------------------------------------------------------------


def _migrate_inference(cfg: dict[str, Any]) -> bool:
    """``engine: vllm + engines.vllm: {...}`` → ``engine: {kind: vllm, ...}``."""
    inference = cfg.get("inference")
    if not isinstance(inference, dict):
        return False
    if "engine" in inference and isinstance(inference["engine"], dict):
        return False  # already migrated

    engine_kind = inference.get("engine") if isinstance(inference.get("engine"), str) else None
    engines_block = inference.get("engines")
    if not engine_kind or not isinstance(engines_block, dict):
        return False

    engine_payload = engines_block.get(engine_kind)
    if not isinstance(engine_payload, dict):
        return False

    inference["engine"] = {"kind": engine_kind, **engine_payload}
    inference.pop("engines", None)
    return True


def _migrate_training_adapter(cfg: dict[str, Any]) -> bool:
    """``type: qlora + qlora: {...}`` → ``adapter: {kind: qlora, ...}``."""
    training = cfg.get("training")
    if not isinstance(training, dict):
        return False
    if "adapter" in training and isinstance(training["adapter"], dict):
        return False  # already migrated

    type_value = training.get("type")
    if not isinstance(type_value, str):
        return False
    if type_value not in ("lora", "qlora", "adalora"):
        return False

    adapter_block = training.get(type_value)
    if not isinstance(adapter_block, dict):
        return False

    training["adapter"] = {"kind": type_value, **adapter_block}
    training.pop("type", None)
    for k in ("lora", "qlora", "adalora"):
        training.pop(k, None)
    return True


def _migrate_dataset_source(cfg: dict[str, Any]) -> bool:
    """``source_type: local + source_local: {...}`` → ``source: {kind: local, ...}``."""
    datasets = cfg.get("datasets")
    if not isinstance(datasets, dict):
        return False

    changed = False
    for ds_name, ds_block in datasets.items():
        if not isinstance(ds_block, dict):
            continue
        if "source" in ds_block and isinstance(ds_block["source"], dict):
            continue

        source_type = ds_block.get("source_type")
        if source_type == "local":
            payload = ds_block.get("source_local")
        elif source_type == "huggingface":
            payload = ds_block.get("source_hf")
        else:
            continue

        if not isinstance(payload, dict):
            continue

        ds_block["source"] = {"kind": source_type, **payload}
        ds_block.pop("source_type", None)
        ds_block.pop("source_local", None)
        ds_block.pop("source_hf", None)
        _ = ds_name
        changed = True
    return changed


def _strip_auto_ids(cfg: dict[str, Any]) -> bool:
    """Strip explicit ``id:`` lines from evaluator/validator plugin entries
    when the value matches the hash-based auto-id."""
    changed = False

    def _autogen(plugin: str, params: dict[str, Any]) -> str:
        payload = json.dumps(params or {}, sort_keys=True, default=str).encode()
        return f"{plugin}_{hashlib.md5(payload, usedforsecurity=False).hexdigest()[:8]}"

    # evaluation.evaluators.plugins[]
    eval_block = cfg.get("evaluation", {}).get("evaluators", {})
    plugins = eval_block.get("plugins", []) if isinstance(eval_block, dict) else []
    for entry in plugins if isinstance(plugins, list) else []:
        if not isinstance(entry, dict):
            continue
        plugin = entry.get("plugin")
        params = entry.get("params") or {}
        if isinstance(plugin, str) and entry.get("id") == _autogen(plugin, params):
            entry.pop("id")
            changed = True

    # datasets.<name>.validations.plugins[]
    datasets = cfg.get("datasets") or {}
    for _name, ds in (datasets.items() if isinstance(datasets, dict) else []):
        validations = ds.get("validations") if isinstance(ds, dict) else None
        plugins = validations.get("plugins") if isinstance(validations, dict) else None
        for entry in plugins or []:
            if not isinstance(entry, dict):
                continue
            plugin = entry.get("plugin")
            params = entry.get("params") or {}
            if isinstance(plugin, str) and entry.get("id") == _autogen(plugin, params):
                entry.pop("id")
                changed = True

    return changed


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def _process_one(path: Path, *, dry_run: bool) -> bool:
    """Process a single YAML file. Returns True if any change was made."""
    try:
        original = path.read_text(encoding="utf-8")
    except OSError as exc:
        print(f"[skip] {path}: {exc}", file=sys.stderr)
        return False

    try:
        cfg = yaml.safe_load(original)
    except yaml.YAMLError as exc:
        print(f"[skip] {path}: YAML parse error: {exc}", file=sys.stderr)
        return False

    if not isinstance(cfg, dict):
        return False

    changed = False
    changed |= _migrate_inference(cfg)
    changed |= _migrate_training_adapter(cfg)
    changed |= _migrate_dataset_source(cfg)
    changed |= _strip_auto_ids(cfg)

    if not changed:
        return False

    new_text = yaml.safe_dump(
        cfg, sort_keys=False, default_flow_style=False, allow_unicode=True,
    )

    if dry_run:
        print(f"[would-migrate] {path}")
    else:
        path.write_text(new_text, encoding="utf-8")
        print(f"[migrated] {path}")
    return True


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("paths", nargs="+", type=Path, help="YAML file(s) or directory(ies)")
    ap.add_argument("--dry-run", action="store_true", help="Print changes without writing")
    args = ap.parse_args()

    files: list[Path] = []
    for p in args.paths:
        if p.is_dir():
            files.extend(p.rglob("*.yaml"))
            files.extend(p.rglob("*.yml"))
        elif p.is_file():
            files.append(p)
        else:
            print(f"[skip] {p}: not found", file=sys.stderr)

    total = sum(_process_one(f, dry_run=args.dry_run) for f in files)
    suffix = " (dry-run)" if args.dry_run else ""
    print(f"\n[done] {total}/{len(files)} file(s) {'would change' if args.dry_run else 'migrated'}{suffix}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
