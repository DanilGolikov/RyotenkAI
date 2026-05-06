from __future__ import annotations

import importlib
import os
from pathlib import Path

from pydantic import ValidationError

from ryotenkai_control.api.schemas.config_validate import ConfigCheck, ConfigValidationResult


def _loc_to_path(loc: tuple) -> str:
    """Turn a Pydantic ``loc`` tuple into a dotted path the UI uses."""
    return ".".join(str(p) for p in loc)


def validate_config(config_path: Path) -> ConfigValidationResult:
    checks: list[ConfigCheck] = []
    field_errors: dict[str, list[str]] = {}
    cfg = None
    try:
        from ryotenkai_shared.config.loader import load_pipeline_config

        cfg = load_pipeline_config(config_path)
        checks.append(ConfigCheck(label="YAML schema valid (Pydantic)", status="ok"))
    except ValidationError as exc:
        # Pydantic gives us structured per-field errors via ``.errors()``.
        # Collect them into ``field_errors`` so the UI can paint each bad
        # field red + show the specific message inline, and keep a
        # coarse check for the banner.
        for err in exc.errors():
            loc = err.get("loc") or ()
            if not loc:
                continue
            path = _loc_to_path(loc)
            msg = str(err.get("msg") or "").strip()
            field_errors.setdefault(path, []).append(msg)
        summary = f"{len(exc.errors())} field error(s)" if exc.errors() else str(exc)
        checks.append(ConfigCheck(label="YAML schema", status="fail", detail=summary))
    except Exception as exc:
        checks.append(ConfigCheck(label="YAML schema", status="fail", detail=str(exc)))

    if cfg is not None:
        from ryotenkai_shared.config import DatasetSourceLocal

        # Dataset paths
        for ds_name, ds_cfg in cfg.datasets.items():
            source = ds_cfg.source
            if isinstance(source, DatasetSourceLocal):
                train_path = source.local_paths.train
                eval_path = getattr(source.local_paths, "eval", None)
                if Path(train_path).exists():
                    checks.append(
                        ConfigCheck(label=f"Dataset '{ds_name}' train path exists", status="ok", detail=train_path)
                    )
                else:
                    checks.append(
                        ConfigCheck(label=f"Dataset '{ds_name}' train path not found", status="fail", detail=train_path)
                    )
                if eval_path:
                    if Path(eval_path).exists():
                        checks.append(
                            ConfigCheck(label=f"Dataset '{ds_name}' eval path exists", status="ok", detail=eval_path)
                        )
                    else:
                        checks.append(
                            ConfigCheck(
                                label=f"Dataset '{ds_name}' eval path not found", status="fail", detail=eval_path
                            )
                        )
            else:
                checks.append(ConfigCheck(label=f"Dataset '{ds_name}' (HuggingFace — path check skipped)", status="ok"))

        # Env vars
        hf_token = os.environ.get("HF_TOKEN", "").strip()
        if hf_token:
            checks.append(ConfigCheck(label="HF_TOKEN found", status="ok"))
        else:
            checks.append(ConfigCheck(label="HF_TOKEN not set", status="fail"))

        # Phase 14.D+F — provider-driven secret check.
        # `get_active_provider_name()` raises when `training.provider` is
        # unset. That's expected at runtime (you can't launch without a
        # provider), but during the UI's pre-save validation we don't
        # want that to surface as a 500. Tolerate "not yet chosen" — the
        # dedicated provider-required check below still nags the user.
        try:
            active_provider = cfg.get_active_provider_name() if hasattr(cfg, "get_active_provider_name") else None
        except ValueError:
            active_provider = None

        # Resolve required secrets via the same registry the
        # startup validator uses. Each provider declares its own
        # tuple — adding a third provider with credentials = one
        # registry update, no edits here.
        from ryotenkai_control.pipeline.bootstrap.startup_validator import (
            _resolve_required_secrets_for_provider,
        )

        required_secrets = _resolve_required_secrets_for_provider(active_provider) if active_provider else ()
        runpod_key = os.environ.get("RUNPOD_API_KEY", "").strip()
        if "RUNPOD_API_KEY" in required_secrets:
            if runpod_key:
                checks.append(ConfigCheck(label="RUNPOD_API_KEY found", status="ok"))
            else:
                checks.append(
                    ConfigCheck(
                        label=f"RUNPOD_API_KEY not set (required for {active_provider} provider)",
                        status="fail",
                    )
                )
        elif runpod_key:
            checks.append(ConfigCheck(label="RUNPOD_API_KEY found (optional)", status="ok"))
        else:
            checks.append(ConfigCheck(label="RUNPOD_API_KEY not set (optional)", status="warn"))

        # Eval plugins
        eval_cfg = getattr(cfg, "evaluation", None)
        if eval_cfg and getattr(eval_cfg, "enabled", False):
            plugins_cfg = getattr(getattr(eval_cfg, "evaluators", None), "plugins", []) or []
            for plug_cfg in plugins_cfg:
                plugin_type = getattr(plug_cfg, "type", None)
                if plugin_type:
                    try:
                        importlib.import_module(f"ryotenkai_control.evaluation.plugins.builtins.{plugin_type}")
                        checks.append(ConfigCheck(label=f"Eval plugin '{plugin_type}' importable", status="ok"))
                    except ImportError:
                        checks.append(
                            ConfigCheck(label=f"Eval plugin '{plugin_type}' not found (custom?)", status="warn")
                        )
        else:
            checks.append(ConfigCheck(label="Evaluation disabled — plugin check skipped", status="ok"))

        # Reward ↔ strategy compatibility. Each training phase with a
        # reward_plugin must reference a community reward plugin whose
        # [plugin].supported_strategies contains the phase's strategy_type.
        # Unlike pure schema checks, this one needs the community catalog,
        # so it lives here (not in Pydantic validators).
        _check_reward_strategy_compat(cfg, checks, field_errors)

        # Stage consistency
        inference_enabled = getattr(getattr(cfg, "inference", None), "enabled", False)
        eval_enabled = eval_cfg and getattr(eval_cfg, "enabled", False)
        if eval_enabled and not inference_enabled:
            checks.append(
                ConfigCheck(
                    label="Stage consistency",
                    status="warn",
                    detail="evaluation.enabled=true but inference.enabled=false — ModelEvaluator needs a live runtime",
                )
            )
        else:
            checks.append(ConfigCheck(label="Stage consistency", status="ok"))

    ok = not any(check.status == "fail" for check in checks)
    return ConfigValidationResult(
        ok=ok,
        config_path=str(config_path),
        checks=checks,
        field_errors=field_errors,
    )


def _check_reward_strategy_compat(
    cfg: object,
    checks: list[ConfigCheck],
    field_errors: dict[str, list[str]],
) -> None:
    """Ensure every training phase's ``reward_plugin`` is compatible with
    the phase's ``strategy_type``.

    Cross-validator — lives outside Pydantic because it needs the
    community catalog (reward plugin's ``supported_strategies`` comes
    from the manifest, not the YAML). Mirrors existing validator pattern
    in :mod:`src.config.validators.runtime`.
    """
    training = getattr(cfg, "training", None)
    if training is None:
        return
    strategies = getattr(training, "strategies", None) or []
    if not strategies:
        return

    # Load once per validation — cheap and uses the mtime-fingerprint
    # cache already in CommunityCatalog.
    from ryotenkai_community.catalog import catalog

    catalog.ensure_loaded()
    reward_plugins = {entry.manifest.plugin.id: entry.manifest.plugin for entry in catalog.plugins("reward")}

    any_issue = False
    for idx, strat in enumerate(strategies):
        params = getattr(strat, "params", {}) or {}
        reward_id = params.get("reward_plugin") if isinstance(params, dict) else None
        if not reward_id:
            continue
        strategy_type = (getattr(strat, "strategy_type", "") or "").lower()
        path = f"training.strategies.{idx}.params.reward_plugin"

        spec = reward_plugins.get(reward_id)
        if spec is None:
            any_issue = True
            msg = (
                f"Reward plugin '{reward_id}' is not in the community catalog. "
                f"Install it under community/reward/ or pick another plugin."
            )
            field_errors.setdefault(path, []).append(msg)
            checks.append(
                ConfigCheck(
                    label=f"Strategy #{idx + 1} ({strategy_type}) reward plugin missing",
                    status="fail",
                    detail=f"{reward_id} (not found)",
                )
            )
            continue

        supported = [s.lower() for s in spec.supported_strategies]
        if strategy_type not in supported:
            any_issue = True
            msg = (
                f"Reward plugin '{reward_id}' supports {supported} but "
                f"strategy_type is {strategy_type!r}. Either pick a compatible "
                f"reward plugin or change the strategy."
            )
            field_errors.setdefault(path, []).append(msg)
            checks.append(
                ConfigCheck(
                    label=f"Strategy #{idx + 1} reward/strategy mismatch",
                    status="fail",
                    detail=f"{reward_id} supports {supported}, phase uses {strategy_type}",
                )
            )
        else:
            checks.append(
                ConfigCheck(
                    label=f"Strategy #{idx + 1} reward plugin compatible",
                    status="ok",
                    detail=f"{reward_id} → {strategy_type}",
                )
            )

    if not any_issue and not any(
        (getattr(s, "params", None) or {}).get("reward_plugin")
        if isinstance(getattr(s, "params", None), dict)
        else False
        for s in strategies
    ):
        # No reward plugins in use — not a failure, just a neutral note.
        checks.append(ConfigCheck(label="No reward plugins configured", status="ok"))


__all__ = ["validate_config"]
