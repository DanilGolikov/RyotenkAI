"""``ryotenkai config <verb>`` — read-only config inspection + validation.

`config validate <path>` runs the full set of static + lightweight env
checks defined in :mod:`src.api.services.config_service`. `config show`
re-emits the parsed/expanded config (useful with ``-o yaml | jq``-style
flows). `config schema` dumps the JSON Schema for the pipeline config.
`config explain` prints a human summary of model + training + dataset
choices for a given file.
"""

from __future__ import annotations

import json
from typing import Annotated

import typer

from src.cli.common_options import RequiredConfigOpt
from src.cli.context import CLIContext
from src.cli.errors import die
from src.cli.renderer import get_renderer

config_app = typer.Typer(
    no_args_is_help=True,
    help="Validate, show, explain, or dump the schema of pipeline configs.",
    rich_markup_mode=None,
    context_settings={"help_option_names": ["-h", "--help"]},
    pretty_exceptions_enable=False,
)


@config_app.command("validate")
def validate_cmd(
    ctx: typer.Context,
    config: RequiredConfigOpt,
) -> None:
    """Static pre-flight checks for a pipeline config (no network calls)."""
    from src.api.services import config_service

    state = ctx.ensure_object(CLIContext)
    renderer = get_renderer(state)

    try:
        result = config_service.validate_config(config)
    except FileNotFoundError as exc:
        raise die(str(exc))

    if state.is_machine_readable:
        renderer.emit({
            "ok": result.ok,
            "config_path": result.config_path,
            "checks": [c.model_dump() for c in result.checks],
            "field_errors": result.field_errors,
        })
    else:
        marker = {"ok": "[OK]", "warn": "[WARN]", "fail": "[FAIL]"}
        for check in result.checks:
            line = f"  {marker[check.status]} {check.label}"
            if check.detail:
                line += f"  ({check.detail})"
            renderer.text(line)
        for loc, msgs in result.field_errors.items():
            for msg in msgs:
                renderer.text(f"  [FAIL] {loc}: {msg}")
        renderer.text("")
        renderer.text("Result: ready to run" if result.ok else "Result: not ready — fix errors above")
    renderer.flush()
    if not result.ok:
        raise typer.Exit(code=1)


@config_app.command("show")
def show_cmd(
    ctx: typer.Context,
    config: RequiredConfigOpt,
) -> None:
    """Print the parsed pipeline config (model_dump)."""
    from src.utils.config import load_config

    state = ctx.ensure_object(CLIContext)
    renderer = get_renderer(state)
    cfg = load_config(config)
    renderer.emit(cfg.model_dump(mode="json"))
    renderer.flush()


@config_app.command("explain")
def explain_cmd(
    ctx: typer.Context,
    config: RequiredConfigOpt,
) -> None:
    """Show a short human-readable summary of model / dataset / training."""
    from src.config.datasets.constants import SOURCE_TYPE_HUGGINGFACE
    from src.utils.config import load_config

    state = ctx.ensure_object(CLIContext)
    renderer = get_renderer(state)
    cfg = load_config(config)

    model_cfg = cfg.model
    training_cfg = cfg.training
    strategies = [s.strategy_type for s in training_cfg.get_strategy_chain()]
    default_ds = cfg.get_primary_dataset()
    train_ref = default_ds.get_display_train_ref()
    if default_ds.get_source_type() == SOURCE_TYPE_HUGGINGFACE:
        assert default_ds.source_hf is not None
        eval_ref = default_ds.source_hf.eval_id or None
    else:
        assert default_ds.source_local is not None
        eval_ref = default_ds.source_local.local_paths.eval or None

    payload = {
        "model": {"name": model_cfg.name, "training_type": training_cfg.type},
        "strategies": strategies,
        "dataset": {"train": train_ref, "eval": eval_ref},
    }

    if state.is_machine_readable:
        renderer.emit(payload)
    else:
        renderer.kv(
            {"Model": model_cfg.name, "Training": training_cfg.type,
             "Strategies": " → ".join(s.upper() for s in strategies) if strategies else "-"},
            title="Model",
        )
        renderer.text("")
        renderer.kv({"Train": train_ref, "Eval": eval_ref or "-"}, title="Dataset")
    renderer.flush()


@config_app.command("schema")
def schema_cmd(
    ctx: typer.Context,
    indent: Annotated[
        int, typer.Option("--indent", help="JSON indent (text mode only).")
    ] = 2,
) -> None:
    """Print the JSON Schema for ``PipelineConfig``."""
    from src.utils.config import PipelineConfig

    state = ctx.ensure_object(CLIContext)
    renderer = get_renderer(state)
    schema = PipelineConfig.model_json_schema()
    if state.is_machine_readable:
        renderer.emit(schema)
    else:
        renderer.text(json.dumps(schema, indent=indent))
    renderer.flush()


__all__ = ["config_app"]
