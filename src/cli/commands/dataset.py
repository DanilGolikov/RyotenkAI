"""``ryotenkai dataset <verb>`` — dataset-only operations.

Currently a single verb: ``validate``. Runs the orchestrator's stage 0
(dataset validation) without spawning the rest of the pipeline.

NR-04 (plan B): on RESEACRH branch ``DatasetValidator`` no longer
silently picks up "default" plugins — validation must be explicit. If
the user passes a config with no per-dataset ``[validation.plugins]``
entries this command **must not** silently report success. Instead we
detect the empty-plugin case up front and exit with a helpful message
pointing at ``ryotenkai plugin ls --kind validation``.
"""

from __future__ import annotations

import typer

from src.cli.common_options import RequiredConfigOpt
from src.cli.context import CLIContext
from src.cli.errors import die
from src.cli.renderer import get_renderer

dataset_app = typer.Typer(
    no_args_is_help=True,
    help="Dataset-level operations (validation, ...).",
    rich_markup_mode=None,
    context_settings={"help_option_names": ["-h", "--help"]},
    pretty_exceptions_enable=False,
)


@dataset_app.command("validate")
def validate_cmd(
    ctx: typer.Context,
    config: RequiredConfigOpt,
) -> None:
    """Run dataset validation (Stage 0) for the configured datasets.

    Exits non-zero with a clear message when the config declares no
    validation plugins — the orchestrator no longer treats this as
    "everything's fine, nothing to check".
    """
    from src.utils.config import load_config

    state = ctx.ensure_object(CLIContext)
    renderer = get_renderer(state)

    cfg = load_config(config)
    declared = _datasets_with_validation(cfg)
    if not declared:
        raise die(
            "no validation plugins configured for any dataset",
            hint=(
                "add at least one entry under `datasets.<name>.validations.plugins` "
                "in your config, then run `ryotenkai plugin ls --kind validation` "
                "to discover available validators."
            ),
            code=2,
        )

    from src.pipeline.orchestrator import PipelineOrchestrator  # heavy: lazy

    orchestrator = PipelineOrchestrator(config)
    stage = orchestrator.stages[0]
    result = stage.run(orchestrator.context)

    if result.is_success():
        if state.is_machine_readable:
            renderer.emit({"ok": True, "validated_datasets": declared})
        else:
            renderer.text(f"Dataset validation passed ({len(declared)} datasets)")
        renderer.flush()
        return

    err = result.unwrap_err()
    if state.is_machine_readable:
        renderer.emit({"ok": False, "error": str(err)})
    else:
        renderer.text(f"Validation failed: {err}")
    renderer.flush()
    raise typer.Exit(code=1)


def _datasets_with_validation(cfg) -> list[str]:  # type: ignore[no-untyped-def]
    """Names of datasets that declare at least one validation plugin."""
    declared: list[str] = []
    for name, ds in cfg.datasets.items():
        validations = getattr(ds, "validations", None)
        plugins = getattr(validations, "plugins", None) if validations else None
        if plugins:
            declared.append(name)
    return declared


__all__ = ["dataset_app"]
