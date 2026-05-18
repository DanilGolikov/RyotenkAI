"""``ryotenkai model <verb>`` — Model Registry promotion gate.

Currently a single verb: ``promote``. Moves an existing alias on a
registered MLflow model to a target version. The default flow is the
``challenger -> champion`` promotion gate described in Phase M5:

* The pod-trainer auto-assigns ``challenger`` on a successful publish.
* ``ryotenkai model promote --name <n> --version <v>`` (default
  ``--alias champion``) attaches the second pointer to the same
  version after operator review.

Aliases are movable: re-running the command against a different
version transparently reassigns the pointer. There is no atomic
"swap" -- MLflow's set_registered_model_alias is the only primitive.
"""

from __future__ import annotations

from typing import Annotated

import typer

from ryotenkai_control.cli.common_options import ConfigOpt
from ryotenkai_control.cli.context import CLIContext
from ryotenkai_control.cli.errors import die, wrap_command
from ryotenkai_control.cli.renderer import get_renderer

model_app = typer.Typer(
    no_args_is_help=True,
    help="Model Registry operations (alias promotion).",
    rich_markup_mode=None,
    context_settings={"help_option_names": ["-h", "--help"]},
    pretty_exceptions_enable=False,
)


def _resolve_tracking_uri(config_path) -> str:  # type: ignore[no-untyped-def]
    """Resolve the MLflow tracking URI for the promote command.

    Reads the pipeline YAML pointed to by ``--config`` and pulls
    ``integrations.mlflow.tracking_uri`` (or ``local_tracking_uri`` as
    a fallback). Raises a user-friendly ``die`` when the URI is
    missing -- promotion against an unknown server would silently
    register on the wrong target.

    :param config_path: Resolved :class:`pathlib.Path` to the
        pipeline config YAML, or ``None`` to fall back to env
        variables.
    :returns: The MLflow tracking URI string.
    """
    import os as _os

    env_uri = _os.environ.get("MLFLOW_TRACKING_URI")
    if env_uri and env_uri.strip():
        return env_uri.strip()

    if config_path is None:
        raise die(
            "no MLflow tracking URI available",
            hint=(
                "pass --config pointing at your pipeline YAML, or export "
                "MLFLOW_TRACKING_URI before running this command."
            ),
            code=2,
        )

    from ryotenkai_control.cli.errors import load_config_or_die

    cfg = load_config_or_die(config_path)
    mlflow_cfg = getattr(getattr(cfg, "integrations", None), "mlflow", None)
    if mlflow_cfg is None:
        raise die(
            "pipeline config does not declare integrations.mlflow",
            hint=(
                "add an ``integrations: { mlflow: { tracking_uri: ... } }`` "
                "block to your config, or set MLFLOW_TRACKING_URI."
            ),
            code=2,
        )
    uri = (
        getattr(mlflow_cfg, "tracking_uri", None)
        or getattr(mlflow_cfg, "local_tracking_uri", None)
    )
    if not uri:
        raise die(
            "integrations.mlflow has neither tracking_uri nor local_tracking_uri",
            code=2,
        )
    return uri


def _resolve_model_version_from_run_id(
    tracking_uri: str,
    run_id: str,
) -> tuple[str, str]:
    """Look up the registered model name + version produced by ``run_id``.

    Searches ``mlflow.search_model_versions(filter="run_id = '<id>'")``
    and returns the first match. Raises if the run produced zero or
    multiple model versions (the caller must disambiguate by passing
    ``--name``/``--version`` explicitly).

    :param tracking_uri: Resolved MLflow tracking URI.
    :param run_id: MLflow run id (the value of ``MLFLOW_RUN_ID`` or
        ``state.root_mlflow_run_id`` / ``state.pipeline_attempt_mlflow_run_id``).
    :returns: ``(registered_name, version)`` tuple.
    :raises typer.Exit: when zero or >1 versions match.
    """
    from ryotenkai_shared.infrastructure.mlflow.registry import (
        MlflowModelRegistry,
    )

    registry = MlflowModelRegistry(tracking_uri=tracking_uri)
    client = registry.client
    versions = client.search_model_versions(filter_string=f"run_id = '{run_id}'")

    if not versions:
        raise die(
            f"no registered model version found for run_id={run_id!r}",
            hint=(
                "the run may not have called ModelPublisher.publish, "
                "or the run_id is wrong. Use --name + --version explicitly."
            ),
            code=2,
        )
    if len(versions) > 1:
        rendered = ", ".join(f"{v.name}@v{v.version}" for v in versions)
        raise die(
            f"run_id={run_id!r} produced multiple model versions: {rendered}",
            hint="pass --name and --version explicitly to disambiguate.",
            code=2,
        )

    return versions[0].name, str(versions[0].version)


@model_app.command("promote")
@wrap_command
def promote_cmd(
    ctx: typer.Context,
    name: Annotated[
        str | None,
        typer.Option(
            "--name",
            help=(
                "Registered model name (e.g. ryotenkai/exp/family). "
                "Mutually exclusive with --run-id."
            ),
        ),
    ] = None,
    version: Annotated[
        str | None,
        typer.Option(
            "--version",
            help=(
                "Model version. Required with --name; ignored when "
                "resolving from --run-id."
            ),
        ),
    ] = None,
    run_id: Annotated[
        str | None,
        typer.Option(
            "--run-id",
            help=(
                "MLflow run id (e.g. value of MLFLOW_RUN_ID or "
                "state.root_mlflow_run_id from pipeline_state.json). "
                "Resolves to the registered model version produced by "
                "that run."
            ),
        ),
    ] = None,
    alias: Annotated[
        str,
        typer.Option(
            "--alias",
            help="Target alias (default: champion).",
        ),
    ] = "champion",
    config: ConfigOpt = None,
) -> None:
    """Promote a model version to a target alias.

    Two input modes (mutually exclusive):

    1. ``--name X --version N`` — explicit, no lookup.
    2. ``--run-id <mlflow_run_id>`` — looks up the model version produced
       by that run via ``search_model_versions(filter="run_id = '...'"))``.

    Tip: get ``run_id`` from ``pipeline_state.json`` under the run
    directory (``root_mlflow_run_id`` or
    ``pipeline_attempt_mlflow_run_id``) or from the MLflow UI.

    Reads the MLflow tracking URI from the pipeline config (or the
    ``MLFLOW_TRACKING_URI`` env override) and points ``--alias`` at the
    resolved version.

    Exit codes:

    * 0 -- alias set successfully.
    * 1 -- MLflow rejected the request (unknown version, server error).
    * 2 -- user input incomplete or ambiguous (multiple/zero model
      versions for the run, missing tracking URI, unreadable config).
    """
    state = ctx.ensure_object(CLIContext)
    renderer = get_renderer(state)

    # Enforce mutual exclusivity of the two input modes.
    explicit_mode = name is not None or version is not None
    run_id_mode = run_id is not None
    if explicit_mode and run_id_mode:
        raise die(
            "--name/--version and --run-id are mutually exclusive",
            code=2,
        )
    if not explicit_mode and not run_id_mode:
        raise die(
            "must specify either --name + --version, or --run-id",
            code=2,
        )
    if explicit_mode and (name is None or version is None):
        raise die(
            "--name and --version must be provided together",
            code=2,
        )

    tracking_uri = _resolve_tracking_uri(config)

    # Resolve (name, version) from whichever mode the caller used.
    if run_id_mode:
        assert run_id is not None
        resolved_name, resolved_version = _resolve_model_version_from_run_id(
            tracking_uri,
            run_id,
        )
    else:
        assert name is not None and version is not None
        resolved_name, resolved_version = name, version

    from ryotenkai_shared.infrastructure.mlflow.registry import (
        MlflowModelRegistry,
    )

    registry = MlflowModelRegistry(tracking_uri=tracking_uri)
    try:
        registry.set_alias(resolved_name, alias, resolved_version)
    except Exception as exc:
        if state.is_machine_readable:
            renderer.emit({"ok": False, "error": str(exc)})
            renderer.flush()
            raise typer.Exit(code=1) from exc
        raise die(
            f"failed to set alias {alias!r} on {resolved_name}@v{resolved_version}: {exc}",
            code=1,
        ) from exc

    payload = {
        "ok": True,
        "name": resolved_name,
        "version": resolved_version,
        "alias": alias,
        "tracking_uri": tracking_uri,
    }
    if state.is_machine_readable:
        renderer.emit(payload)
    else:
        renderer.text(
            f"Promoted {resolved_name} v{resolved_version} -> @{alias} (tracking_uri={tracking_uri})",
        )
    renderer.flush()


__all__ = ["model_app"]
