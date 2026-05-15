"""``ryotenkai project <verb>`` — workspace project lifecycle.

Surfaces the workspace project registry (``src.workspace.projects``) +
project service (``src.api.services.project_service``) as a CLI noun.

The ``use`` verb persists a "current project" pointer in
``~/.ryotenkai/cli-context.json`` (see :mod:`src.cli_state.context_store`).
Subsequent commands honouring ``--project`` / ``RYOTENKAI_PROJECT``
fall through to that pointer when nothing is supplied.
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer

from ryotenkai_control.cli.common_options import DryRunOpt, YesOpt
from ryotenkai_control.cli.context import CLIContext
from ryotenkai_control.cli.errors import die, wrap_command
from ryotenkai_control.cli.renderer import get_renderer

project_app = typer.Typer(
    no_args_is_help=True,
    help="Manage workspace projects (create / list / use / run / env / rm).",
    rich_markup_mode=None,
    context_settings={"help_option_names": ["-h", "--help"]},
    pretty_exceptions_enable=False,
)


def _registry():  # type: ignore[no-untyped-def]
    """Return a fresh :class:`ProjectRegistry` instance.

    Lazy-imported so ``ryotenkai project --help`` doesn't pay for the
    workspace import chain.
    """
    from ryotenkai_control.workspace.projects import ProjectRegistry

    return ProjectRegistry()


# ---------------------------------------------------------------------------
# ls / show
# ---------------------------------------------------------------------------


@project_app.command("ls")
@wrap_command
def ls_cmd(ctx: typer.Context) -> None:
    """List registered projects."""
    from ryotenkai_control.api.services import project_service

    state = ctx.ensure_object(CLIContext)
    renderer = get_renderer(state)
    summaries = project_service.list_summaries(_registry())

    if state.is_machine_readable:
        renderer.emit([s.model_dump() for s in summaries])
    elif not summaries:
        renderer.text("No projects registered.")
    else:
        renderer.table(
            headers=["ID", "Name", "Path", "Created"],
            rows=[(s.id, s.name, s.path, s.created_at) for s in summaries],
        )
    renderer.flush()


@project_app.command("show")
@wrap_command
def show_cmd(
    ctx: typer.Context,
    project_id: Annotated[str, typer.Argument(help="Project id.")],
) -> None:
    """Show project details (config + metadata)."""
    from ryotenkai_control.api.services import project_service

    state = ctx.ensure_object(CLIContext)
    renderer = get_renderer(state)
    try:
        detail = project_service.get_detail(_registry(), project_id)
    except project_service.ProjectServiceError as exc:
        raise die(str(exc))

    if state.is_machine_readable:
        renderer.emit(detail.model_dump())
    else:
        renderer.kv(
            {
                "ID": detail.id,
                "Name": detail.name,
                "Path": detail.path,
                "Description": detail.description or "-",
                "Created": detail.created_at,
                "Updated": detail.updated_at,
            },
            title=f"Project {detail.id}",
        )
    renderer.flush()


# ---------------------------------------------------------------------------
# use / current
# ---------------------------------------------------------------------------


@project_app.command("use")
@wrap_command
def use_cmd(
    project_id: Annotated[str, typer.Argument(help="Project id to make active.")],
    dry_run: DryRunOpt = False,
) -> None:
    """Persist ``project_id`` as the active project for follow-up commands."""
    from ryotenkai_control.api.services import project_service
    from ryotenkai_control.cli_state import context_store

    # Verify the project exists before persisting — avoids creating a
    # ghost context that resolves to nothing.
    try:
        project_service.get_detail(_registry(), project_id)
    except project_service.ProjectServiceError as exc:
        raise die(str(exc), hint="run `ryotenkai project ls` to see registered ids")

    if dry_run:
        typer.echo(f"dry-run: would set current project to {project_id!r}")
        return

    context_store.set_current_project(project_id)
    typer.echo(f"current project: {project_id}")


@project_app.command("current")
@wrap_command
def current_cmd(ctx: typer.Context) -> None:
    """Print the persisted current project (if any)."""
    from ryotenkai_control.cli_state import context_store

    state = ctx.ensure_object(CLIContext)
    renderer = get_renderer(state)
    persisted = context_store.load_context()
    if state.is_machine_readable:
        renderer.emit(
            {
                "current_project_id": persisted.current_project_id,
                "set_at": persisted.set_at,
            }
        )
    else:
        if persisted.current_project_id:
            renderer.text(persisted.current_project_id)
        else:
            renderer.text("(no project selected — run `ryotenkai project use <id>`)")
    renderer.flush()


# ---------------------------------------------------------------------------
# create / rm
# ---------------------------------------------------------------------------


@project_app.command("create")
@wrap_command
def create_cmd(
    name: Annotated[str, typer.Argument(help="Human-readable project name.")],
    project_id: Annotated[
        str | None,
        typer.Option(
            "--id",
            help="Slug used on disk (default: derived from name).",
        ),
    ] = None,
    path: Annotated[
        Path | None,
        typer.Option(
            "--path",
            help="Project directory path (default: ~/.ryotenkai/<id>).",
            file_okay=False,
            dir_okay=True,
            resolve_path=True,
        ),
    ] = None,
    description: Annotated[
        str,
        typer.Option("--description", help="Optional one-line description."),
    ] = "",
    dry_run: DryRunOpt = False,
) -> None:
    """Create a new project (registers + writes metadata file)."""
    from ryotenkai_control.api.services import project_service

    if dry_run:
        typer.echo(f"dry-run: would create project {project_id or '<derived from name>'} " f"at {path or '<default>'}")
        return
    try:
        summary = project_service.create_project(
            _registry(),
            name=name,
            project_id=project_id,
            path=str(path) if path else None,
            description=description,
        )
    except project_service.ProjectServiceError as exc:
        raise die(str(exc))
    typer.echo(f"created: {summary.id} → {summary.path}")


@project_app.command("rm")
@wrap_command
def rm_cmd(
    project_id: Annotated[str, typer.Argument(help="Project id.")],
    yes: YesOpt = False,
    dry_run: DryRunOpt = False,
) -> None:
    """Unregister a project (does not delete the project directory)."""
    from ryotenkai_control.api.services import project_service

    if (
        not yes
        and not dry_run
        and not typer.confirm(
            f"Unregister project {project_id!r}?",
        )
    ):
        raise die("aborted by user", code=2)

    if dry_run:
        typer.echo(f"dry-run: would unregister project {project_id!r}")
        return

    try:
        project_service.unregister(_registry(), project_id)
    except project_service.ProjectServiceError as exc:
        raise die(str(exc))
    typer.echo(f"unregistered: {project_id}")


# ---------------------------------------------------------------------------
# env
# ---------------------------------------------------------------------------


@project_app.command("env")
@wrap_command
def env_cmd(
    ctx: typer.Context,
    project_id: Annotated[str, typer.Argument(help="Project id.")],
) -> None:
    """Print the project's persisted env vars (no secret unmasking)."""
    from ryotenkai_control.api.services import project_service

    state = ctx.ensure_object(CLIContext)
    renderer = get_renderer(state)
    try:
        env = project_service.read_env(_registry(), project_id)
    except project_service.ProjectServiceError as exc:
        raise die(str(exc))
    if state.is_machine_readable:
        renderer.emit(env)
    else:
        for key, value in sorted(env.items()):
            renderer.text(f"{key}={value}")
    renderer.flush()


# ---------------------------------------------------------------------------
# run — start a pipeline using the project's current config + env
# ---------------------------------------------------------------------------


@project_app.command("run")
@wrap_command
def run_cmd(
    project_id: Annotated[
        str | None,
        typer.Argument(
            help="Project id (default: persisted via `project use`).",
        ),
    ] = None,
    dry_run: DryRunOpt = False,
) -> None:
    """Launch the project's current config as a fresh pipeline run.

    Thin convenience wrapper over ``ryotenkai run start --project <id>``
    — both paths funnel through
    :func:`src.workspace.projects.adapter.resolve_project_launch_inputs`,
    so env merge and metadata stamping are identical in both surfaces.
    """
    from ryotenkai_control.cli_state import context_store

    resolved_id = project_id or context_store.get_current_project()
    if not resolved_id:
        raise die(
            "no project specified",
            hint="pass <project_id> or run `ryotenkai project use <id>` first",
        )

    # Delegate to the same code path as ``run start --project`` — the
    # adapter resolves the project's filesystem, integration refs, and
    # env mapping in one place. Keeps both CLI surfaces in lock-step.
    from ryotenkai_control.cli.commands.run import _spawn_worker

    _spawn_worker(
        mode="start",
        config=None,  # adapter pulls configs/current.yaml
        run_dir=None,
        restart_from_stage=None,
        project_id=resolved_id,
        dry_run=dry_run,
    )


__all__ = ["project_app"]
