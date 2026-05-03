"""``ryotenkai job <verb>`` — talk to the in-pod runner from the CLI.

The launch path (``ryotenkai run start``) opens an SSH tunnel and
submits a job to the in-pod runner; everything afterwards happens
in the trainer's process and lands as events on the runner's
WebSocket. ``ryotenkai job ...`` lets a fresh shell rebuild the
SSH tunnel from the persisted ``attempts/<n>/job_submission.json``
and read live state straight from the runner — no marker-file
detour, no pipeline_state.json lag.

Sub-commands:

- ``status <run-dir>``   GET /api/v1/jobs/{id} — current FSM snapshot
- ``stop <run-dir>``     POST /stop — graceful shutdown request
- ``events <run-dir> [--follow] [--since N]``
                         WebSocket subscribe — replay + live tail of
                         structured events
- ``logs <run-dir> [--follow] [--tail N] [--stream stdout|stderr]``
                         Filtered tail of trainer stdout / stderr —
                         Phase 7.x file-tail fallback for when
                         ``RunnerEventCallback`` self-disabled
- ``metrics <run-dir>``  Latest GPU/RAM ``health_snapshot`` event

Every command is a thin wrapper around :class:`SSHTunnelManager`
+ :class:`JobClient` — same primitives the pipeline launcher
uses, no duplicate connection plumbing.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Annotated, Any

import typer

from ryotenkai_control.cli.context import CLIContext
from ryotenkai_control.cli.errors import die
from ryotenkai_control.cli.renderer import get_renderer

job_app = typer.Typer(
    no_args_is_help=True,
    help="Inspect, monitor, and control jobs on the in-pod runner.",
    rich_markup_mode=None,
    context_settings={"help_option_names": ["-h", "--help"]},
    pretty_exceptions_enable=False,
)


# ---------------------------------------------------------------------------
# Submission lookup helpers
# ---------------------------------------------------------------------------


def _resolve_attempt_dir(run_dir: Path, attempt: int | None) -> Path:
    """Pick which ``attempts/<n>/`` directory to read from.

    Defaults to the highest-numbered attempt — the one most likely
    to be running. Explicit ``--attempt N`` overrides; missing
    attempt or empty ``runs/`` is a hard error so the user doesn't
    silently consult the wrong job's state.
    """
    runs = sorted(
        (run_dir / "attempts").glob("attempt_*"),
        key=lambda p: int(p.name.split("_")[-1]) if p.name.split("_")[-1].isdigit() else 0,
    )
    if not runs:
        raise die(
            f"no attempts/ subdirectories under {run_dir}",
            hint="did the pipeline get past the launcher? "
            "check pipeline_state.json",
        )
    if attempt is None:
        return runs[-1]
    target = run_dir / "attempts" / f"attempt_{attempt}"
    if not target.is_dir():
        raise die(
            f"attempt_{attempt} not found under {run_dir}",
            hint=f"available: {[p.name for p in runs]}",
        )
    return target


def _load_submission(attempt_dir: Path) -> Any:
    """Wrap :func:`load_job_submission` with a CLI-friendly error."""
    from ryotenkai_control.pipeline.state.job_submission import (
        JobSubmissionLoadError,
        load_job_submission,
    )

    try:
        return load_job_submission(attempt_dir)
    except JobSubmissionLoadError as exc:
        raise die(
            f"cannot read job submission for {attempt_dir.name}: {exc}",
            hint="this attempt may pre-date Phase 6.3a (no job_server "
            "metadata persisted) or never reached the launcher",
        )


# ---------------------------------------------------------------------------
# Tunnel + client lifecycle
# ---------------------------------------------------------------------------


async def _with_runner(submission, fn):  # type: ignore[no-untyped-def]
    """Open the SSH tunnel, build a :class:`JobClient`, run ``fn``,
    tear everything down — and don't leak a port if anything inside
    raises. ``fn`` receives ``(client, job_id)``.

    Kept as an async helper so callers can ``asyncio.run(_with_runner(...))``
    once and stay sync-shaped.
    """
    from ryotenkai_shared.utils.clients.job_client import JobClient
    from ryotenkai_shared.utils.clients.ssh_tunnel import (
        SSHTunnelEndpoint,
        SSHTunnelManager,
    )

    endpoint = SSHTunnelEndpoint(
        host=submission.ssh_host,
        port=submission.ssh_port,
        username=submission.ssh_username,
        key_path=submission.ssh_key_path,
    )
    tunnel = SSHTunnelManager(endpoint)
    await tunnel.open()
    try:
        client = JobClient(tunnel.base_url)
        try:
            return await fn(client, submission.job_id)
        finally:
            await client.aclose()
    finally:
        await tunnel.close()


# ---------------------------------------------------------------------------
# status
# ---------------------------------------------------------------------------


@job_app.command("status")
def status_cmd(
    ctx: typer.Context,
    run_dir: Annotated[
        Path,
        typer.Argument(
            help="Run directory (the one containing pipeline_state.json).",
            file_okay=False, dir_okay=True, exists=True,
        ),
    ],
    attempt: Annotated[
        int | None,
        typer.Option(
            "--attempt", "-a",
            help="Specific attempt number to query (default: latest).",
        ),
    ] = None,
) -> None:
    """Print the current FSM snapshot for a run's job."""
    state = ctx.ensure_object(CLIContext)
    renderer = get_renderer(state)

    submission = _load_submission(_resolve_attempt_dir(run_dir, attempt))

    async def _go(client, job_id):  # type: ignore[no-untyped-def]
        return await client.get_status(job_id)

    try:
        snapshot = asyncio.run(_with_runner(submission, _go))
    except Exception as exc:  # noqa: BLE001 — surface anything unexpected
        raise die(f"failed to query runner: {exc}")

    if state.is_machine_readable:
        renderer.emit({
            "submission": submission.to_dict(),
            "snapshot": snapshot,
        })
        renderer.flush()
        return

    renderer.heading(f"Job {submission.job_id}")
    renderer.kv({
        "Provider": submission.provider_name,
        "Pod": submission.pod_id or "—",
        "Pod SSH": f"{submission.ssh_username}@{submission.ssh_host}:{submission.ssh_port}",
        "Created": submission.created_at_iso,
    })
    renderer.text("")
    renderer.kv(
        {
            "State": snapshot.get("state", "unknown"),
            "Sequence": snapshot.get("sequence", "?"),
            "Started": snapshot.get("started_at", "?"),
            "Updated": snapshot.get("updated_at", "?"),
            "Last event offset": snapshot.get("last_event_offset", "?"),
            "Message": snapshot.get("message") or "—",
        },
        title="FSM",
    )
    renderer.flush()


# ---------------------------------------------------------------------------
# stop
# ---------------------------------------------------------------------------


@job_app.command("stop")
def stop_cmd(
    ctx: typer.Context,
    run_dir: Annotated[
        Path,
        typer.Argument(
            help="Run directory to stop.",
            file_okay=False, dir_okay=True, exists=True,
        ),
    ],
    attempt: Annotated[
        int | None,
        typer.Option("--attempt", "-a", help="Specific attempt to stop."),
    ] = None,
    grace: Annotated[
        float | None,
        typer.Option(
            "--grace",
            help="Override the runner's default SIGTERM grace window (seconds).",
        ),
    ] = None,
) -> None:
    """Request graceful stop of the trainer subprocess."""
    state = ctx.ensure_object(CLIContext)
    renderer = get_renderer(state)
    submission = _load_submission(_resolve_attempt_dir(run_dir, attempt))

    async def _go(client, job_id):  # type: ignore[no-untyped-def]
        return await client.request_stop(job_id, grace_seconds=grace)

    try:
        result = asyncio.run(_with_runner(submission, _go))
    except Exception as exc:  # noqa: BLE001
        raise die(f"failed to request stop: {exc}")

    if state.is_machine_readable:
        renderer.emit(result)
        renderer.flush()
        return
    renderer.heading(f"Stop requested for {submission.job_id}")
    renderer.kv({
        "Job ID": result.get("job_id", submission.job_id),
        "State": result.get("state", "?"),
        "Sequence": result.get("sequence", "?"),
    })
    renderer.flush()


# ---------------------------------------------------------------------------
# events
# ---------------------------------------------------------------------------


@job_app.command("events")
def events_cmd(
    ctx: typer.Context,
    run_dir: Annotated[
        Path,
        typer.Argument(
            help="Run directory.",
            file_okay=False, dir_okay=True, exists=True,
        ),
    ],
    attempt: Annotated[
        int | None,
        typer.Option("--attempt", "-a"),
    ] = None,
    since: Annotated[
        int,
        typer.Option(
            "--since",
            help="Replay events from this offset (default: 0 = oldest buffered).",
            min=0,
        ),
    ] = 0,
    follow: Annotated[
        bool,
        typer.Option(
            "--follow", "-f",
            help="Keep the WebSocket open for live events. "
                 "Default: drop after the buffered backlog is replayed.",
        ),
    ] = False,
    limit: Annotated[
        int | None,
        typer.Option(
            "--limit", "-n",
            help="Stop after this many events (only meaningful with --follow).",
        ),
    ] = None,
) -> None:
    """Subscribe to the runner's event stream and print each event.

    Without ``--follow`` the command exits as soon as the server-side
    replay catches up with the live cursor — useful for "what
    happened so far?" snapshots in scripts. With ``--follow`` it
    blocks until a terminal-state event arrives or the user
    interrupts.
    """
    state = ctx.ensure_object(CLIContext)
    renderer = get_renderer(state)
    submission = _load_submission(_resolve_attempt_dir(run_dir, attempt))

    machine = state.is_machine_readable

    async def _go(client, job_id):  # type: ignore[no-untyped-def]
        emitted: list[dict[str, Any]] = []
        async for event in client.subscribe_events(
            job_id,
            since=since,
            max_reconnect_attempts=0 if not follow else None,
        ):
            emitted.append(event)
            # Text mode prints one line per event in real-time so the
            # user gets a live tail. Machine mode buffers and emits
            # one combined payload at the end — JsonRenderer's
            # one-emit-per-command contract.
            if not machine:
                _format_event_line(event, renderer)
            if limit is not None and len(emitted) >= limit:
                break
        return emitted

    try:
        events = asyncio.run(_with_runner(submission, _go))
    except KeyboardInterrupt:
        renderer.text("\n[interrupted]")
        renderer.flush()
        return
    except Exception as exc:  # noqa: BLE001
        raise die(f"event subscription failed: {exc}")

    if machine:
        renderer.emit(events)
    elif not events:
        renderer.text("(no events)")
    renderer.flush()


def _format_event_line(event: dict[str, Any], renderer) -> None:  # type: ignore[no-untyped-def]
    """Collapse one event to a one-liner for text mode. Machine modes
    bypass this and emit the raw dict so downstream tooling parses
    the same shape the runner serialised."""
    offset = event.get("offset", "?")
    kind = event.get("kind", "?")
    payload = event.get("payload") or {}
    payload_summary = ", ".join(f"{k}={v}" for k, v in list(payload.items())[:4])
    if len(payload) > 4:
        payload_summary += ", ..."
    renderer.text(f"[{offset:>6}] {kind:<24} {payload_summary}")


def _format_log_line(event: dict[str, Any], renderer) -> None:  # type: ignore[no-untyped-def]
    """Render one ``trainer_log`` event as ``<stream> line``.

    Angle brackets rather than square brackets so the underlying
    Rich-backed renderer doesn't interpret ``[stdout]`` as markup
    and silently strip the tag. Matches how the operator would
    have seen the output if they had ``tail -f``'d ``training.log``.
    """
    payload = event.get("payload") or {}
    stream = payload.get("kind", "stdout")
    line = payload.get("line", "")
    renderer.text(f"<{stream}> {line}")


# ---------------------------------------------------------------------------
# logs
# ---------------------------------------------------------------------------


_VALID_LOG_STREAMS: frozenset[str] = frozenset({"stdout", "stderr"})


@job_app.command("logs")
def logs_cmd(
    ctx: typer.Context,
    run_dir: Annotated[
        Path,
        typer.Argument(
            help="Run directory.",
            file_okay=False, dir_okay=True, exists=True,
        ),
    ],
    attempt: Annotated[
        int | None,
        typer.Option("--attempt", "-a", help="Specific attempt to query."),
    ] = None,
    tail: Annotated[
        int,
        typer.Option(
            "--tail", "-n",
            help="Stop after this many lines have been emitted.",
            min=1, max=10_000,
        ),
    ] = 200,
    follow: Annotated[
        bool,
        typer.Option(
            "--follow", "-f",
            help="Keep streaming live until interrupted or ``--tail`` lines emitted.",
        ),
    ] = False,
    stream: Annotated[
        list[str] | None,
        typer.Option(
            "--stream", "-s",
            help="Filter to ``stdout``, ``stderr``, or both (default both).",
        ),
    ] = None,
) -> None:
    """Tail trainer stdout / stderr through the runner.

    The runner publishes every line of trainer output as a
    ``trainer_log`` event; this command filters and renders them.
    Use ``--stream stderr`` to focus on errors only, or ``--tail 50``
    for a quick "what's the trainer doing right now" snapshot.

    This is the file-tail fallback for cases where the structured
    ``RunnerEventCallback`` channel has self-disabled — the
    supervisor's stdout/stderr pump runs unconditionally so logs
    keep flowing.
    """
    state = ctx.ensure_object(CLIContext)
    renderer = get_renderer(state)
    submission = _load_submission(_resolve_attempt_dir(run_dir, attempt))

    requested_streams: set[str] = set(stream or _VALID_LOG_STREAMS)
    invalid = requested_streams - _VALID_LOG_STREAMS
    if invalid:
        raise die(
            f"unknown stream(s): {sorted(invalid)} "
            f"(must be a subset of {sorted(_VALID_LOG_STREAMS)})",
        )
    streams = requested_streams & _VALID_LOG_STREAMS
    machine = state.is_machine_readable

    async def _go(client, job_id):  # type: ignore[no-untyped-def]
        emitted: list[dict[str, Any]] = []
        async for event in client.subscribe_events(
            job_id,
            since=0,
            max_reconnect_attempts=0 if not follow else None,
        ):
            if event.get("kind") != "trainer_log":
                continue
            payload = event.get("payload") or {}
            if payload.get("kind") not in streams:
                continue
            emitted.append(event)
            if not machine:
                _format_log_line(event, renderer)
            if len(emitted) >= tail:
                break
        return emitted

    try:
        events = asyncio.run(_with_runner(submission, _go))
    except KeyboardInterrupt:
        renderer.text("\n[interrupted]")
        renderer.flush()
        return
    except Exception as exc:  # noqa: BLE001
        raise die(f"log stream failed: {exc}")

    if machine:
        renderer.emit(events)
    elif not events:
        renderer.text("(no logs)")
    renderer.flush()


# ---------------------------------------------------------------------------
# metrics
# ---------------------------------------------------------------------------


@job_app.command("metrics")
def metrics_cmd(
    ctx: typer.Context,
    run_dir: Annotated[
        Path,
        typer.Argument(
            help="Run directory.",
            file_okay=False, dir_okay=True, exists=True,
        ),
    ],
    attempt: Annotated[
        int | None,
        typer.Option("--attempt", "-a"),
    ] = None,
) -> None:
    """Print the most-recent GPU / RAM / CPU snapshot.

    Walks the runner's WebSocket replay and keeps only the latest
    ``health_snapshot`` event. If none is in the buffer the user
    sees an explicit "no snapshot yet" — the alternative would be
    blocking until one arrives, which a one-shot CLI shouldn't do.
    """
    state = ctx.ensure_object(CLIContext)
    renderer = get_renderer(state)
    submission = _load_submission(_resolve_attempt_dir(run_dir, attempt))

    async def _go(client, job_id):  # type: ignore[no-untyped-def]
        latest: dict[str, Any] | None = None
        try:
            async for event in client.subscribe_events(
                job_id, since=0, max_reconnect_attempts=0,
            ):
                if event.get("kind") == "health_snapshot":
                    latest = event
        except Exception:  # noqa: BLE001 — best-effort, we just want the latest
            pass
        return latest

    try:
        latest = asyncio.run(_with_runner(submission, _go))
    except Exception as exc:  # noqa: BLE001
        raise die(f"failed to fetch metrics: {exc}")

    if latest is None:
        if state.is_machine_readable:
            renderer.emit(None)
        else:
            renderer.text("no health_snapshot in buffer yet")
        renderer.flush()
        return

    if state.is_machine_readable:
        renderer.emit(latest)
        renderer.flush()
        return

    payload = latest.get("payload") or {}
    renderer.heading(f"Latest snapshot (offset {latest.get('offset')})")
    renderer.kv({
        "GPU util %": payload.get("gpu_util_percent", "—"),
        "GPU mem %": payload.get("gpu_memory_percent", "—"),
        "CPU %": payload.get("cpu_percent", "—"),
        "RAM used GB": payload.get("ram_used_gb", "—"),
        "RAM total GB": payload.get("ram_total_gb", "—"),
        "Timestamp": latest.get("timestamp", "—"),
    })
    renderer.flush()
