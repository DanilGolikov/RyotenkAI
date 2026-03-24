"""RyotenkAI TUI — single App entry point."""

from __future__ import annotations

import contextlib
import os
from datetime import UTC, datetime
from typing import TYPE_CHECKING

from textual import work
from textual.app import App, ComposeResult
from textual.widgets import Label
from textual.worker import Worker, WorkerState

from src.pipeline.state import PipelineState, StageRunState
from src.tui.launch import (
    ActiveLaunch,
    LaunchRequest,
    LaunchResult,
    build_train_command,
    execute_launch_subprocess,
    interrupt_launch_process,
)

if TYPE_CHECKING:
    from pathlib import Path

_DEFAULT_INTERVAL = 5.0
_ACTIVE_LAUNCH_STATUSES = {"launching", "running", "stopping"}


def _resolve_next_attempt_no(state: PipelineState) -> int:
    """Return the next attempt number that a new launch should create."""
    if not state.attempts:
        return 1
    return max(attempt.attempt_no for attempt in state.attempts) + 1


def _resolve_running_attempt_no(state: PipelineState) -> int | None:
    """Return the currently running attempt number, if any."""
    if state.active_attempt_id:
        for attempt in state.attempts:
            if attempt.attempt_id == state.active_attempt_id and attempt.status == StageRunState.STATUS_RUNNING:
                return attempt.attempt_no
    for attempt in reversed(state.attempts):
        if attempt.status == StageRunState.STATUS_RUNNING:
            return attempt.attempt_no
    return None


class RyotenkaiApp(App):
    """Interactive TUI for pipeline run inspection.

    Opens the runs-list browser by default.
    If `initial_run_dir` is provided, opens a useful screen for that run directly.
    """

    TITLE = "ryotenkai"
    SUB_TITLE = "pipeline run inspector"

    CSS = """
    #status-bar {
        height: 1;
        padding: 0 1;
        color: $text-muted;
    }
    """

    def __init__(
        self,
        runs_dir: Path,
        initial_run_dir: Path | None = None,
        interval: float = _DEFAULT_INTERVAL,
    ) -> None:
        super().__init__()
        self._runs_dir = runs_dir
        self._initial_run_dir = initial_run_dir
        self._interval = interval
        self._active_launches: dict[Path, ActiveLaunch] = {}
        self._launch_workers: dict[Path, Worker[LaunchResult]] = {}

    def compose(self) -> ComposeResult:
        yield Label("")

    def on_mount(self) -> None:
        if self._initial_run_dir is not None:
            if not self.open_running_attempt_for_run(self._initial_run_dir):
                from src.tui.screens.run_detail import RunDetailScreen

                self.push_screen(RunDetailScreen(self._initial_run_dir))
        else:
            from src.tui.screens.runs_list import RunsListScreen

            self.push_screen(RunsListScreen(self._runs_dir))

    @property
    def active_launch(self) -> ActiveLaunch | None:
        if not self._active_launches:
            return None
        return next(reversed(self._active_launches.values()))

    def get_active_launch_for_run(self, run_dir: Path) -> ActiveLaunch | None:
        launch = self._active_launches.get(run_dir.expanduser().resolve())
        if launch is None or launch.status not in _ACTIVE_LAUNCH_STATUSES:
            return None
        return launch

    def _read_run_lock_pid(self, run_dir: Path) -> int | None:
        lock_path = run_dir.expanduser().resolve() / "run.lock"
        try:
            raw = lock_path.read_text(encoding="utf-8").strip()
        except OSError:
            return None
        if not raw:
            return None
        first_line = raw.splitlines()[0].strip()
        candidate = first_line.split("=", maxsplit=1)[-1].strip()
        try:
            return int(candidate)
        except ValueError:
            return None

    def _resolve_interrupt_pid(self, run_dir: Path) -> int | None:
        launch = self.get_active_launch_for_run(run_dir)
        if launch is not None and launch.pid is not None and launch.status in {"launching", "running", "stopping"}:
            return launch.pid
        pid = self._read_run_lock_pid(run_dir)
        if pid is None:
            return None
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return None
        except OSError:
            return None
        return pid

    def can_interrupt_run(self, run_dir: Path) -> bool:
        return self._resolve_interrupt_pid(run_dir) is not None

    def _predict_launched_attempt_no(self, run_dir: Path) -> int:
        from src.pipeline.state import PipelineStateStore

        resolved_run_dir = run_dir.expanduser().resolve()
        try:
            state = PipelineStateStore(resolved_run_dir).load()
        except Exception:
            return 1
        return _resolve_next_attempt_no(state)

    def start_launch(self, request: LaunchRequest) -> bool:
        normalized = request.validate()
        current = self.get_active_launch_for_run(normalized.run_dir)
        if current is not None:
            self.notify(f"This run is already launched in the TUI: {normalized.run_dir.name}", severity="warning")
            return False
        if self._active_launches and normalized.mode != "new_run":
            self.notify("Parallel launch in the TUI is only allowed for new run for now", severity="warning")
            return False

        command = tuple(build_train_command(normalized))
        expected_attempt_no = self._predict_launched_attempt_no(normalized.run_dir)
        self._active_launches[normalized.run_dir] = ActiveLaunch(request=normalized, command=command)
        self._launch_workers[normalized.run_dir] = self._run_launch_worker(normalized)
        self.notify(f"Launch started: {normalized.run_dir.name}", severity="information")
        self.open_attempt_for_run(normalized.run_dir, expected_attempt_no)
        return True

    def open_attempt_for_run(self, run_dir: Path, attempt_no: int | None = None) -> None:
        from src.pipeline.state import PipelineStateStore
        from src.tui.screens.attempt_detail import AttemptDetailScreen

        resolved_run_dir = run_dir.expanduser().resolve()
        resolved_attempt_no = attempt_no
        if resolved_attempt_no is None:
            try:
                state = PipelineStateStore(resolved_run_dir).load()
                if state.attempts:
                    resolved_attempt_no = state.attempts[-1].attempt_no
            except Exception:
                resolved_attempt_no = None
        if resolved_attempt_no is None:
            resolved_attempt_no = 1
        self.push_screen(AttemptDetailScreen(resolved_run_dir, resolved_attempt_no))

    def _get_running_attempt_no(self, run_dir: Path) -> int | None:
        from src.pipeline.state import PipelineStateStore

        resolved_run_dir = run_dir.expanduser().resolve()
        try:
            state = PipelineStateStore(resolved_run_dir).load()
        except Exception:
            return None
        return _resolve_running_attempt_no(state)

    def can_open_running_attempt_for_run(self, run_dir: Path) -> bool:
        return self._get_running_attempt_no(run_dir) is not None

    def open_running_attempt_for_run(self, run_dir: Path) -> bool:
        resolved_run_dir = run_dir.expanduser().resolve()
        attempt_no = self._get_running_attempt_no(resolved_run_dir)
        if attempt_no is None:
            self.notify("This run has no running attempt right now", severity="warning")
            return False
        self.open_attempt_for_run(resolved_run_dir, attempt_no)
        return True

    def _mark_launch_started(self, run_dir: Path, pid: int | None) -> None:
        launch = self._active_launches.get(run_dir.expanduser().resolve())
        if launch is None:
            return
        launch.pid = pid
        launch.status = "running"

    def interrupt_active_launch_for_run(self, run_dir: Path) -> bool:
        launch = self.get_active_launch_for_run(run_dir)
        pid = self._resolve_interrupt_pid(run_dir)
        if pid is None:
            self.notify("No active process found to stop", severity="warning")
            return False
        if launch is not None and launch.status not in _ACTIVE_LAUNCH_STATUSES:
            self.notify("This launch is no longer active", severity="warning")
            return False
        if not interrupt_launch_process(pid):
            self.notify("Failed to send stop signal to the process", severity="error")
            return False
        if launch is not None:
            launch.status = "stopping"
        self.notify(f"Stop signal sent: {run_dir.expanduser().resolve().name}", severity="information")
        with contextlib.suppress(Exception):
            self.screen.refresh_bindings()
        return True

    def _finish_launch(self, result: LaunchResult) -> None:
        run_dir = result.request.run_dir.expanduser().resolve()
        launch = self._active_launches.get(run_dir)
        if launch is None:
            return
        if result.is_success:
            launch.status = "completed"
        elif result.return_code in {130, -2}:
            launch.status = "interrupted"
        else:
            launch.status = "failed"
        launch.pid = result.pid
        launch.return_code = result.return_code
        launch.finished_at = result.finished_at
        launch.output_tail = list(result.output_tail)
        if result.is_success:
            self.notify(f"Launch finished: {result.request.run_dir.name}", severity="information")
        elif result.return_code in {130, -2}:
            self.notify(f"Launch stopped: {result.request.run_dir.name}", severity="warning")
        else:
            tail = result.output_tail[-1] if result.output_tail else "unknown error"
            self.notify(f"Launch failed with error: {tail}", severity="error")
        self._launch_workers.pop(run_dir, None)
        self._active_launches.pop(run_dir, None)
        with contextlib.suppress(Exception):
            self.screen.refresh_bindings()

    @work(thread=True, exclusive=False)
    def _run_launch_worker(self, request: LaunchRequest) -> LaunchResult:
        return execute_launch_subprocess(
            request,
            on_started=lambda pid: self.call_from_thread(self._mark_launch_started, request.run_dir, pid),
        )

    def on_worker_state_changed(self, event: Worker.StateChanged) -> None:
        run_dir = next(
            (path for path, worker in self._launch_workers.items() if worker == event.worker),
            None,
        )
        if run_dir is None:
            return
        if event.state == WorkerState.SUCCESS:
            result = event.worker.result
            if result is None:
                return
            self._finish_launch(result)
        elif event.state == WorkerState.ERROR:
            launch = self._active_launches.get(run_dir)
            if launch is None:
                return
            self._finish_launch(
                LaunchResult(
                    request=launch.request,
                    command=tuple(launch.command),
                    return_code=1,
                    output_tail=(str(event.worker.error),),
                    pid=launch.pid,
                    started_at=launch.started_at,
                    finished_at=datetime.now(UTC).isoformat(),
                ),
            )
