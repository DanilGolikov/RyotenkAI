"""RyotenkAI TUI — single App entry point."""

from __future__ import annotations

import contextlib
from datetime import UTC, datetime
from typing import TYPE_CHECKING

from textual import work
from textual.app import App, ComposeResult
from textual.widgets import Label
from textual.worker import Worker, WorkerState

from src.tui.adapters.app_state import (
    predict_attempt_for_launch,
    resolve_attempt_to_open,
    resolve_interrupt_pid,
    running_attempt_for_run,
)
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
_DEFAULT_NOTIFICATION_TIMEOUT = 3.0
_NOTIFICATION_TIMEOUT_MULTIPLIER = 3.0

class RyotenkaiApp(App):
    """Interactive TUI for pipeline run inspection.

    Opens the runs-list browser by default.
    If `initial_run_dir` is provided, opens a useful screen for that run directly.
    """

    TITLE = "ryotenkai"
    SUB_TITLE = "pipeline run inspector"
    NOTIFICATION_TIMEOUT = _DEFAULT_NOTIFICATION_TIMEOUT * _NOTIFICATION_TIMEOUT_MULTIPLIER

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

    def _resolve_interrupt_pid(self, run_dir: Path) -> int | None:
        launch = self.get_active_launch_for_run(run_dir)
        launch_pid = None
        if launch is not None and launch.status in {"launching", "running", "stopping"}:
            launch_pid = launch.pid
        return resolve_interrupt_pid(run_dir, launch_pid)

    def can_interrupt_run(self, run_dir: Path) -> bool:
        return self._resolve_interrupt_pid(run_dir) is not None

    def _predict_launched_attempt_no(self, run_dir: Path) -> int:
        return predict_attempt_for_launch(run_dir)

    def start_launch(self, request: LaunchRequest) -> bool:
        normalized = request.validate()
        current = self.get_active_launch_for_run(normalized.run_dir)
        if current is not None:
            self.notify(f"This run is already launched in the TUI: {normalized.run_dir.name}", severity="warning")
            return False

        command = tuple(build_train_command(normalized))
        expected_attempt_no = self._predict_launched_attempt_no(normalized.run_dir)
        self._active_launches[normalized.run_dir] = ActiveLaunch(request=normalized, command=command)
        self._launch_workers[normalized.run_dir] = self._run_launch_worker(normalized)
        self.notify(f"Launch started: {normalized.run_dir.name}", severity="information")
        self.open_attempt_for_run(normalized.run_dir, expected_attempt_no)
        return True

    def open_attempt_for_run(self, run_dir: Path, attempt_no: int | None = None) -> None:
        from src.tui.screens.attempt_detail import AttemptDetailScreen

        resolved_run_dir = run_dir.expanduser().resolve()
        resolved_attempt_no = resolve_attempt_to_open(resolved_run_dir, attempt_no)
        self.push_screen(AttemptDetailScreen(resolved_run_dir, resolved_attempt_no))

    def _get_running_attempt_no(self, run_dir: Path) -> int | None:
        return running_attempt_for_run(run_dir)

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
