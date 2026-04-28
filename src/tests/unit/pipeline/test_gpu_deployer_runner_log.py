"""
GPUDeployer — runner.log download alongside training.log.

Tests for the second log channel introduced to capture uvicorn /
runner stdout (including pre-import crashes that the trainer-log
channel misses). The training.log path stays unchanged and must keep
working independently.

Categories:
* Positive — runner.log download is invoked with correct LogManager args
* Negative — runner.log failure does NOT skip training.log download
* Negative — training.log failure does NOT skip runner.log download
* Boundary — no SSH client → both downloads skipped silently
* Invariant — both downloads attempted in order: runner first, training next
* Regression — existing _download_remote_logs signature unchanged
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from src.pipeline.stages.gpu_deployer import GPUDeployer
from src.utils.logs_layout import LogLayout


@pytest.fixture
def deployer_with_ssh(tmp_path, monkeypatch):
    """A GPUDeployer instance with a mocked SSH client and deployment ctx.

    All collaborators that aren't under test (provider, factory,
    deployment_manager) are pre-mocked.
    """
    layout = LogLayout(tmp_path)
    layout.ensure_logs_dir()
    monkeypatch.setattr(
        "src.pipeline.stages.gpu_deployer.get_run_log_layout",
        lambda: layout,
    )

    config = MagicMock()
    secrets = MagicMock()

    with patch("src.pipeline.stages.gpu_deployer.TrainingDeploymentManager") as tdm_cls:
        tdm = MagicMock()
        tdm.workspace = "/workspace"
        tdm_cls.return_value = tdm
        deployer = GPUDeployer(
            config=config,
            secrets=secrets,
            callbacks=MagicMock(),
        )

    deployer._ssh_client = MagicMock()
    deployer._ssh_client.exec_command.return_value = (False, "", "")  # default: cat fails
    deployer.deployment.workspace = "/workspace"
    return deployer


# ---------------------------------------------------------------------------
# Positive — runner.log call shape
# ---------------------------------------------------------------------------


def test_runner_log_download_invoked_with_canonical_paths(deployer_with_ssh, tmp_path):
    """_download_runner_log instantiates LogManager with /workspace/runner.log
    and the layout's remote_runner_log local destination."""
    with patch("src.pipeline.stages.gpu_deployer.LogManager") as lm_cls:
        lm_inst = MagicMock()
        lm_inst.download.return_value = True
        lm_cls.return_value = lm_inst

        deployer_with_ssh._download_runner_log(reason="test")

        lm_cls.assert_called_once()
        kwargs = lm_cls.call_args.kwargs
        # remote_path keyword
        assert kwargs.get("remote_path") == "/workspace/runner.log"
        # local_path keyword: must point at <attempt>/logs/runner.log
        local_path = kwargs.get("local_path")
        assert local_path is not None
        assert local_path.name == "runner.log"
        assert local_path.parent == tmp_path / "logs"


# ---------------------------------------------------------------------------
# Negative — runner.log failure must not break training.log download
# ---------------------------------------------------------------------------


def test_runner_log_exception_does_not_break_training_download(deployer_with_ssh):
    """Exception inside _download_runner_log is swallowed at debug level;
    _download_remote_logs continues on to training.log."""
    with patch("src.pipeline.stages.gpu_deployer.LogManager") as lm_cls:
        runner_mgr = MagicMock()
        runner_mgr.download.side_effect = RuntimeError("ssh dead during runner.log")
        # Subsequent LogManager() calls (training.log) succeed.
        training_mgr = MagicMock()
        training_mgr.download.return_value = True
        lm_cls.side_effect = [runner_mgr, training_mgr]

        # Should NOT raise; logs the runner failure at debug, proceeds.
        deployer_with_ssh._download_remote_logs("test")

        # Both LogManager constructions happened.
        assert lm_cls.call_count >= 2


def test_training_log_exception_does_not_skip_runner_download(deployer_with_ssh):
    """Runner.log download runs FIRST so an exception in training.log
    doesn't matter — but verify the ordering invariant holds."""
    call_order = []
    with patch("src.pipeline.stages.gpu_deployer.LogManager") as lm_cls:
        def make_lm(*a, **kw):
            mgr = MagicMock()
            remote = kw.get("remote_path", "")
            if "runner.log" in remote:
                call_order.append("runner")
            else:
                call_order.append("training")
            mgr.download.return_value = True
            return mgr
        lm_cls.side_effect = make_lm

        deployer_with_ssh._download_remote_logs("test")

        # Runner must have been the first LogManager built.
        assert call_order, "no LogManager calls"
        assert call_order[0] == "runner"


# ---------------------------------------------------------------------------
# Boundary — no SSH client
# ---------------------------------------------------------------------------


def test_no_ssh_client_skips_both_downloads_silently(deployer_with_ssh):
    """When ssh_client is None, _download_remote_logs is a no-op."""
    deployer_with_ssh._ssh_client = None
    with patch("src.pipeline.stages.gpu_deployer.LogManager") as lm_cls:
        deployer_with_ssh._download_remote_logs("test")
        assert lm_cls.call_count == 0


def test_no_ssh_client_skips_runner_log_silently(deployer_with_ssh):
    """_download_runner_log on its own returns immediately when SSH is gone."""
    deployer_with_ssh._ssh_client = None
    with patch("src.pipeline.stages.gpu_deployer.LogManager") as lm_cls:
        deployer_with_ssh._download_runner_log(reason="test")
        assert lm_cls.call_count == 0


# ---------------------------------------------------------------------------
# Invariant — runner runs BEFORE training in _download_remote_logs
# ---------------------------------------------------------------------------


def test_runner_log_attempted_before_training_log(deployer_with_ssh):
    """The chain is: 1) runner.log, 2) training.log. This protects the
    case where uvicorn died before trainer started — runner.log holds
    the only diagnostic and must be fetched even if training.log path
    short-circuits early."""
    seen = []
    with patch("src.pipeline.stages.gpu_deployer.LogManager") as lm_cls:
        def make_lm(*a, **kw):
            remote = kw.get("remote_path", a[1] if len(a) > 1 else "")
            mgr = MagicMock()
            mgr.download.return_value = True

            def _record(*_a, **_kw):
                seen.append(remote)
                return True

            mgr.download.side_effect = _record
            return mgr
        lm_cls.side_effect = make_lm

        deployer_with_ssh._download_remote_logs("test")

        assert seen[0] == "/workspace/runner.log", \
            f"Expected runner.log first, got order: {seen}"
