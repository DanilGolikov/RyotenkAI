"""GPUDeployer — on-failure SCP pull of pod-side logs.

Phase 3 PR-3.3 (transport-unification-v2): the LogManager-based
download path is gone. The deployer now uses
``self._ssh_client.download_file(remote, local)`` directly for the
on-failure log channel — same intent (best-effort recovery of
runner.log + trainer.stdio.log when deployment fails before
TrainingMonitor takes over) but with a much smaller surface.

Categories:
* Positive — both channels are attempted (SCP, not LogManager)
* Negative — first-channel failure does not skip the second
* Boundary — no SSH client → both skipped silently
* Invariant — runner.log fetched first
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from ryotenkai_control.pipeline.stages.gpu_deployer import GPUDeployer
from ryotenkai_shared.errors import SSHTransferFailedError
from ryotenkai_shared.utils.logs_layout import LogLayout


@pytest.fixture
def deployer_with_ssh(tmp_path, monkeypatch):
    """A GPUDeployer instance with a mocked SSH client + injected layout."""
    layout = LogLayout(tmp_path)
    layout.ensure_logs_dir()
    monkeypatch.setattr(
        "ryotenkai_control.pipeline.stages.gpu_deployer.get_run_log_layout",
        lambda: layout,
    )

    config = MagicMock()
    secrets = MagicMock()

    with patch("ryotenkai_control.pipeline.stages.gpu_deployer.TrainingDeploymentManager") as tdm_cls:
        tdm = MagicMock()
        tdm.workspace = "/workspace"
        tdm_cls.return_value = tdm
        deployer = GPUDeployer(
            config=config,
            secrets=secrets,
            callbacks=MagicMock(),
        )

    deployer._ssh_client = MagicMock()
    # New SSH contract: download_file returns None on success, raises on failure.
    deployer._ssh_client.download_file.return_value = None
    deployer.deployment.workspace = "/workspace"
    return deployer


# ---------------------------------------------------------------------------
# Positive — both channels SCP'd via download_file
# ---------------------------------------------------------------------------


def test_both_channels_scp_via_download_file(deployer_with_ssh) -> None:
    deployer_with_ssh._download_remote_logs(reason="test")
    calls = deployer_with_ssh._ssh_client.download_file.call_args_list
    assert len(calls) == 2
    remote_paths = [call.kwargs["remote_path"] for call in calls]
    # PodLayout writes both files under <workspace>/logs/.
    assert any(p.endswith("runner.log") for p in remote_paths)
    assert any(p.endswith("trainer.stdio.log") for p in remote_paths)


# ---------------------------------------------------------------------------
# Negative — first failure does not skip the second
# ---------------------------------------------------------------------------


def test_runner_log_failure_does_not_skip_trainer_log(deployer_with_ssh) -> None:
    deployer_with_ssh._ssh_client.download_file.side_effect = [
        SSHTransferFailedError(detail="boom", context={"op": "download_file"}),  # runner.log fails
        None,       # trainer.stdio.log succeeds
    ]
    deployer_with_ssh._download_remote_logs(reason="test")
    assert deployer_with_ssh._ssh_client.download_file.call_count == 2


def test_runner_log_exception_does_not_skip_trainer_log(deployer_with_ssh) -> None:
    deployer_with_ssh._ssh_client.download_file.side_effect = [
        RuntimeError("ssh dead"),
        None,
    ]
    deployer_with_ssh._download_remote_logs(reason="test")
    # Both calls were attempted despite the first raising.
    assert deployer_with_ssh._ssh_client.download_file.call_count == 2


# ---------------------------------------------------------------------------
# Boundary — no SSH client
# ---------------------------------------------------------------------------


def test_no_ssh_client_skips_both_downloads_silently(deployer_with_ssh) -> None:
    deployer_with_ssh._ssh_client = None
    deployer_with_ssh._download_remote_logs("test")  # must not raise


# ---------------------------------------------------------------------------
# Invariant — runner.log fetched first
# ---------------------------------------------------------------------------


def test_runner_log_attempted_first(deployer_with_ssh) -> None:
    deployer_with_ssh._download_remote_logs(reason="test")
    calls = deployer_with_ssh._ssh_client.download_file.call_args_list
    assert calls
    assert calls[0].kwargs["remote_path"].endswith("runner.log")
