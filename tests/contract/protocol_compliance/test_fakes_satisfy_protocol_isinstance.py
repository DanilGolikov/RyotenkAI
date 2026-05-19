"""Structural compliance: every canonical fake satisfies its Protocol via isinstance.

Fast structural check — independent of behavioral compliance — that
catches the case where a Protocol gains a new method (or a fake silently
drops one) before the heavyweight per-method tests do.
"""

from __future__ import annotations

import pytest

from ryotenkai_shared.infrastructure.docker import IDockerClient
from ryotenkai_shared.infrastructure.hf_hub import IHFHubClient
from ryotenkai_shared.infrastructure.job_client import IJobClient
from ryotenkai_shared.infrastructure.lifecycle import IPodLifecycleClient
from ryotenkai_shared.infrastructure.runpod_api import IRunPodAPI
from ryotenkai_shared.infrastructure.ssh import ISSHClient
from ryotenkai_shared.infrastructure.trainer_spawner import ITrainerSpawner
from tests._fakes.docker import FakeDockerClient
from tests._fakes.hf_hub import FakeHFHubClient
from tests._fakes.job_client import FakeJobClient
from tests._fakes.lifecycle import FakePodLifecycleClient
from tests._fakes.runpod import FakeRunPodAPI
from tests._fakes.ssh import FakeSSHClient
from tests._fakes.trainer import FakeTrainerSpawner
from tests._harness.clock import Clock, ManualClock, RealClock

pytestmark = [pytest.mark.contract, pytest.mark.compliance]


def test_fake_pod_lifecycle_client_satisfies_protocol() -> None:
    assert isinstance(FakePodLifecycleClient(), IPodLifecycleClient)


def test_fake_runpod_api_satisfies_protocol() -> None:
    assert isinstance(FakeRunPodAPI(), IRunPodAPI)


def test_fake_trainer_spawner_satisfies_protocol() -> None:
    assert isinstance(FakeTrainerSpawner(), ITrainerSpawner)


def test_fake_ssh_client_satisfies_protocol() -> None:
    assert isinstance(FakeSSHClient(), ISSHClient)


def test_fake_hf_hub_client_satisfies_protocol() -> None:
    assert isinstance(FakeHFHubClient(), IHFHubClient)


def test_fake_job_client_satisfies_protocol() -> None:
    assert isinstance(FakeJobClient(), IJobClient)


def test_fake_docker_client_satisfies_protocol() -> None:
    assert isinstance(FakeDockerClient(), IDockerClient)


def test_real_clock_satisfies_protocol() -> None:
    assert isinstance(RealClock(), Clock)


def test_manual_clock_satisfies_protocol() -> None:
    assert isinstance(ManualClock(), Clock)
