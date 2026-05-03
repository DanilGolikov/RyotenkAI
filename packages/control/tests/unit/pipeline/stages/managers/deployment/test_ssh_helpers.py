"""Unit tests for src.pipeline.stages.managers.deployment.ssh_helpers."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from src.pipeline.stages.managers.deployment.ssh_helpers import build_ssh_opts

pytestmark = pytest.mark.unit


def test_build_ssh_opts_alias_mode():
    ssh_client = MagicMock()
    ssh_client.ssh_base_opts = None
    ssh_client._is_alias_mode = True

    assert build_ssh_opts(ssh_client) == "-o StrictHostKeyChecking=no"


def test_build_ssh_opts_explicit_mode():
    ssh_client = MagicMock()
    ssh_client.ssh_base_opts = None
    ssh_client._is_alias_mode = False
    ssh_client.key_path = "/tmp/test_key"
    ssh_client.port = 2222

    assert build_ssh_opts(ssh_client) == "-i /tmp/test_key -p 2222 -o StrictHostKeyChecking=no"


def test_build_ssh_opts_reuses_base_opts_from_ssh_client():
    ssh_client = MagicMock()
    ssh_client.ssh_base_opts = ["-o", "StrictHostKeyChecking=no", "-o", "ControlMaster=auto"]
    ssh_client.key_path = "/tmp/key"
    ssh_client.port = 3333

    result = build_ssh_opts(ssh_client)
    assert "-i /tmp/key" in result
    assert "-p 3333" in result
    assert "ControlMaster=auto" in result
