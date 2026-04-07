"""Tests for PodSnapshot / SshEndpoint domain models."""

from __future__ import annotations

import pytest

from src.providers.runpod.models import PodSnapshot, SshEndpoint, read_ssh_public_key

pytestmark = pytest.mark.unit


class TestSshEndpoint:
    def test_frozen(self) -> None:
        ep = SshEndpoint(host="1.2.3.4", port=2222)
        with pytest.raises(AttributeError):
            ep.host = "5.6.7.8"  # type: ignore[misc]


class TestPodSnapshotFromGraphql:
    def test_running_with_ssh(self) -> None:
        snap = PodSnapshot.from_graphql(
            {
                "id": "pod-1",
                "desiredStatus": "RUNNING",
                "runtime": {
                    "uptimeInSeconds": 42,
                    "ports": [
                        {"ip": "1.2.3.4", "privatePort": 22, "publicPort": 12345, "isIpPublic": True},
                    ],
                },
            }
        )
        assert snap.pod_id == "pod-1"
        assert snap.status == "RUNNING"
        assert snap.uptime_seconds == 42
        assert snap.port_count == 1
        assert snap.ssh_endpoint == SshEndpoint(host="1.2.3.4", port=12345)
        assert snap.is_ready is True
        assert snap.is_terminal is False

    def test_running_without_ssh(self) -> None:
        snap = PodSnapshot.from_graphql(
            {
                "id": "pod-2",
                "desiredStatus": "RUNNING",
                "runtime": {"uptimeInSeconds": 0, "ports": []},
            }
        )
        assert snap.ssh_endpoint is None
        assert snap.is_ready is False

    def test_failed_state(self) -> None:
        snap = PodSnapshot.from_graphql({"id": "pod-3", "desiredStatus": "FAILED", "runtime": {}})
        assert snap.is_terminal is True
        assert snap.is_ready is False

    def test_ignores_non_public_ip(self) -> None:
        snap = PodSnapshot.from_graphql(
            {
                "id": "pod-4",
                "desiredStatus": "RUNNING",
                "runtime": {
                    "uptimeInSeconds": 10,
                    "ports": [{"ip": "10.0.0.1", "privatePort": 22, "publicPort": 2222, "isIpPublic": False}],
                },
            }
        )
        assert snap.ssh_endpoint is None

    def test_ignores_non_ssh_ports(self) -> None:
        snap = PodSnapshot.from_graphql(
            {
                "id": "pod-5",
                "desiredStatus": "RUNNING",
                "runtime": {
                    "uptimeInSeconds": 5,
                    "ports": [{"ip": "1.2.3.4", "privatePort": 8888, "publicPort": 31000, "isIpPublic": True}],
                },
            }
        )
        assert snap.ssh_endpoint is None
        assert snap.port_count == 1

    def test_missing_runtime(self) -> None:
        snap = PodSnapshot.from_graphql({"id": "pod-6", "desiredStatus": "STARTING"})
        assert snap.uptime_seconds == 0
        assert snap.port_count == 0
        assert snap.ssh_endpoint is None

    def test_empty_ip_ignored(self) -> None:
        snap = PodSnapshot.from_graphql(
            {
                "id": "pod-7",
                "desiredStatus": "RUNNING",
                "runtime": {
                    "uptimeInSeconds": 10,
                    "ports": [{"ip": "", "privatePort": 22, "publicPort": 3333, "isIpPublic": True}],
                },
            }
        )
        assert snap.ssh_endpoint is None

    def test_non_integer_public_port_is_ignored(self) -> None:
        snap = PodSnapshot.from_graphql(
            {
                "id": "pod-8",
                "desiredStatus": "RUNNING",
                "runtime": {
                    "uptimeInSeconds": 10,
                    "ports": [{"ip": "1.2.3.4", "privatePort": 22, "publicPort": "2222", "isIpPublic": True}],
                },
            }
        )
        assert snap.ssh_endpoint is None
        assert snap.is_ready is False

    def test_missing_is_ip_public_is_treated_as_non_blocking(self) -> None:
        snap = PodSnapshot.from_graphql(
            {
                "id": "pod-9",
                "desiredStatus": "RUNNING",
                "runtime": {
                    "uptimeInSeconds": 10,
                    "ports": [{"ip": "1.2.3.4", "privatePort": 22, "publicPort": 2222}],
                },
            }
        )
        assert snap.ssh_endpoint == SshEndpoint(host="1.2.3.4", port=2222)
        assert snap.is_ready is True

    def test_runtime_ports_non_list_is_ignored(self) -> None:
        snap = PodSnapshot.from_graphql(
            {
                "id": "pod-10",
                "desiredStatus": "RUNNING",
                "runtime": {"uptimeInSeconds": 10, "ports": {"22/tcp": 2222}},
            }
        )
        assert snap.port_count == 0
        assert snap.ssh_endpoint is None
        assert snap.is_ready is False


class TestReadSshPublicKey:
    def test_returns_none_for_nonexistent(self, tmp_path) -> None:
        assert read_ssh_public_key(str(tmp_path / "id_ed25519")) is None

    def test_reads_pub_file(self, tmp_path) -> None:
        priv = tmp_path / "id_ed25519"
        pub = tmp_path / "id_ed25519.pub"
        priv.write_text("private")
        pub.write_text("ssh-ed25519 AAAA... user@host\n")
        assert read_ssh_public_key(str(priv)) == "ssh-ed25519 AAAA... user@host"

    def test_empty_pub_returns_none(self, tmp_path) -> None:
        priv = tmp_path / "id_ed25519"
        pub = tmp_path / "id_ed25519.pub"
        priv.write_text("private")
        pub.write_text("  \n")
        assert read_ssh_public_key(str(priv)) is None
