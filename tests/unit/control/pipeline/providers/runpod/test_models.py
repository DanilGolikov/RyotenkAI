"""Tests for PodSnapshot / SshEndpoint domain models."""

from __future__ import annotations

import pytest

from ryotenkai_providers.runpod.models import PodSnapshot, SshEndpoint, read_ssh_public_key

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


class TestPodSnapshotPortMappingsFallback:
    """When ``runtime.ports`` is missing/empty the parser falls back to
    ``portMappings`` + ``publicIp``. Pins the four observed sub-shapes."""

    def test_dict_string_key_int_value(self) -> None:
        snap = PodSnapshot.from_graphql(
            {
                "id": "pod-fb-1",
                "desiredStatus": "RUNNING",
                "publicIp": "1.2.3.4",
                "portMappings": {"22": 23828, "8888": 41000},
            }
        )
        assert snap.ssh_endpoint == SshEndpoint(host="1.2.3.4", port=23828)
        assert snap.port_count == 2
        assert snap.is_ready is True

    def test_dict_protocol_key_string_value(self) -> None:
        snap = PodSnapshot.from_graphql(
            {
                "id": "pod-fb-2",
                "desiredStatus": "RUNNING",
                "publicIp": "1.2.3.4",
                "portMappings": {"22/tcp": "23828"},
            }
        )
        assert snap.ssh_endpoint == SshEndpoint(host="1.2.3.4", port=23828)

    def test_dict_nested_hostport_object(self) -> None:
        snap = PodSnapshot.from_graphql(
            {
                "id": "pod-fb-3",
                "desiredStatus": "RUNNING",
                "publicIp": "1.2.3.4",
                "portMappings": {"22": {"hostPort": 23828}},
            }
        )
        assert snap.ssh_endpoint == SshEndpoint(host="1.2.3.4", port=23828)

    def test_list_with_explicit_keys(self) -> None:
        snap = PodSnapshot.from_graphql(
            {
                "id": "pod-fb-4",
                "desiredStatus": "RUNNING",
                "publicIp": "1.2.3.4",
                "portMappings": [
                    {"containerPort": 22, "hostPort": 23828},
                    {"containerPort": 8888, "hostPort": 41000},
                ],
            }
        )
        assert snap.ssh_endpoint == SshEndpoint(host="1.2.3.4", port=23828)
        assert snap.port_count == 2

    def test_dict_int_key(self) -> None:
        snap = PodSnapshot.from_graphql(
            {
                "id": "pod-fb-5",
                "desiredStatus": "RUNNING",
                "publicIp": "1.2.3.4",
                "portMappings": {22: 23828},
            }
        )
        assert snap.ssh_endpoint == SshEndpoint(host="1.2.3.4", port=23828)

    def test_no_publicip_falls_through_even_with_mapping(self) -> None:
        snap = PodSnapshot.from_graphql(
            {
                "id": "pod-fb-6",
                "desiredStatus": "RUNNING",
                "portMappings": {"22": 23828},
            }
        )
        # No publicIp → can't construct an endpoint, but port_count still
        # reports the underlying collection size (caller may surface this).
        assert snap.ssh_endpoint is None
        assert snap.port_count == 1

    def test_runtime_ports_takes_precedence_over_mappings(self) -> None:
        """When both shapes are present (as some SDK versions emit),
        the explicit ``runtime.ports`` list wins over the flat fallback."""
        snap = PodSnapshot.from_graphql(
            {
                "id": "pod-fb-7",
                "desiredStatus": "RUNNING",
                "runtime": {
                    "uptimeInSeconds": 5,
                    "ports": [
                        {"ip": "5.5.5.5", "privatePort": 22, "publicPort": 99999, "isIpPublic": True},
                    ],
                },
                # Different/conflicting fallback values — must be ignored
                "publicIp": "1.2.3.4",
                "portMappings": {"22": 23828},
            }
        )
        assert snap.ssh_endpoint == SshEndpoint(host="5.5.5.5", port=99999)

    def test_unknown_mapping_shape_is_ignored(self) -> None:
        snap = PodSnapshot.from_graphql(
            {
                "id": "pod-fb-8",
                "desiredStatus": "RUNNING",
                "publicIp": "1.2.3.4",
                "portMappings": "this is not a port mapping",
            }
        )
        assert snap.ssh_endpoint is None
        assert snap.port_count == 0

    def test_negative_or_zero_port_rejected(self) -> None:
        snap = PodSnapshot.from_graphql(
            {
                "id": "pod-fb-9",
                "desiredStatus": "RUNNING",
                "publicIp": "1.2.3.4",
                "portMappings": {"22": 0},
            }
        )
        assert snap.ssh_endpoint is None

    def test_empty_publicip_treated_as_missing(self) -> None:
        snap = PodSnapshot.from_graphql(
            {
                "id": "pod-fb-10",
                "desiredStatus": "RUNNING",
                "publicIp": "   ",
                "portMappings": {"22": 23828},
            }
        )
        assert snap.ssh_endpoint is None


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
