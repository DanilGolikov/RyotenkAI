"""Compliance tests for :class:`IHFHubClient`."""

from __future__ import annotations

import os
from typing import Any

import pytest

from ryotenkai_shared.infrastructure.hf_hub import (
    HFAuthError,
    HFNotFoundError,
    HFRateLimitedError,
    HFTransientError,
    IHFHubClient,
)
from tests._fakes.hf_hub import FakeHFHubClient

pytestmark = [
    pytest.mark.contract,
    pytest.mark.compliance,
    pytest.mark.exercises_protocol("IHFHubClient"),
    pytest.mark.uses_fake("FakeHFHubClient"),
    pytest.mark.asyncio,
]


@pytest.fixture(params=["fake", pytest.param("real", marks=pytest.mark.live)])
def hf_client(request: pytest.FixtureRequest, manual_clock: Any) -> IHFHubClient:
    if request.param == "real":
        if os.environ.get("RYOTENKAI_LIVE") != "1":
            pytest.skip("real IHFHubClient requires RYOTENKAI_LIVE=1")
        token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
        if not token:
            pytest.skip(
                "real IHFHubClient compliance requires HF_TOKEN (or "
                "HUGGING_FACE_HUB_TOKEN) pointing at a HuggingFace sandbox repo.",
            )
        from ryotenkai_shared.infrastructure.hf_hub.adapter import HFHubAdapter

        return HFHubAdapter(token=token, endpoint=os.environ.get("HF_ENDPOINT"))
    return FakeHFHubClient(clock=manual_clock)


def _as_fake(client: IHFHubClient) -> FakeHFHubClient:
    if not isinstance(client, FakeHFHubClient):
        pytest.skip(
            "test exercises FakeHFHubClient-only chaos helpers; "
            "real-mode covers Protocol shape only",
        )
    return client


class TestHFHubClientCompliance:
    async def test_isinstance_protocol(self, hf_client: IHFHubClient) -> None:
        assert isinstance(hf_client, IHFHubClient)

    async def test_create_then_info_round_trip(self, hf_client: IHFHubClient) -> None:
        created = await hf_client.create_repo("user/repo-A", private=True)
        assert created.repo_id == "user/repo-A"
        assert created.private is True
        info = await hf_client.repo_info("user/repo-A")
        assert info.repo_id == created.repo_id
        assert info.private is True

    async def test_create_existing_repo_returns_existing(
        self, hf_client: IHFHubClient,
    ) -> None:
        first = await hf_client.create_repo("user/repo-B", private=False)
        again = await hf_client.create_repo("user/repo-B", private=False)
        assert first.repo_id == again.repo_id

    async def test_repo_info_unknown_raises(self, hf_client: IHFHubClient) -> None:
        with pytest.raises(HFNotFoundError):
            await hf_client.repo_info("user/missing")

    async def test_upload_then_download_round_trips(
        self, hf_client: IHFHubClient,
    ) -> None:
        await hf_client.create_repo("user/repo-C")
        await hf_client.upload_file(
            repo_id="user/repo-C",
            path_in_repo="weights.safetensors",
            content=b"\x00\x01\x02hello",
            commit_message="initial upload",
        )
        out = await hf_client.download_file(
            repo_id="user/repo-C", path_in_repo="weights.safetensors",
        )
        assert out == b"\x00\x01\x02hello"

    async def test_download_unknown_raises(self, hf_client: IHFHubClient) -> None:
        await hf_client.create_repo("user/repo-D")
        with pytest.raises(HFNotFoundError):
            await hf_client.download_file(
                repo_id="user/repo-D", path_in_repo="never-uploaded",
            )

    async def test_upload_readme_populates_model_card(
        self, hf_client: IHFHubClient,
    ) -> None:
        await hf_client.create_repo("user/repo-E")
        await hf_client.upload_file(
            repo_id="user/repo-E", path_in_repo="README.md", content=b"# My Model\n",
        )
        card = await hf_client.get_model_card("user/repo-E")
        assert "# My Model" in card

    async def test_get_model_card_without_readme_raises(
        self, hf_client: IHFHubClient,
    ) -> None:
        await hf_client.create_repo("user/repo-F")
        with pytest.raises(HFNotFoundError):
            await hf_client.get_model_card("user/repo-F")

    # -- chaos surface --------------------------------------------------

    async def test_chaos_inject_rate_limit_recovers(
        self, hf_client: IHFHubClient,
    ) -> None:
        fake = _as_fake(hf_client)
        fake.inject_rate_limit(count=2)
        with pytest.raises(HFRateLimitedError):
            await hf_client.create_repo("user/x1")
        with pytest.raises(HFRateLimitedError):
            await hf_client.create_repo("user/x1")
        # 3rd call works.
        await hf_client.create_repo("user/x1")

    async def test_chaos_inject_5xx(self, hf_client: IHFHubClient) -> None:
        fake = _as_fake(hf_client)
        fake.inject_5xx(count=1)
        with pytest.raises(HFTransientError):
            await hf_client.create_repo("user/x2")

    async def test_chaos_inject_auth_failure(self, hf_client: IHFHubClient) -> None:
        fake = _as_fake(hf_client)
        fake.inject_auth_failure()
        with pytest.raises(HFAuthError):
            await hf_client.create_repo("user/x3")
        # Subsequent call succeeds.
        await hf_client.create_repo("user/x3")

    async def test_chaos_corrupted_download(self, hf_client: IHFHubClient) -> None:
        fake = _as_fake(hf_client)
        await hf_client.create_repo("user/x4")
        await hf_client.upload_file(
            repo_id="user/x4",
            path_in_repo="m.bin",
            content=b"AAAABBBBCCCCDDDDEEEE",
        )
        fake.inject_corrupted_download(count=1)
        corrupted = await hf_client.download_file(repo_id="user/x4", path_in_repo="m.bin")
        # Garbage marker is present and bytes != original.
        assert b"CORRUPT" in corrupted
        assert corrupted != b"AAAABBBBCCCCDDDDEEEE"
        # Recovery — full bytes back.
        good = await hf_client.download_file(repo_id="user/x4", path_in_repo="m.bin")
        assert good == b"AAAABBBBCCCCDDDDEEEE"

    async def test_snapshot_is_json_serializable(self, hf_client: IHFHubClient) -> None:
        import json
        fake = _as_fake(hf_client)
        await hf_client.create_repo("user/snap")
        await hf_client.upload_file(
            repo_id="user/snap", path_in_repo="f", content=b"x",
        )
        snap = fake.snapshot()
        json.dumps(snap)
        assert "user/snap" in snap["repos"]


__all__: list[str] = []
