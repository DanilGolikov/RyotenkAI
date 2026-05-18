"""Unit tests: ``ParentRunOpener``.

7-class structure (positive / negative / boundary / invariants /
dependency-errors / regressions / logic-specific).

Coverage:

* All required ``ryotenkai.lineage.*`` + ``ryotenkai.lifecycle.opened_by``
  tags stamped on root.
* Attempt nested under root via ``start_nested_run`` (not start_run).
* ``adopt_root`` delegates to ``client.adopt_run`` (resume path).
* Reserved-prefix guard catches accidental ``mlflow.*`` keys.
* attempt_no / attempt_id validated.
"""

from __future__ import annotations

import pytest

from ryotenkai_control.pipeline.mlflow.lifecycle.opener import ParentRunOpener
from ryotenkai_shared.infrastructure.mlflow.taxonomy import TagKey
from tests._fakes.mlflow_tracking_client import FakeTrackingClient


# ---------------------------------------------------------------------------
# Test helpers
# ---------------------------------------------------------------------------


def _make_opener(client: FakeTrackingClient | None = None) -> ParentRunOpener:
    """Construct a ParentRunOpener with sensible defaults."""
    client = client or FakeTrackingClient()
    return ParentRunOpener(client, opened_by="ci-host:ci-user")


def _open_root(opener: ParentRunOpener, **overrides: str):
    """Open a root run with realistic kwargs (callers may override)."""
    kwargs = {
        "experiment": "exp-default",
        "logical_run_id": "pipeline-2026-05-18-T120000",
        "config_sha256": "abc123def456abc123def456abc123def456abc123def456abc123def456abcd",
        "code_commit": "deadbeef1234",
        "engine_kind": "sft",
        "provider_kind": "single_node",
        "provider_gpu": "H100-80GB",
    }
    kwargs.update(overrides)
    return opener.open(**kwargs)  # type: ignore[arg-type]


# ===========================================================================
# 1. Positive
# ===========================================================================


class TestPositive:
    """Open paths produce well-formed RunHandles with expected tags."""

    def test_open_returns_handle(self) -> None:
        client = FakeTrackingClient()
        opener = _make_opener(client)

        handle = _open_root(opener)

        assert handle.run_id.startswith("fake-")
        assert handle.parent_run_id is None

    def test_open_calls_start_run_once(self) -> None:
        client = FakeTrackingClient()
        opener = _make_opener(client)

        _open_root(opener)

        assert len(client.start_run_calls) == 1
        assert len(client.start_nested_run_calls) == 0

    def test_open_attempt_returns_nested_handle(self) -> None:
        client = FakeTrackingClient()
        opener = _make_opener(client)
        root = _open_root(opener)

        attempt = opener.open_attempt(
            root_run=root,
            logical_run_id="pipeline-1",
            attempt_id="attempt-abc",
            attempt_no=1,
        )

        assert attempt.parent_run_id == root.run_id

    def test_adopt_root_returns_existing_handle(self) -> None:
        client = FakeTrackingClient()
        opener = _make_opener(client)
        original = _open_root(opener)

        adopted = opener.adopt_root(original.run_id)

        assert adopted.run_id == original.run_id
        assert client.adopt_run_calls == [original.run_id]


# ===========================================================================
# 2. Negative
# ===========================================================================


class TestNegative:
    """Bad input rejected at construction or invocation."""

    def test_empty_opened_by_rejected(self) -> None:
        client = FakeTrackingClient()
        with pytest.raises(ValueError, match="opened_by"):
            ParentRunOpener(client, opened_by="")

    def test_open_attempt_zero_no_rejected(self) -> None:
        client = FakeTrackingClient()
        opener = _make_opener(client)
        root = _open_root(opener)

        with pytest.raises(ValueError, match="attempt_no"):
            opener.open_attempt(
                root_run=root,
                logical_run_id="p1",
                attempt_id="a",
                attempt_no=0,
            )

    def test_open_attempt_empty_attempt_id_rejected(self) -> None:
        client = FakeTrackingClient()
        opener = _make_opener(client)
        root = _open_root(opener)

        with pytest.raises(ValueError, match="attempt_id"):
            opener.open_attempt(
                root_run=root,
                logical_run_id="p1",
                attempt_id="",
                attempt_no=1,
            )

    def test_adopt_root_empty_id_rejected(self) -> None:
        client = FakeTrackingClient()
        opener = _make_opener(client)

        with pytest.raises(ValueError, match="root_run_id"):
            opener.adopt_root("")


# ===========================================================================
# 3. Boundary
# ===========================================================================


class TestBoundary:
    """Edge cases on attempt_no and tag values."""

    @pytest.mark.parametrize("n", [1, 2, 5, 99, 1000])
    def test_attempt_no_accepts_any_positive(self, n: int) -> None:
        client = FakeTrackingClient()
        opener = _make_opener(client)
        root = _open_root(opener)

        attempt = opener.open_attempt(
            root_run=root,
            logical_run_id="p1",
            attempt_id=f"a-{n}",
            attempt_no=n,
        )

        attempt_tags = client.get_tags(attempt.run_id)
        assert attempt_tags[TagKey.ATTEMPT_NO.value] == str(n)

    @pytest.mark.parametrize("bad_n", [-1, -100, 0])
    def test_negative_attempt_no_rejected(self, bad_n: int) -> None:
        client = FakeTrackingClient()
        opener = _make_opener(client)
        root = _open_root(opener)

        with pytest.raises(ValueError):
            opener.open_attempt(
                root_run=root,
                logical_run_id="p1",
                attempt_id="a",
                attempt_no=bad_n,
            )


# ===========================================================================
# 4. Invariants
# ===========================================================================


class TestInvariants:
    """All required lineage + lifecycle tags must be present on every root."""

    def test_all_required_root_tags_present(self) -> None:
        client = FakeTrackingClient()
        opener = _make_opener(client)

        handle = _open_root(opener)
        tags = client.get_tags(handle.run_id)

        required_keys = {
            TagKey.LINEAGE_PIPELINE_ID.value,
            TagKey.LINEAGE_RUN_ID.value,
            TagKey.LINEAGE_CONFIG_SHA256.value,
            TagKey.LINEAGE_CODE_COMMIT.value,
            TagKey.LIFECYCLE_OPENED_BY.value,
            TagKey.ENGINE_KIND.value,
            TagKey.PROVIDER_KIND.value,
            TagKey.PROVIDER_GPU.value,
        }
        missing = required_keys - tags.keys()
        assert missing == set(), f"missing required tags: {missing}"

    def test_root_tags_match_input(self) -> None:
        client = FakeTrackingClient()
        opener = _make_opener(client)

        handle = _open_root(
            opener,
            logical_run_id="pipeline-x",
            config_sha256="cfg-sha",
            code_commit="commit-y",
            engine_kind="dpo",
            provider_kind="runpod",
            provider_gpu="A100-80GB",
        )
        tags = client.get_tags(handle.run_id)

        assert tags[TagKey.LINEAGE_PIPELINE_ID.value] == "pipeline-x"
        assert tags[TagKey.LINEAGE_RUN_ID.value] == "pipeline-x"
        assert tags[TagKey.LINEAGE_CONFIG_SHA256.value] == "cfg-sha"
        assert tags[TagKey.LINEAGE_CODE_COMMIT.value] == "commit-y"
        assert tags[TagKey.ENGINE_KIND.value] == "dpo"
        assert tags[TagKey.PROVIDER_KIND.value] == "runpod"
        assert tags[TagKey.PROVIDER_GPU.value] == "A100-80GB"

    def test_opened_by_stamped(self) -> None:
        client = FakeTrackingClient()
        opener = ParentRunOpener(client, opened_by="prod-host:alice")

        handle = _open_root(opener)
        tags = client.get_tags(handle.run_id)

        assert tags[TagKey.LIFECYCLE_OPENED_BY.value] == "prod-host:alice"

    def test_tags_passed_via_start_run_not_set_tags(self) -> None:
        """Tags must hit the server as part of start_run -- not via a
        second set_tags round-trip -- so the run is never observable
        without lineage tags.
        """
        client = FakeTrackingClient()
        opener = _make_opener(client)

        _open_root(opener)

        # No set_tags should have been called during open.
        assert client.set_tags_calls == []
        # Tags were passed in the start_run call itself.
        assert client.start_run_calls[0].tags

    def test_attempt_tags_via_start_nested_run(self) -> None:
        client = FakeTrackingClient()
        opener = _make_opener(client)
        root = _open_root(opener)
        client.set_tags_calls.clear()  # reset

        opener.open_attempt(
            root_run=root,
            logical_run_id="p1",
            attempt_id="a-1",
            attempt_no=1,
        )

        # Attempt-time tags applied via start_nested_run -- not set_tags.
        assert client.set_tags_calls == []
        assert len(client.start_nested_run_calls) == 1
        attempt_tags = client.start_nested_run_calls[0].tags
        assert TagKey.ATTEMPT_ID.value in attempt_tags
        assert TagKey.ATTEMPT_NO.value in attempt_tags


# ===========================================================================
# 5. Dependency errors
# ===========================================================================


class TestDependencyErrors:
    """Tracking client failure during open should propagate (caller wraps)."""

    def test_start_run_failure_propagates(self) -> None:
        client = FakeTrackingClient()
        client.set_unavailable(True)
        opener = _make_opener(client)

        with pytest.raises(Exception):
            _open_root(opener)

    def test_start_nested_failure_propagates(self) -> None:
        client = FakeTrackingClient()
        opener = _make_opener(client)
        root = _open_root(opener)
        client.set_unavailable(True)

        with pytest.raises(Exception):
            opener.open_attempt(
                root_run=root,
                logical_run_id="p1",
                attempt_id="a",
                attempt_no=1,
            )

    def test_adopt_unknown_run_id_propagates(self) -> None:
        client = FakeTrackingClient()
        opener = _make_opener(client)

        with pytest.raises(KeyError):
            opener.adopt_root("never-created")


# ===========================================================================
# 6. Regressions
# ===========================================================================


class TestRegressions:
    """Behaviours that were broken in the legacy MLflowAttemptManager."""

    def test_tags_set_only_once_per_open(self) -> None:
        """Legacy code re-stamped lineage tags every time the trainer
        sub-process re-attached. The new opener writes them ONCE via
        start_run.tags."""
        client = FakeTrackingClient()
        opener = _make_opener(client)

        handle = _open_root(opener)

        # Exactly one start_run call. No follow-up set_tags.
        assert len(client.start_run_calls) == 1
        assert client.set_tags_calls == []
        # The captured call holds the full tag set.
        captured = client.start_run_calls[0]
        assert TagKey.LINEAGE_PIPELINE_ID.value in captured.tags
        assert captured.run_id == handle.run_id

    def test_attempt_run_name_includes_attempt_no(self) -> None:
        """Run names must encode the retry ordinal so MLflow UI sorts
        attempts deterministically."""
        client = FakeTrackingClient()
        opener = _make_opener(client)
        root = _open_root(opener)

        opener.open_attempt(
            root_run=root,
            logical_run_id="pipeline-42",
            attempt_id="aid",
            attempt_no=3,
        )

        call = client.start_nested_run_calls[0]
        assert call.name == "pipeline-42_attempt_3"

    def test_attempt_run_parent_id_matches_root(self) -> None:
        client = FakeTrackingClient()
        opener = _make_opener(client)
        root = _open_root(opener)

        attempt = opener.open_attempt(
            root_run=root,
            logical_run_id="p",
            attempt_id="a",
            attempt_no=1,
        )

        assert attempt.parent_run_id == root.run_id
        assert client.start_nested_run_calls[0].parent_run_id == root.run_id

    def test_no_mlflow_namespace_in_emitted_tags(self) -> None:
        """The opener must not emit any 'mlflow.*' tag -- those collide
        with MLflow system reserved tags and were a recurring bug.
        """
        client = FakeTrackingClient()
        opener = _make_opener(client)

        handle = _open_root(opener)
        tags = client.get_tags(handle.run_id)

        for key in tags:
            assert not key.startswith("mlflow."), (
                f"opener leaked reserved 'mlflow.*' tag: {key}"
            )


# ===========================================================================
# 7. Logic-specific
# ===========================================================================


class TestLogicSpecific:
    """Tag-shape pins: keys come from TagKey enum (no string literals)."""

    def test_lineage_run_id_mirrors_logical_run_id(self) -> None:
        """LINEAGE_PIPELINE_ID and LINEAGE_RUN_ID currently mirror each
        other (the logical run id IS the pipeline id). Pin to catch a
        future schema drift.
        """
        client = FakeTrackingClient()
        opener = _make_opener(client)

        handle = _open_root(opener, logical_run_id="logical-7")
        tags = client.get_tags(handle.run_id)

        assert (
            tags[TagKey.LINEAGE_PIPELINE_ID.value]
            == tags[TagKey.LINEAGE_RUN_ID.value]
        )
        assert tags[TagKey.LINEAGE_PIPELINE_ID.value] == "logical-7"

    def test_attempt_no_is_string_not_int(self) -> None:
        """MLflow tags are string-valued. attempt_no must be stringified."""
        client = FakeTrackingClient()
        opener = _make_opener(client)
        root = _open_root(opener)

        attempt = opener.open_attempt(
            root_run=root,
            logical_run_id="p",
            attempt_id="a",
            attempt_no=42,
        )
        tags = client.get_tags(attempt.run_id)

        assert isinstance(tags[TagKey.ATTEMPT_NO.value], str)
        assert tags[TagKey.ATTEMPT_NO.value] == "42"

    def test_attempt_id_stamped_verbatim(self) -> None:
        client = FakeTrackingClient()
        opener = _make_opener(client)
        root = _open_root(opener)

        attempt = opener.open_attempt(
            root_run=root,
            logical_run_id="p",
            attempt_id="unique-attempt-uuid-here",
            attempt_no=1,
        )
        tags = client.get_tags(attempt.run_id)

        assert tags[TagKey.ATTEMPT_ID.value] == "unique-attempt-uuid-here"

    def test_root_call_passes_empty_params(self) -> None:
        """The opener stamps tags, not params, on open. Params come later
        via metric_sink / dedicated config logger."""
        client = FakeTrackingClient()
        opener = _make_opener(client)

        _open_root(opener)

        assert client.start_run_calls[0].params == {}
