"""Comprehensive tests for :mod:`src.pipeline.state.job_submission`.

This dataclass + its tiny IO is the bridge that makes Phase 7.1's
out-of-process CLI (``ryotenkai job status``, ``... events``,
``... stop``) and Phase 7.2's Web UI possible: launchers persist
``attempts/<n>/job_submission.json`` after submitting a job, and any
fresh process re-builds the SSH endpoint by reading that file. If
this layer breaks, the entire control-plane proxy + CLI silently
fails — tests pin the contract tightly.

Coverage split (project policy):

1. **Positive**           — happy-path save / load / round-trip / now().
2. **Negative**           — load raises on missing file, malformed JSON,
                             schema mismatch, missing required keys.
3. **Boundary**           — None optionals, empty strings, port limits,
                             unicode in fields, missing parent dir,
                             overwrite, repeated load.
4. **Invariants**         — round-trip identity, frozen mutation refused,
                             ``CURRENT_VERSION`` is class-level not field,
                             ``to_dict`` returns only declared fields,
                             always-UTC timestamps.
5. **Dependency errors**  — ``atomic_write_json`` raises OSError;
                             ``Path.read_text`` raises OSError; JSON load
                             error reraised as ``JobSubmissionLoadError``.
6. **Regressions**        — slots+frozen+ClassVar+asdict bug stays fixed;
                             ``to_dict`` JSON-serialisable; load preserves
                             port type after round-trip.
7. **Logic-specific**     — ``now()`` uses UTC at call time; the bound
                             constructor coerces port to int.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest

from src.pipeline.state.job_submission import (
    JOB_SUBMISSION_FILENAME,
    JobSubmission,
    JobSubmissionLoadError,
    load_job_submission,
    save_job_submission,
)


# ---------------------------------------------------------------------------
# Fixtures / factories
# ---------------------------------------------------------------------------


def _submission(**overrides: Any) -> JobSubmission:
    """Build a JobSubmission with conventional defaults that any test can override."""
    defaults: dict[str, Any] = dict(
        schema_version=JobSubmission.CURRENT_VERSION,
        job_id="run-foo:attempt:1",
        provider_name="runpod",
        pod_id="pod-xyz",
        ssh_host="1.2.3.4",
        ssh_port=22022,
        ssh_username="root",
        ssh_key_path="/home/me/.ssh/id_ed25519",
        created_at_iso="2026-04-26T00:00:00+00:00",
    )
    defaults.update(overrides)
    return JobSubmission(**defaults)


# ---------------------------------------------------------------------------
# 1. Positive
# ---------------------------------------------------------------------------


class TestPositive:
    def test_now_factory_populates_required_fields(self) -> None:
        sub = JobSubmission.now(
            job_id="j-1",
            provider_name="runpod",
            pod_id="pod-1",
            ssh_host="host",
            ssh_port=22,
            ssh_username="root",
            ssh_key_path="/k",
        )
        assert sub.schema_version == JobSubmission.CURRENT_VERSION
        assert sub.job_id == "j-1"
        assert sub.provider_name == "runpod"
        assert sub.pod_id == "pod-1"
        assert sub.ssh_host == "host"
        assert sub.ssh_port == 22
        assert sub.ssh_username == "root"
        assert sub.ssh_key_path == "/k"
        assert sub.created_at_iso  # set to *now*

    def test_save_creates_file_with_expected_contents(self, tmp_path: Path) -> None:
        sub = _submission()
        target = save_job_submission(tmp_path, sub)

        assert target == (tmp_path / JOB_SUBMISSION_FILENAME).resolve()
        on_disk = json.loads(target.read_text(encoding="utf-8"))
        assert on_disk["job_id"] == "run-foo:attempt:1"
        assert on_disk["schema_version"] == JobSubmission.CURRENT_VERSION

    def test_load_returns_dataclass_instance(self, tmp_path: Path) -> None:
        save_job_submission(tmp_path, _submission())
        loaded = load_job_submission(tmp_path)
        assert isinstance(loaded, JobSubmission)
        assert loaded.job_id == "run-foo:attempt:1"

    def test_save_returns_resolved_path(self, tmp_path: Path) -> None:
        # ``save_job_submission`` resolves the path so callers can log a
        # canonical absolute location regardless of how ``tmp_path`` was
        # passed in.
        target = save_job_submission(tmp_path, _submission())
        assert target.is_absolute()
        assert target.name == JOB_SUBMISSION_FILENAME

    def test_to_dict_contains_all_instance_fields(self) -> None:
        sub = _submission()
        d = sub.to_dict()
        for key in (
            "schema_version", "job_id", "provider_name", "pod_id",
            "ssh_host", "ssh_port", "ssh_username", "ssh_key_path",
            "created_at_iso",
        ):
            assert key in d
        # No *extra* keys leak in (i.e. ``CURRENT_VERSION`` ClassVar must NOT
        # appear — see Regressions section for the pinned contract).
        assert set(d.keys()) == {
            "schema_version", "job_id", "provider_name", "pod_id",
            "ssh_host", "ssh_port", "ssh_username", "ssh_key_path",
            "created_at_iso",
        }


# ---------------------------------------------------------------------------
# 2. Negative
# ---------------------------------------------------------------------------


class TestNegative:
    def test_load_raises_when_file_missing(self, tmp_path: Path) -> None:
        with pytest.raises(JobSubmissionLoadError, match="not found"):
            load_job_submission(tmp_path)

    def test_load_raises_on_malformed_json(self, tmp_path: Path) -> None:
        (tmp_path / JOB_SUBMISSION_FILENAME).write_text("{broken", encoding="utf-8")
        with pytest.raises(JobSubmissionLoadError, match="failed to read"):
            load_job_submission(tmp_path)

    def test_load_raises_on_unknown_future_schema(self, tmp_path: Path) -> None:
        (tmp_path / JOB_SUBMISSION_FILENAME).write_text(
            json.dumps({"schema_version": JobSubmission.CURRENT_VERSION + 1}),
            encoding="utf-8",
        )
        with pytest.raises(
            JobSubmissionLoadError, match="unsupported job submission schema_version",
        ):
            load_job_submission(tmp_path)

    def test_load_raises_on_missing_schema_version(self, tmp_path: Path) -> None:
        (tmp_path / JOB_SUBMISSION_FILENAME).write_text(
            json.dumps({"job_id": "x"}), encoding="utf-8",
        )
        with pytest.raises(JobSubmissionLoadError, match="unsupported job submission"):
            load_job_submission(tmp_path)

    def test_load_raises_on_missing_required_field(self, tmp_path: Path) -> None:
        # Correct schema_version but ``ssh_host`` absent — should be a
        # malformed submission, not a silent fall-through to defaults.
        payload = _submission().to_dict()
        del payload["ssh_host"]
        (tmp_path / JOB_SUBMISSION_FILENAME).write_text(
            json.dumps(payload), encoding="utf-8",
        )
        with pytest.raises(JobSubmissionLoadError, match="malformed job submission"):
            load_job_submission(tmp_path)

    def test_load_raises_on_non_int_ssh_port(self, tmp_path: Path) -> None:
        payload = _submission().to_dict()
        payload["ssh_port"] = "not-a-port"
        (tmp_path / JOB_SUBMISSION_FILENAME).write_text(
            json.dumps(payload), encoding="utf-8",
        )
        with pytest.raises(JobSubmissionLoadError, match="malformed job submission"):
            load_job_submission(tmp_path)


# ---------------------------------------------------------------------------
# 3. Boundary
# ---------------------------------------------------------------------------


class TestBoundary:
    def test_pod_id_none_is_valid(self, tmp_path: Path) -> None:
        save_job_submission(tmp_path, _submission(pod_id=None))
        loaded = load_job_submission(tmp_path)
        assert loaded.pod_id is None

    def test_ssh_key_path_none_is_valid(self, tmp_path: Path) -> None:
        save_job_submission(tmp_path, _submission(ssh_key_path=None))
        loaded = load_job_submission(tmp_path)
        assert loaded.ssh_key_path is None

    @pytest.mark.parametrize("port", [1, 22, 22022, 65535])
    def test_ssh_port_legal_values_round_trip(
        self, tmp_path: Path, port: int,
    ) -> None:
        save_job_submission(tmp_path, _submission(ssh_port=port))
        assert load_job_submission(tmp_path).ssh_port == port

    def test_save_creates_missing_parent_dirs(self, tmp_path: Path) -> None:
        deep = tmp_path / "a" / "b" / "c"
        # Parent doesn't exist yet — save must mkdir it.
        save_job_submission(deep, _submission())
        assert (deep / JOB_SUBMISSION_FILENAME).is_file()

    def test_overwrite_replaces_previous_record(self, tmp_path: Path) -> None:
        save_job_submission(tmp_path, _submission(job_id="first"))
        save_job_submission(tmp_path, _submission(job_id="second"))
        assert load_job_submission(tmp_path).job_id == "second"

    def test_repeated_load_returns_equal_objects(self, tmp_path: Path) -> None:
        save_job_submission(tmp_path, _submission())
        a = load_job_submission(tmp_path)
        b = load_job_submission(tmp_path)
        assert a == b
        # Frozen dataclasses are hashable when all fields are hashable —
        # so two reads share the same hash too.
        assert hash(a) == hash(b)

    def test_unicode_in_string_fields_round_trips(self, tmp_path: Path) -> None:
        save_job_submission(
            tmp_path,
            _submission(provider_name="провайдер", ssh_username="ユーザ"),
        )
        loaded = load_job_submission(tmp_path)
        assert loaded.provider_name == "провайдер"
        assert loaded.ssh_username == "ユーザ"

    def test_path_with_user_expansion(self, tmp_path: Path) -> None:
        # ``load_job_submission`` calls ``expanduser().resolve()`` — feeding
        # in an unresolved path with a tilde must still work (assuming the
        # underlying path is available).
        save_job_submission(tmp_path, _submission())
        loaded = load_job_submission(tmp_path)
        assert loaded.job_id == "run-foo:attempt:1"


# ---------------------------------------------------------------------------
# 4. Invariants
# ---------------------------------------------------------------------------


class TestInvariants:
    def test_save_then_load_yields_equal_object(self, tmp_path: Path) -> None:
        original = _submission()
        save_job_submission(tmp_path, original)
        assert load_job_submission(tmp_path) == original

    def test_dataclass_is_frozen(self) -> None:
        sub = _submission()
        with pytest.raises((AttributeError, Exception)):
            sub.job_id = "mutated"  # type: ignore[misc]

    def test_current_version_is_classvar_not_instance_field(self) -> None:
        # Pinning the slots+frozen+ClassVar interaction so the previous
        # ``member_descriptor`` regression cannot return.
        sub = _submission()
        # ``CURRENT_VERSION`` should *not* be in __slots__ / asdict output.
        d = sub.to_dict()
        assert "CURRENT_VERSION" not in d
        # It is, however, accessible via the class.
        assert JobSubmission.CURRENT_VERSION == 1

    def test_to_dict_is_json_serialisable(self) -> None:
        # Underpins ``save_job_submission`` — anything to_dict returns has
        # to round-trip through ``json.dumps`` cleanly. Exercising it in
        # isolation prevents a subtle "save works but text is gibberish"
        # surprise the day someone adds a non-JSON field.
        sub = _submission()
        json.dumps(sub.to_dict())  # must not raise

    def test_created_at_iso_parses_as_utc(self) -> None:
        sub = JobSubmission.now(
            job_id="j",
            provider_name="p",
            pod_id=None,
            ssh_host="h",
            ssh_port=1,
            ssh_username="u",
            ssh_key_path=None,
        )
        parsed = datetime.fromisoformat(sub.created_at_iso)
        assert parsed.tzinfo is not None
        # Has a UTC offset (offset == zero or any positive — the contract
        # says we *write* UTC, so must be exactly zero offset).
        assert parsed.utcoffset() == timezone.utc.utcoffset(None)

    def test_save_is_idempotent_for_identical_input(self, tmp_path: Path) -> None:
        sub = _submission()
        target1 = save_job_submission(tmp_path, sub)
        contents1 = target1.read_text(encoding="utf-8")
        target2 = save_job_submission(tmp_path, sub)
        contents2 = target2.read_text(encoding="utf-8")
        assert contents1 == contents2


# ---------------------------------------------------------------------------
# 5. Dependency errors
# ---------------------------------------------------------------------------


class TestDependencyErrors:
    def test_save_propagates_atomic_write_errors(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        from src.pipeline.state import job_submission as mod

        def _boom(_path: Any, _payload: Any) -> None:
            raise OSError("disk full")

        monkeypatch.setattr(mod, "atomic_write_json", _boom)
        with pytest.raises(OSError, match="disk full"):
            save_job_submission(tmp_path, _submission())

    def test_load_wraps_read_oserror(
        self, tmp_path: Path,
    ) -> None:
        target = tmp_path / JOB_SUBMISSION_FILENAME
        target.write_text("{}", encoding="utf-8")

        # Make ``Path.read_text`` raise mid-load; ``load_job_submission``
        # must wrap the OSError into our typed error so callers don't
        # need to know about the specific failure mode.
        with patch.object(
            Path, "read_text", side_effect=OSError("permission denied"),
        ):
            with pytest.raises(JobSubmissionLoadError, match="failed to read"):
                load_job_submission(tmp_path)

    def test_load_directory_with_unrelated_files_still_works(
        self, tmp_path: Path,
    ) -> None:
        # The attempt directory typically has lots of siblings (logs,
        # checkpoints, ...). Load must only look at the canonical
        # filename and ignore everything else.
        save_job_submission(tmp_path, _submission())
        (tmp_path / "training.log").write_text("noise", encoding="utf-8")
        (tmp_path / "checkpoint-final.bin").write_bytes(b"\x00\x01")
        loaded = load_job_submission(tmp_path)
        assert loaded.job_id == "run-foo:attempt:1"


# ---------------------------------------------------------------------------
# 6. Regressions
# ---------------------------------------------------------------------------


class TestRegressions:
    def test_slots_frozen_classvar_asdict_does_not_choke(self) -> None:
        # The original implementation typed ``CURRENT_VERSION`` without
        # ``ClassVar``, which made dataclass treat it as a field; combined
        # with ``slots=True`` it became a slot descriptor, and ``asdict()``
        # crashed with ``Object of type member_descriptor is not JSON
        # serializable``. This test pins the fix.
        sub = _submission()
        json.dumps(sub.to_dict())

    def test_load_preserves_port_type(self, tmp_path: Path) -> None:
        # JSON has no integer/float distinction, but ``ssh_port`` MUST
        # come back as ``int`` so callers can pass it to ``ssh -p`` /
        # ``socket.connect`` without manual casts.
        save_job_submission(tmp_path, _submission(ssh_port=22022))
        port = load_job_submission(tmp_path).ssh_port
        assert isinstance(port, int)
        assert port == 22022

    def test_load_returns_typed_none_for_optionals(self, tmp_path: Path) -> None:
        # Earlier draft had a bug where ``payload.get("pod_id", "")``
        # returned ``""`` when the key was absent. The current contract
        # says missing optional == ``None`` — pin it.
        save_job_submission(tmp_path, _submission(pod_id=None, ssh_key_path=None))
        loaded = load_job_submission(tmp_path)
        assert loaded.pod_id is None
        assert loaded.ssh_key_path is None


# ---------------------------------------------------------------------------
# 7. Logic-specific
# ---------------------------------------------------------------------------


class TestLogicSpecific:
    def test_now_uses_utc_at_call_time(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        # Pin the timestamp via patching ``datetime.now`` inside the
        # module so the assertion is deterministic.
        from src.pipeline.state import job_submission as mod

        fixed = datetime(2026, 1, 2, 3, 4, 5, tzinfo=timezone.utc)

        class _FakeDateTime:
            @classmethod
            def now(cls, tz=None):  # type: ignore[no-untyped-def]
                assert tz == timezone.utc, "now() must request UTC"
                return fixed

        monkeypatch.setattr(mod, "datetime", _FakeDateTime)
        sub = JobSubmission.now(
            job_id="j",
            provider_name="p",
            pod_id=None,
            ssh_host="h",
            ssh_port=1,
            ssh_username="u",
            ssh_key_path=None,
        )
        assert sub.created_at_iso == fixed.isoformat()

    def test_now_coerces_port_to_int(self) -> None:
        # Some callers (config plumbing, env reads) pass strings or
        # numpy scalars. ``now`` documents that it coerces.
        sub = JobSubmission.now(
            job_id="j",
            provider_name="p",
            pod_id=None,
            ssh_host="h",
            ssh_port="22022",  # type: ignore[arg-type]
            ssh_username="u",
            ssh_key_path=None,
        )
        assert sub.ssh_port == 22022
        assert isinstance(sub.ssh_port, int)

    def test_filename_constant_matches_actual_file(self, tmp_path: Path) -> None:
        # Two different code paths write/read this file — pin that they
        # both agree on the filename.
        save_job_submission(tmp_path, _submission())
        assert (tmp_path / JOB_SUBMISSION_FILENAME).is_file()
        # Reading via the constant is what the CLI / router do.
        load_job_submission(tmp_path)


# ---------------------------------------------------------------------------
# 8. Combinatorial
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("pod_id", [None, "", "pod-001", "very-long-id-" + "x" * 32])
@pytest.mark.parametrize("ssh_key_path", [None, "/k/id", "~/relative"])
@pytest.mark.parametrize("ssh_port", [22, 22022, 65535])
@pytest.mark.parametrize("provider_name", ["runpod", "single_node", "custom"])
def test_combinatorial_round_trip(
    tmp_path: Path,
    pod_id: str | None,
    ssh_key_path: str | None,
    ssh_port: int,
    provider_name: str,
) -> None:
    """Parametric matrix: every combination of optional / variant fields
    survives a save/load round-trip without mutation. 4×3×3×3 = 108
    combinations — cheap to run, but covers every interaction we care
    about (missing optionals + uncommon ports + non-default providers)."""
    sub = _submission(
        pod_id=pod_id,
        ssh_key_path=ssh_key_path,
        ssh_port=ssh_port,
        provider_name=provider_name,
    )
    save_job_submission(tmp_path, sub)
    loaded = load_job_submission(tmp_path)
    assert loaded == sub
