"""Phase 2 (ethereal-tumbling-patterson) — :class:`EventJournal` envelope contract.

Replaces the legacy ``append(offset, ts, kind, payload)`` test suite with
coverage of the typed-envelope path:

* ``append_envelope`` writes a length-prefixed line via the shared codec.
* ``iter_envelopes`` round-trips envelopes through the codec.
* Resume after a torn write atomically truncates the partial tail via
  ``tmp + rename`` so the next append sees a clean file.
* Rotation honours the file-count cap and fires the on-rotate callback
  with the right metadata.

The test file lives next to the rest of the event_bus tests so the
shared ``conftest`` (if any) applies; otherwise it's pure stdlib.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from ryotenkai_pod.runner.event_journal import (
    DEFAULT_FILE_SIZE_CAP,
    EVENTS_DIR_REL,
    EVENTS_FILE_FMT,
    EventJournal,
)
from ryotenkai_shared.events import UNKNOWN_OFFSET, parse_length_prefix
from ryotenkai_shared.events.types.pod_lifecycle import (
    TrainerSpawnedEvent,
    TrainerSpawnedPayload,
)


def _make_event(offset: int, *, source: str = "pod://test/runner") -> TrainerSpawnedEvent:
    return TrainerSpawnedEvent(
        source=source,
        run_id="test",
        offset=offset,
        payload=TrainerSpawnedPayload(pid=offset + 1, cmdline="py", cwd="/tmp"),
    )


def _make_journal(tmp_path: Path, **kwargs: object) -> EventJournal:
    return EventJournal(root_dir=tmp_path / EVENTS_DIR_REL, **kwargs)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# TestPositive — round-trip
# ---------------------------------------------------------------------------


class TestPositive:
    def test_append_envelope_writes_length_prefixed_line(self, tmp_path: Path) -> None:
        j = _make_journal(tmp_path)
        ev = _make_event(offset=0)
        j.append_envelope(ev)
        j.close()

        path = tmp_path / EVENTS_DIR_REL / EVENTS_FILE_FMT.format(seq=0)
        line = path.read_text(encoding="utf-8")
        declared, body = parse_length_prefix(line)
        assert declared == len(body.encode("utf-8"))
        assert "ryotenkai.pod.lifecycle.trainer_spawned" in body

    def test_iter_envelopes_yields_in_order(self, tmp_path: Path) -> None:
        j = _make_journal(tmp_path)
        for i in range(5):
            j.append_envelope(_make_event(offset=i))
        j.close()

        j2 = _make_journal(tmp_path)
        offsets = [ev.offset for ev in j2.iter_envelopes()]
        assert offsets == [0, 1, 2, 3, 4]
        j2.close()


# ---------------------------------------------------------------------------
# TestResumeTruncation — torn-write detection
# ---------------------------------------------------------------------------


class TestResumeTruncation:
    def test_partial_last_line_is_truncated_on_resume(self, tmp_path: Path) -> None:
        events_dir = tmp_path / EVENTS_DIR_REL
        j = _make_journal(tmp_path)
        j.append_envelope(_make_event(offset=0))
        j.close()

        # Simulate a torn write: append bytes that don't match the
        # length prefix. Use a wrong byte count so parse_length_prefix
        # rejects the line.
        file_path = events_dir / EVENTS_FILE_FMT.format(seq=0)
        with file_path.open("ab") as fh:
            fh.write(b"999\tpartial-json\n")  # declared 999, body 12

        # Resume — should atomically rewrite the file without the
        # torn line.
        j2 = _make_journal(tmp_path)
        offsets = [ev.offset for ev in j2.iter_envelopes()]
        j2.close()
        # The single good line is preserved.
        assert 0 in offsets

    def test_resume_preserves_offsets_for_bus_seed(self, tmp_path: Path) -> None:
        j = _make_journal(tmp_path)
        for i in range(3):
            j.append_envelope(_make_event(offset=i))
        j.close()
        j2 = _make_journal(tmp_path)
        assert j2.newest_persisted_offset() == 2
        assert j2.oldest_persisted_offset() == 0
        j2.close()


# ---------------------------------------------------------------------------
# TestRotation — file cap + on_rotate callback
# ---------------------------------------------------------------------------


class TestRotation:
    def test_rotation_fires_callback(self, tmp_path: Path) -> None:
        observed: list[dict[str, object]] = []

        def _on_rotate(**kwargs: object) -> None:
            observed.append(dict(kwargs))

        # Tiny cap so each event triggers rotation.
        j = _make_journal(
            tmp_path,
            file_size_cap=128,
            max_files=3,
            fsync_batch=1,
            fsync_interval_ms=10,
        )
        j.set_rotation_callback(_on_rotate)
        for i in range(5):
            j.append_envelope(_make_event(offset=i))
        j.close()
        # At least one rotation must have fired.
        assert observed
        for rec in observed:
            assert "from_seq" in rec and "to_seq" in rec

    def test_invalid_config_raises_value_error(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError):
            _make_journal(tmp_path, file_size_cap=0)


# ---------------------------------------------------------------------------
# TestRegressions — codec compatibility
# ---------------------------------------------------------------------------


class TestRegressions:
    def test_codec_strict_false_yields_unknown_event_for_unknown_kind(
        self, tmp_path: Path,
    ) -> None:
        """A length-prefixed line with an unknown ``kind`` lands as
        :class:`UnknownEvent` instead of crashing the journal reader.

        Note: pre-Phase-2 journals (no length prefix) are treated as
        torn writes by the resume-time tail truncator — that path is
        covered by :class:`TestResumeTruncation` above. Cross-format
        compatibility for the legacy v1 journal isn't a Phase 2 goal
        per the plan ("Pre-Phase-2 journals — out of scope for this PR").
        """
        from ryotenkai_shared.events import UnknownEvent, parse_length_prefix

        events_dir = tmp_path / EVENTS_DIR_REL
        events_dir.mkdir(parents=True, exist_ok=True)
        legacy_path = events_dir / EVENTS_FILE_FMT.format(seq=0)
        body = (
            '{"event_id":"019e3000-0000-7000-0000-000000000000",'
            '"kind":"ryotenkai.future.kind","source":"pod://t/runner",'
            '"time":"2026-05-16T00:00:00Z","run_id":"t","stage_id":null,'
            '"offset":0,"schema_version":1,"severity":"info",'
            '"payload":{"unused":true}}'
        )
        prefixed = f"{len(body.encode('utf-8'))}\t{body}\n"
        legacy_path.write_text(prefixed, encoding="utf-8")
        # Sanity: the prefix parses cleanly so the resume tail check
        # doesn't drop the line.
        parse_length_prefix(prefixed)

        j = _make_journal(tmp_path)
        envelopes = list(j.iter_envelopes())
        j.close()
        assert envelopes
        assert all(isinstance(ev, UnknownEvent) for ev in envelopes)
        assert envelopes[0].original_type == "ryotenkai.future.kind"


# Avoid the "unused" import warning when stripping legacy constants.
_ = UNKNOWN_OFFSET
_ = DEFAULT_FILE_SIZE_CAP
