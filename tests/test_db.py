from pathlib import Path

from cloud.backend.db import EventRecord, SQLiteStore


def test_sqlite_store_roundtrip(tmp_path: Path):
    store = SQLiteStore(db_path=tmp_path / "test.db")
    record = EventRecord(
        event_id="e1",
        site_id="dev-site-01",
        label="motion",
        track_id=1,
        started_at="2026-01-01T00:00:00+00:00",
        detected_at="2026-01-01T00:00:03+00:00",
    )
    store.save(record)

    recent = store.list_recent()
    assert len(recent) == 1
    assert recent[0].event_id == "e1"
    assert recent[0].assigned is False


def test_sqlite_store_assign(tmp_path: Path):
    store = SQLiteStore(db_path=tmp_path / "test.db")
    record = EventRecord(
        event_id="e2",
        site_id="dev-site-01",
        label="motion",
        track_id=2,
        started_at="2026-01-01T00:00:00+00:00",
        detected_at="2026-01-01T00:00:03+00:00",
    )
    store.save(record)

    assert store.assign("e2") is True
    assert store.assign("nonexistent") is False
    assert store.list_recent()[0].assigned is True
