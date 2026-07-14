import sqlite3
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


def test_sqlite_store_update_description_patches_existing_event(tmp_path: Path):
    store = SQLiteStore(db_path=tmp_path / "test.db")
    record = EventRecord(
        event_id="e3",
        site_id="dev-site-01",
        label="person",
        track_id=1,
        started_at="2026-01-01T00:00:00+00:00",
        detected_at="2026-01-01T00:00:03+00:00",
        description="Person detected and tracked for 5s (daytime).",
        severity="medium",
    )
    store.save(record)

    ok = store.update_description("e3", "A person in a red jacket walks past the gate.", "high")

    assert ok is True
    updated = store.list_recent()[0]
    assert updated.description == "A person in a red jacket walks past the gate."
    assert updated.severity == "high"


def test_sqlite_store_update_description_returns_false_for_missing_event(tmp_path: Path):
    store = SQLiteStore(db_path=tmp_path / "test.db")
    assert store.update_description("nonexistent", "text", "high") is False


def test_sqlite_store_migrates_pre_existing_db_missing_new_columns(tmp_path: Path):
    """Regression test: a DB created before org_id/description/severity were
    added to the schema must not crash on startup.
    """
    db_path = tmp_path / "legacy.db"
    conn = sqlite3.connect(db_path)
    conn.execute(
        """
        CREATE TABLE events (
            event_id TEXT PRIMARY KEY,
            site_id TEXT NOT NULL,
            label TEXT NOT NULL,
            track_id INTEGER NOT NULL,
            started_at TEXT NOT NULL,
            detected_at TEXT NOT NULL,
            assigned INTEGER NOT NULL DEFAULT 0
        )
        """
    )
    conn.execute(
        "INSERT INTO events VALUES ('e1','site1','person',1,'2026-01-01T00:00:00Z','2026-01-01T00:00:03Z',0)"
    )
    conn.commit()
    conn.close()

    store = SQLiteStore(db_path=db_path)

    recs = store.list_recent()
    assert len(recs) == 1
    assert recs[0].event_id == "e1"
    assert recs[0].org_id == "default"
    assert recs[0].description is None
