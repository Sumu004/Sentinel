from pathlib import Path

from cloud.backend.db import SQLiteStore


def test_heartbeat_not_silent_when_recent(tmp_path: Path):
    store = SQLiteStore(db_path=tmp_path / "test.db")
    store.record_heartbeat("site-01")
    statuses = store.site_statuses(silent_threshold_s=60)
    assert len(statuses) == 1
    assert statuses[0]["site_id"] == "site-01"
    assert statuses[0]["silent"] is False


def test_heartbeat_silent_when_threshold_is_zero(tmp_path: Path):
    store = SQLiteStore(db_path=tmp_path / "test.db")
    store.record_heartbeat("site-01")
    # any elapsed time exceeds a 0-second threshold
    statuses = store.site_statuses(silent_threshold_s=0)
    assert statuses[0]["silent"] is True


def test_heartbeat_updates_existing_site(tmp_path: Path):
    store = SQLiteStore(db_path=tmp_path / "test.db")
    store.record_heartbeat("site-01")
    store.record_heartbeat("site-01")
    statuses = store.site_statuses(silent_threshold_s=60)
    assert len(statuses) == 1  # not duplicated
