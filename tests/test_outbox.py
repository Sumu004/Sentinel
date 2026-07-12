from pathlib import Path

from edge.outbox import Outbox, retry_pending


def test_enqueue_and_pending(tmp_path: Path):
    from edge.events import Event

    outbox = Outbox(db_path=tmp_path / "outbox.db")
    event = Event(
        event_id="e1",
        site_id="dev-site-01",
        label="motion",
        track_id=1,
        started_at="2026-01-01T00:00:00+00:00",
        detected_at="2026-01-01T00:00:03+00:00",
    )
    outbox.enqueue(event)
    assert len(outbox) == 1

    pending = outbox.pending()
    assert len(pending) == 1
    assert pending[0].payload["event_id"] == "e1"


def test_retry_pending_removes_on_success(tmp_path: Path):
    from edge.events import Event

    outbox = Outbox(db_path=tmp_path / "outbox.db")
    outbox.enqueue(
        Event(
            event_id="e1",
            site_id="s1",
            label="motion",
            track_id=1,
            started_at="2026-01-01T00:00:00+00:00",
            detected_at="2026-01-01T00:00:03+00:00",
        )
    )

    sent = retry_pending(outbox, sender=lambda payload: True)
    assert sent == 1
    assert len(outbox) == 0


def test_retry_pending_keeps_failed_items(tmp_path: Path):
    from edge.events import Event

    outbox = Outbox(db_path=tmp_path / "outbox.db")
    outbox.enqueue(
        Event(
            event_id="e1",
            site_id="s1",
            label="motion",
            track_id=1,
            started_at="2026-01-01T00:00:00+00:00",
            detected_at="2026-01-01T00:00:03+00:00",
        )
    )

    sent = retry_pending(outbox, sender=lambda payload: False)
    assert sent == 0
    assert len(outbox) == 1
    assert outbox.pending()[0].attempts == 1
