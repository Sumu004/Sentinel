"""Store-and-forward outbox.

A failed cloud send queues here and gets retried instead of being dropped.
"""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from config import settings
from edge.events import Event


@dataclass(frozen=True)
class OutboxItem:
    id: int
    payload: dict
    queued_at: str
    attempts: int


class Outbox:
    def __init__(self, db_path: Path | None = None):
        self._db_path = db_path or (settings.data_dir / "outbox.db")
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self._db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_schema(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS outbox (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    payload TEXT NOT NULL,
                    queued_at TEXT NOT NULL,
                    attempts INTEGER NOT NULL DEFAULT 0
                )
                """
            )

    def enqueue(self, event: Event) -> None:
        payload = json.dumps(
            {
                "event_id": event.event_id,
                "site_id": event.site_id,
                "label": event.label,
                "track_id": event.track_id,
                "started_at": event.started_at,
                "detected_at": event.detected_at,
                "description": event.description,
                "severity": event.severity,
            }
        )
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO outbox (payload, queued_at, attempts) VALUES (?, ?, 0)",
                (payload, datetime.now(timezone.utc).isoformat()),
            )

    def pending(self, limit: int = 50) -> list[OutboxItem]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM outbox ORDER BY id LIMIT ?", (limit,)
            ).fetchall()
        return [
            OutboxItem(id=r["id"], payload=json.loads(r["payload"]), queued_at=r["queued_at"], attempts=r["attempts"])
            for r in rows
        ]

    def mark_sent(self, item_id: int) -> None:
        with self._connect() as conn:
            conn.execute("DELETE FROM outbox WHERE id = ?", (item_id,))

    def mark_failed(self, item_id: int) -> None:
        with self._connect() as conn:
            conn.execute("UPDATE outbox SET attempts = attempts + 1 WHERE id = ?", (item_id,))

    def __len__(self) -> int:
        with self._connect() as conn:
            row = conn.execute("SELECT COUNT(*) AS c FROM outbox").fetchone()
        return row["c"]


def retry_pending(outbox: Outbox, sender) -> int:
    """Attempts to resend every queued item via `sender(payload) -> bool`.
    Returns the number successfully sent.
    """
    sent = 0
    for item in outbox.pending():
        if sender(item.payload):
            outbox.mark_sent(item.id)
            sent += 1
        else:
            outbox.mark_failed(item.id)
    return sent
