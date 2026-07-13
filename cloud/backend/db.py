"""Event storage — pluggable backend (DECISIONS.md D8).

SQLiteStore is the default: free, zero setup, file-based. DynamoDBStore targets
the same interface for when a real deployment needs it (Lambda + DynamoDB are
always-free at dev scale per D8 — keep them; it's API Gateway that gets
replaced by this very FastAPI app). Swapping backend is one env var
(SENTINEL_STORAGE_BACKEND), not a rewrite of app.py.

This also fixes the original lambda_function.py inefficiency: that code ran
`table.scan()` twice per invocation. SQLiteStore uses an indexed query; the
DynamoDB stub documents the GSI a real deployment needs instead of a scan.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Protocol

from config import settings


@dataclass(frozen=True)
class EventRecord:
    event_id: str
    site_id: str
    label: str
    track_id: int
    started_at: str
    detected_at: str
    assigned: bool = False
    org_id: str = "default"
    description: str | None = None
    severity: str | None = None


class EventStore(Protocol):
    def save(self, event: EventRecord) -> None: ...
    def list_recent(self, limit: int = 100) -> list[EventRecord]: ...
    def assign(self, event_id: str) -> bool: ...


class SQLiteStore:
    """Default store. Free, no server process, no account — see DECISIONS.md D8."""

    def __init__(self, db_path=None):
        self._db_path = db_path or settings.db_path
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
                CREATE TABLE IF NOT EXISTS events (
                    event_id TEXT PRIMARY KEY,
                    site_id TEXT NOT NULL,
                    label TEXT NOT NULL,
                    track_id INTEGER NOT NULL,
                    started_at TEXT NOT NULL,
                    detected_at TEXT NOT NULL,
                    assigned INTEGER NOT NULL DEFAULT 0,
                    org_id TEXT NOT NULL DEFAULT 'default',
                    description TEXT,
                    severity TEXT
                )
                """
            )
            self._migrate_events_columns(conn)
            conn.execute("CREATE INDEX IF NOT EXISTS idx_events_detected_at ON events(detected_at)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_events_org_id ON events(org_id)")
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS heartbeats (
                    site_id TEXT PRIMARY KEY,
                    last_seen TEXT NOT NULL
                )
                """
            )

    def _migrate_events_columns(self, conn: sqlite3.Connection) -> None:
        """`CREATE TABLE IF NOT EXISTS` silently no-ops on a pre-existing
        database — a DB created before org_id/description/severity were added
        to the schema keeps its old columns forever unless we add them here.
        Runs on every startup; each ALTER is a no-op once the column exists.
        """
        existing = {row["name"] for row in conn.execute("PRAGMA table_info(events)").fetchall()}
        migrations = {
            "org_id": "ALTER TABLE events ADD COLUMN org_id TEXT NOT NULL DEFAULT 'default'",
            "description": "ALTER TABLE events ADD COLUMN description TEXT",
            "severity": "ALTER TABLE events ADD COLUMN severity TEXT",
        }
        for column, ddl in migrations.items():
            if column not in existing:
                conn.execute(ddl)

    def record_heartbeat(self, site_id: str) -> None:
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO heartbeats (site_id, last_seen) VALUES (?, ?) "
                "ON CONFLICT(site_id) DO UPDATE SET last_seen = excluded.last_seen",
                (site_id, utcnow_iso()),
            )

    def site_statuses(self, silent_threshold_s: float) -> list[dict]:
        """A site with no heartbeat within the threshold is flagged `silent` —
        VISION.md's "silence is an alarm": camera unplugged, tampered, or
        powered off all look identical to the backend, and all three deserve
        the same alert.
        """
        now = datetime.now(timezone.utc)
        with self._connect() as conn:
            rows = conn.execute("SELECT * FROM heartbeats").fetchall()
        statuses = []
        for r in rows:
            last_seen = datetime.fromisoformat(r["last_seen"])
            age_s = (now - last_seen).total_seconds()
            statuses.append(
                {
                    "site_id": r["site_id"],
                    "last_seen": r["last_seen"],
                    "seconds_since_heartbeat": round(age_s, 1),
                    "silent": age_s > silent_threshold_s,
                }
            )
        return statuses

    def save(self, event: EventRecord) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO events
                    (event_id, site_id, label, track_id, started_at, detected_at, assigned, org_id, description, severity)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    event.event_id,
                    event.site_id,
                    event.label,
                    event.track_id,
                    event.started_at,
                    event.detected_at,
                    int(event.assigned),
                    event.org_id,
                    event.description,
                    event.severity,
                ),
            )

    def list_recent(self, limit: int = 100, org_id: str | None = None) -> list[EventRecord]:
        with self._connect() as conn:
            if org_id:
                rows = conn.execute(
                    "SELECT * FROM events WHERE org_id = ? ORDER BY detected_at DESC LIMIT ?", (org_id, limit)
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT * FROM events ORDER BY detected_at DESC LIMIT ?", (limit,)
                ).fetchall()
        return [
            EventRecord(
                event_id=r["event_id"],
                site_id=r["site_id"],
                label=r["label"],
                track_id=r["track_id"],
                started_at=r["started_at"],
                detected_at=r["detected_at"],
                assigned=bool(r["assigned"]),
                org_id=r["org_id"],
                description=r["description"],
                severity=r["severity"],
            )
            for r in rows
        ]

    def assign(self, event_id: str) -> bool:
        with self._connect() as conn:
            cur = conn.execute("UPDATE events SET assigned = 1 WHERE event_id = ?", (event_id,))
            return cur.rowcount > 0


class DynamoDBStore:
    """Phase-2.5 target. Lambda + DynamoDB are always-free at dev scale (D8) —
    this is here so the swap from SQLiteStore is mechanical once a real
    multi-site deployment exists. Requires a table with `event_id` as the
    partition key and a GSI on `assigned` (avoids the original code's full
    table.scan() on every poll).
    """

    def __init__(self, table_name: str | None = None, region: str | None = None):
        import boto3  # local import: boto3 stays optional for local-only dev

        self._table = boto3.resource("dynamodb", region_name=region or settings.dynamodb_region).Table(
            table_name or settings.dynamodb_table
        )

    def save(self, event: EventRecord) -> None:
        self._table.put_item(
            Item={
                "event_id": event.event_id,
                "site_id": event.site_id,
                "label": event.label,
                "track_id": event.track_id,
                "started_at": event.started_at,
                "detected_at": event.detected_at,
                "assigned": event.assigned,
            }
        )

    def list_recent(self, limit: int = 100) -> list[EventRecord]:
        # Real deployment: query the assigned-GSI sorted by detected_at instead
        # of a scan. Left as a query stub since it needs the GSI provisioned.
        response = self._table.scan(Limit=limit)
        items = sorted(response.get("Items", []), key=lambda i: i["detected_at"], reverse=True)
        return [EventRecord(**item) for item in items]

    def assign(self, event_id: str) -> bool:
        try:
            self._table.update_item(
                Key={"event_id": event_id},
                UpdateExpression="SET assigned = :v",
                ExpressionAttributeValues={":v": True},
            )
            return True
        except Exception:
            return False


def make_store() -> EventStore:
    if settings.storage_backend == "sqlite":
        return SQLiteStore()
    if settings.storage_backend == "dynamodb":
        return DynamoDBStore()
    raise ValueError(f"Unknown SENTINEL_STORAGE_BACKEND: {settings.storage_backend!r}")


def utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat()
