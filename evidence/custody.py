"""Chain-of-custody log.

Every access to evidence (capture, view, export, anchor) gets an
append-only, signed, hash-chained entry — altering or deleting a past
entry breaks every entry after it.
"""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from config import settings
from evidence.signing import _load_or_create_key

VALID_ACTIONS = {"captured", "signed", "viewed", "exported", "anchored", "assigned"}


@dataclass(frozen=True)
class CustodyEntry:
    seq: int
    clip_name: str
    action: str
    actor: str
    timestamp: str
    prev_entry_hash: str
    entry_hash: str
    signature_hex: str


class CustodyLog:
    """Append-only log backed by SQLite. Each row's `entry_hash` covers its
    own fields plus the previous row's `entry_hash`.
    """

    def __init__(self, db_path: Path | None = None):
        self._db_path = db_path or (settings.data_dir / "custody_log.db")
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._key: Ed25519PrivateKey = _load_or_create_key()
        self._init_schema()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self._db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_schema(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS custody (
                    seq INTEGER PRIMARY KEY AUTOINCREMENT,
                    clip_name TEXT NOT NULL,
                    action TEXT NOT NULL,
                    actor TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    prev_entry_hash TEXT NOT NULL,
                    entry_hash TEXT NOT NULL,
                    signature_hex TEXT NOT NULL
                )
                """
            )

    def _last_hash(self, conn: sqlite3.Connection) -> str:
        row = conn.execute("SELECT entry_hash FROM custody ORDER BY seq DESC LIMIT 1").fetchone()
        return row["entry_hash"] if row else "0" * 64

    def record(self, clip_name: str, action: str, actor: str = "system") -> CustodyEntry:
        if action not in VALID_ACTIONS:
            raise ValueError(f"Unknown custody action {action!r}; expected one of {VALID_ACTIONS}")

        with self._connect() as conn:
            prev_hash = self._last_hash(conn)
            timestamp = datetime.now(timezone.utc).isoformat()

            payload = json.dumps(
                {
                    "clip_name": clip_name,
                    "action": action,
                    "actor": actor,
                    "timestamp": timestamp,
                    "prev_entry_hash": prev_hash,
                },
                sort_keys=True,
            ).encode()
            entry_hash = sha256_file_bytes(payload)
            signature = self._key.sign(bytes.fromhex(entry_hash))

            conn.execute(
                """
                INSERT INTO custody (clip_name, action, actor, timestamp, prev_entry_hash, entry_hash, signature_hex)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (clip_name, action, actor, timestamp, prev_hash, entry_hash, signature.hex()),
            )
            seq = conn.execute("SELECT last_insert_rowid() AS seq").fetchone()["seq"]

        return CustodyEntry(
            seq=seq,
            clip_name=clip_name,
            action=action,
            actor=actor,
            timestamp=timestamp,
            prev_entry_hash=prev_hash,
            entry_hash=entry_hash,
            signature_hex=signature.hex(),
        )

    def history(self, clip_name: str | None = None) -> list[CustodyEntry]:
        with self._connect() as conn:
            if clip_name:
                rows = conn.execute(
                    "SELECT * FROM custody WHERE clip_name = ? ORDER BY seq", (clip_name,)
                ).fetchall()
            else:
                rows = conn.execute("SELECT * FROM custody ORDER BY seq").fetchall()
        return [CustodyEntry(**dict(r)) for r in rows]

    def verify_chain(self) -> bool:
        """Walks the whole log, recomputing each entry's hash and checking it
        matches both the stored hash and the next row's prev_entry_hash.
        """
        entries = self.history()
        prev_hash = "0" * 64
        for entry in entries:
            if entry.prev_entry_hash != prev_hash:
                return False
            payload = json.dumps(
                {
                    "clip_name": entry.clip_name,
                    "action": entry.action,
                    "actor": entry.actor,
                    "timestamp": entry.timestamp,
                    "prev_entry_hash": entry.prev_entry_hash,
                },
                sort_keys=True,
            ).encode()
            if sha256_file_bytes(payload) != entry.entry_hash:
                return False
            prev_hash = entry.entry_hash
        return True


def sha256_file_bytes(data: bytes) -> str:
    import hashlib

    return hashlib.sha256(data).hexdigest()
