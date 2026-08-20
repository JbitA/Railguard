from __future__ import annotations

import json
import sqlite3
import threading
import time
from pathlib import Path

from .ack import telemetry_ack_key

def _stable_identity_payload(payload: str) -> str:
    """Canonicalize telemetry while excluding transport-local queue counters.

    Retries may observe a different current spool depth even though the sensor record
    identity/content is unchanged. Every other field is immutable for a given ACK key.
    """
    obj = json.loads(payload, parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)))
    if not isinstance(obj, dict):
        raise ValueError("outbox payload must be a JSON object")
    health = obj.get("health")
    if isinstance(health, dict):
        health = dict(health)
        health.pop("spool_depth", None)
        health.pop("spool_dropped", None)
        obj = dict(obj)
        obj["health"] = health
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), allow_nan=False)


class DurableOutbox:
    """Thread-safe bounded SQLite FIFO retained until application-level DB ACK."""

    def __init__(self, path: str | Path, max_records: int = 200_000):
        if max_records < 1:
            raise ValueError("max_records must be positive")
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.max_records = int(max_records)
        self.db = sqlite3.connect(self.path, check_same_thread=False)
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.execute("PRAGMA synchronous=NORMAL")
        self.db.execute("CREATE TABLE IF NOT EXISTS outbox (id INTEGER PRIMARY KEY AUTOINCREMENT, payload TEXT NOT NULL)")
        columns = {str(row[1]) for row in self.db.execute("PRAGMA table_info(outbox)").fetchall()}
        if "ack_key" not in columns:
            self.db.execute("ALTER TABLE outbox ADD COLUMN ack_key TEXT")
        if "last_sent" not in columns:
            self.db.execute("ALTER TABLE outbox ADD COLUMN last_sent REAL NOT NULL DEFAULT 0")
        if "attempts" not in columns:
            self.db.execute("ALTER TABLE outbox ADD COLUMN attempts INTEGER NOT NULL DEFAULT 0")
        self.db.execute("CREATE INDEX IF NOT EXISTS outbox_ack_key_idx ON outbox(ack_key)")
        self.db.execute("CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value INTEGER NOT NULL)")
        self.db.execute("INSERT OR IGNORE INTO meta(key,value) VALUES ('dropped_records',0)")
        self._backfill_ack_keys()
        self.db.commit()
        self.lock = threading.Lock()

    def _backfill_ack_keys(self) -> None:
        for row_id, payload in self.db.execute("SELECT id,payload FROM outbox WHERE ack_key IS NULL").fetchall():
            try:
                obj = json.loads(str(payload))
                key = telemetry_ack_key(str(obj["device_id"]), str(obj["ts"]), int(obj["seq"]))
            except Exception:
                # Legacy/unparseable rows cannot ever receive a trustworthy DB ACK.
                # Drop them explicitly and account for the loss instead of wedging
                # the spool forever after a schema migration.
                self.db.execute("DELETE FROM outbox WHERE id=?", (int(row_id),))
                self.db.execute("UPDATE meta SET value=value+1 WHERE key='dropped_records'")
            else:
                self.db.execute("UPDATE outbox SET ack_key=? WHERE id=?", (key, int(row_id)))

    def enqueue(self, payload: str, ack_key: str | None = None) -> None:
        if ack_key is None:
            obj = json.loads(payload)
            ack_key = telemetry_ack_key(str(obj["device_id"]), str(obj["ts"]), int(obj["seq"]))
        with self.lock:
            # An idempotency identity is immutable. Exact semantic replays are a no-op
            # (spool counters may differ because they are transport-local); any other
            # content change under the same ACK key is an integrity error, not an upsert.
            existing = self.db.execute("SELECT id,payload FROM outbox WHERE ack_key=? ORDER BY id LIMIT 1", (ack_key,)).fetchone()
            if existing:
                if _stable_identity_payload(str(existing[1])) != _stable_identity_payload(payload):
                    raise ValueError(f"conflicting telemetry payload for immutable ACK identity {ack_key!r}")
            else:
                self.db.execute("INSERT INTO outbox(payload,ack_key,last_sent,attempts) VALUES (?,?,0,0)", (payload, ack_key))
            count = int(self.db.execute("SELECT count(*) FROM outbox").fetchone()[0])
            excess = max(0, count - self.max_records)
            if excess:
                self.db.execute("DELETE FROM outbox WHERE id IN (SELECT id FROM outbox ORDER BY id LIMIT ?)", (excess,))
                self.db.execute("UPDATE meta SET value=value+? WHERE key='dropped_records'", (excess,))
            self.db.commit()

    def depth(self) -> int:
        with self.lock:
            return int(self.db.execute("SELECT count(*) FROM outbox").fetchone()[0])

    def dropped_records(self) -> int:
        with self.lock:
            row = self.db.execute("SELECT value FROM meta WHERE key='dropped_records'").fetchone()
            return int(row[0] if row else 0)

    def batch(self, limit: int) -> list[tuple[int, str]]:
        with self.lock:
            return [(int(i), str(payload)) for i, payload in self.db.execute(
                "SELECT id,payload FROM outbox ORDER BY id LIMIT ?", (int(limit),)
            ).fetchall()]

    def ready_batch(self, limit: int, *, retry_after_s: float, now: float | None = None) -> list[tuple[int, str]]:
        now = time.time() if now is None else float(now)
        threshold = now - float(retry_after_s)
        with self.lock:
            return [(int(i), str(payload)) for i, payload in self.db.execute(
                "SELECT id,payload FROM outbox WHERE last_sent<=? ORDER BY id LIMIT ?",
                (threshold, int(limit)),
            ).fetchall()]

    def mark_sent(self, row_id: int, *, sent_at: float | None = None) -> None:
        sent_at = time.time() if sent_at is None else float(sent_at)
        with self.lock:
            self.db.execute("UPDATE outbox SET last_sent=?, attempts=attempts+1 WHERE id=?", (sent_at, int(row_id)))
            self.db.commit()

    def acknowledge(self, ack_key: str) -> bool:
        with self.lock:
            cur = self.db.execute("DELETE FROM outbox WHERE ack_key=?", (ack_key,))
            self.db.commit()
            return cur.rowcount > 0

    def attempts(self, ack_key: str) -> int:
        with self.lock:
            row = self.db.execute("SELECT attempts FROM outbox WHERE ack_key=? ORDER BY id LIMIT 1", (ack_key,)).fetchone()
            return int(row[0]) if row else 0

    def delete(self, row_id: int) -> None:
        """Administrative/test helper; production deletion should use acknowledge()."""
        with self.lock:
            self.db.execute("DELETE FROM outbox WHERE id=?", (int(row_id),))
            self.db.commit()

    def close(self) -> None:
        with self.lock:
            self.db.close()
