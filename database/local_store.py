"""Durable checkpoint + outbox, committed together before PostgreSQL IO."""
import json
from pathlib import Path
import sqlite3


class LocalStore:
    def __init__(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        # Hold a process lock for the lifetime of the worker, including on Windows.
        self.lock = path.with_suffix(path.suffix + ".lock").open("a+b")
        try:
            self.lock.seek(0)
            if self.lock.read(1) == b"":
                self.lock.write(b"0")
                self.lock.flush()
            self.lock.seek(0)
            import os
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(self.lock.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(self.lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            self.lock.close()
            raise RuntimeError("STATE_PATH is already in use by another worker") from exc
        try:
            self.conn = sqlite3.connect(path)
            self.conn.execute("PRAGMA journal_mode=WAL")
            self.conn.execute("PRAGMA synchronous=FULL")
            self.conn.executescript("""
                CREATE TABLE IF NOT EXISTS checkpoint (
                    id INTEGER PRIMARY KEY CHECK (id=1), revision INTEGER NOT NULL,
                    source_key TEXT NOT NULL, payload TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS outbox (
                    event_id TEXT PRIMARY KEY, payload TEXT NOT NULL);
            """)
        except BaseException:
            self.lock.close()
            raise

    def load(self):
        row = self.conn.execute("SELECT revision, source_key, payload FROM checkpoint WHERE id=1").fetchone()
        return {"revision": row[0], "source_key": row[1], "payload": json.loads(row[2])} if row else None

    def save(self, source_key, payload, events=(), *, revision=None):
        # Serialize first; never commit half a checkpoint on serialization errors.
        encoded = json.dumps(payload, allow_nan=False, separators=(",", ":"))
        records = [(e.event_id, json.dumps(e.to_dict(), allow_nan=False)) for e in events]
        old = self.load()
        if old and old["source_key"] != source_key:
            raise ValueError("STATE_PATH belongs to another source/configuration; use a separate state file")
        revision = revision if revision is not None else (old["revision"] + 1 if old else 1)
        with self.conn:
            self.conn.execute("INSERT OR REPLACE INTO checkpoint VALUES (1, ?, ?, ?)",
                              (revision, source_key, encoded))
            self.conn.executemany("INSERT INTO outbox VALUES (?, ?)", records)

    def pending(self, limit=500):
        rows = self.conn.execute("SELECT payload FROM outbox ORDER BY rowid LIMIT ?", (limit,)).fetchall()
        return [json.loads(row[0]) for row in rows]

    def pending_count(self):
        return self.conn.execute("SELECT COUNT(*) FROM outbox").fetchone()[0]

    def acknowledge(self, events):
        with self.conn:
            self.conn.executemany("DELETE FROM outbox WHERE event_id=?", [(e["event_id"],) for e in events])

    def close(self):
        self.conn.close()
        self.lock.close()
