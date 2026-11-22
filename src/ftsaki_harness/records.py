"""SQLite transactions keep batch results and cursors atomic."""
import json
import sqlite3
from pathlib import Path
from .policy import canonical

class Records:
    def __init__(self, path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(path)
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.execute("CREATE TABLE IF NOT EXISTS records (key TEXT PRIMARY KEY, value TEXT NOT NULL)")
        self.db.commit()

    def get(self, key, default=None):
        row = self.db.execute("SELECT value FROM records WHERE key=?", (key,)).fetchone()
        return json.loads(row[0]) if row else default

    def commit(self, entries):
        with self.db:
            for key, value in entries.items():
                encoded = canonical(value)
                inserted = self.db.execute("INSERT OR IGNORE INTO records VALUES (?,?)", (key, encoded))
                if not inserted.rowcount:
                    existing = self.db.execute("SELECT value FROM records WHERE key=?", (key,)).fetchone()[0]
                    if canonical(json.loads(existing)) != encoded:
                        raise ValueError("Immutable record conflict: " + key)

    def export(self):
        return {key: json.loads(value) for key, value in self.db.execute("SELECT key,value FROM records ORDER BY key")}

    def close(self):
        self.db.close()
