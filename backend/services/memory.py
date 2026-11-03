import sqlite3
import json
from typing import List, Dict, Optional
from config import MEMORY_DB, MEMORY_WINDOW


class ConversationMemory:
    def __init__(self, db_path: Optional[str] = None):
        self.db_path = db_path or str(MEMORY_DB)
        self._init_db()

    def _init_db(self):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS conversations (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    human TEXT NOT NULL,
                    ai TEXT NOT NULL,
                    sources TEXT,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_session ON conversations(session_id, timestamp)
            """)

    def add_turn(self, session_id: str, human: str, ai: str, sources: List[Dict]):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "INSERT INTO conversations (session_id, human, ai, sources) VALUES (?, ?, ?, ?)",
                (session_id, human, ai, json.dumps(sources)),
            )

    def get_history(self, session_id: str) -> List[Dict]:
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                """SELECT human, ai, sources FROM conversations
                   WHERE session_id = ?
                   ORDER BY timestamp DESC
                   LIMIT ?""",
                (session_id, MEMORY_WINDOW),
            ).fetchall()

        return [
            {
                "human": r["human"],
                "ai": r["ai"],
                "sources": json.loads(r["sources"]) if r["sources"] else [],
            }
            for r in reversed(rows)
        ]

    def clear_session(self, session_id: str):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("DELETE FROM conversations WHERE session_id = ?", (session_id,))