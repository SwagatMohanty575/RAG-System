"""
short_term_memory.py

Session-isolated short-term memory storing ONLY user queries
and topic context. Assistant responses are NOT stored to prevent
pollution of future answers.
"""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Optional

from config import SHORT_TERM_MEMORY_WINDOW


@dataclass
class ConversationQuery:
    """Only the user query and topic, not the response."""
    query: str
    topic: Optional[str]  # Extracted topic (e.g., "chapter 3", "Marlowe")
    timestamp: str

    def to_dict(self) -> Dict:
        return {
            "query": self.query,
            "topic": self.topic,
            "timestamp": self.timestamp,
        }

    @staticmethod
    def from_dict(data: Dict) -> "ConversationQuery":
        return ConversationQuery(
            query=data["query"],
            topic=data.get("topic"),
            timestamp=data["timestamp"],
        )


class ShortTermMemory:
    """
    Session-specific short-term memory with sliding window.
    
    CRITICAL: Only stores USER QUERIES, not assistant responses.
    This prevents wrong answers from polluting future responses.
    """

    def __init__(self, db_path: str) -> None:
        self.db_path = db_path
        self.window_size = SHORT_TERM_MEMORY_WINDOW
        self._init_db()

    def _init_db(self) -> None:
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS stm_queries (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    query TEXT NOT NULL,
                    topic TEXT,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_stm_query 
                ON stm_queries(timestamp)
            """)

    def add_query(self, query: str, topic: Optional[str] = None) -> None:
        """
        Add only the user query (not the response) to STM.
        
        Topic is extracted to help with follow-up understanding.
        """
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "INSERT INTO stm_queries (query, topic) VALUES (?, ?)",
                (query, topic),
            )

            # Enforce sliding window
            conn.execute(
                """DELETE FROM stm_queries
                   WHERE id NOT IN (
                       SELECT id FROM stm_queries
                       ORDER BY timestamp DESC
                       LIMIT ?
                   )""",
                (self.window_size,),
            )

    def get_recent_queries(self, limit: Optional[int] = None) -> List[ConversationQuery]:
        """Retrieve recent user queries (oldest first)."""
        limit = limit or self.window_size
        
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                """SELECT query, topic, timestamp
                   FROM stm_queries
                   ORDER BY timestamp DESC
                   LIMIT ?""",
                (limit,),
            ).fetchall()

        return [
            ConversationQuery(
                query=r["query"],
                topic=r["topic"],
                timestamp=r["timestamp"],
            )
            for r in reversed(rows)
        ]

    def get_current_topic(self) -> Optional[str]:
        """Get the most recent topic for follow-up context."""
        queries = self.get_recent_queries(limit=1)
        return queries[0].topic if queries else None

    def clear(self) -> None:
        with sqlite3.connect(self.db_path) as conn: 
            conn.execute("DELETE FROM stm_queries")

    def count(self) -> int:
        with sqlite3.connect(self.db_path) as conn:
            row = conn.execute("SELECT COUNT(*) FROM stm_queries").fetchone()
            return row[0] if row else 0