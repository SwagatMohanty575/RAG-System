"""
session_models.py

Data model for a single session's metadata.json file. Put this file at
backend/services/session_models.py (next to session_manager.py).

Kept separate from the existing models.py (which holds your FastAPI
Pydantic request/response schemas) so the two don't get tangled — this
one is a plain dataclass used internally by SessionManager, not part of
any API request/response body directly (main.py can still wrap it in a
Pydantic model for the /sessions endpoint response if you want one).
"""

from dataclasses import dataclass, asdict
from typing import Optional, Dict, Any
from datetime import datetime, timezone


@dataclass
class SessionMetadata:
    session_id: str
    chat_title: str
    pdf_filename: str
    created_at: str
    last_opened_at: str
    message_count: int = 0
    short_term_memory_path: Optional[str] = None 

    @staticmethod
    def new(session_id: str, chat_title: str, pdf_filename: str) -> "SessionMetadata":
        now = datetime.now(timezone.utc).isoformat()
        return SessionMetadata(
            session_id=session_id,
            chat_title=chat_title,
            pdf_filename=pdf_filename,
            created_at=now,
            last_opened_at=now,
            message_count=0,
        )

    def touch(self) -> None:
        """Call whenever the session is opened/activated."""
        self.last_opened_at = datetime.now(timezone.utc).isoformat()

    def increment_message_count(self, by: int = 1) -> None:
        self.message_count += by

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> "SessionMetadata":
        return SessionMetadata(**data)