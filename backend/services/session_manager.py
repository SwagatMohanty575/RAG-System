"""
session_manager.py

Put this file at: backend/services/session_manager.py
"""

import json
import logging
import gc
import shutil
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional
from config import (
    SESSION_DOCUMENT_INDEX_FILE,
    SESSION_ENTITY_INDEX_FILE,
)

from config import (
    SESSIONS_DIR,
    SESSION_PDF_SUBDIR,
    SESSION_CHROMA_SUBDIR,
    SESSION_MEMORY_DB_NAME,
    SESSION_METADATA_FILE,
)
from services.session_models import SessionMetadata
from services.vector import ChromaVectorStore
from services.memory import ConversationMemory
from services.short_term_memory import ShortTermMemory
from services.document_index import DocumentIndex
from services.entity_index import EntityIndex
from config import SESSION_STM_DB_NAME, ENABLE_SHORT_TERM_MEMORY

logger = logging.getLogger(__name__)


class SessionNotFoundError(Exception):
    pass


@dataclass
class ActiveSession:
    session_id: str
    metadata: SessionMetadata
    vector_store: ChromaVectorStore
    memory: ConversationMemory
    pdf_path: Path
    short_term_memory: Optional[ShortTermMemory] = None


class SessionManager:
    def __init__(self, ingestor, chunker, embedder, guard):
        self._ingestor = ingestor
        self._chunker = chunker
        self._embedder = embedder
        self._guard = guard
        self._active: Optional[ActiveSession] = None

    # ------------------------------------------------------------------
    # Creation
    # ------------------------------------------------------------------

    def create_session(self, uploaded_pdf_path: str, original_filename: str, chat_title: Optional[str] = None) -> ActiveSession:
        session_id = self._generate_session_id()
        session_dir = SESSIONS_DIR / session_id
        pdf_dir = session_dir / SESSION_PDF_SUBDIR
        chroma_dir = session_dir / SESSION_CHROMA_SUBDIR

        try:
            pdf_dir.mkdir(parents=True, exist_ok=False)
            chroma_dir.mkdir(parents=True, exist_ok=False)

            dest_pdf_path = pdf_dir / original_filename
            shutil.copy2(uploaded_pdf_path, dest_pdf_path)

            metadata = SessionMetadata.new(
                session_id=session_id,
                chat_title=chat_title or self._title_from_filename(original_filename),
                pdf_filename=original_filename,
            )
            self._write_metadata(session_dir, metadata)

            vector_store = ChromaVectorStore(persist_directory=str(chroma_dir))
            memory = ConversationMemory(db_path=str(session_dir / SESSION_MEMORY_DB_NAME))
            #Initialize short-term memory
            stm_path = str(session_dir / SESSION_STM_DB_NAME)
            short_term_memory = ShortTermMemory(db_path = stm_path) if ENABLE_SHORT_TERM_MEMORY else None
            metadata.short_term_memory_path = stm_path

            self._run_ingestion_pipeline(dest_pdf_path, original_filename, vector_store)

            if self._active:
                self._close_active_session()

            active = ActiveSession(
                session_id=session_id,
                metadata=metadata,
                vector_store=vector_store,
                memory=memory,
                pdf_path=dest_pdf_path,
                short_term_memory=short_term_memory,
            )
            self._active = active
            logger.info("Created and activated new session %s (%s)", session_id, original_filename)
            return active

        except Exception:
            logger.exception("Failed to create session %s — cleaning up partial folder", session_id)
            if session_dir.exists():
                shutil.rmtree(session_dir, ignore_errors=True)
            raise

    def _run_ingestion_pipeline(self, pdf_path: Path, source_file: str, vector_store: ChromaVectorStore) -> int:
        pages = self._ingestor.extract(str(pdf_path))
        session_dir = pdf_path.parent.parent
        document_index_path = session_dir / SESSION_DOCUMENT_INDEX_FILE
        DocumentIndex.build(pages=pages, path=document_index_path, source_file=source_file)
        entity_index_path = session_dir / SESSION_ENTITY_INDEX_FILE
        EntityIndex.build(pages=pages, path=entity_index_path, source_file=source_file)
        chunks = self._chunker.chunk_pages(pages)
        for c in chunks:
            c["source_file"] = source_file
        chunks = self._guard.scrub_chunks(chunks)
        texts = [c["text"] for c in chunks]
        embeddings = self._embedder.embed(texts)
        vector_store.add_chunks(chunks, embeddings)
        return len(chunks)

    # ------------------------------------------------------------------
    # Loading / switching
    # ------------------------------------------------------------------

    def switch_session(self, session_id: str) -> ActiveSession:
        if self._active and self._active.session_id == session_id:
            return self._active

        session_dir = SESSIONS_DIR / session_id
        if not session_dir.exists():
            raise SessionNotFoundError(f"No session found with id '{session_id}'")

        metadata = self._read_metadata(session_dir)
        metadata.touch()
        self._write_metadata(session_dir, metadata)

        vector_store = ChromaVectorStore(persist_directory=str(session_dir / SESSION_CHROMA_SUBDIR))
        memory = ConversationMemory(db_path=str(session_dir / SESSION_MEMORY_DB_NAME))
         # Load short-term memory
        stm_path = str(session_dir / SESSION_STM_DB_NAME)
        short_term_memory = ShortTermMemory(db_path=stm_path) if ENABLE_SHORT_TERM_MEMORY and Path(stm_path).exists() else None
        pdf_dir = session_dir / SESSION_PDF_SUBDIR
        pdf_files = list(pdf_dir.glob("*"))
        pdf_path = pdf_files[0] if pdf_files else pdf_dir / metadata.pdf_filename

        if self._active:
            self._close_active_session()

        active = ActiveSession(
            session_id=session_id,
            metadata=metadata,
            vector_store=vector_store,
            memory=memory,
            pdf_path=pdf_path,
             short_term_memory=short_term_memory,
        )
        self._active = active
        logger.info("Switched to session %s", session_id)
        return active

    def get_active_session(self) -> Optional[ActiveSession]:
        return self._active

    # ------------------------------------------------------------------
    # Listing
    # ------------------------------------------------------------------

    def list_sessions(self) -> List[SessionMetadata]:
        sessions: List[SessionMetadata] = []
        if not SESSIONS_DIR.exists():
            return sessions

        for entry in SESSIONS_DIR.iterdir():
            if not entry.is_dir():
                continue
            metadata_path = entry / SESSION_METADATA_FILE
            if not metadata_path.exists():
                logger.warning("Skipping session folder without metadata.json: %s", entry)
                continue
            try:
                sessions.append(self._read_metadata(entry))
            except Exception:
                logger.exception("Failed to read metadata for session folder %s", entry)

        sessions.sort(key=lambda m: m.last_opened_at, reverse=True)
        return sessions

    # ------------------------------------------------------------------
    # Deletion
    # ------------------------------------------------------------------

    def delete_session(self, session_id: str) -> None:
        session_dir = SESSIONS_DIR / session_id
        if not session_dir.exists():
            raise SessionNotFoundError(f"No session found with id '{session_id}'")

        if self._active and self._active.session_id == session_id:
            self._close_active_session()

        self._remove_session_dir(session_dir)
        logger.info("Deleted session %s", session_id)

    # ------------------------------------------------------------------
    # Message count bookkeeping
    # ------------------------------------------------------------------

    def record_message(self, session_id: str) -> None:
        session_dir = SESSIONS_DIR / session_id
        metadata = self._read_metadata(session_dir)
        metadata.increment_message_count()
        self._write_metadata(session_dir, metadata)
        if self._active and self._active.session_id == session_id:
            self._active.metadata = metadata

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _generate_session_id() -> str:
        return f"sess_{uuid.uuid4().hex[:16]}"

    @staticmethod
    def _title_from_filename(filename: str) -> str:
        return Path(filename).stem.replace("_", " ").replace("-", " ").strip() or "Untitled chat"

    @staticmethod
    def _read_metadata(session_dir: Path) -> SessionMetadata:
        with open(session_dir / SESSION_METADATA_FILE, "r", encoding="utf-8") as f:
            return SessionMetadata.from_dict(json.load(f))

    @staticmethod
    def _write_metadata(session_dir: Path, metadata: SessionMetadata) -> None:
        with open(session_dir / SESSION_METADATA_FILE, "w", encoding="utf-8") as f:
            json.dump(metadata.to_dict(), f, indent=2)

    def _close_active_session(self) -> None:
        if not self._active:
            return

        try:
            self._active.vector_store.close()
        except Exception:
            logger.exception("Failed to close Chroma client for session %s", self._active.session_id)
        finally:
            self._active = None
            gc.collect()

    @staticmethod
    def _remove_session_dir(session_dir: Path) -> None:
        last_error = None
        for _ in range(5):
            try:
                shutil.rmtree(session_dir)
                return
            except PermissionError as exc:
                last_error = exc
                gc.collect()
                time.sleep(0.25)

        if last_error:
            raise last_error
