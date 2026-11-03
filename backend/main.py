import os
from pathlib import Path
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

from models import ChatRequest, ChatResponse, IngestRequest, IngestResponse, HealthResponse, SourceCitation
from services.ingest import PDFIngestor
from services.chunk import SectionAwareChunker
from services.embed import OllamaEmbedder
from services.guard import PIIGuard
from services.retrieve import MMRRetriever
from services.client import NvidiaLLM
from services.session_manager import SessionManager, SessionNotFoundError

from config import (
    UPLOAD_DIR,
    SESSIONS_DIR,
    SESSION_DOCUMENT_INDEX_FILE,
    SESSION_ENTITY_INDEX_FILE,
    SESSION_SUMMARY_INDEX_FILE,
)

from services.document_index import DocumentIndex
from services.entity_index import EntityIndex
from services.summary_index import SummaryIndex
from services.query_router import QueryRouter
from services.query_contextualizer import QueryContextualizer


# =====================================================
# HELPER FUNCTIONS
# =====================================================

FALLBACK = "I don't have that information in the uploaded document."


def _session_file(session_id: str, filename: str) -> Path:
    return SESSIONS_DIR / session_id / filename


def _load_document_index(active) -> DocumentIndex:
    path = _session_file(active.session_id, SESSION_DOCUMENT_INDEX_FILE)

    if path.exists():
        return DocumentIndex.load(path)

    # Backward compatibility for older sessions
    pages = app.state.ingest.extract(str(active.pdf_path))

    return DocumentIndex.build(
        pages=pages,
        path=path,
        source_file=active.metadata.pdf_filename,
    )


def _load_entity_index(active) -> EntityIndex:
    path = _session_file(active.session_id, SESSION_ENTITY_INDEX_FILE)

    if path.exists():
        return EntityIndex.load(path)

    # Backward compatibility for older sessions
    pages = app.state.ingest.extract(str(active.pdf_path))

    return EntityIndex.build(
        pages=pages,
        path=path,
        source_file=active.metadata.pdf_filename,
    )


def _make_sources(chunks, limit: int = 10):
    sources = []
    seen = set()

    for chunk in chunks[:limit]:
        metadata = chunk.get("metadata", {})

        key = (
            metadata.get("source_file"),
            metadata.get("page_number"),
            metadata.get("section_title"),
        )

        if key in seen:
            continue

        seen.add(key)

        preview = chunk.get("text", "")

        sources.append(
            SourceCitation(
                file_name=str(metadata.get("source_file", "")),
                page_number=int(metadata.get("page_number", 0) or 0),
                section_title=metadata.get("section_title"),
                chunk_preview=preview[:300] + "..." if len(preview) > 300 else preview,
            )
        )

    return sources


def _structure_sources(index: DocumentIndex):
    sources = []

    for chapter in index.chapters[:10]:
        sources.append(
            SourceCitation(
                file_name=index.data.get("source_file", ""),
                page_number=int(chapter["start_page"]),
                section_title=f"CHAPTER {chapter['number']}: {chapter['title']}",
                chunk_preview=f"Chapter {chapter['number']}: {chapter['title']}",
            )
        )

    return sources


# =====================================================
# APP LIFESPAN
# =====================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.ingest = PDFIngestor()
    app.state.chunk = SectionAwareChunker()
    app.state.embed = OllamaEmbedder()
    app.state.guard = PIIGuard()
    app.state.client = NvidiaLLM()

    # Replaces app.state.vector / app.state.retrieve / app.state.memory.
    # There is no single global instance of any of these anymore.
    app.state.session_manager = SessionManager(
        ingestor=app.state.ingest,
        chunker=app.state.chunk,
        embedder=app.state.embed,
        guard=app.state.guard,
    )

    yield


# =====================================================
# FASTAPI APP
# =====================================================

app = FastAPI(
    title="RAG Employee Handbook API",
    description="Context-aware Q&A with source citations, PII guardrails, and conversation memory.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# =====================================================
# HEALTH
# =====================================================

@app.get("/health", response_model=HealthResponse)
async def health():
    active = app.state.session_manager.get_active_session()

    return HealthResponse(
        status="healthy",
        api_ready=app.state.client.health_check(),
        vector_db_ready=active.vector_store.count() > 0 if active else False,
        model=app.state.client.model,
    )


# =====================================================
# INGEST
# =====================================================

@app.post("/ingest", response_model=IngestResponse)
async def ingest_pdf(request: IngestRequest):
    """Creates a brand-new, isolated session for this PDF."""
    try:
        file_name = os.path.basename(request.file_path)

        active = app.state.session_manager.create_session(
            uploaded_pdf_path=request.file_path,
            original_filename=file_name,
        )

        return IngestResponse(
            status="success",
            chunks_indexed=active.vector_store.count(),
            file_name=file_name,
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# =====================================================
# CHAT
# =====================================================

@app.post(
    "/chat",
    response_model=ChatResponse,
)
async def chat(
    request: ChatRequest,
):

    try:

        # =====================================================
        # ACTIVE SESSION
        # =====================================================

        active = (
            app.state.session_manager
            .get_active_session()
        )

        if active is None:

            raise HTTPException(
                status_code=400,
                detail=(
                    "No active session. "
                    "Upload a document or "
                    "select a chat first."
                ),
            )

        session_id = (
            active.session_id
        )

        # =====================================================
        # CONVERSATION HISTORY
        # =====================================================

        history = (
            active.memory
            .get_history(
                session_id
            )
        )

        # =====================================================
        # QUERY CONTEXTUALIZATION
        # =====================================================

        effective_query = (
            QueryContextualizer.resolve(
                query=request.query,
                history=history,
                client=app.state.client,
            )
        )

        print(
            "[DEBUG] Original query:",
            request.query,
        )

        print(
            "[DEBUG] Effective query:",
            effective_query,
        )

        # =====================================================
        # QUERY ROUTING
        # =====================================================

        query_type = (
            QueryRouter.route(
                effective_query
            )
        )

        print(
            "[DEBUG] Query type:",
            query_type,
        )

        # =====================================================
        # LOAD INDEXES
        # =====================================================

        document_index = (
            _load_document_index(
                active
            )
        )

        entity_index = (
            _load_entity_index(
                active
            )
        )

        chunks = []

        answer = None

        sources = []

        # =====================================================
        # 1. STRUCTURE QUESTIONS
        # =====================================================

        if (
            query_type
            == QueryRouter.STRUCTURE
        ):

            answer = (
                document_index
                .answer_structure(
                    effective_query
                )
            )

            sources = (
                _structure_sources(
                    document_index
                )
            )

        # =====================================================
        # 2. GLOBAL DOCUMENT QUESTIONS
        # =====================================================

        elif (
            query_type
            == QueryRouter.GLOBAL_SUMMARY
        ):

            summary_path = (
                _session_file(
                    session_id,
                    SESSION_SUMMARY_INDEX_FILE,
                )
            )

            summary_index = (
                SummaryIndex(
                    summary_path
                )
            )

            summary_data = (
                summary_index.ensure(
                    vector_store=(
                        active.vector_store
                    ),
                    client=(
                        app.state.client
                    ),
                    document_index=(
                        document_index
                    ),
                )
            )

            chunks = (
                summary_index
                .as_context_chunks(
                    summary_data
                )
            )

            if not chunks:

                answer = FALLBACK

                sources = []

            else:

                raw_answer = (
                    app.state.client
                    .generate(
                        query=request.query,
                        resolved_query=(
                            effective_query
                        ),
                        query_type=(
                            query_type
                        ),
                        context_chunks=(
                            chunks
                        ),
                        conversation_history=(
                            history
                        ),
                        short_term_memory=(
                            active
                            .short_term_memory
                        ),
                    )
                )

                answer = (
                    app.state.guard
                    .scrub_response(
                        raw_answer
                    )
                )

                sources = (
                    _make_sources(
                        chunks,
                        limit=10,
                    )
                )

        # =====================================================
        # 3. ENTITY LIST
        # =====================================================

        elif (
            query_type
            == QueryRouter.ENTITY_LIST
        ):

            entities = (
                entity_index
                .list_entities()
            )

            if entities:

                names = [
                    item["name"]
                    for item
                    in entities[:50]
                ]

                answer = (
                    f"I found "
                    f"{len(entities)} "
                    f"named character/entity "
                    f"entries in the document:"
                    "\n\n"
                    +
                    "\n".join(
                        f"- {name}"
                        for name in names
                    )
                )

                sources = []

                for item in entities[:10]:

                    page = (
                        item["pages"][0]
                        if item["pages"]
                        else 0
                    )

                    sources.append(
                        SourceCitation(
                            file_name=(
                                entity_index
                                .data
                                .get(
                                    "source_file",
                                    "",
                                )
                            ),
                            page_number=int(
                                page
                            ),
                            section_title=(
                                "Character mentions"
                            ),
                            chunk_preview=(
                                f"{item['name']} "
                                f"is mentioned "
                                f"{item['mentions']} "
                                f"time(s)."
                            ),
                        )
                    )

            else:

                # If no entities were found,
                # fall back to normal RAG.
                query_type = (
                    QueryRouter.LOCAL_QA
                )

        # =====================================================
        # 4. ENTITY / RELATIONSHIP
        # =====================================================

        if (
            answer is None
            and
            query_type
            in {
                QueryRouter.ENTITY,
                QueryRouter.RELATIONSHIP,
            }
        ):

            names = (
                entity_index.resolve(
                    effective_query
                )
            )

            all_chunks = (
                active.vector_store
                .get_all_chunks()
            )

            if names:

                chunks = (
                    entity_index
                    .collect_evidence(
                        all_chunks,
                        names,
                        limit=12,
                    )
                )

            # -------------------------------------------------
            # Fallback if entity index didn't resolve it.
            # -------------------------------------------------

            if not chunks:

                retriever = (
                    MMRRetriever(
                        active.vector_store,
                        app.state.embed,
                    )
                )

                chunks = (
                    retriever.retrieve(
                        effective_query,
                        conversation_history=(
                            history
                        ),
                    )
                )

            if chunks:

                raw_answer = (
                    app.state.client
                    .generate(
                        query=request.query,
                        resolved_query=(
                            effective_query
                        ),
                        query_type=(
                            query_type
                        ),
                        context_chunks=(
                            chunks
                        ),
                        conversation_history=(
                            history
                        ),
                        short_term_memory=(
                            active
                            .short_term_memory
                        ),
                    )
                )

                answer = (
                    app.state.guard
                    .scrub_response(
                        raw_answer
                    )
                )

                sources = (
                    _make_sources(
                        chunks
                    )
                )

            else:

                answer = FALLBACK

                sources = []

        # =====================================================
        # 5. NORMAL LOCAL QA
        # =====================================================

        if answer is None:

            retriever = (
                MMRRetriever(
                    active.vector_store,
                    app.state.embed,
                )
            )

            chunks = (
                retriever.retrieve(
                    effective_query,
                    conversation_history=(
                        history
                    ),
                )
            )

            if not chunks:

                answer = FALLBACK

                sources = []

            else:

                raw_answer = (
                    app.state.client
                    .generate(
                        query=request.query,
                        resolved_query=(
                            effective_query
                        ),
                        query_type=(
                            QueryRouter.LOCAL_QA
                        ),
                        context_chunks=(
                            chunks
                        ),
                        conversation_history=(
                            history
                        ),
                        short_term_memory=(
                            active
                            .short_term_memory
                        ),
                    )
                )

                answer = (
                    app.state.guard
                    .scrub_response(
                        raw_answer
                    )
                )

                sources = (
                    _make_sources(
                        chunks
                    )
                )

        # =====================================================
        # STORE MEMORY
        # =====================================================

        active.memory.add_turn(
            session_id=session_id,
            human=request.query,
            ai=answer,
            sources=[
                source.model_dump()
                for source in sources
            ],
        )

        # =====================================================
        # SHORT TERM MEMORY
        # =====================================================

        if active.short_term_memory:

            active.short_term_memory.add_query(
                query=request.query,
                topic=effective_query[
                    :200
                ],
            )

        # =====================================================
        # SESSION MESSAGE COUNT
        # =====================================================

        app.state.session_manager.record_message(
            session_id
        )

        # =====================================================
        # RESPONSE
        # =====================================================

        return ChatResponse(
            answer=answer,
            sources=sources,
            session_id=session_id,
        )

    except HTTPException:

        raise

    except Exception as e:

        print(
            "[ERROR] Chat failed:",
            repr(e),
        )

        raise HTTPException(
            status_code=500,
            detail=str(e),
        )

# =====================================================
# CLEAR MEMORY
# =====================================================

@app.post("/clear-memory")
async def clear_memory():

    active = (
        app.state.session_manager
        .get_active_session()
    )

    if active is None:

        raise HTTPException(
            status_code=400,
            detail="No active session.",
        )

    # Clear long-term conversation memory.
    active.memory.clear_session(
        active.session_id
    )

    # Clear short-term query memory.
    if active.short_term_memory:

        active.short_term_memory.clear()

    return {
        "status": "cleared",
        "session_id": active.session_id,
    }

# =====================================================
# STATS
# =====================================================

@app.get("/stats")
async def stats():
    active = app.state.session_manager.get_active_session()

    if active is None:
        return {
            "total_chunks": 0,
            "memory_db": None,
            "active_session": None,
        }

    return {
        "total_chunks": active.vector_store.count(),
        "memory_db": str(active.memory.db_path),
        "active_session": active.session_id,
    }


# =====================================================
# SESSION MANAGEMENT
# =====================================================

@app.get("/sessions")
async def list_sessions():
    sessions = app.state.session_manager.list_sessions()

    return {
        "sessions": [s.to_dict() for s in sessions]
    }


@app.post("/sessions/{session_id}/activate")
async def activate_session(session_id: str):
    try:
        active = app.state.session_manager.switch_session(session_id)

    except SessionNotFoundError:
        raise HTTPException(
            status_code=404,
            detail=f"Session '{session_id}' not found",
        )

    return {
        "session_id": active.session_id,
        "metadata": active.metadata.to_dict(),
        "history": active.memory.get_history(active.session_id),
    }


@app.delete("/sessions/{session_id}")
async def delete_session(session_id: str):
    try:
        app.state.session_manager.delete_session(session_id)

    except SessionNotFoundError:
        raise HTTPException(
            status_code=404,
            detail=f"Session '{session_id}' not found",
        )

    return {
        "status": "deleted",
        "session_id": session_id,
    }


@app.get("/sessions/active")
async def get_active_session_info():
    active = app.state.session_manager.get_active_session()

    if active is None:
        return {"active": False}

    return {
        "active": True,
        "session_id": active.session_id,
        "metadata": active.metadata.to_dict(),
    }