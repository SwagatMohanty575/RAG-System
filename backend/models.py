from pydantic import BaseModel, Field
from typing import List, Optional


class ChatRequest(BaseModel):
    query: str = Field(..., min_length=1)
    session_id: str = Field(default="default")


class SourceCitation(BaseModel):
    file_name: str
    page_number: int
    section_title: Optional[str] = None
    chunk_preview: str


class ChatResponse(BaseModel):
    answer: str
    sources: List[SourceCitation]
    session_id: str


class IngestRequest(BaseModel):
    file_path: str = Field(..., description="Absolute path to the PDF file")


class IngestResponse(BaseModel):
    status: str
    chunks_indexed: int
    file_name: str


class HealthResponse(BaseModel):
    status: str
    api_ready: bool
    vector_db_ready: bool
    model: str