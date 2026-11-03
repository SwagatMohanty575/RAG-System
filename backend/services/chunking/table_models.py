"""
table_models.py

Data models for table-specific chunking.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any


@dataclass
class TableRow:
    """A single row from a spreadsheet with full metadata."""
    row_index: int
    data: Dict[str, Any]           # Column name → value
    sheet_name: str
    workbook_name: str
    headers: List[str]
    primary_key: Optional[str]
    
    def to_semantic_text(self) -> str:
        """Generate natural language from this row."""
        sentences = []
        pk_value = self.data.get(self.primary_key) if self.primary_key else None
        
        for header, value in self.data.items():
            if value is None or str(value).strip() == "":
                continue
            if header == self.primary_key:
                continue
            
            if pk_value:
                sentences.append(f"{pk_value}'s {header.lower()} is {value}.")
            else:
                sentences.append(f"The {header.lower()} is {value}.")
        
        return " ".join(sentences) if sentences else f"Row data: {self.data}"


@dataclass
class TableChunk:
    """A chunk containing one or more table rows."""
    text: str                       # Semantic text for embedding
    page_number: int                # Sheet index
    section_title: str              # Sheet name
    source_file: str                # Workbook filename
    chunk_index: int
    
    # Table-specific metadata
    workbook: str
    sheet: str
    row_start: int
    row_end: int
    column_names: List[str]
    primary_key: Optional[str]
    rows_data: List[Dict[str, Any]]  # Structured rows for retrieval
    schema_relationships: List[Dict[str, Any]] = field(default_factory=list)
    row_relationships: List[Dict[str, Any]] = field(default_factory=list)
    cross_row_relationships: List[Dict[str, Any]] = field(default_factory=list)
    document_id: str = ""
    session_id: str = ""
    
    # Compatibility fields
    chapter: Optional[str] = None
    section: Optional[str] = None
    subsection: Optional[str] = None
    heading: Optional[str] = None
    page_start: int = 0
    page_end: int = 0
    paragraph_ids: List[str] = field(default_factory=list)
    previous_chunk: Optional[str] = None
    next_chunk: Optional[str] = None
    parent_section: Optional[str] = None
    estimated_tokens: int = 0
    character_count: int = 0

    def __post_init__(self) -> None:
        if not self.character_count:
            self.character_count = len(self.text)
        if not self.estimated_tokens:
            self.estimated_tokens = self.character_count // 4

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dict compatible with embedding pipeline."""
        return {
            # Required existing fields
            "text": self.text,
            "page_number": self.page_number,
            "section_title": self.section_title,
            "source_file": self.source_file,
            "chunk_index": self.chunk_index,
            # Table-specific metadata (stored in Chroma metadata)
            "workbook": self.workbook,
            "sheet": self.sheet,
            "row_start": self.row_start,
            "row_end": self.row_end,
            "column_names": self.column_names,
            "primary_key": self.primary_key,
            "rows_data": self.rows_data,
            "schema_relationships": self.schema_relationships,
            "row_relationships": self.row_relationships,
            "cross_row_relationships": self.cross_row_relationships,
            "document_id": self.document_id,
            "session_id": self.session_id,
            # Compatibility fields
            "chapter": self.chapter,
            "section": self.section,
            "subsection": self.subsection,
            "heading": self.heading,
            "page_start": self.page_start,
            "page_end": self.page_end,
            "paragraph_ids": self.paragraph_ids,
            "previous_chunk": self.previous_chunk,
            "next_chunk": self.next_chunk,
            "parent_section": self.parent_section,
            "estimated_tokens": self.estimated_tokens,
            "character_count": self.character_count,
        }
