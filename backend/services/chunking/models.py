"""
Data models for the HASPC chunking pipeline.
Enhanced with rich metadata for better retrieval.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any


@dataclass
class Heading:
    """Represents a detected document heading with hierarchy level."""
    text: str
    level: int
    page_number: int
    raw_line: str


@dataclass
class Paragraph:
    """A single paragraph unit extracted from a page."""
    text: str
    page_number: int
    heading: Optional[Heading] = None
    chapter: Optional[str] = None
    section: Optional[str] = None
    subsection: Optional[str] = None
    source_file: str = ""
    paragraph_id: str = ""
    estimated_tokens: int = 0
    character_count: int = 0
    # NEW: Enhanced metadata fields for downstream chunk enrichment
    part_title: Optional[str] = None
    chapter_number: Optional[str] = None
    chapter_title: Optional[str] = None
    page_type: str = "chapter_content"
    is_chapter_start: bool = False
    characters_mentioned: List[str] = field(default_factory=list)
    has_dialogue: bool = False

    def __post_init__(self) -> None:
        if not self.character_count:
            self.character_count = len(self.text)
        if not self.estimated_tokens:
            self.estimated_tokens = self.character_count // 4


@dataclass
class Chunk:
    """Final chunk ready for embedding and vector storage."""
    text: str
    page_number: int
    section_title: str
    source_file: str
    chunk_index: int
    
    # HASPC enriched metadata
    document_id: str = ""
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
    
    # NEW: Enhanced structural metadata
    part_title: Optional[str] = None
    chapter_number: Optional[str] = None
    chapter_title: Optional[str] = None
    page_type: str = "chapter_content"
    is_chapter_start: bool = False
    characters_mentioned: List[str] = field(default_factory=list)
    has_dialogue: bool = False

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dict compatible with embedding pipeline."""
        return {
            # Required existing fields
            "text": self.text,
            "page_number": self.page_number,
            "section_title": self.section_title,
            "source_file": self.source_file,
            "chunk_index": self.chunk_index,
            # Enriched metadata
            "document_id": self.document_id,
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
            # NEW enhanced metadata
            "part_title": self.part_title,
            "chapter_number": self.chapter_number,
            "chapter_title": self.chapter_title,
            "page_type": self.page_type,
            "is_chapter_start": self.is_chapter_start,
            "characters_mentioned": self.characters_mentioned,
            "has_dialogue": self.has_dialogue,
        }