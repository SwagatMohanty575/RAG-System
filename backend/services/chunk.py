"""
Chunking module — document-type-aware router.

Routes documents to the correct chunking engine:
- PDF, DOCX, PPTX → HASPC (paragraph-based semantic chunking)
- Excel, CSV → TableChunker (row-preserving structured chunking)

Maintains backward compatibility with main.py.
"""

from typing import List, Dict
from pathlib import Path

from services.chunking import HASPCChunker, TableChunker


class SectionAwareChunker:
    """
    Document-type-aware chunking router.
    
    Automatically detects whether the input is table data or prose
    and dispatches to the appropriate chunking engine.
    """
    
    def __init__(self) -> None:
        self._haspc = HASPCChunker()
        self._table = TableChunker()
    
    def chunk_pages(self, pages: List[Dict]) -> List[Dict]:
        """
        Chunk pages using the appropriate engine.
        
        Args:
            pages: List of page dicts from ingest.py/parsers.
                   Table pages have _is_table=True flag.
            
        Returns:
            List of chunk dicts compatible with embedding pipeline.
        """
        if not pages:
            return []
        
        # Check if this is table data
        is_table = any(page.get("_is_table") for page in pages)
        
        if is_table:
            return self._table.chunk_pages(pages)
        else:
            return self._haspc.chunk_pages(pages)