"""
HASPC Pipeline — Main entry point for the chunking engine.
"""

from __future__ import annotations

from typing import List, Dict

from config import (
    MIN_PARAGRAPH_SIZE,
    MAX_PARAGRAPH_SIZE,
    TARGET_CHUNK_TOKENS,
    MAX_CHUNK_TOKENS,
    MIN_CHUNK_TOKENS,
    MIN_OVERLAP,
    SMALL_OVERLAP,
    MAX_OVERLAP,
    OVERLAP_CHUNK_THRESHOLD,
    HEADING_PATTERNS,
    MERGE_SMALL_PARAGRAPHS,
    MERGE_ACROSS_SECTIONS,
)

from .models import Heading, Paragraph, Chunk
from .hierarchy import HierarchyDetector
from .paragraph import ParagraphExtractor
from .semantic import ChunkMerger
from .overlap import SmartOverlap


class HASPCChunker:
    """
    Hierarchical Adaptive Semantic Paragraph Chunker.
    
    Production-grade document chunking that preserves hierarchy,
    semantic continuity, and paragraph integrity.
    
    Compatible with existing embedding pipeline. Output format:
    List[Dict] with fields: text, page_number, section_title,
    source_file, chunk_index, plus enriched metadata.
    """
    
    def __init__(self) -> None:
        self.detector = HierarchyDetector(HEADING_PATTERNS)
        self.extractor = ParagraphExtractor(
            min_size=MIN_PARAGRAPH_SIZE,
            max_size=MAX_PARAGRAPH_SIZE,
        )
        self.merger = ChunkMerger(
            target_tokens=TARGET_CHUNK_TOKENS,
            max_tokens=MAX_CHUNK_TOKENS,
            min_tokens=MIN_CHUNK_TOKENS,
            merge_across_sections=MERGE_ACROSS_SECTIONS,
        )
        self.overlap = SmartOverlap(
            min_overlap=MIN_OVERLAP,
            small_overlap=SMALL_OVERLAP,
            max_overlap=MAX_OVERLAP,
            threshold=OVERLAP_CHUNK_THRESHOLD,
            max_tokens=MAX_CHUNK_TOKENS,
        )
    
    def chunk_pages(self, pages: List[Dict]) -> List[Dict]:
        """
        Process pages through the full HASPC pipeline.
        
        Args:
            pages: List of dicts with keys: text, page_number,
                   section_title, source_file
                   
        Returns:
            List of dicts compatible with existing embedding pipeline.
        """
        # Step 1: Detect headings and build hierarchy context per page
        page_contexts = self._build_page_contexts(pages)
        
        # Step 2: Extract paragraphs with hierarchy metadata
        all_paragraphs: List[Paragraph] = []
        for page, context in zip(pages, page_contexts):
            paras = self.extractor.extract_from_page(
                page_text=page["text"],
                page_number=page["page_number"],
                source_file=page["source_file"],
                heading_context=context,
                page_metadata=self._page_metadata(page),
            )
            all_paragraphs.extend(paras)
        
        if not all_paragraphs:
            return []
        
        # Step 3: Merge paragraphs into chunks
        chunks = self.merger.merge(all_paragraphs)
        
        # Step 4: Apply smart overlap
        chunks = self.overlap.apply(chunks)
        
        # Step 5: Convert to dict format
        return [c.to_dict() for c in chunks]
    
    def _build_page_contexts(
        self,
        pages: List[Dict],
    ) -> List[Dict[str, str]]:
        """
        Build hierarchy context for each page.
        
        Tracks headings across pages to maintain continuity.
        """
        contexts: List[Dict] = []
        current_headings: List[Heading] = []
        
        for page in pages:
            # Detect headings in this page
            lines = page["text"].splitlines()
            page_headings: List[Heading] = []
            
            for line in lines:
                heading = self.detector.detect_heading(
                    line=line,
                    page_number=page["page_number"],
                )
                if heading:
                    page_headings.append(heading)
            
            # Update running heading context
            if page_headings:
                # Remove old headings at same or deeper levels
                for new_h in page_headings:
                    current_headings = [
                        h for h in current_headings 
                        if h.level < new_h.level
                    ]
                    current_headings.append(new_h)
            
            # Build context from current headings
            context = self.detector.build_hierarchy_context(current_headings)
            parser_section = page.get("section_title")
            if page.get("chapter_title") or page.get("chapter_number"):
                chapter_label = " ".join(
                    part for part in [page.get("chapter_number"), page.get("chapter_title")]
                    if part
                )
                context["chapter"] = chapter_label or parser_section
                context["section"] = parser_section or chapter_label
            elif parser_section and context.get("section") is None:
                context["section"] = parser_section
            contexts.append(context)
        
        return contexts

    @staticmethod
    def _page_metadata(page: Dict) -> Dict:
        keys = (
            "part_title",
            "chapter_number",
            "chapter_title",
            "page_type",
            "is_chapter_start",
            "characters_mentioned",
            "has_dialogue",
        )
        return {key: page.get(key) for key in keys if key in page}
