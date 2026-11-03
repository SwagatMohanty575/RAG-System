"""
Dynamic chunk merging based on semantic coherence and token targets.
"""

from __future__ import annotations

from typing import List, Optional

from .models import Paragraph, Chunk


class ChunkMerger:
    """
    Merges consecutive paragraphs into chunks until target token size.
    
    Respects hierarchy boundaries:
    - Never merges across chapter boundaries
    - Prefers not to merge across section boundaries
    - Small paragraphs may be merged; large ones stay independent
    """
    
    def __init__(
        self,
        target_tokens: int,
        max_tokens: int,
        min_tokens: int,
        merge_across_sections: bool,
    ) -> None:
        self.target_tokens = target_tokens
        self.max_tokens = max_tokens
        self.min_tokens = min_tokens
        self.merge_across_sections = merge_across_sections
    
    def merge(self, paragraphs: List[Paragraph]) -> List[Chunk]:
        """
        Merge paragraphs into optimally-sized chunks.
        
        Algorithm:
        1. Group paragraphs by chapter (hard boundary)
        2. Within chapter, greedily merge until target_tokens
        3. Respect section boundaries unless paragraph is tiny
        4. Ensure no chunk exceeds max_tokens
        """
        if not paragraphs:
            return []
        
        # Group by chapter
        chapter_groups = self._group_by_chapter(paragraphs)
        
        chunks: List[Chunk] = []
        global_index = 0
        
        for chapter_paras in chapter_groups:
            chapter_chunks = self._merge_within_chapter(chapter_paras)
            
            for chunk in chapter_chunks:
                chunk.chunk_index = global_index
                chunks.append(chunk)
                global_index += 1
        
        # Link neighbors
        self._link_neighbors(chunks)
        
        return chunks
    
    def _group_by_chapter(
        self,
        paragraphs: List[Paragraph],
    ) -> List[List[Paragraph]]:
        """Group paragraphs by chapter boundary."""
        groups: List[List[Paragraph]] = []
        current: List[Paragraph] = []
        current_chapter: Optional[str] = None
        
        for para in paragraphs:
            if para.chapter != current_chapter and current:
                groups.append(current)
                current = []
            current_chapter = para.chapter
            current.append(para)
        
        if current:
            groups.append(current)
            
        return groups
    
    def _merge_within_chapter(
        self,
        paragraphs: List[Paragraph],
    ) -> List[Chunk]:
        """Merge paragraphs within a single chapter."""
        if not paragraphs:
            return []
        
        chunks: List[Chunk] = []
        buffer: List[Paragraph] = []
        buffer_tokens = 0
        
        for para in paragraphs:
            para_tokens = para.estimated_tokens
            
            # Check if adding this paragraph would exceed max
            would_exceed = (buffer_tokens + para_tokens) > self.max_tokens
            
            # Check section boundary (soft boundary)
            section_boundary = False
            if buffer and not self.merge_across_sections:
                last = buffer[-1]
                if (last.section != para.section or 
                    last.subsection != para.subsection):
                    section_boundary = True
            
            # Start new chunk if:
            # 1. Would exceed max_tokens, OR
            # 2. Hit section boundary AND buffer already has decent size
            if would_exceed or (section_boundary and buffer_tokens >= self.target_tokens // 2):
                if buffer:
                    chunks.append(self._build_chunk(buffer))
                buffer = [para]
                buffer_tokens = para_tokens
            else:
                buffer.append(para)
                buffer_tokens += para_tokens
        
        # Flush remaining buffer
        if buffer:
            # If too small and we have previous chunk, try to merge back
            if (buffer_tokens < self.min_tokens and chunks and 
                self._can_merge(chunks[-1], buffer)):
                chunks[-1] = self._extend_chunk(chunks[-1], buffer)
            else:
                chunks.append(self._build_chunk(buffer))
        
        return chunks
        
    def _build_chunk(self, paragraphs: List[Paragraph]) -> Chunk:
        """Build a Chunk from a list of paragraphs."""
        combined_text = "\n\n".join(p.text for p in paragraphs)
        
        # Determine parent section
        parent = None
        for p in reversed(paragraphs):
            if p.subsection:
                parent = p.subsection
                break
            elif p.section:
                parent = p.section
                break
        
        # Determine heading
        heading = None
        for p in paragraphs:
            if p.heading:
                heading = p.heading.text if hasattr(p.heading, "text") else str(p.heading)
                break
        
        page_numbers = [p.page_number for p in paragraphs]
        
        # Aggregate characters_mentioned across all paragraphs
        all_chars = []
        for p in paragraphs:
            chars = getattr(p, "characters_mentioned", [])
            if chars:
                all_chars.extend(chars)
        
        chunk_text = self._with_context_header(combined_text, paragraphs[0], all_chars)

        return Chunk(
            text=chunk_text,
            page_number=page_numbers[0],
            section_title=paragraphs[0].section or paragraphs[0].chapter or "Unknown",
            source_file=paragraphs[0].source_file,
            chunk_index=0,  # Set later
            document_id=paragraphs[0].source_file,
            chapter=paragraphs[0].chapter,
            section=paragraphs[0].section,
            subsection=paragraphs[0].subsection,
            heading=heading,
            page_start=min(page_numbers),
            page_end=max(page_numbers),
            paragraph_ids=[p.paragraph_id for p in paragraphs],
            parent_section=parent,
            estimated_tokens=len(chunk_text) // 4,
            character_count=len(chunk_text),
            # NEW: Pass through enhanced metadata
            part_title=getattr(paragraphs[0], "part_title", None),
            chapter_number=getattr(paragraphs[0], "chapter_number", None),
            chapter_title=getattr(paragraphs[0], "chapter_title", None),
            page_type=getattr(paragraphs[0], "page_type", "chapter_content"),
            is_chapter_start=getattr(paragraphs[0], "is_chapter_start", False),
            characters_mentioned=list(set(all_chars)),
            has_dialogue=any(getattr(p, "has_dialogue", False) for p in paragraphs),
        )
    
    def _can_merge(self, chunk: Chunk, paragraphs: List[Paragraph]) -> bool:
        """Check if paragraphs can be merged into existing chunk."""
        added_tokens = sum(p.estimated_tokens for p in paragraphs)
        return (chunk.estimated_tokens + added_tokens) <= self.max_tokens
    
    def _extend_chunk(
        self,
        chunk: Chunk,
        paragraphs: List[Paragraph],
    ) -> Chunk:
        """Extend an existing chunk with additional paragraphs."""
        all_paras = chunk.paragraph_ids + [p.paragraph_id for p in paragraphs]
        raw_text = self._strip_context_header(chunk.text) + "\n\n" + "\n\n".join(p.text for p in paragraphs)
        
        # Merge characters_mentioned from existing chunk + new paragraphs
        existing_chars = getattr(chunk, "characters_mentioned", []) or []
        new_chars = []
        for p in paragraphs:
            chars = getattr(p, "characters_mentioned", [])
            if chars:
                new_chars.extend(chars)
        merged_chars = list(set(existing_chars + new_chars))
        
        chunk_text = self._with_context_header(raw_text, paragraphs[0], merged_chars, existing_chunk=chunk)

        return Chunk(
            text=chunk_text,
            page_number=chunk.page_number,
            section_title=chunk.section_title,
            source_file=chunk.source_file,
            chunk_index=chunk.chunk_index,
            document_id=chunk.document_id,
            chapter=chunk.chapter,
            section=chunk.section,
            subsection=chunk.subsection,
            heading=chunk.heading,
            page_start=min(chunk.page_start, min(p.page_number for p in paragraphs)),
            page_end=max(chunk.page_end, max(p.page_number for p in paragraphs)),
            paragraph_ids=all_paras,
            previous_chunk=chunk.previous_chunk,
            next_chunk=chunk.next_chunk,
            parent_section=chunk.parent_section,
            estimated_tokens=len(chunk_text) // 4,
            character_count=len(chunk_text),
            # NEW: Preserve and merge enhanced metadata
            part_title=chunk.part_title,
            chapter_number=chunk.chapter_number,
            chapter_title=chunk.chapter_title,
            page_type=chunk.page_type,
            is_chapter_start=chunk.is_chapter_start,
            characters_mentioned=merged_chars,
            has_dialogue=chunk.has_dialogue or any(getattr(p, "has_dialogue", False) for p in paragraphs),
        )

    def _with_context_header(
        self,
        text: str,
        first_paragraph: Paragraph,
        characters: List[str],
        existing_chunk: Optional[Chunk] = None,
    ) -> str:
        """Add concise structure metadata to text before embedding/storage."""
        part_title = existing_chunk.part_title if existing_chunk else first_paragraph.part_title
        chapter_number = existing_chunk.chapter_number if existing_chunk else first_paragraph.chapter_number
        chapter_title = existing_chunk.chapter_title if existing_chunk else first_paragraph.chapter_title
        page_type = existing_chunk.page_type if existing_chunk else first_paragraph.page_type

        header_parts = []
        if part_title:
            header_parts.append(f"Part: {part_title}")
        if chapter_number or chapter_title:
            chapter = " ".join(part for part in [chapter_number, chapter_title] if part)
            header_parts.append(f"Chapter: {chapter}")
            if chapter_number:
                header_parts.append(f"Chapter number: {chapter_number}")
            if chapter_title:
                header_parts.append(f"Chapter title: {chapter_title}")
        elif first_paragraph.section or first_paragraph.chapter:
            header_parts.append(f"Section: {first_paragraph.section or first_paragraph.chapter}")
        if page_type:
            header_parts.append(f"Page type: {page_type}")
        if characters:
            header_parts.append(f"Characters mentioned: {', '.join(sorted(set(characters))[:12])}")

        if not header_parts:
            return text

        return "Document structure:\n" + "\n".join(header_parts) + "\n\n" + text

    @staticmethod
    def _strip_context_header(text: str) -> str:
        marker = "\n\n"
        if text.startswith("Document structure:") and marker in text:
            return text.split(marker, 1)[1]
        return text

    def _link_neighbors(self, chunks: List[Chunk]) -> None:
        """Set previous_chunk and next_chunk IDs."""
        for i, chunk in enumerate(chunks):
            if i > 0:
                chunk.previous_chunk = chunks[i - 1].paragraph_ids[0] if chunks[i - 1].paragraph_ids else None
            if i < len(chunks) - 1:
                chunk.next_chunk = chunks[i + 1].paragraph_ids[0] if chunks[i + 1].paragraph_ids else None
