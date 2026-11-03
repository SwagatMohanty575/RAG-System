"""
Smart overlap application based on chunk size and content type.
"""

from __future__ import annotations

import re
from typing import List, Optional

from .models import Chunk


class SmartOverlap:
    """
    Applies context overlap between consecutive chunks.
    
    Rules:
    - Tiny chunks (< threshold): 0 overlap
    - Medium chunks: small overlap
    - Large chunks: larger overlap
    - Never crosses chapter boundaries
    """
    
    def __init__(
        self,
        min_overlap: int,
        small_overlap: int,
        max_overlap: int,
        threshold: int,
        max_tokens: Optional[int] = None,
    ) -> None:
        self.min_overlap = min_overlap
        self.small_overlap = small_overlap
        self.max_overlap = max_overlap
        self.threshold = threshold
        self.max_tokens = max_tokens
    
    def apply(self, chunks: List[Chunk]) -> List[Chunk]:
        """
        Apply smart overlap to a list of chunks.
        
        Overlap text is prepended from the previous chunk's end,
        truncated at sentence boundaries when possible.
        """
        if len(chunks) <= 1:
            return chunks
        
        result: List[Chunk] = [chunks[0]]
        
        for i in range(1, len(chunks)):
            prev_chunk = result[-1]
            current = chunks[i]
            
            # Skip overlap if chapter boundary
            if prev_chunk.chapter != current.chapter:
                result.append(current)
                continue
            
            overlap_size = self._determine_overlap_size(current)
            
            if overlap_size > 0:
                if self.max_tokens is not None:
                    remaining_tokens = self.max_tokens - current.estimated_tokens
                    overlap_size = min(overlap_size, max(0, remaining_tokens * 4))

                overlap_text = self._extract_overlap(prev_chunk.text, overlap_size)
                if overlap_text:
                    current.text = overlap_text + "\n\n---\n\n" + current.text
                    current.character_count = len(current.text)
                    current.estimated_tokens = current.character_count // 4
            
            result.append(current)
        
        return result
    
    def _determine_overlap_size(self, chunk: Chunk) -> int:
        """Determine overlap size based on chunk characteristics."""
        if chunk.character_count < self.threshold:
            return self.min_overlap  # 0
        
        if chunk.character_count < self.threshold * 2:
            return self.small_overlap
        
        return self.max_overlap
    
    def _extract_overlap(self, text: str, max_chars: int) -> str:
        """
        Extract overlap text from the end of a chunk.
        
        Tries to break at sentence boundary. Falls back to
        clause boundary, then word boundary.
        """
        if len(text) <= max_chars:
            return text
        
        # Start from the end, look back max_chars
        start_idx = len(text) - max_chars
        
        # Try sentence boundary
        sentence_match = re.search(r"[.!?]\s+", text[start_idx:])
        if sentence_match:
            pos = start_idx + sentence_match.end()
            return text[pos:].strip()
        
        # Try clause boundary
        clause_match = re.search(r"[;,]\s+", text[start_idx:])
        if clause_match:
            pos = start_idx + clause_match.end()
            return text[pos:].strip()
        
        # Fallback: word boundary
        word_match = re.search(r"\s+", text[start_idx:])
        if word_match:
            pos = start_idx + word_match.end()
            return text[pos:].strip()
        
        # Last resort: hard cut
        return text[start_idx:].strip()
