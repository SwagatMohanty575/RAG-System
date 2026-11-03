"""
Document hierarchy detection using multiple heuristics.
"""

from __future__ import annotations

import re
from typing import List, Optional, Pattern

from .models import Heading


class HierarchyDetector:
    """
    Detects document hierarchy levels using multiple heuristic patterns.
    
    Supports numbered sections, roman numerals, markdown headings,
    uppercase headings, and chapter markers.
    """
    
    def __init__(self, patterns: List[str]) -> None:
        self._compiled: List[Pattern] = [re.compile(p, re.MULTILINE) for p in patterns]
    
    def detect_heading(self, line: str, page_number: int) -> Optional[Heading]:
        """
        Analyze a single line and return a Heading if it matches any pattern.
        
        Returns None if the line is not recognized as a heading.
        """
        stripped = line.strip()
        if not stripped:
            return None
            
        for level, pattern in enumerate(self._compiled):
            if pattern.match(stripped):
                # Determine semantic level based on pattern characteristics
                semantic_level = self._infer_level(stripped, level)
                return Heading(
                    text=self._clean_heading_text(stripped),
                    level=semantic_level,
                    page_number=page_number,
                    raw_line=stripped,
                )
        return None
    
    def _infer_level(self, text: str, pattern_index: int) -> int:
        """
        Infer hierarchy depth from heading text characteristics.
        
        Levels: 0=document, 1=chapter, 2=section, 3=subsection, 4=topic
        """
        text = text.strip()
        
        # Chapter markers are always level 1
        if re.match(r"^(Chapter|CHAPTER)\s+\d+", text, re.IGNORECASE):
            return 1
        
        # Markdown headings: #=1, ##=2, ###=3, ####+=4
        markdown_match = re.match(r"^(#{1,6})\s+", text)
        if markdown_match:
            hashes = len(markdown_match.group(1))
            return min(hashes, 4)
        
        # Numbered sections: depth by dot count
        numbered_match = re.match(r"^(\d+(?:\.\d+)*)[:\.\s]", text)
        if numbered_match:
            parts = numbered_match.group(1).split(".")
            depth = len(parts)
            return min(depth + 1, 4)  # 1.x -> section(2), 1.1.x -> subsection(3)
        
        # Roman numerals: typically chapters or major sections
        if re.match(r"^[IVXivx]+[:\.\s]", text):
            return 1
        
        # Uppercase short lines: likely topic headers
        if re.match(r"^[A-Z][A-Z\s]{2,50}[A-Z][:\.\s]?$", text):
            return 4
        
        # Default based on pattern registration order
        return min(pattern_index + 1, 4)
    
    def _clean_heading_text(self, text: str) -> str:
        """Remove markdown hashes and normalize heading text."""
        text = re.sub(r"^#{1,6}\s*", "", text)
        text = re.sub(r"^[•\-\*]\s*", "", text)
        return text.strip()
    
    def build_hierarchy_context(
        self,
        headings: List[Heading],
    ) -> dict[str, Optional[str]]:
        """
        Build a running context of current chapter/section/subsection.
        
        Returns a dict with keys: chapter, section, subsection, topic.
        """
        context = {
            "chapter": None,
            "section": None,
            "subsection": None,
            "topic": None,
        }
        
        if not headings:
            return context
            
        # Use the most recent heading at each level
        for h in headings:
            if h.level == 1:
                context["chapter"] = h.text
                context["section"] = None
                context["subsection"] = None
                context["topic"] = None
            elif h.level == 2:
                context["section"] = h.text
                context["subsection"] = None
                context["topic"] = None
            elif h.level == 3:
                context["subsection"] = h.text
                context["topic"] = None
            elif h.level == 4:
                context["topic"] = h.text
                
        return context
    