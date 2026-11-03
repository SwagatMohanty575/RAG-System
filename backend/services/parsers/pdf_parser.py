"""
pdf_parser.py

Enhanced PDF parser with rich metadata extraction:
- Part titles (PART ONE, PART TWO, etc.)
- Chapter numbers and titles
- Character name detection
- Page type classification
"""

import re
import fitz
from pathlib import Path
from typing import List, Dict, Optional, Set

from services.parsers.base import BaseParser


class PDFParser(BaseParser):
    """PDF parser with rich structural and semantic metadata."""

    def __init__(self):
        # Patterns for structural elements
        self.part_pattern = re.compile(r'^#?\s*PART\s+(ONE|TWO|THREE|FOUR|FIVE|THE\s+\w+)', re.IGNORECASE)
        self.chapter_pattern = re.compile(r'^#?\s*CHAPTER\s+(ONE|TWO|THREE|FOUR|FIVE|SIX|SEVEN|EIGHT|NINE|TEN|\d+)', re.IGNORECASE)
        self.dialogue_pattern = re.compile(r'"[^"]{3,200}"')  # Quoted speech
        self.name_pattern = re.compile(r'\b([A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+)?)\b')  # Capitalized names

    def extract(self, file_path: str) -> List[Dict]:
        doc = fitz.open(str(file_path))
        pages = []
        current_part: Optional[str] = None
        current_chapter_num: Optional[str] = None
        current_chapter_title: Optional[str] = None

        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            text = page.get_text("text").strip()

            if not text:
                continue

            # Detect structural elements
            part_title = self._detect_part(text)
            if part_title:
                current_part = part_title

            chapter_info = self._detect_chapter(text)
            if chapter_info:
                current_chapter_num, current_chapter_title = chapter_info
                is_chapter_start = True
                page_type = "chapter_header"
            else:
                is_chapter_start = False
                page_type = self._classify_page_type(text, page_num, len(doc))

            # Detect characters mentioned on this page
            characters = self._extract_character_names(text)

            # Detect dialogue presence
            has_dialogue = bool(self.dialogue_pattern.search(text))

            # Build section title from chapter info
            if current_chapter_num and current_chapter_title:
                section_title = f"{current_chapter_num}: {current_chapter_title}"
            elif current_chapter_num:
                section_title = current_chapter_num
            else:
                section_title = self._detect_section(text) or f"Page {page_num + 1}"

            pages.append({
                # Core content
                "page_number": page_num + 1,
                "text": text,
                "section_title": section_title,
                "source_file": self._source_name(file_path),
                
                # Enhanced structural metadata
                "part_title": current_part,
                "chapter_number": current_chapter_num,
                "chapter_title": current_chapter_title,
                "page_type": page_type,
                "is_chapter_start": is_chapter_start,
                
                # Semantic metadata
                "characters_mentioned": list(characters),
                "has_dialogue": has_dialogue,
                
                # Internal flag for chunker
                "_is_table": False,
            })

        doc.close()
        return pages

    def _detect_part(self, text: str) -> Optional[str]:
        """Detect PART ONE/TWO/THREE headers."""
        lines = text.split('\n')
        for line in lines[:5]:  # Usually at top of page
            match = self.part_pattern.match(line.strip())
            if match:
                return match.group(0).strip()
        return None

    def _detect_chapter(self, text: str) -> Optional[tuple]:
        """Detect CHAPTER X and its title. Returns (chapter_num, chapter_title)."""
        lines = text.split('\n')
        for i, line in enumerate(lines):
            line_stripped = line.strip()
            match = self.chapter_pattern.match(line_stripped)
            if match:
                chapter_num = match.group(0).strip()
                # Title is next non-empty, non-header line
                for j in range(i + 1, min(i + 5, len(lines))):
                    title = lines[j].strip()
                    if title and len(title) > 2 and not title.startswith('#') and not self.chapter_pattern.match(title):
                        return (chapter_num, title)
                return (chapter_num, None)
        return None

    def _classify_page_type(self, text: str, page_num: int, total_pages: int) -> str:
        """Classify the type of page content."""
        if page_num == 0:
            return "title_page"
        if page_num >= total_pages - 1:
            return "epilogue"
        if "THE END" in text:
            return "ending"
        return "chapter_content"

    def _extract_character_names(self, text: str) -> Set[str]:
        """Extract potential character names from text."""
        # Find capitalized word pairs (First Last)
        names = set()
        
        # Common titles that precede names
        title_patterns = [
            r'(?:Mr|Mrs|Ms|Miss|Dr|Inspector|Lady|Lord|Sir)\s+([A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+)?)',
            r'([A-Z][a-zA-Z]+\s+[A-Z][a-zA-Z]+)',  # First Last
        ]
        
        for pattern in title_patterns:
            for match in re.finditer(pattern, text):
                name = match.group(1).strip()
                if len(name) > 3 and name not in self._get_stop_names():
                    names.add(name)
        
        return names

    @staticmethod
    def _get_stop_names() -> Set[str]:
        """Common words that look like names but aren't."""
        return {
            "The", "And", "But", "For", "Not", "Yet", "She", "Her", "His",
            "Him", "They", "Them", "This", "That", "These", "Those", "With",
            "From", "Into", "Through", "During", "Before", "After", "Above",
            "Below", "Between", "Among", "Within", "Without", "Against",
            "About", "Under", "Over", "Again", "Further", "Then", "Once",
            "Here", "There", "When", "Where", "Why", "How", "All", "Any",
            "Both", "Each", "Few", "More", "Most", "Other", "Some", "Such",
            "No", "Nor", "Only", "Own", "Same", "So", "Than", "Too", "Very",
            "Just", "Now", "Also", "Back", "Still", "Even", "Well", "Away",
            "Down", "Off", "On", "Out", "Up", "One", "Two", "Three", "Four",
            "Five", "Six", "Seven", "Eight", "Nine", "Ten", "Elmwood", "London",
        }

    def _detect_section(self, text: str) -> Optional[str]:
        """Fallback section detection."""
        lines = text.split('\n')
        for line in lines[:10]:
            line = line.strip()
            if len(line) > 3 and len(line) < 100:
                if line.isupper() or re.match(r'^\d+[\.\s]', line):
                    return line
        return None