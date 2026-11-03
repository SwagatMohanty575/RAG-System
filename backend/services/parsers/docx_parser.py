"""
docx_parser.py

Handles .docx and .doc files using python-docx. Each heading in the
Word document becomes a section boundary — content between two headings
is grouped as one "page" so the chunker sees logical sections rather
than arbitrary page breaks.
"""

from typing import List, Dict, Optional
from docx import Document

from services.parsers.base import BaseParser


class DOCXParser(BaseParser):
    def extract(self, file_path: str) -> List[Dict]:
        doc = Document(str(file_path))
        source = self._source_name(file_path)
        pages = []

        current_section: Optional[str] = None
        current_lines: List[str] = []
        page_number = 1

        def flush():
            nonlocal page_number
            text = "\n".join(current_lines).strip()
            if text:
                pages.append({
                    "page_number": page_number,
                    "text": text,
                    "section_title": current_section or f"Section {page_number}",
                    "source_file": source,
                })
                page_number += 1
            current_lines.clear()

        for para in doc.paragraphs:
            style_name = para.style.name.lower() if para.style else ""
            text = para.text.strip()

            if not text:
                continue

            if "heading" in style_name:
                flush()
                current_section = text
            else:
                current_lines.append(text)

        flush()
        return pages