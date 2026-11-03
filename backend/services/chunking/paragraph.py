"""
Paragraph extraction and semantic splitting.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

from .models import Paragraph


class ParagraphExtractor:
    """
    Extracts paragraphs from page text while preserving structure.

    Paragraphs are the primary chunking unit. Large paragraphs are split
    at natural semantic boundaries when possible, with a word-boundary
    fallback so oversized text cannot pass through unchanged.
    """

    def __init__(
        self,
        min_size: int,
        max_size: int,
    ) -> None:
        self.min_size = min_size
        self.max_size = max_size
        self._split_patterns = [
            (r"(?<=[.!?])\s+(?=[A-Z])", "sentence"),
            (r"(?<=[.!?])\s*[\n\r]+(?=\s*[\-\*\u2022]\s)", "bullet"),
            (r"(?<=[.!?])\s*[\n\r]+(?=\s*\d+[\.)\]]\s)", "numbered"),
            (r"\n\s*\n", "paragraph"),
            (r"(?<=[;:,])\s+", "clause"),
        ]

    def extract_from_page(
        self,
        page_text: str,
        page_number: int,
        source_file: str,
        heading_context: dict,
        page_metadata: Optional[Dict[str, Any]] = None,
    ) -> List[Paragraph]:
        raw_blocks = re.split(r"\n\s*\n", page_text.strip())
        metadata = page_metadata or {}

        paragraphs: List[Paragraph] = []
        for idx, block in enumerate(raw_blocks):
            block = block.strip()
            if len(block) < self.min_size:
                continue

            para = self._create_paragraph(
                text=block,
                page_number=page_number,
                source_file=source_file,
                heading_context=heading_context,
                page_metadata=metadata,
                index=idx,
            )

            if para.character_count > self.max_size:
                paragraphs.extend(self._semantic_split(para))
            else:
                paragraphs.append(para)

        return paragraphs

    def _create_paragraph(
        self,
        text: str,
        page_number: int,
        source_file: str,
        heading_context: dict,
        page_metadata: Dict[str, Any],
        index: int,
    ) -> Paragraph:
        return Paragraph(
            text=text,
            page_number=page_number,
            chapter=heading_context.get("chapter"),
            section=heading_context.get("section"),
            subsection=heading_context.get("subsection"),
            source_file=source_file,
            paragraph_id=f"{source_file}_p{page_number}_para{index}",
            part_title=page_metadata.get("part_title"),
            chapter_number=page_metadata.get("chapter_number"),
            chapter_title=page_metadata.get("chapter_title"),
            page_type=page_metadata.get("page_type", "chapter_content"),
            is_chapter_start=bool(page_metadata.get("is_chapter_start", False)),
            characters_mentioned=list(page_metadata.get("characters_mentioned") or []),
            has_dialogue=bool(page_metadata.get("has_dialogue", False)),
        )

    def _semantic_split(self, paragraph: Paragraph) -> List[Paragraph]:
        text = paragraph.text
        splits: List[tuple[int, str]] = []

        for pattern, split_type in self._split_patterns:
            for match in re.finditer(pattern, text):
                pos = match.end()
                if 0 < pos < len(text):
                    splits.append((pos, split_type))

        if not splits:
            return self._hard_split(paragraph)

        splits.sort(key=lambda x: x[0])

        pieces: List[str] = []
        last_cut = 0

        for pos, _ in splits:
            segment = text[last_cut:pos].strip()
            if len(segment) >= self.min_size:
                pieces.append(segment)
                last_cut = pos

        remainder = text[last_cut:].strip()
        if len(remainder) >= self.min_size:
            pieces.append(remainder)

        if not pieces:
            return self._hard_split(paragraph)

        result: List[Paragraph] = []
        for idx, piece in enumerate(pieces):
            if len(piece) > self.max_size:
                result.extend(
                    self._hard_split(
                        self._clone_with_text(
                            paragraph=paragraph,
                            text=piece,
                            paragraph_id=f"{paragraph.paragraph_id}_s{idx}",
                        )
                    )
                )
            else:
                result.append(
                    self._clone_with_text(
                        paragraph=paragraph,
                        text=piece,
                        paragraph_id=f"{paragraph.paragraph_id}_s{idx}",
                    )
                )

        return result

    def _hard_split(self, paragraph: Paragraph) -> List[Paragraph]:
        text = paragraph.text.strip()
        if len(text) <= self.max_size:
            return [paragraph]

        pieces: List[str] = []
        start = 0
        while start < len(text):
            end = min(start + self.max_size, len(text))
            if end < len(text):
                word_boundary = text.rfind(" ", start, end)
                if word_boundary > start + self.min_size:
                    end = word_boundary

            piece = text[start:end].strip()
            if piece:
                pieces.append(piece)
            start = max(end, start + 1)

        return [
            self._clone_with_text(
                paragraph=paragraph,
                text=piece,
                paragraph_id=f"{paragraph.paragraph_id}_h{idx}",
            )
            for idx, piece in enumerate(pieces)
            if len(piece) >= self.min_size
        ]

    def _clone_with_text(self, paragraph: Paragraph, text: str, paragraph_id: str) -> Paragraph:
        return Paragraph(
            text=text,
            page_number=paragraph.page_number,
            heading=paragraph.heading,
            chapter=paragraph.chapter,
            section=paragraph.section,
            subsection=paragraph.subsection,
            source_file=paragraph.source_file,
            paragraph_id=paragraph_id,
            part_title=paragraph.part_title,
            chapter_number=paragraph.chapter_number,
            chapter_title=paragraph.chapter_title,
            page_type=paragraph.page_type,
            is_chapter_start=paragraph.is_chapter_start,
            characters_mentioned=list(paragraph.characters_mentioned),
            has_dialogue=paragraph.has_dialogue,
        )
