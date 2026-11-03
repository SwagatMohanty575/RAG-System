"""
pptx_parser.py

Handles .pptx files using python-pptx. Each slide becomes one "page".
The slide title (if present) becomes the section_title. Text from all
shapes on the slide is concatenated as the page text. Empty slides
(no extractable text) are skipped.
"""

from typing import List, Dict, Optional
from pptx import Presentation


from services.parsers.base import BaseParser


class PPTXParser(BaseParser):
    def extract(self, file_path: str) -> List[Dict]:
        prs = Presentation(str(file_path))
        source = self._source_name(file_path)
        pages = []

        for slide_num, slide in enumerate(prs.slides, start=1):
            title: Optional[str] = None
            text_parts: List[str] = []

            for shape in slide.shapes:
                if not shape.has_text_frame:
                    continue
                shape_text = shape.text_frame.text.strip()
                if not shape_text:
                    continue
                # treat the first shape with the slide's title placeholder
                # as the section heading; everything else is body text
                if shape.shape_id == 1 or (
                    hasattr(shape, "placeholder_format")
                    and shape.placeholder_format is not None
                    and shape.placeholder_format.idx == 0
                ):
                    title = shape_text
                else:
                    text_parts.append(shape_text)

            # If no body text but there's a title, still include it
            full_text = "\n".join(filter(None, [title] + text_parts)).strip()
            if not full_text:
                continue

            pages.append({
                "page_number": slide_num,
                "text": full_text,
                "section_title": title or f"Slide {slide_num}",
                "source_file": source,
            })

        return pages