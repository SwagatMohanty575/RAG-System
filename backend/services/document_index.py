"""
Deterministic document-structure index.

This index is NOT vector-based.

It is used for questions such as:

- How many chapters are there?
- What are the chapter names?
- What is Chapter 4 about structurally?
- Where does Chapter 6 begin?

These are document-structure questions and should not depend
on semantic vector similarity.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Dict, List, Optional


WORD_NUMBERS = {
    "ONE": 1,
    "TWO": 2,
    "THREE": 3,
    "FOUR": 4,
    "FIVE": 5,
    "SIX": 6,
    "SEVEN": 7,
    "EIGHT": 8,
    "NINE": 9,
    "TEN": 10,
    "ELEVEN": 11,
    "TWELVE": 12,
    "THIRTEEN": 13,
    "FOURTEEN": 14,
    "FIFTEEN": 15,
    "SIXTEEN": 16,
    "SEVENTEEN": 17,
    "EIGHTEEN": 18,
    "NINETEEN": 19,
    "TWENTY": 20,
}


def chapter_number_from_label(
    label: Optional[str],
) -> Optional[int]:

    if not label:
        return None

    match = re.search(
        r"CHAPTER\s+([A-Z]+|\d+)",
        str(label),
        re.IGNORECASE,
    )

    if not match:
        return None

    value = match.group(1).upper()

    if value.isdigit():
        return int(value)

    return WORD_NUMBERS.get(value)


class DocumentIndex:

    def __init__(
        self,
        path: str | Path,
    ) -> None:

        self.path = Path(path)

        self.data: Dict = {}

    # ---------------------------------------------------------
    # BUILD
    # ---------------------------------------------------------

    @classmethod
    def build(
        cls,
        pages: List[Dict],
        path: str | Path,
        source_file: str = "",
    ) -> "DocumentIndex":

        index = cls(path)

        chapters: Dict[int, Dict] = {}

        current_chapter: Optional[int] = None

        epilogue_pages: List[int] = []

        total_pages = max(
            (
                int(
                    page.get(
                        "page_number",
                        0,
                    )
                )
                for page in pages
            ),
            default=0,
        )

        for page in pages:

            page_number = int(
                page.get(
                    "page_number",
                    0,
                )
            )

            text = str(
                page.get(
                    "text",
                    "",
                )
            )

            chapter_number = page.get(
                "chapter_number"
            )

            chapter_title = page.get(
                "chapter_title"
            )

            numeric_chapter = (
                chapter_number_from_label(
                    chapter_number
                )
            )

            # -------------------------------------------------
            # Chapter detected
            # -------------------------------------------------

            if numeric_chapter is not None:

                current_chapter = numeric_chapter

                chapter = chapters.setdefault(
                    numeric_chapter,
                    {
                        "number": numeric_chapter,
                        "label": (
                            chapter_number
                            or f"CHAPTER {numeric_chapter}"
                        ),
                        "title": (
                            chapter_title
                            or ""
                        ),
                        "part": page.get(
                            "part_title"
                        ),
                        "start_page": page_number,
                        "end_page": page_number,
                        "page_count": 0,
                    },
                )

                chapter["start_page"] = min(
                    chapter["start_page"],
                    page_number,
                )

                chapter["end_page"] = max(
                    chapter["end_page"],
                    page_number,
                )

                if (
                    chapter_title
                    and not chapter["title"]
                ):
                    chapter["title"] = (
                        chapter_title
                    )

                if page.get("part_title"):
                    chapter["part"] = page[
                        "part_title"
                    ]

            # -------------------------------------------------
            # Detect epilogue
            # -------------------------------------------------

            is_epilogue = (
                "EPILOGUE"
                in text[:500].upper()
                or
                str(
                    page.get(
                        "page_type",
                        "",
                    )
                ).lower()
                == "epilogue"
            )

            if is_epilogue:
                epilogue_pages.append(
                    page_number
                )

            # -------------------------------------------------
            # Extend current chapter range
            # -------------------------------------------------

            if (
                current_chapter is not None
                and numeric_chapter is None
                and not is_epilogue
            ):

                chapter = chapters.get(
                    current_chapter
                )

                if chapter:

                    chapter["end_page"] = max(
                        chapter["end_page"],
                        page_number,
                    )

        # -----------------------------------------------------
        # Page counts
        # -----------------------------------------------------

        for chapter in chapters.values():

            chapter["page_count"] = (
                chapter["end_page"]
                - chapter["start_page"]
                + 1
            )

        ordered_chapters = [
            chapters[number]
            for number in sorted(chapters)
        ]

        # -----------------------------------------------------
        # Build parts
        # -----------------------------------------------------

        parts = {}

        for chapter in ordered_chapters:

            part_name = (
                chapter.get("part")
                or "Unassigned"
            )

            part = parts.setdefault(
                part_name,
                {
                    "title": part_name,
                    "chapters": [],
                },
            )

            part["chapters"].append(
                chapter
            )

        index.data = {
            "source_file": source_file,
            "total_pages": total_pages,
            "chapter_count": len(
                ordered_chapters
            ),
            "chapters": ordered_chapters,
            "parts": list(
                parts.values()
            ),
            "epilogue_pages": sorted(
                set(epilogue_pages)
            ),
        }

        index.save()

        return index

    # ---------------------------------------------------------
    # LOAD
    # ---------------------------------------------------------

    @classmethod
    def load(
        cls,
        path: str | Path,
    ) -> "DocumentIndex":

        index = cls(path)

        with open(
            index.path,
            "r",
            encoding="utf-8",
        ) as file:

            index.data = json.load(file)

        return index

    # ---------------------------------------------------------
    # SAVE
    # ---------------------------------------------------------

    def save(self) -> None:

        self.path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        with open(
            self.path,
            "w",
            encoding="utf-8",
        ) as file:

            json.dump(
                self.data,
                file,
                indent=2,
                ensure_ascii=False,
            )

    # ---------------------------------------------------------
    # PROPERTIES
    # ---------------------------------------------------------

    @property
    def chapter_count(self) -> int:

        return int(
            self.data.get(
                "chapter_count",
                0,
            )
        )

    @property
    def chapters(self) -> List[Dict]:

        return list(
            self.data.get(
                "chapters",
                [],
            )
        )

    # ---------------------------------------------------------
    # FIND CHAPTER
    # ---------------------------------------------------------

    def find_chapter(
        self,
        number: int,
    ) -> Optional[Dict]:

        for chapter in self.chapters:

            if chapter["number"] == number:
                return chapter

        return None

    # ---------------------------------------------------------
    # STRUCTURE ANSWER
    # ---------------------------------------------------------

    def answer_structure(
        self,
        query: str,
    ) -> str:

        q = query.lower()

        # ---------------------------------------------
        # How many chapters?
        # ---------------------------------------------

        if (
            re.search(
                r"\bhow many\b.*\bchapter",
                q,
            )
            or
            "number of chapters" in q
        ):

            if self.data.get(
                "epilogue_pages"
            ):

                return (
                    f"The document contains "
                    f"{self.chapter_count} chapters "
                    f"followed by an epilogue."
                )

            return (
                f"The document contains "
                f"{self.chapter_count} chapters."
            )

        # ---------------------------------------------
        # Specific chapter
        # ---------------------------------------------

        match = re.search(
            r"\bchapter\s+(\d+)\b",
            q,
        )

        if (
            match
            and any(
                word in q
                for word in (
                    "title",
                    "name",
                    "where",
                    "begin",
                    "start",
                )
            )
        ):

            number = int(
                match.group(1)
            )

            chapter = self.find_chapter(
                number
            )

            if chapter:

                answer = (
                    f"Chapter {number} "
                    f"is titled "
                    f"\"{chapter['title']}\" "
                    f"and begins on page "
                    f"{chapter['start_page']}."
                )

                if chapter.get("part"):

                    answer += (
                        f" It is part of "
                        f"{chapter['part']}."
                    )

                return answer

        # ---------------------------------------------
        # List chapters
        # ---------------------------------------------

        if any(
            word in q
            for word in (
                "list",
                "names",
                "titles",
                "what are",
                "chapters",
            )
        ):

            lines = [
                "The document has these chapters:"
            ]

            for chapter in self.chapters:

                lines.append(
                    f"{chapter['number']}. "
                    f"{chapter['title']} "
                    f"(starts on page "
                    f"{chapter['start_page']})"
                )

            if self.data.get(
                "epilogue_pages"
            ):

                lines.append(
                    "Epilogue: present."
                )

            return "\n".join(lines)

        return self.to_outline()

    # ---------------------------------------------------------
    # OUTLINE
    # ---------------------------------------------------------

    def to_outline(self) -> str:

        lines = [
            (
                "Document: "
                + str(
                    self.data.get(
                        "source_file",
                        "uploaded document",
                    )
                )
            ),
            (
                "Total pages: "
                + str(
                    self.data.get(
                        "total_pages",
                        0,
                    )
                )
            ),
            (
                "Chapters: "
                + str(
                    self.chapter_count
                )
            ),
        ]

        for chapter in self.chapters:

            lines.append(
                f"Chapter "
                f"{chapter['number']}: "
                f"{chapter['title']} "
                f"(pages "
                f"{chapter['start_page']}-"
                f"{chapter['end_page']})"
            )

        if self.data.get(
            "epilogue_pages"
        ):

            lines.append(
                "Epilogue: present."
            )

        return "\n".join(lines)