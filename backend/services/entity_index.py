"""
Per-session character/entity index.

The index is created from parser metadata.

It does NOT replace document evidence.
For final answers, supporting document chunks are still retrieved.
"""

from __future__ import annotations

import json
import re

from collections import defaultdict
from pathlib import Path
from typing import Dict, List


class EntityIndex:

    def __init__(
        self,
        path: str | Path,
    ) -> None:

        self.path = Path(path)

        self.data: Dict = {
            "entities": {},
            "aliases": {},
        }

    # ---------------------------------------------------------
    # BUILD
    # ---------------------------------------------------------

    @classmethod
    def build(
        cls,
        pages: List[Dict],
        path: str | Path,
        source_file: str = "",
    ) -> "EntityIndex":

        index = cls(path)

        entities = defaultdict(
            lambda: {
                "name": "",
                "mentions": 0,
                "pages": [],
                "chapters": [],
            }
        )

        for page in pages:

            page_number = int(
                page.get(
                    "page_number",
                    0,
                )
            )

            chapter = page.get(
                "chapter_number"
            )

            names = page.get(
                "characters_mentioned",
                [],
            ) or []

            for raw_name in names:

                name = re.sub(
                    r"\s+",
                    " ",
                    str(raw_name),
                ).strip()

                if not name:
                    continue

                if len(name) < 3:
                    continue

                item = entities[name]

                item["name"] = name

                item["mentions"] += 1

                item["pages"].append(
                    page_number
                )

                if chapter:

                    item["chapters"].append(
                        chapter
                    )

        # Keep entities that appear repeatedly
        # or are already full names.
        filtered = {}

        for name, item in entities.items():

            if (
                item["mentions"] >= 2
                or len(name.split()) >= 2
            ):

                item["pages"] = sorted(
                    set(item["pages"])
                )

                item["chapters"] = sorted(
                    set(item["chapters"])
                )

                filtered[name] = item

        # -----------------------------------------------------
        # Alias mapping
        # -----------------------------------------------------

        aliases = {}

        ambiguous = set()

        for name in filtered:

            parts = name.split()

            candidates = {
                name.lower()
            }

            if len(parts) >= 2:

                candidates.add(
                    parts[0].lower()
                )

                candidates.add(
                    parts[-1].lower()
                )

            for alias in candidates:

                if (
                    alias in aliases
                    and aliases[alias] != name
                ):

                    ambiguous.add(alias)

                else:

                    aliases[alias] = name

        # Remove ambiguous aliases.
        for alias in ambiguous:

            aliases.pop(
                alias,
                None,
            )

        index.data = {
            "source_file": source_file,
            "entities": filtered,
            "aliases": aliases,
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
    ) -> "EntityIndex":

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
    # RESOLVE ENTITY
    # ---------------------------------------------------------

    def resolve(
        self,
        query: str,
    ) -> List[str]:

        q = query.lower()

        found = []

        # First check complete names.
        for name in self.data.get(
            "entities",
            {},
        ):

            if name.lower() in q:

                found.append(
                    name
                )

        # Then check aliases.
        for (
            alias,
            canonical,
        ) in self.data.get(
            "aliases",
            {},
        ).items():

            if re.search(
                rf"\b{re.escape(alias)}\b",
                q,
            ):

                found.append(
                    canonical
                )

        return list(
            dict.fromkeys(found)
        )

    # ---------------------------------------------------------
    # LIST ENTITIES
    # ---------------------------------------------------------

    def list_entities(
        self,
    ) -> List[Dict]:

        return sorted(
            self.data.get(
                "entities",
                {},
            ).values(),
            key=lambda item: (
                -item["mentions"],
                item["name"].lower(),
            ),
        )

    # ---------------------------------------------------------
    # FIND SUPPORTING CHUNKS
    # ---------------------------------------------------------

    def collect_evidence(
        self,
        chunks: List[Dict],
        names: List[str],
        limit: int = 12,
    ) -> List[Dict]:

        if not names:
            return []

        names_lower = [
            name.lower()
            for name in names
        ]

        scored = []

        for chunk in chunks:

            text = str(
                chunk.get(
                    "text",
                    "",
                )
            ).lower()

            hits = 0

            for name in names_lower:

                if name in text:
                    hits += 1

            if hits:

                page_number = (
                    chunk.get(
                        "metadata",
                        {},
                    ).get(
                        "page_number",
                        0,
                    )
                )

                scored.append(
                    (
                        hits,
                        page_number,
                        chunk,
                    )
                )

        scored.sort(
            key=lambda item: (
                -item[0],
                item[1],
            )
        )

        return [
            item[2]
            for item in scored[:limit]
        ]