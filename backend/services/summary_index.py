"""
Lazy hierarchical summary cache.

Chapter summaries are generated only when a global/document
question is asked.

After generation they are stored in the current session.
"""

from __future__ import annotations

import json

from pathlib import Path
from typing import Dict, List


class SummaryIndex:

    def __init__(
        self,
        path: str | Path,
    ) -> None:

        self.path = Path(path)

    # ---------------------------------------------------------
    # LOAD
    # ---------------------------------------------------------

    def load(self) -> Dict:

        if not self.path.exists():

            return {
                "chapters": []
            }

        with open(
            self.path,
            "r",
            encoding="utf-8",
        ) as file:

            return json.load(file)

    # ---------------------------------------------------------
    # SAVE
    # ---------------------------------------------------------

    def save(
        self,
        data: Dict,
    ) -> None:

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
                data,
                file,
                indent=2,
                ensure_ascii=False,
            )

    # ---------------------------------------------------------
    # GROUP CHUNKS
    # ---------------------------------------------------------

    @staticmethod
    def group_chunks(
        chunks: List[Dict],
    ) -> List[Dict]:

        groups: Dict[str, Dict] = {}

        for chunk in chunks:

            metadata = chunk.get(
                "metadata",
                {},
            )

            chapter_number = (
                metadata.get(
                    "chapter_number"
                )
                or
                metadata.get(
                    "chapter"
                )
            )

            chapter_title = (
                metadata.get(
                    "chapter_title"
                )
                or
                metadata.get(
                    "section_title"
                )
            )

            if not chapter_number:

                section = str(
                    metadata.get(
                        "section_title",
                        "",
                    )
                )

                if (
                    "epilogue"
                    in section.lower()
                ):

                    key = "EPILOGUE"

                else:

                    key = "OTHER"

            else:

                key = str(
                    chapter_number
                )

            group = groups.setdefault(
                key,
                {
                    "key": key,
                    "title": (
                        chapter_title
                        or key
                    ),
                    "chunks": [],
                    "page_start": metadata.get(
                        "page_number",
                        0,
                    ),
                    "page_end": metadata.get(
                        "page_number",
                        0,
                    ),
                    "source_file": metadata.get(
                        "source_file",
                        "",
                    ),
                },
            )

            group["chunks"].append(
                chunk
            )

            page = int(
                metadata.get(
                    "page_number",
                    0,
                )
                or 0
            )

            group["page_start"] = min(
                group["page_start"],
                page,
            )

            group["page_end"] = max(
                group["page_end"],
                page,
            )

        result = list(
            groups.values()
        )

        result.sort(
            key=lambda item:
                item["page_start"]
        )

        for group in result:

            group["chunks"].sort(
                key=lambda chunk: (
                    int(
                        chunk.get(
                            "metadata",
                            {},
                        ).get(
                            "page_number",
                            0,
                        )
                        or 0
                    ),
                    int(
                        chunk.get(
                            "metadata",
                            {},
                        ).get(
                            "chunk_index",
                            0,
                        )
                        or 0
                    ),
                )
            )

        return result

    # ---------------------------------------------------------
    # BATCH CHUNKS
    # ---------------------------------------------------------

    @staticmethod
    def make_batches(
        chunks: List[Dict],
        max_chars: int = 14000,
    ) -> List[str]:

        batches = []

        current = []

        current_size = 0

        for chunk in chunks:

            metadata = chunk.get(
                "metadata",
                {},
            )

            text = (
                f"[Page "
                f"{metadata.get('page_number')}, "
                f"Section: "
                f"{metadata.get('section_title')}]\n"
                f"{chunk.get('text', '')}"
            )

            if (
                current
                and
                current_size + len(text)
                > max_chars
            ):

                batches.append(
                    "\n\n---\n\n".join(
                        current
                    )
                )

                current = []

                current_size = 0

            current.append(
                text
            )

            current_size += len(text)

        if current:

            batches.append(
                "\n\n---\n\n".join(
                    current
                )
            )

        return batches

    # ---------------------------------------------------------
    # ENSURE SUMMARY
    # ---------------------------------------------------------

    def ensure(
        self,
        vector_store,
        client,
        document_index,
    ) -> Dict:

        cached = self.load()

        # Already generated.
        if cached.get(
            "chapters"
        ):

            return cached

        all_chunks = (
            vector_store.get_all_chunks()
        )

        groups = self.group_chunks(
            all_chunks
        )

        summaries = []

        for group in groups:

            partial_summaries = []

            batches = self.make_batches(
                group["chunks"]
            )

            for batch in batches:

                prompt = f"""
Summarize ONLY the document excerpts below.

Preserve important:

- characters
- relationships
- causes
- events
- discoveries
- outcomes
- important decisions

Do not add outside knowledge.

Do not speculate.

Chapter/Section:
{group['title']}

DOCUMENT EXCERPTS:

{batch}
""".strip()

                summary = client.complete(
                    prompt,
                    system_prompt=(
                        "You are a factual document "
                        "summarizer. "
                        "Use ONLY the supplied "
                        "document excerpts."
                    ),
                    max_tokens=700,
                    temperature=0.1,
                )

                partial_summaries.append(
                    summary.strip()
                )

            # -------------------------------------------------
            # Single batch
            # -------------------------------------------------

            if len(
                partial_summaries
            ) == 1:

                final_summary = (
                    partial_summaries[0]
                )

            # -------------------------------------------------
            # Multiple batches
            # -------------------------------------------------

            else:

                reduce_prompt = f"""
Combine the following partial summaries into one
coherent factual summary of:

{group['title']}

Use ONLY the supplied summaries.

Do not invent facts.

PARTIAL SUMMARIES:

{chr(10).join(
    f"- {summary}"
    for summary in partial_summaries
)}
""".strip()

                final_summary = client.complete(
                    reduce_prompt,
                    system_prompt=(
                        "You are a factual "
                        "document summarizer. "
                        "Use only supplied summaries."
                    ),
                    max_tokens=900,
                    temperature=0.1,
                ).strip()

            summaries.append(
                {
                    "key": group["key"],
                    "title": group["title"],
                    "summary": final_summary,
                    "page_start": group[
                        "page_start"
                    ],
                    "page_end": group[
                        "page_end"
                    ],
                    "source_file": group[
                        "source_file"
                    ],
                }
            )

        data = {
            "source_file": (
                document_index.data.get(
                    "source_file",
                    "",
                )
            ),
            "chapter_count": (
                document_index.chapter_count
            ),
            "chapters": summaries,
        }

        self.save(data)

        return data

    # ---------------------------------------------------------
    # CONVERT TO CONTEXT
    # ---------------------------------------------------------

    def as_context_chunks(
        self,
        data: Dict,
    ) -> List[Dict]:

        chunks = []

        for item in data.get(
            "chapters",
            [],
        ):

            chunks.append(
                {
                    "text": item[
                        "summary"
                    ],
                    "metadata": {
                        "source_file": item.get(
                            "source_file",
                            "",
                        ),
                        "page_number": item.get(
                            "page_start",
                            0,
                        ),
                        "section_title": item.get(
                            "title",
                            "",
                        ),
                        "page_end": item.get(
                            "page_end",
                            0,
                        ),
                    },
                }
            )

        return chunks