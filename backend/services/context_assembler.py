"""
Builds the final document-grounded prompt.

Conversation history is only used for context.
Previous assistant answers are NOT treated as document evidence.
"""

from __future__ import annotations

from typing import Dict, List, Optional

from services.short_term_memory import (
    ShortTermMemory,
)


class ContextAssembler:

    @staticmethod
    def assemble(
        query: str,
        context_chunks: List[Dict],
        short_term_memory: Optional[
            ShortTermMemory
        ],
        resolved_query: Optional[str] = None,
        query_type: str = "LOCAL_QA",
    ) -> str:

        parts: List[str] = []

        # -----------------------------------------------------
        # Query type
        # -----------------------------------------------------

        parts.append(
            f"""
=== QUERY TYPE ===
{query_type}
""".strip()
        )

        # -----------------------------------------------------
        # Original question
        # -----------------------------------------------------

        parts.append(
            f"""
=== CURRENT USER QUESTION ===
{query}
""".strip()
        )

        # -----------------------------------------------------
        # Resolved follow-up
        # -----------------------------------------------------

        if (
            resolved_query
            and
            resolved_query.strip()
            != query.strip()
        ):

            parts.append(
                f"""
=== RESOLVED FOLLOW-UP QUESTION ===
{resolved_query}
""".strip()
            )

        # -----------------------------------------------------
        # Document evidence
        # -----------------------------------------------------

        parts.append(
            "=== DOCUMENT EVIDENCE ==="
        )

        if context_chunks:

            for i, chunk in enumerate(
                context_chunks,
                start=1,
            ):

                metadata = chunk.get(
                    "metadata",
                    {},
                )

                source = (
                    f"[Source: "
                    f"{metadata.get('source_file', 'document')}, "
                    f"Page "
                    f"{metadata.get('page_number', '?')}, "
                    f"Section: "
                    f"{metadata.get('section_title', '')}]"
                )

                parts.append(
                    f"""
Excerpt {i}: {source}

{chunk.get('text', '')}
""".strip()
                )

        else:

            parts.append(
                "No document evidence was retrieved."
            )

        # -----------------------------------------------------
        # Recent user queries
        # -----------------------------------------------------

        if short_term_memory:

            recent = (
                short_term_memory
                .get_recent_queries()
            )

            if recent:

                parts.append(
                    """
=== RECENT USER QUERIES
(CONTEXT ONLY — NOT FACTUAL EVIDENCE) ===
""".strip()
                )

                for item in recent:

                    parts.append(
                        f"- {item.query}"
                    )

        # -----------------------------------------------------
        # Rules
        # -----------------------------------------------------

        parts.append(
            """
=== ANSWERING RULES ===

1. Answer using ONLY the DOCUMENT EVIDENCE above.

2. Recent user queries are ONLY for understanding
   conversational follow-ups.

3. Do NOT treat previous assistant answers as
   document evidence.

4. Do NOT use outside knowledge.

5. If the document evidence does not contain the answer,
   say exactly:

"I don't have that information in the uploaded document."

6. Preserve names, relationships, dates and events accurately.

7. Include source references naturally when evidence
   is available.
""".strip()
        )

        return "\n\n".join(
            parts
        )