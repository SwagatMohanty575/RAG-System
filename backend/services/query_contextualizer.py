"""
Resolves conversational references such as:

this character
that character
this person
he
she
they
it

The rewritten query is only used for retrieval.
The final answer still has to come from document evidence.
"""

from __future__ import annotations

import re

from typing import Dict, List


class QueryContextualizer:

    FOLLOWUP_PATTERN = re.compile(
        r"\b("
        r"this|"
        r"that|"
        r"these|"
        r"those|"
        r"he|"
        r"she|"
        r"him|"
        r"her|"
        r"they|"
        r"them|"
        r"it|"
        r"this character|"
        r"that character|"
        r"the character|"
        r"this person|"
        r"that person"
        r")\b",
        re.IGNORECASE,
    )

    @classmethod
    def resolve(
        cls,
        query: str,
        history: List[Dict],
        client,
    ) -> str:

        # No history = nothing to resolve.
        if not history:

            return query

        # No reference word = no contextualization needed.
        if not cls.FOLLOWUP_PATTERN.search(
            query
        ):

            return query

        recent = history[-3:]

        conversation = []

        for turn in recent:

            conversation.append(
                f"User: {turn.get('human', '')}"
            )

            conversation.append(
                "Assistant: "
                + turn.get(
                    "ai",
                    "",
                )[:3000]
            )

        prompt = f"""
Resolve the user's follow-up question into a standalone
retrieval query.

Rules:

1. Do NOT answer the question.

2. Use ONLY names and facts explicitly present
   in the conversation.

3. Do NOT invent a person, relationship,
   event, or name.

4. Preserve the user's original intent.

5. If the reference cannot be resolved confidently,
   return the original question unchanged.

6. Return ONLY the rewritten query.

CONVERSATION:

{chr(10).join(conversation)}

CURRENT QUESTION:

{query}
""".strip()

        try:

            rewritten = client.complete(
                prompt,
                system_prompt=(
                    "You are a retrieval-query "
                    "contextualizer. "
                    "Rewrite questions only. "
                    "Never answer the question."
                ),
                max_tokens=180,
                temperature=0.0,
            )

            rewritten = rewritten.strip()

            if not rewritten:
                return query

            if len(rewritten) > 500:
                return query

            return rewritten

        except Exception:

            # Never allow contextualization failure
            # to break normal RAG.
            return query