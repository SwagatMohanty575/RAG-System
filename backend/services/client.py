from typing import Dict, List, Optional

from openai import OpenAI

from config import (
    NVIDIA_API_KEY,
    NVIDIA_BASE_URL,
    NVIDIA_MODEL,
    SYSTEM_PROMPT,
    ANSWER_MAX_TOKENS,
)


class NvidiaLLM:

    def __init__(self):

        if not NVIDIA_API_KEY:

            raise RuntimeError(
                "NVIDIA_API_KEY is not configured. "
                "Add it to your .env file."
            )

        self.client = OpenAI(
            base_url=NVIDIA_BASE_URL,
            api_key=NVIDIA_API_KEY,
        )

        self.model = NVIDIA_MODEL

        print(
            f"[DEBUG] NvidiaLLM initialized "
            f"with model: {self.model}"
        )

    # =========================================================
    # GENERIC COMPLETION
    # =========================================================

    def complete(
        self,
        user_prompt: str,
        system_prompt: Optional[str] = None,
        max_tokens: int = 800,
        temperature: float = 0.2,
    ) -> str:

        response = (
            self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            system_prompt
                            or SYSTEM_PROMPT
                        ),
                    },
                    {
                        "role": "user",
                        "content": user_prompt,
                    },
                ],
                temperature=temperature,
                max_tokens=max_tokens,
            )
        )

        answer = (
            response
            .choices[0]
            .message
            .content
        )

        if not answer:

            raise RuntimeError(
                "NVIDIA API returned "
                "an empty response."
            )

        return answer.strip()

    # =========================================================
    # RAG GENERATION
    # =========================================================

    def generate(
        self,
        query: str,
        context_chunks: List[Dict],
        conversation_history: List[Dict],
        short_term_memory=None,
        resolved_query: Optional[str] = None,
        query_type: str = "LOCAL_QA",
    ) -> str:

        from services.context_assembler import (
            ContextAssembler,
        )

        prompt = (
            ContextAssembler.assemble(
                query=query,
                context_chunks=context_chunks,
                short_term_memory=(
                    short_term_memory
                ),
                resolved_query=(
                    resolved_query
                ),
                query_type=query_type,
            )
        )

        max_tokens = (
            ANSWER_MAX_TOKENS.get(
                query_type,
                ANSWER_MAX_TOKENS[
                    "LOCAL_QA"
                ],
            )
        )

        return self.complete(
            prompt,
            system_prompt=SYSTEM_PROMPT,
            max_tokens=max_tokens,
            temperature=0.3,
        )

    # =========================================================
    # HEALTH
    # =========================================================

    def health_check(
        self,
    ) -> bool:

        try:

            return bool(
                NVIDIA_API_KEY
                and NVIDIA_BASE_URL
                and NVIDIA_MODEL
            )

        except Exception:

            return False