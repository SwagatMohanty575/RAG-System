"""
MMR retrieval for normal document-passage questions.

Structure questions, global questions, entity questions and
relationship questions are routed separately.
"""

from __future__ import annotations

import re

from typing import Dict, List, Optional

import numpy as np

from services.vector import ChromaVectorStore
from services.embed import OllamaEmbedder

from config import (
    RETRIEVER_K,
    RETRIEVER_FETCH_K,
    RETRIEVER_LAMBDA,
)


class MMRRetriever:

    def __init__(
        self,
        vector_store: ChromaVectorStore,
        embedder: OllamaEmbedder,
    ) -> None:

        self.vs = vector_store

        self.embedder = embedder

        self.k = RETRIEVER_K

        self.fetch_k = RETRIEVER_FETCH_K

        self.lambda_param = (
            RETRIEVER_LAMBDA
        )

    # =========================================================
    # RETRIEVE
    # =========================================================

    def retrieve(
        self,
        query: str,
        conversation_history: Optional[
            List[Dict]
        ] = None,
    ) -> List[Dict]:

        search_query = (
            self._build_search_query(
                query,
                conversation_history,
            )
        )

        query_embedding = np.asarray(
            self.embedder.embed_query(
                search_query
            ),
            dtype=np.float32,
        )

        # -----------------------------------------------------
        # IMPORTANT:
        #
        # Retrieve a large candidate pool first.
        # MMR then chooses the best diverse chunks.
        # -----------------------------------------------------

        n_results = max(
            self.k,
            self.fetch_k,
        )

        candidates = self.vs.query(
            query_embedding.tolist(),
            n_results=n_results,
        )

        if not candidates:
            return []

        key_terms = (
            self._extract_key_terms(
                search_query
            )
        )

        chapter_terms = (
            self._chapter_terms(
                search_query
            )
        )

        # -----------------------------------------------------
        # Boost exact keyword/chapter matches.
        # -----------------------------------------------------

        for candidate in candidates:

            text = candidate.get(
                "text",
                "",
            ).lower()

            metadata = candidate.get(
                "metadata",
                {},
            )

            metadata_text = " ".join(
                str(value)
                for value in metadata.values()
            ).lower()

            searchable = (
                text
                + " "
                + metadata_text
            )

            exact_boost = 0.0

            # Proper names / exact terms.
            for term in key_terms:

                if (
                    len(term) >= 3
                    and term in searchable
                ):

                    exact_boost += 0.08

            # Chapter boost.
            for term in chapter_terms:

                if term in searchable:

                    exact_boost += 0.15

            candidate[
                "_exact_match_boost"
            ] = min(
                exact_boost,
                0.35,
            )

        # -----------------------------------------------------
        # MMR
        # -----------------------------------------------------

        if len(candidates) <= self.k:

            return self._strip_embeddings(
                candidates
            )

        reranked = self._mmr_rerank(
            query_emb=query_embedding,
            candidates=candidates,
            top_k=self.k,
        )

        return self._strip_embeddings(
            reranked
        )

    # =========================================================
    # SEARCH QUERY
    # =========================================================

    def _build_search_query(
        self,
        query: str,
        conversation_history: Optional[
            List[Dict]
        ],
    ) -> str:

        if not conversation_history:

            return query

        previous_questions = [
            turn.get(
                "human",
                "",
            )
            for turn in conversation_history[-2:]
            if turn.get(
                "human"
            )
        ]

        if not previous_questions:

            return query

        return (
            "\n".join(
                previous_questions
            )
            + "\n"
            + query
        )

    # =========================================================
    # KEY TERMS
    # =========================================================

    @staticmethod
    def _extract_key_terms(
        query: str,
    ) -> List[str]:

        stopwords = {
            "what",
            "when",
            "where",
            "which",
            "who",
            "whom",
            "whose",
            "how",
            "why",
            "does",
            "did",
            "was",
            "were",
            "the",
            "and",
            "for",
            "from",
            "with",
            "about",
            "this",
            "that",
            "these",
            "those",
            "chapter",
            "document",
            "tell",
            "explain",
            "give",
            "please",
        }

        terms = []

        # Quoted phrases.
        quoted = re.findall(
            r'"([^"]+)"',
            query,
        )

        for phrase in quoted:

            if phrase.strip():

                terms.append(
                    phrase.lower().strip()
                )

        # Words / numbers.
        tokens = re.findall(
            r"\b[A-Za-z0-9][A-Za-z0-9'-]*\b",
            query,
        )

        for token in tokens:

            normalized = token.lower()

            if (
                normalized in stopwords
            ):
                continue

            if len(normalized) < 3:
                continue

            terms.append(
                normalized
            )

        return list(
            dict.fromkeys(
                terms
            )
        )

    # =========================================================
    # CHAPTER TERMS
    # =========================================================

    @staticmethod
    def _chapter_terms(
        query: str,
    ) -> List[str]:

        match = re.search(
            r"\bchapter\s+(\d+)\b",
            query,
            re.IGNORECASE,
        )

        if not match:

            return []

        number = int(
            match.group(1)
        )

        words = {
            1: "one",
            2: "two",
            3: "three",
            4: "four",
            5: "five",
            6: "six",
            7: "seven",
            8: "eight",
            9: "nine",
            10: "ten",
            11: "eleven",
            12: "twelve",
            13: "thirteen",
            14: "fourteen",
            15: "fifteen",
            16: "sixteen",
            17: "seventeen",
            18: "eighteen",
            19: "nineteen",
            20: "twenty",
        }

        terms = [
            f"chapter {number}",
            f"chapter {words.get(number, number)}",
            f"chapter number: chapter {number}",
            f"chapter: chapter {number}",
        ]

        if number in words:

            terms.extend(
                [
                    f"chapter {words[number]}",
                    f"chapter number: chapter {words[number]}",
                    f"chapter: chapter {words[number]}",
                ]
            )

        return terms

    # =========================================================
    # MMR
    # =========================================================

    def _mmr_rerank(
        self,
        query_emb: np.ndarray,
        candidates: List[Dict],
        top_k: int,
    ) -> List[Dict]:

        candidate_embeddings = np.asarray(
            [
                candidate[
                    "embedding"
                ]
                for candidate in candidates
            ],
            dtype=np.float32,
        )

        query_norm = np.linalg.norm(
            query_emb
        )

        if query_norm == 0:

            return candidates[:top_k]

        candidate_norms = np.linalg.norm(
            candidate_embeddings,
            axis=1,
            keepdims=True,
        )

        candidate_norms[
            candidate_norms == 0
        ] = 1.0

        query_emb = (
            query_emb
            / query_norm
        )

        candidate_embeddings = (
            candidate_embeddings
            / candidate_norms
        )

        query_similarity = (
            candidate_embeddings
            @ query_emb
        )

        # Add exact-match boosts.
        for i, candidate in enumerate(
            candidates
        ):

            query_similarity[i] += (
                candidate.get(
                    "_exact_match_boost",
                    0.0,
                )
            )

        selected = []

        remaining = set(
            range(
                len(candidates)
            )
        )

        # Best initial candidate.
        first_index = int(
            np.argmax(
                query_similarity
            )
        )

        selected.append(
            first_index
        )

        remaining.remove(
            first_index
        )

        while (
            len(selected) < top_k
            and remaining
        ):

            best_index = None

            best_score = -float(
                "inf"
            )

            for index in remaining:

                relevance = (
                    query_similarity[
                        index
                    ]
                )

                diversity_penalty = max(
                    candidate_embeddings[
                        index
                    ]
                    @ candidate_embeddings[
                        selected_index
                    ]
                    for selected_index
                    in selected
                )

                score = (
                    self.lambda_param
                    * relevance
                    -
                    (
                        1
                        - self.lambda_param
                    )
                    * diversity_penalty
                )

                if score > best_score:

                    best_score = score

                    best_index = index

            selected.append(
                best_index
            )

            remaining.remove(
                best_index
            )

        return [
            candidates[index]
            for index in selected
        ]

    # =========================================================
    # REMOVE EMBEDDINGS
    # =========================================================

    @staticmethod
    def _strip_embeddings(
        chunks: List[Dict],
    ) -> List[Dict]:

        return [
            {
                key: value
                for key, value
                in chunk.items()
                if key
                not in {
                    "embedding",
                    "_exact_match_boost",
                }
            }
            for chunk in chunks
        ]