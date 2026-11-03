"""
Routes questions before normal vector retrieval.
"""

from __future__ import annotations

import re


class QueryRouter:

    STRUCTURE = "STRUCTURE"

    GLOBAL_SUMMARY = "GLOBAL_SUMMARY"

    ENTITY = "ENTITY"

    ENTITY_LIST = "ENTITY_LIST"

    RELATIONSHIP = "RELATIONSHIP"

    LOCAL_QA = "LOCAL_QA"

    @classmethod
    def route(
        cls,
        query: str,
    ) -> str:

        q = query.lower().strip()

        # =====================================================
        # STRUCTURE
        # =====================================================

        if (
            re.search(
                r"\bhow many\b.*\bchapter",
                q,
            )
            or
            "number of chapters" in q
        ):

            return cls.STRUCTURE

        if any(
            phrase in q
            for phrase in (
                "list the chapters",
                "chapter names",
                "chapter titles",
                "what chapters are",
                "which chapters are",
                "table of contents",
                "document outline",
                "outline of the document",
            )
        ):

            return cls.STRUCTURE

        # =====================================================
        # GLOBAL DOCUMENT
        # =====================================================

        if any(
            phrase in q
            for phrase in (
                "whole document",
                "entire document",
                "full document",
                "complete document",
                "overall story",
                "whole story",
                "entire story",
                "complete story",
                "summarize the document",
                "summary of the document",
                "summarize everything",
                "explain the whole",
                "explain the entire",
            )
        ):

            return cls.GLOBAL_SUMMARY

        # =====================================================
        # ENTITY LIST
        # =====================================================

        if any(
            phrase in q
            for phrase in (
                "all characters",
                "list characters",
                "who are the characters",
                "characters in the document",
                "characters in this story",
            )
        ):

            return cls.ENTITY_LIST

        # =====================================================
        # RELATIONSHIP
        # =====================================================

        if any(
            phrase in q
            for phrase in (
                "relationship between",
                "relation between",
                "related to",
                "connection between",
                "married to",
                "husband of",
                "wife of",
                "father of",
                "mother of",
                "son of",
                "daughter of",
                "brother of",
                "sister of",
                "affair with",
                "lover of",
            )
        ):

            return cls.RELATIONSHIP

        # =====================================================
        # ENTITY
        # =====================================================

        if any(
            phrase in q
            for phrase in (
                "who is",
                "who was",
                "what is the name of",
                "name of this character",
                "name of the character",
                "identify this character",
                "identify the character",
                "which character is",
            )
        ):

            return cls.ENTITY

        # =====================================================
        # NORMAL RAG
        # =====================================================

        return cls.LOCAL_QA