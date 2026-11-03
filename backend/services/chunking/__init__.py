"""
Hierarchical Adaptive Semantic Paragraph Chunking (HASPC) Engine.

This package provides a production-grade document chunking system
that preserves hierarchy, semantic continuity, and paragraph integrity.
"""

from services.chunking.pipeline import HASPCChunker
from services.chunking.table_chunker import TableChunker

__all__ = ["HASPCChunker", "TableChunker"]