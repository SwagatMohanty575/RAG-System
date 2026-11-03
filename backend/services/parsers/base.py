"""
base.py

Abstract base class every parser must implement. Each parser receives
a file path and returns List[Dict] in the exact same shape ingest.py
has always produced — so chunk.py, HASPC, and everything downstream
never needs to know what file type was parsed.
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Dict


class BaseParser(ABC):
    @abstractmethod
    def extract(self, file_path: str) -> List[Dict]:
        """
        Parse the file and return a list of page-like dicts, each with:
            page_number  : int
            text         : str
            section_title: str
            source_file  : str  (just the filename, not the full path)
        """
        pass

    @staticmethod
    def _source_name(file_path: str) -> str:
        return Path(file_path).name