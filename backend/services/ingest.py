"""
ingest.py

Now a pure router — inspects the file extension and delegates to the
right parser. All parsers return the same List[Dict] shape, so chunk.py,
HASPC, and everything downstream is completely unaffected.
"""

from pathlib import Path
from typing import List, Dict

from services.parsers.pdf_parser import PDFParser
from services.parsers.docx_parser import DOCXParser
from services.parsers.excel_parser import ExcelParser
from services.parsers.pptx_parser import PPTXParser


SUPPORTED_EXTENSIONS = {
    ".pdf":  PDFParser,
    ".docx": DOCXParser,
    ".doc":  DOCXParser,
    ".xlsx": ExcelParser,
    ".xls":  ExcelParser,
    ".csv":  ExcelParser,
    ".pptx": PPTXParser,
}


class PDFIngestor:
    """
    Kept named PDFIngestor so session_manager.py and main.py don't need
    any changes — but it now handles all supported formats, not just PDF.
    """

    def extract(self, file_path: str) -> List[Dict]:
        ext = Path(file_path).suffix.lower()

        parser_class = SUPPORTED_EXTENSIONS.get(ext)
        if parser_class is None:
            supported = ", ".join(SUPPORTED_EXTENSIONS.keys())
            raise ValueError(
                f"Unsupported file type '{ext}'. "
                f"Supported formats: {supported}"
            )

        parser = parser_class()
        return parser.extract(file_path)