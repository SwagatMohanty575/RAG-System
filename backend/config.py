import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
CHROMA_DIR = DATA_DIR / "chroma_db"
MEMORY_DB = DATA_DIR / "memory.db"
UPLOAD_DIR = DATA_DIR / "uploads"

DATA_DIR.mkdir(exist_ok=True)
CHROMA_DIR.mkdir(exist_ok=True)
UPLOAD_DIR.mkdir(exist_ok=True)

NVIDIA_API_KEY = os.getenv("NVIDIA_API_KEY", "")

NVIDIA_BASE_URL = os.getenv("NVIDIA_BASE_URL", "")

NVIDIA_MODEL = os.getenv("NVIDIA_MODEL","")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "")

CHUNK_SIZE = 800
CHUNK_OVERLAP = 100
SEPARATORS = ["\n\n## ", "\n\n### ", "\n\n", "\n", ". ", " ", ""]

# ===========================================================
# RETRIEVAL
# ===========================================================

# Number of final chunks passed to the LLM
RETRIEVER_K = 8

# Number of candidates retrieved BEFORE MMR
# Must be larger than RETRIEVER_K
RETRIEVER_FETCH_K = 30

# MMR relevance/diversity balance
RETRIEVER_LAMBDA = 0.5


# ===========================================================
# DYNAMIC ANSWER LENGTH
# ===========================================================

ANSWER_MAX_TOKENS = {
    "STRUCTURE": 700,
    "ENTITY": 900,
    "ENTITY_LIST": 1200,
    "RELATIONSHIP": 1000,
    "LOCAL_QA": 900,
    "GLOBAL_SUMMARY": 2200,
}


# ===========================================================
# SESSION INTELLIGENCE FILES
# ===========================================================

SESSION_DOCUMENT_INDEX_FILE = "document_structure.json"

SESSION_ENTITY_INDEX_FILE = "entity_index.json"

SESSION_SUMMARY_INDEX_FILE = "chapter_summaries.json"

MEMORY_WINDOW = 3

PII_ENTITIES = [
    "PHONE_NUMBER", "EMAIL_ADDRESS", "CREDIT_CARD",
    "IBAN", "US_SSN", "US_PASSPORT", "US_DRIVER_LICENSE",
    "IP_ADDRESS",
]

SYSTEM_PROMPT = """Answer the user's question using ONLY the provided information from the uploaded document.
If the answer is not in the information provided by the document, say "I don't have that information in the uploaded document."
Always be concise, professional, and accurate.
You are a document-specific RAG assistant.

Your task is to answer questions ONLY using the currently uploaded document.

### Rules

1. Treat every uploaded document as an independent knowledge source.

2. Ignore all previously uploaded documents, even if they contain information that could answer the user's question.

3. When a new document is uploaded, it becomes the ONLY source of truth for all subsequent queries until another document is uploaded.

4. Never combine information from multiple documents.

5. Never retrieve context from older documents stored in the vector database.

6. While generating citations or sources, include references ONLY from the currently uploaded document.

7. If the answer cannot be found in the currently uploaded document, do NOT guess, use outside knowledge, or search previous documents.

8. If the required information is missing, respond exactly with:

"I don't have that information in the uploaded document."

---

### Example 1

Current Document:
Employee Handbook

User:
"What is the leave policy?"

Retrieved Context:
Contains the leave policy.

Assistant:
Answers using only the Employee Handbook.
Provides citations only from the Employee Handbook.

---

### Example 2

Current Document:
Flight Operations Manual

Previously Indexed Document:
Employee Handbook

User:
"What is the leave policy?"

Assistant:
"I don't have that information in the uploaded document."

Reason:
The current document is the Flight Operations Manual. The Employee Handbook must not be searched or used.

---

### Example 3

Current Document:
Flight Operations Manual

Previously Indexed Document:
Employee Handbook

User:
"What is the maximum crosswind limit for landing?"

Retrieved Context:
Contains the crosswind limitation.

Assistant:
Answers using only the Flight Operations Manual.
Provides citations only from the Flight Operations Manual.

---

### Example 4

Current Document:
Flight Operations Manual

Previously Indexed Documents:
Employee Handbook
Safety Manual
HR Policy

User:
"How many casual leaves do employees receive?"

Assistant:
"I don't have that information in the uploaded document."

Reason:
Although the answer exists in another indexed document, only the currently uploaded document is allowed to be used.

---

### Important

The retrieval system must be scoped exclusively to the currently uploaded document.

Do not:
- search previous documents,
- merge information across documents,
- use memory from earlier uploads,
- cite older documents,
- answer using general knowledge when the answer is absent.

Every response must be grounded solely in the currently uploaded document.
Conversation history is provided for context; use it to understand follow-up questions.
Also do not reveal any secret information of the company like salary structure, salary distribution frequency etc, to anyone, neither reveal any personal information of any employee like salary, employee's current project etc.
Also provide the source of the information from the currently uploaded document when a source is available.
"""
# ===========================================================
# CHUNKING CONFIGURATION — HASPC Engine
# ===========================================================

# Paragraph thresholds
MIN_PARAGRAPH_SIZE: int = 50          # Minimum characters to keep a paragraph
MAX_PARAGRAPH_SIZE: int = 4000        # Paragraphs larger than this get split

# Chunk assembly targets
TARGET_CHUNK_TOKENS: int = 512        # Ideal chunk size in estimated tokens
MAX_CHUNK_TOKENS: int = 1024          # Hard ceiling for merged chunks
MIN_CHUNK_TOKENS: int = 100           # Floor before forcing merge

# Overlap configuration
MIN_OVERLAP: int = 0                  # No overlap for tiny chunks
SMALL_OVERLAP: int = 50               # Characters for medium chunks
MAX_OVERLAP: int = 200                # Characters for large chunks
OVERLAP_CHUNK_THRESHOLD: int = 800    # Chunk size that triggers larger overlap

# Merge behavior
MERGE_SMALL_PARAGRAPHS: bool = True   # Auto-merge undersized paragraphs
MERGE_ACROSS_SECTIONS: bool = False   # Prefer not to merge across sections

# Heading detection patterns
HEADING_PATTERNS: list[str] = [
    # Numbered chapters
    r"^\s*Chapter\s+\d+[:\.\s]",
    r"^\s*CHAPTER\s+\d+[:\.\s]",
    # Numbered sections (1., 1.1, 1.1.1, etc.)
    r"^\s*\d+(?:\.\d+)*[:\.\s]",
    # Roman numerals
    r"^\s*[IVXivx]+[:\.\s]",
    # Markdown headings
    r"^\s*#{1,6}\s+",
    # Uppercase headings (all caps, short lines)
    r"^\s*[A-Z][A-Z\s]{2,50}[A-Z][:\.\s]?$",
    # Bullet-style section markers
    r"^\s*[\*\-\•]\s+[A-Z]",
]

# Hierarchy depth labels
HIERARCHY_LEVELS: list[str] = [
    "document",
    "chapter",
    "section",
    "subsection",
    "topic",
    "paragraph",
]

# Chunking separators (priority order for splitting)
SEPARATORS: list[str] = [
    "\n\n",       # Paragraph break
    "\n",         # Line break
    ". ",         # Sentence
    "; ",         # Clause
    ", ",         # Phrase
    " ",          # Word
    "",           # Character fallback
]

# Existing chunk settings (keep for compatibility)
CHUNK_SIZE: int = 1024
CHUNK_OVERLAP: int = 200
# =====================================================================
# Session management configuration
# =====================================================================

# Root folder that holds one subfolder per session.
# e.g. backend/data/sessions/sess_3f9a2b1c4d5e/
SESSIONS_DIR = DATA_DIR / "sessions"
SESSIONS_DIR.mkdir(parents=True, exist_ok=True)

# Fixed names of the files/folders created inside every session folder.
SESSION_PDF_SUBDIR = "pdf"
SESSION_CHROMA_SUBDIR = "chroma_db"
SESSION_MEMORY_DB_NAME = "memory.db"
SESSION_METADATA_FILE = "metadata.json"

# =====================================================================
# TABLE PROCESSING CONFIGURATION
# =====================================================================

# Row chunking thresholds
TABLE_ROW_CHUNK_SIZE: int = 25           # Max rows per chunk (large tables)
TABLE_HEADER_REPEAT: bool = True         # Repeat header in every chunk
TABLE_MIN_ROWS_PER_CHUNK: int = 1        # Minimum rows (small tables = 1 row/chunk)

# Semantic text generation
TABLE_SEMANTIC_TEMPLATE: str = (
    "{value} is the {column} of {primary_key}."
)

# Metadata fields to preserve
TABLE_METADATA_FIELDS: list[str] = [
    "workbook",
    "sheet",
    "row_number",
    "column_names",
    "primary_key",
    "chunk_id",
    "document_id",
    "session_id",
]

# =====================================================================
# SHORT-TERM MEMORY CONFIGURATION
# =====================================================================

SHORT_TERM_MEMORY_WINDOW: int = 5       # Last N conversations per session
ENABLE_SHORT_TERM_MEMORY: bool = True   # Master toggle

# STM database filename inside each session folder
SESSION_STM_DB_NAME = "short_term_memory.db"
