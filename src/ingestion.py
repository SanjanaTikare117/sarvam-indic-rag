"""
ingestion.py
------------
Document loading and chunking for Indic language documents.
Handles plain text, PDF, and raw string inputs.
Supports Hindi, Kannada, and other Indic scripts.
"""

import re
from pathlib import Path
from typing import Optional

try:
    import PyPDF2
    PDF_SUPPORT = True
except ImportError:
    PDF_SUPPORT = False


# --- Script detection ---

SCRIPT_RANGES = {
    "hindi":   (0x0900, 0x097F),   # Devanagari
    "kannada": (0x0C80, 0x0CFF),
    "tamil":   (0x0B80, 0x0BFF),
    "telugu":  (0x0C00, 0x0C7F),
    "bengali": (0x0980, 0x09FF),
    "english": (0x0041, 0x007A),
}


def detect_language(text: str) -> str:
    """Detect dominant script/language from Unicode ranges."""
    counts = {lang: 0 for lang in SCRIPT_RANGES}
    for ch in text:
        cp = ord(ch)
        for lang, (lo, hi) in SCRIPT_RANGES.items():
            if lo <= cp <= hi:
                counts[lang] += 1
    dominant = max(counts, key=counts.get)
    return dominant if counts[dominant] > 0 else "unknown"


# --- Sentence-aware chunking for Indic text ---

# Devanagari uses । (U+0964) as full stop; Kannada uses same.
# We split on these + standard punctuation.
INDIC_SENTENCE_ENDINGS = re.compile(r'(?<=[।॥\.\!\?])\s+')


def chunk_text(
    text: str,
    chunk_size: int = 200,
    overlap: int = 40,
) -> list[str]:
    """
    Split text into overlapping chunks.
    Tries to split on Indic/English sentence boundaries first,
    then falls back to character-level chunking.
    """
    text = text.strip()
    if not text:
        return []

    # Split into sentences
    sentences = INDIC_SENTENCE_ENDINGS.split(text)
    sentences = [s.strip() for s in sentences if s.strip()]

    chunks = []
    current = ""

    for sentence in sentences:
        if len(current) + len(sentence) <= chunk_size:
            current += (" " if current else "") + sentence
        else:
            if current:
                chunks.append(current.strip())
            # Overlap: carry last `overlap` chars into next chunk
            current = current[-overlap:] + " " + sentence if overlap else sentence

    if current.strip():
        chunks.append(current.strip())

    # If no sentence boundaries found, fall back to sliding window
    if not chunks:
        for i in range(0, len(text), chunk_size - overlap):
            chunks.append(text[i:i + chunk_size])

    return chunks


# --- Loaders ---

def load_text_file(path: str | Path) -> str:
    """Load a plain text file (UTF-8)."""
    return Path(path).read_text(encoding="utf-8")


def load_pdf(path: str | Path) -> str:
    """Extract text from a PDF using PyPDF2."""
    if not PDF_SUPPORT:
        raise ImportError("PyPDF2 not installed. Run: pip install PyPDF2")
    text = []
    with open(path, "rb") as f:
        reader = PyPDF2.PdfReader(f)
        for page in reader.pages[:10]:
            page_text = page.extract_text()
            if page_text:
                text.append(page_text)
    return "\n".join(text)


def load_raw_strings(docs: list[str]) -> list[str]:
    """Pass-through for raw string lists (used in demo/testing)."""
    return docs


# --- Main ingestion pipeline ---

def ingest_documents(
    sources: list[str | Path],
    chunk_size: int = 200,
    overlap: int = 40,
    language_override: Optional[str] = None,
) -> list[dict]:
    """
    Load documents from file paths or raw strings.
    Returns a list of chunk dicts:
        {
            "text": str,
            "language": str,
            "source": str,
            "chunk_id": int,
        }
    """
    records = []
    chunk_id = 0

    for source in sources:
        # Determine if it's a file path or raw string
        if isinstance(source, Path) or (isinstance(source, str) and Path(source).exists()):
            path = Path(source)
            if path.suffix.lower() == ".pdf":
                raw_text = load_pdf(path)
            else:
                raw_text = load_text_file(path)
            source_name = path.name
        else:
            # Raw string
            raw_text = source
            source_name = "raw"

        chunks = chunk_text(raw_text, chunk_size=chunk_size, overlap=overlap)

        for chunk in chunks:
            lang = language_override or detect_language(chunk)
            records.append({
                "text": chunk,
                "language": lang,
                "source": source_name,
                "chunk_id": chunk_id,
            })
            chunk_id += 1

    return records


# --- Quick sanity check ---
if __name__ == "__main__":
    sample_hindi = [
        "भारत एक विविधताओं से भरा देश है। यहाँ अनेक भाषाएं, धर्म और संस्कृतियाँ एक साथ रहती हैं।",
        "भारतीय संविधान 26 जनवरी 1950 को लागू हुआ था। यह दुनिया का सबसे बड़ा लिखित संविधान है।",
    ]
    sample_kannada = [
        "ಕರ್ನಾಟಕ ದಕ್ಷಿಣ ಭಾರತದ ಒಂದು ರಾಜ್ಯ. ಇದರ ರಾಜಧಾನಿ ಬೆಂಗಳೂರು.",
    ]

    docs = ingest_documents(sample_hindi + sample_kannada)
    for d in docs:
        print(f"[{d['language']}] chunk_id={d['chunk_id']} | {d['text'][:60]}...")