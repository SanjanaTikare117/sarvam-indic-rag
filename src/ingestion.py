"""
ingestion.py
------------
Document loading and chunking for Indic language documents.
Supports:
- PDF (text + OCR fallback)
- Images (OCR)
- Plain text files
- Raw strings
"""

import re
from pathlib import Path
from typing import Optional, List, Union

import pytesseract
import cv2
from pdf2image import convert_from_path

# Set Tesseract path (IMPORTANT)
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# Optional PDF support
try:
    import PyPDF2
    PDF_SUPPORT = True
except ImportError:
    PDF_SUPPORT = False



# ============================================================
# OCR FUNCTIONS
# ============================================================

def extract_text_from_image(image_path: str) -> str:
    """Extract text from image using OCR."""
    img = cv2.imread(image_path)

    if img is None:
        return ""

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)[1]

    text = pytesseract.image_to_string(
        gray,
        lang="eng+hin+kan",
        config="--oem 3 --psm 6"
    )

    return text.strip()

def extract_text_from_pdf(path: str) -> str:
    """
    Extract text from PDF.
    First tries normal extraction, then OCR fallback.
    """
    if not PDF_SUPPORT:
        raise ImportError("Install PyPDF2 and pdf2image")

    text = []

    # ---- Try normal extraction ----
    with open(path, "rb") as f:
        reader = PyPDF2.PdfReader(f)
        for page in reader.pages[:10]:
            page_text = page.extract_text()
            if page_text:
                text.append(page_text)

    full_text = "\n".join(text).strip()

    if full_text:
        return full_text

    # ---- OCR fallback ----
    print("⚠️ No text found in PDF, using OCR...")

    images = convert_from_path(
        path,
        poppler_path=r"C:\Release-25.12.0-0\poppler-25.12.0\Library\bin"
    )
    ocr_text = []

    for i, img in enumerate(images):
        temp_path = f"temp_page_{i}.png"
        img.save(temp_path, "PNG")
        page_text = extract_text_from_image(temp_path)
        ocr_text.append(page_text)

    return "\n".join(ocr_text).strip()

# LANGUAGE DETECTION
# ============================================================

SCRIPT_RANGES = {
    "hindi":   (0x0900, 0x097F),
    "kannada": (0x0C80, 0x0CFF),
    "tamil":   (0x0B80, 0x0BFF),
    "telugu":  (0x0C00, 0x0C7F),
    "bengali": (0x0980, 0x09FF),
    "english": (0x0041, 0x007A),
}


def detect_language(text: str) -> str:
    counts = {lang: 0 for lang in SCRIPT_RANGES}

    for ch in text:
        cp = ord(ch)
        for lang, (lo, hi) in SCRIPT_RANGES.items():
            if lo <= cp <= hi:
                counts[lang] += 1

    dominant = max(counts, key=counts.get)
    return dominant if counts[dominant] > 0 else "unknown"


# ============================================================
# CHUNKING
# ============================================================

INDIC_SENTENCE_ENDINGS = re.compile(r'(?<=[।॥\.\!\?])\s+')


def chunk_text(text: str, chunk_size: int = 200, overlap: int = 40) -> List[str]:
    text = text.strip()
    if not text:
        return []

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
            current = current[-overlap:] + " " + sentence if overlap else sentence

    if current.strip():
        chunks.append(current.strip())

    # fallback
    if not chunks:
        for i in range(0, len(text), chunk_size - overlap):
            chunks.append(text[i:i + chunk_size])

    return chunks


# ============================================================
# MAIN INGESTION
# ============================================================

def ingest_documents(
    sources: List[Union[str, Path]],
    chunk_size: int = 200,
    overlap: int = 40,
    language_override: Optional[str] = None,
) -> List[dict]:

    records = []
    chunk_id = 0

    for source in sources:

        # ---------- FILE INPUT ----------
        if isinstance(source, Path) or (isinstance(source, str) and Path(source).exists()):
            path = Path(source)
            suffix = path.suffix.lower()

            if suffix == ".pdf":
                raw_text = extract_text_from_pdf(str(path))

            elif suffix in [".png", ".jpg", ".jpeg"]:
                raw_text = extract_text_from_image(str(path))

            else:
                raw_text = path.read_text(encoding="utf-8")

            source_name = path.name

        # ---------- RAW STRING ----------
        else:
            raw_text = str(source)
            source_name = "raw"

        # ---------- SAFETY ----------
        if not raw_text.strip():
            print(f"⚠️ Skipping empty document: {source_name}")
            continue

        # ---------- CHUNK ----------
        chunks = chunk_text(raw_text, chunk_size, overlap)

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


# ============================================================
# TEST
# ============================================================

if __name__ == "__main__":
    docs = [
        "भारत की राजधानी नई दिल्ली है।",
        "ಕರ್ನಾಟಕದ ರಾಜಧಾನಿ ಬೆಂಗಳೂರು."
    ]

    result = ingest_documents(docs)

    for r in result:
        print(f"[{r['language']}] {r['text']}")