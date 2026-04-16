Sarvam Indic RAG — Multilingual Document Understanding System
A retrieval-augmented generation (RAG) pipeline for Indian language documents, built using Sarvam AI's open-source models. Supports Hindi and Kannada out of the box, with architecture ready for all major Indic scripts.

ARCHITECTURE
docs/ (PDF, TXT)
     │
     ▼
ingestion.py        ← Script-aware chunking (Devanagari, Kannada, Tamil, Telugu...)
     │
     ▼
embeddings.py       ← multilingual-e5-base (768-dim vectors)
     │
     ▼
retriever.py        ← FAISS vector index
     │
     ▼
generator.py        ← sarvamai/sarvam-1 (optional, requires GPU)
     │
     ▼
pipeline.py         ← IndicRAGPipeline orchestrator
     │
     ▼
app.py              ← FastAPI REST API
## Roadmap
- [x] Hindi + Kannada PDF ingestion
- [x] OCR fallback for scanned image PDFs
- [x] REST API with FastAPI
- [ ] Tamil and Telugu PDF samples
- [ ] Streaming API responses
- [ ] Full generation mode with sarvam-1 on GPU
Features
Indic script detection — auto-detects Hindi, Kannada, Tamil, Telugu, Bengali, English
Sentence-aware chunking — splits on । (Devanagari) and ॥ in addition to standard punctuation
PDF ingestion — upload PDFs via REST API, chunks and indexes automatically
Cross-language retrieval — query in Hindi, retrieve relevant Kannada chunks and vice versa
Retrieval-only mode — runs fast on CPU without needing the full LLM
Retrieval evaluation — built-in accuracy scoring for retrieval quality
- **OCR support** — scanned Kannada/Hindi image PDFs automatically fall back to Tesseract OCR (eng+hin+kan)
Quick Start
1. Install dependencies
bashpip install -r requirements.txt
2. Start the API server
bashpython -m uvicorn app:app --reload
3. Ingest a document
bashcurl -X POST "http://127.0.0.1:8000/ingest" \
  -F "file=@docs/hindi_sample.pdf"
4. Query in Hindi
bashcurl -X POST "http://127.0.0.1:8000/query" \
  -H "Content-Type: application/json" \
  -d '{"query": "भारत की राजधानी क्या है?"}'
5. Query in Kannada
bashcurl -X POST "http://127.0.0.1:8000/query" \
  -H "Content-Type: application/json" \
  -d '{"query": "ಕರ್ನಾಟಕದ ರಾಜಧಾನಿ ಯಾವುದು?"}'

  API Endpoints
MethodEndpointDescriptionPOST/ingestUpload and index a PDF or text filePOST/queryQuery the RAG pipelineGET/docsInteractive Swagger UI

Requirements
fastapi
uvicorn
sentence-transformers
faiss-cpu
PyPDF2
transformers
accelerate
torch
pydantic
huggingface-hub
fpdf2

Project Structure
Sarvam/
├── src/
│   ├── embeddings.py     # multilingual-e5-base wrapper
│   ├── ingestion.py      # PDF/text loading + Indic chunking
│   ├── retriever.py      # FAISS index build + search
│   ├── generator.py      # sarvam-1 generation wrapper
│   └── pipeline.py       # IndicRAGPipeline orchestrator
├── docs/                 # Sample Hindi + Kannada PDFs
├── app.py                # FastAPI server
├── run_demo.py           # Quick terminal demo (no server needed)
├── make_pdf.py           # Script to generate clean test PDFs
└── requirements.txt

Models Used
ModelPurposeintfloat/multilingual-e5-baseMultilingual embeddings (50+ languages)sarvamai/sarvam-1Indic LLM for answer generation (optional)

Retrieval Evaluation
bashpython -m src.pipeline
Sample output:
Retrieval accuracy: 5/5 = 100%
  ✓ 'भारत की राजधानी' → expected=hindi, got=hindi (score=0.921)
  ✓ 'ಕರ್ನಾಟಕ ರಾಜ್ಯ'  → expected=kannada, got=kannada (score=0.908)
  <p align="center">
  <img src="https://github.com/user-attachments/assets/6e767460-0c59-4db5-8f0c-31b6407b4e88" width="800"/>
</p>
