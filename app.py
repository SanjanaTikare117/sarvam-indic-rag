from fastapi import FastAPI
from pydantic import BaseModel
from src.pipeline import IndicRAGPipeline
from fastapi import FastAPI, UploadFile, File
import shutil, os

app = FastAPI()

# load once (very important)
pipe = IndicRAGPipeline(retrieval_only=True)

# sample ingestion (later replace with files)
pipe.ingest(raw_docs=[
    "भारत की राजधानी नई दिल्ली है।",
    "ಕರ್ನಾಟಕ ದಕ್ಷಿಣ ಭಾರತದ ರಾಜ್ಯ. ರಾಜಧಾನಿ ಬೆಂಗಳೂರು.",
])

class QueryRequest(BaseModel):
    query: str

@app.post("/query")
def query_rag(request: QueryRequest):
    result = pipe.query(request.query)
    top_chunk = result["retrieved"][0]["text"] if result["retrieved"] else "No answer found"
    return {
        "answer": top_chunk,
        "retrieved": result["retrieved"]
    }
@app.post("/ingest")
def ingest_pdf(file: UploadFile = File(...)):
    # Save uploaded file temporarily
    temp_path = f"F:/Sarvam/docs/{file.filename}"
    os.makedirs("F:/Sarvam/docs", exist_ok=True)
    with open(temp_path, "wb") as f:
        shutil.copyfileobj(file.file, f)
    
    # Ingest it
    pipe.ingest(file_paths=[temp_path])
    return {"status": "ingested", "file": file.filename}