"""
retriever.py
------------
FAISS-based vector store for Indic document retrieval.
Supports:
  - Building index from chunk records
  - Saving / loading index + metadata from disk
  - Top-k retrieval with language filtering
"""

import json
import pickle
from pathlib import Path
from typing import Optional

import faiss
import numpy as np

from .embeddings import IndicEmbedder


class IndicRetriever:
    """
    Wraps a FAISS IndexFlatIP (inner product on normalized vectors = cosine)
    with chunk metadata for Indic document retrieval.
    """

    def __init__(self, embedder: IndicEmbedder):
        self.embedder = embedder
        self.index: Optional[faiss.Index] = None
        self.chunks: list[dict] = []   # parallel to index rows

    # ------------------------------------------------------------------
    # Building the index
    # ------------------------------------------------------------------

    def build(self, chunk_records: list[dict], batch_size: int = 16) -> None:
        """
        Build FAISS index from a list of chunk dicts (from ingestion.py).
        Each dict must have at least {"text": str}.
        """
        texts = [c["text"] for c in chunk_records]
        self.chunks = chunk_records

        print(f"Embedding {len(texts)} chunks...")
        embeddings = self.embedder.embed_passages(texts, batch_size=batch_size)

        dim = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dim)
        self.index.add(embeddings)
        print(f"FAISS index built: {self.index.ntotal} vectors, dim={dim}")

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, directory: str | Path) -> None:
        """Save FAISS index + chunk metadata to disk."""
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)

        faiss.write_index(self.index, str(directory / "index.faiss"))
        with open(directory / "chunks.pkl", "wb") as f:
            pickle.dump(self.chunks, f)
        print(f"Retriever saved to {directory}/")

    def load(self, directory: str | Path) -> None:
        """Load FAISS index + chunk metadata from disk."""
        directory = Path(directory)
        self.index = faiss.read_index(str(directory / "index.faiss"))
        with open(directory / "chunks.pkl", "rb") as f:
            self.chunks = pickle.load(f)
        print(f"Retriever loaded: {self.index.ntotal} vectors from {directory}/")

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------

    def retrieve(
        self,
        query: str,
        top_k: int = 3,
        language_filter: Optional[str] = None,
        score_threshold: float = 0.0,
    ) -> list[dict]:
        """
        Retrieve top_k chunks for a query.

        Args:
            query:           Natural language query (any Indic language or English)
            top_k:           Number of results to return
            language_filter: If set, only return chunks of this language
                             e.g. "hindi" or "kannada"
            score_threshold: Minimum cosine similarity to include

        Returns:
            List of dicts with keys: text, language, source, chunk_id, score
        """
        if self.index is None:
            raise RuntimeError("Index not built. Call build() or load() first.")

        q_vec = self.embedder.embed_query(query)

        # Over-fetch if filtering by language so we still get top_k after filter
        fetch_k = top_k * 4 if language_filter else top_k
        scores, indices = self.index.search(q_vec, min(fetch_k, self.index.ntotal))

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0:
                continue
            chunk = self.chunks[idx].copy()
            chunk["score"] = float(score)

            if float(score) < score_threshold:
                continue
            if language_filter and chunk.get("language") != language_filter:
                continue

            results.append(chunk)
            if len(results) == top_k:
                break

        return results

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def language_distribution(self) -> dict[str, int]:
        """Count chunks per language in the index."""
        dist: dict[str, int] = {}
        for chunk in self.chunks:
            lang = chunk.get("language", "unknown")
            dist[lang] = dist.get(lang, 0) + 1
        return dist


if __name__ == "__main__":
    from ingestion import ingest_documents

    hindi_docs = [
        "भारत एक विविधताओं से भरा देश है। यहाँ अनेक भाषाएं, धर्म और संस्कृतियाँ एक साथ रहती हैं।",
        "भारतीय संविधान 26 जनवरी 1950 को लागू हुआ था। यह दुनिया का सबसे बड़ा लिखित संविधान है।",
        "भारत की राजधानी नई दिल्ली है। यहाँ संसद भवन और राष्ट्रपति भवन स्थित हैं।",
    ]
    kannada_docs = [
        "ಕರ್ನಾಟಕ ದಕ್ಷಿಣ ಭಾರತದ ಒಂದು ರಾಜ್ಯ. ಇದರ ರಾಜಧಾನಿ ಬೆಂಗಳೂರು.",
        "ಹಂಪಿ ಕರ್ನಾಟಕದ ಪ್ರಮುಖ ಐತಿಹಾಸಿಕ ಸ್ಥಳ. ಇದು ವಿಜಯನಗರ ಸಾಮ್ರಾಜ್ಯದ ರಾಜಧಾನಿ ಆಗಿತ್ತು.",
    ]

    chunks = ingest_documents(hindi_docs + kannada_docs)
    embedder = IndicEmbedder()
    retriever = IndicRetriever(embedder)
    retriever.build(chunks)

    print("\nLanguage distribution:", retriever.language_distribution())

    query = "भारत की राजधानी कहाँ है?"
    results = retriever.retrieve(query, top_k=2)
    print(f"\nQuery: {query}")
    for r in results:
        print(f"  [{r['language']}] score={r['score']:.3f} | {r['text'][:70]}")