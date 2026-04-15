"""
embeddings.py
-------------
Handles document and query embedding using multilingual-e5-base.
This model handles Devanagari (Hindi) and Kannada scripts natively,
making it ideal for Indic RAG pipelines.

Why multilingual-e5 and NOT Sarvam-1 for embeddings:
  Sarvam-1 is a generative LLM (decoder-only). It does not produce
  sentence embeddings directly. multilingual-e5 is a dedicated bi-encoder
  trained on 100+ languages including Indic scripts, and gives strong
  cross-lingual retrieval (Hindi query → Kannada doc, etc).
"""

import numpy as np
from sentence_transformers import SentenceTransformer

DEFAULT_EMBED_MODEL = "intfloat/multilingual-e5-small"


class IndicEmbedder:
    """
    Wrapper around multilingual-e5 for passage and query embedding.

    e5 models require prefixes:
        - "passage: " for documents being indexed
        - "query: "   for queries at retrieval time
    These prefixes are handled automatically here.
    """

    def __init__(self, model_name: str = DEFAULT_EMBED_MODEL, device: str = "cpu"):
        print(f"Loading embedding model: {model_name}")
        self.model = SentenceTransformer(model_name, device=device)
        self.model_name = model_name
        print("Embedding model ready.")

    def embed_passages(
        self,
        texts: list[str],
        batch_size: int = 16,
        show_progress: bool = True,
    ) -> np.ndarray:
        """
        Embed a list of document passages.
        Adds 'passage: ' prefix as required by e5.
        Returns L2-normalized float32 array of shape (N, dim).
        """
        prefixed = ["passage: " + t for t in texts]
        embeddings = self.model.encode(
            prefixed,
            batch_size=batch_size,
            show_progress_bar=show_progress,
            normalize_embeddings=True,   # cosine = inner product on normalized vecs
            convert_to_numpy=True,
        )
        return embeddings.astype(np.float32)

    def embed_query(self, query: str) -> np.ndarray:
        """
        Embed a single query string.
        Adds 'query: ' prefix as required by e5.
        Returns L2-normalized float32 array of shape (1, dim).
        """
        prefixed = "query: " + query
        embedding = self.model.encode(
            [prefixed],
            normalize_embeddings=True,
            convert_to_numpy=True,
        )
        return embedding.astype(np.float32)

    @property
    def embedding_dim(self) -> int:
        return self.model.get_sentence_embedding_dimension()


if __name__ == "__main__":
    embedder = IndicEmbedder()
    print(f"Embedding dimension: {embedder.embedding_dim}")

    test_texts = [
        "भारत की राजधानी नई दिल्ली है।",
        "ಕರ್ನಾಟಕದ ರಾಜಧಾನಿ ಬೆಂಗಳೂರು.",
    ]
    vecs = embedder.embed_passages(test_texts, show_progress=False)
    print(f"Passage embeddings shape: {vecs.shape}")

    q_vec = embedder.embed_query("What is the capital of India?")
    print(f"Query embedding shape: {q_vec.shape}")

    # Cross-lingual similarity sanity check
    sim_hindi = float(q_vec @ vecs[0])
    sim_kannada = float(q_vec @ vecs[1])
    print(f"Similarity to Hindi doc (capital of India):   {sim_hindi:.3f}")
    print(f"Similarity to Kannada doc (capital of KA):   {sim_kannada:.3f}")
    print("Hindi doc should score higher for this English query.")