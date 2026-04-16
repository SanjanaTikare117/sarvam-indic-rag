"""
pipeline.py
-----------
Main RAG orchestration for the Sarvam Indic Document Understanding system.
"""

from pathlib import Path
from typing import Optional

from .ingestion import ingest_documents, detect_language
from .embeddings import IndicEmbedder
from .retriever import IndicRetriever
from .generator import SarvamGenerator


class IndicRAGPipeline:
    def __init__(
        self,
        embed_model: str = "intfloat/multilingual-e5-base",
        sarvam_model: str = "sarvamai/sarvam-1",
        retrieval_only: bool = False,
        device: str = "cpu",
    ):
        self.retrieval_only = retrieval_only

        self.embedder = IndicEmbedder(model_name=embed_model, device=device)
        self.retriever = IndicRetriever(self.embedder)

        self.generator: Optional[SarvamGenerator] = None
        if not retrieval_only:
            self.generator = SarvamGenerator(model_name=sarvam_model)

    # ---------------------------
    # INGEST
    # ---------------------------
    def ingest(
        self,
        raw_docs: Optional[list[str]] = None,
        file_paths: Optional[list[str | Path]] = None,
        chunk_size: int = 200,
        overlap: int = 40,
    ) -> int:

        sources = []
        if raw_docs:
            sources.extend(raw_docs)
        if file_paths:
            sources.extend([Path(p) for p in file_paths])

        if not sources:
            raise ValueError("Provide at least one of raw_docs or file_paths.")

        chunks = ingest_documents(sources, chunk_size=chunk_size, overlap=overlap)

        # ✅ CRITICAL FIX (your previous crash)
        if not chunks:
            raise ValueError(
                "No text extracted — likely scanned PDF without OCR or empty file."
            )

        self.retriever.build(chunks)
        return len(chunks)

    # ---------------------------
    # QUERY
    # ---------------------------
    def query(
        self,
        question: str,
        top_k: int = 3,
        language_filter: Optional[str] = None,
        max_new_tokens: int = 200,
    ) -> dict:

        query_lang = detect_language(question)

        lang_map = {
            "hindi": "hindi",
            "kannada": "kannada",
            "english": "english",
        }
        prompt_lang = lang_map.get(query_lang, "hindi")

        retrieved = self.retriever.retrieve(
            question,
            top_k=top_k,
            language_filter=language_filter,
        )

        context = "\n".join(
            f"[{c.get('language','?')}] {c['text']}" for c in retrieved
        )

        answer = ""
        if not self.retrieval_only and self.generator:
            gen_result = self.generator.generate(
                query=question,
                context_chunks=retrieved,
                query_language=prompt_lang,
                max_new_tokens=max_new_tokens,
            )
            answer = gen_result["answer"]

        return {
            "question": question,
            "language": query_lang,
            "retrieved": retrieved,
            "answer": answer,
            "context": context,
        }

    # ---------------------------
    # SAVE / LOAD
    # ---------------------------
    def save_index(self, directory: str | Path) -> None:
        self.retriever.save(directory)

    def load_index(self, directory: str | Path) -> None:
        self.retriever.load(directory)

    # ---------------------------
    # EVAL
    # ---------------------------
    def evaluate_retrieval(self, eval_pairs: list[tuple[str, str]]) -> dict:

        correct = 0
        results = []

        for query, expected_lang in eval_pairs:
            hits = self.retriever.retrieve(query, top_k=1)

            if not hits:
                predicted_lang = "none"
                score = 0.0
            else:
                predicted_lang = hits[0].get("language", "unknown")
                score = hits[0]["score"]

            match = predicted_lang == expected_lang
            correct += int(match)

            results.append({
                "query": query,
                "expected": expected_lang,
                "predicted": predicted_lang,
                "score": score,
                "correct": match,
            })

        accuracy = correct / len(eval_pairs) if eval_pairs else 0.0

        return {
            "accuracy": accuracy,
            "correct": correct,
            "total": len(eval_pairs),
            "results": results,
        }


# ------------------------------------------------------------------
# Quick smoke test (retrieval-only, no GPU needed)
# ------------------------------------------------------------------
if __name__ == "__main__":
    hindi_docs = [
        "भारत एक विविधताओं से भरा देश है। यहाँ अनेक भाषाएं, धर्म और संस्कृतियाँ एक साथ रहती हैं।",
        "भारतीय संविधान 26 जनवरी 1950 को लागू हुआ था। यह दुनिया का सबसे बड़ा लिखित संविधान है।",
        "भारत की राजधानी नई दिल्ली है। यहाँ संसद भवन और राष्ट्रपति भवन स्थित हैं।",
        "हिंदी भारत की राजभाषा है। इसे देवनागरी लिपि में लिखा जाता है।",
        "गंगा नदी भारत की सबसे पवित्र नदी मानी जाती है।",
    ]
    kannada_docs = [
        "ಕರ್ನಾಟಕ ದಕ್ಷಿಣ ಭಾರತದ ಒಂದು ರಾಜ್ಯ. ಇದರ ರಾಜಧಾನಿ ಬೆಂಗಳೂರು.",
        "ಕನ್ನಡ ಭಾಷೆಯು ಕರ್ನಾಟಕದ ಅಧಿಕೃತ ಭಾಷೆ. ಇದು ದ್ರಾವಿಡ ಭಾಷಾ ಕುಟುಂಬಕ್ಕೆ ಸೇರಿದೆ.",
        "ಮೈಸೂರು ದಸರಾ ಹಬ್ಬ ವಿಶ್ವ ಪ್ರಸಿದ್ಧ. ಇದನ್ನು ನಾಡ ಹಬ್ಬ ಎಂದು ಕರೆಯುತ್ತಾರೆ.",
        "ಹಂಪಿ ಕರ್ನಾಟಕದ ಪ್ರಮುಖ ಐತಿಹಾಸಿಕ ಸ್ಥಳ.",
        "ಕಾವೇರಿ ನದಿ ಕರ್ನಾಟಕ ಮತ್ತು ತಮಿಳುನಾಡಿನ ಜೀವನಾಡಿ.",
    ]

    print("=" * 60)
    print("Sarvam Indic RAG — Retrieval-Only Smoke Test")
    print("=" * 60)

    pipe = IndicRAGPipeline(retrieval_only=True)
    n = pipe.ingest(raw_docs=hindi_docs + kannada_docs)
    print(f"\nIndexed {n} chunks\n")

    test_queries = [
        "भारत की राजधानी कहाँ है?",
        "गंगा नदी कहाँ बहती है?",
        "ಕರ್ನಾಟಕದ ರಾಜಧಾನಿ ಯಾವುದು?",
        "ಹಂಪಿ ಯಾಕೆ ಪ್ರಸಿದ್ಧ?",
        "What is the capital of Karnataka?",
    ]

    for q in test_queries:
        result = pipe.query(q, top_k=2)
        print(f"Query [{result['language']}]: {q}")
        for r in result["retrieved"]:
            print(f"  ↳ [{r['language']}] score={r['score']:.3f} | {r['text'][:65]}")
        print()

    # Evaluation
    eval_pairs = [
        ("भारत की राजधानी", "hindi"),
        ("संविधान लागू", "hindi"),
        ("ಕರ್ನಾಟಕ ರಾಜ್ಯ", "kannada"),
        ("ಮೈಸೂರು ದಸರಾ", "kannada"),
        ("ಹಂಪಿ ಇತಿಹಾಸ", "kannada"),
    ]
    eval_result = pipe.evaluate_retrieval(eval_pairs)
    print(f"\nRetrieval accuracy: {eval_result['correct']}/{eval_result['total']} = {eval_result['accuracy']*100:.0f}%")
    for r in eval_result["results"]:
        status = "✓" if r["correct"] else "✗"
        print(f"  {status} '{r['query'][:30]}' → expected={r['expected']}, got={r['predicted']} (score={r['score']:.3f})")