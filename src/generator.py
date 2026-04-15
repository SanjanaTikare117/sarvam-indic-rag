"""
generator.py
------------
Answer generation using Sarvam-1 (sarvamai/sarvam-1).

Sarvam-1 is a 2B parameter decoder-only LLM trained primarily on
Indian language data. It understands Hindi, Kannada, and several
other Indic languages natively.

Note on hardware:
  - float16 recommended (requires ~4GB GPU VRAM)
  - CPU inference is possible but slow; use max_new_tokens <= 100
  - Set device_map="cpu" if no GPU is available
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import Optional

SARVAM_MODEL = "sarvamai/sarvam-1"

# Prompt templates for different query languages.
# Sarvam-1 responds better when the instruction language matches
# the query language — this is a key Indic-specific nuance.
PROMPT_TEMPLATES = {
    "hindi": (
        "नीचे दी गई जानकारी के आधार पर प्रश्न का उत्तर दें।\n\n"
        "संदर्भ:\n{context}\n\n"
        "प्रश्न: {query}\n\n"
        "उत्तर:"
    ),
    "kannada": (
        "ಕೆಳಗೆ ನೀಡಿದ ಮಾಹಿತಿಯ ಆಧಾರದ ಮೇಲೆ ಪ್ರಶ್ನೆಗೆ ಉತ್ತರಿಸಿ.\n\n"
        "ಸಂದರ್ಭ:\n{context}\n\n"
        "ಪ್ರಶ್ನೆ: {query}\n\n"
        "ಉತ್ತರ:"
    ),
    "english": (
        "Answer the question based only on the context below.\n\n"
        "Context:\n{context}\n\n"
        "Question: {query}\n\n"
        "Answer:"
    ),
}


class SarvamGenerator:
    """
    Wraps Sarvam-1 for RAG-style answer generation.
    """

    def __init__(
        self,
        model_name: str = SARVAM_MODEL,
        device_map: str = "auto",
        torch_dtype=torch.float16,
    ):
        print(f"Loading tokenizer: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        print(f"Loading model: {model_name}  (this may take 3-5 min on first run)")
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch_dtype,
            device_map=device_map,
            trust_remote_code=True,
        )
        self.model.eval()
        print("Sarvam-1 model ready.")

    def _build_prompt(
        self,
        query: str,
        context: str,
        query_language: str = "hindi",
    ) -> str:
        template = PROMPT_TEMPLATES.get(query_language, PROMPT_TEMPLATES["hindi"])
        return template.format(context=context, query=query)

    def generate(
        self,
        query: str,
        context_chunks: list[dict],
        query_language: str = "hindi",
        max_new_tokens: int = 200,
        do_sample: bool = False,
        temperature: float = 1.0,
    ) -> dict:
        """
        Generate an answer given a query and retrieved context chunks.

        Args:
            query:          The user's question
            context_chunks: List of chunk dicts from retriever.retrieve()
            query_language: "hindi" | "kannada" | "english"
            max_new_tokens: Max tokens to generate
            do_sample:      Use sampling (False = greedy, more factual)
            temperature:    Sampling temperature (only used if do_sample=True)

        Returns:
            Dict with keys: answer, prompt, num_input_tokens
        """
        # Build context string from retrieved chunks
        context = "\n".join(
            f"[{c.get('language', '?')}] {c['text']}" for c in context_chunks
        )

        prompt = self._build_prompt(query, context, query_language)

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        input_len = inputs["input_ids"].shape[1]

        with torch.no_grad():
            output = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                temperature=temperature if do_sample else 1.0,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        # Decode only newly generated tokens (skip prompt)
        generated_ids = output[0][input_len:]
        answer = self.tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

        return {
            "answer": answer,
            "prompt": prompt,
            "num_input_tokens": input_len,
        }


if __name__ == "__main__":
    # Smoke test — requires GPU and the model downloaded
    gen = SarvamGenerator()

    sample_chunks = [
        {
            "text": "भारतीय संविधान 26 जनवरी 1950 को लागू हुआ था।",
            "language": "hindi",
        }
    ]
    result = gen.generate(
        query="भारत का संविधान कब लागू हुआ?",
        context_chunks=sample_chunks,
        query_language="hindi",
        max_new_tokens=80,
    )
    print("Answer:", result["answer"])