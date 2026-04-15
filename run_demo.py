from src.pipeline import IndicRAGPipeline

pipe = IndicRAGPipeline(retrieval_only=True)

pipe.ingest(raw_docs=[
    "भारत की राजधानी नई दिल्ली है।",
    "ಕರ್ನಾಟಕ ದಕ್ಷಿಣ ಭಾರತದ ರಾಜ್ಯ. ರಾಜಧಾನಿ ಬೆಂಗಳೂರು.",
])

result = pipe.query("भारत की राजधानी कहाँ है?")

print("\n=== RESULT ===")
print(result)

print("\n=== TOP RETRIEVED ===")
print(result["retrieved"][0]["text"])