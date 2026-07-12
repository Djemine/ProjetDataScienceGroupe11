from rag_engine import RAGEngine

engine = RAGEngine()
query = "Quels aliments donner à un enfant de moins de 5 ans ?"
results = engine.retrieve(query, top_k=5)

for i, r in enumerate(results, 1):
    print(f"\n--- Résultat {i} : {r['source']} (score={r['score']:.3f}) ---")
    print(r['text'][:400])