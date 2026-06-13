import ollama
from ingest import AtlasIngestor
from reranker import AtlasReRanker

class AtlasRAG:
    def __init__(self):
        print("--- Initializing RAG System ---")
        self.ingestor = AtlasIngestor()
        
        # Check if DB is empty and add sample data if it is
        if self.ingestor.collection.count() == 0:
            print("[INFO]: DB is empty. Loading sample ML knowledge base...")
            sample_kb = [
                "PyTorch is an open source machine learning framework based on the Torch library.",
                "TensorFlow is a free and open-source software library for machine learning and artificial intelligence.",
                "RAG stands for Retrieval-Augmented Generation, a technique to provide external data to LLMs.",
                "A Cross-Encoder is a type of deep learning model that processes pairs of inputs simultaneously to determine relevance.",
                "Vector databases like ChromaDB store high-dimensional embeddings for fast similarity search.",
                "Ollama allows you to run large language models locally on your machine.",
                "Two-stage RAG uses a fast retriever (Stage 1) and a precise re-ranker (Stage 2) for better accuracy."
            ]
            self.ingestor.add_documents(sample_kb)

        self.ranker = AtlasReRanker()
        self.model = "llama3.2:1b"

    def ask(self, query):
        print(f"\n[QUERY]: {query}")
        
        # --- Stage 1: Retrieval ---
        candidates = self.ingestor.search(query, n_results=10)
        if not candidates:
            return {"answer": "I couldn't find any relevant documents in the database.", "source_documents": []}
        
        # --- Stage 2: Re-ranking (PyTorch) ---
        print(f"[STAGE 2]: Re-ranking {len(candidates)} candidates using PyTorch Cross-Encoder...")
        ranked_results = self.ranker.rerank(query, candidates, top_n=3)
        context_docs = [res['document'] for res in ranked_results]
        
        # --- Stage 3: Generation (Ollama) ---
        print("[STAGE 3]: Generating final answer with Ollama...")
        context_text = "\n\n".join(context_docs)
        
        prompt = f"Context:\n{context_text}\n\nQuestion: {query}\n\nAnswer concisely based on the context:"

        try:
            response = ollama.chat(model=self.model, messages=[
                {'role': 'user', 'content': prompt}
            ])
            return {
                "answer": response['message']['content'],
                "source_documents": ranked_results
            }
        except Exception as e:
            return {
                "answer": f"ERROR: Could not connect to Ollama ({str(e)})",
                "source_documents": []
            }

if __name__ == "__main__":
    rag = AtlasRAG()
    while True:
        try:
            user_input = input("\nAsk Atlas (or type 'exit'): ")
            if user_input.lower() == 'exit': break
            if not user_input.strip(): continue
                
            result = rag.ask(user_input)
            print(f"\n--- RESPONSE ---\n{result['answer']}")
            
            if result['source_documents']:
                print("\n--- SOURCES (Re-ranked) ---")
                for i, src in enumerate(result['source_documents']):
                    print(f"{i+1}. [{src['score']:.4f}] {src['document']}")
        except KeyboardInterrupt:
            break
