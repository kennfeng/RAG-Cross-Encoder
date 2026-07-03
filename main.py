import ollama
from ingest import AtlasIngestor
from langchain_adapters import (
    ChromaRetrieverAdapter,
    CrossEncoderRerankerAdapter,
    OllamaLLMWrapper,
)
from rag_pipeline import LangChainRAG
from reranker import AtlasReRanker


class AtlasRAG:
    def __init__(
        self,
        ingestor=None,
        ranker=None,
        model=None,
        llm_client=None,
        sample_docs=None,
        n_results=10,
        top_n=3,
    ):
        print("--- Initializing RAG System ---")
        self.ingestor = ingestor or AtlasIngestor()

        # Check if DB is empty and add sample data if it is
        if self.ingestor.collection.count() == 0:
            print("[INFO]: DB is empty. Loading sample ML knowledge base...")
            sample_kb = sample_docs or [
                "PyTorch is an open source machine learning framework based on the Torch library.",
                "TensorFlow is a free and open-source software library for machine learning and artificial intelligence.",
                "RAG stands for Retrieval-Augmented Generation, a technique to provide external data to LLMs.",
                "A Cross-Encoder is a type of deep learning model that processes pairs of inputs simultaneously to determine relevance.",
                "Vector databases like ChromaDB store high-dimensional embeddings for fast similarity search.",
                "Ollama allows you to run large language models locally on your machine.",
                "Two-stage RAG uses a fast retriever (Stage 1) and a precise re-ranker (Stage 2) for better accuracy.",
            ]
            self.ingestor.add_documents(sample_kb)

        self.ranker = ranker or AtlasReRanker()
        self.model = model or "llama3.2:1b"
        self.llm_client = llm_client or ollama

        retriever = ChromaRetrieverAdapter(self.ingestor, n_results=n_results)
        reranker = CrossEncoderRerankerAdapter(self.ranker, top_n=top_n)
        llm = OllamaLLMWrapper(model_name=self.model, client=self.llm_client)
        self.pipeline = LangChainRAG(retriever, reranker, llm)

    def ask(self, query):
        print(f"\n[QUERY]: {query}")
        try:
            return self.pipeline.ask(query)
        except Exception as e:
            return {
                "answer": f"ERROR: Could not connect to Ollama ({str(e)})",
                "source_documents": [],
            }


if __name__ == "__main__":
    rag = LangChainRAG.from_defaults()
    while True:
        try:
            user_input = input("\nAsk Atlas (or type 'exit'): ")
            if user_input.lower() == "exit":
                break
            if not user_input.strip():
                continue

            result = rag.ask(user_input)
            print(f"\n--- RESPONSE ---\n{result['answer']}")

            if result["source_documents"]:
                print("\n--- SOURCES (Re-ranked) ---")
                for i, src in enumerate(result["source_documents"]):
                    print(f"{i+1}. [{src['score']:.4f}] {src['document']}")
        except KeyboardInterrupt:
            break
