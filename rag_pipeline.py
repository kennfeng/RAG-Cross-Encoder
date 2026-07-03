from ingest import AtlasIngestor
from reranker import AtlasReRanker
from langchain_adapters import (
    ChromaRetrieverAdapter,
    CrossEncoderRerankerAdapter,
    OllamaLLMWrapper,
)


class LangChainRAG:
    def __init__(self, retriever, reranker, llm):
        self.retriever = retriever
        self.reranker = reranker
        self.llm = llm

    @classmethod
    def from_defaults(
        cls,
        db_path="./atlas_db",
        model_name="BAAI/bge-reranker-base",
        llm_model_name="llama3.2:1b",
        n_results=10,
        top_n=3,
        sample_docs=None,
    ):
        ingestor = AtlasIngestor(db_path=db_path)
        if ingestor.collection.count() == 0:
            sample_kb = sample_docs or [
                "PyTorch is an open source machine learning framework based on the Torch library.",
                "TensorFlow is a free and open-source software library for machine learning and artificial intelligence.",
                "RAG stands for Retrieval-Augmented Generation, a technique to provide external data to LLMs.",
                "A Cross-Encoder is a type of deep learning model that processes pairs of inputs simultaneously to determine relevance.",
                "Vector databases like ChromaDB store high-dimensional embeddings for fast similarity search.",
                "Ollama allows you to run large language models locally on your machine.",
                "Two-stage RAG uses a fast retriever (Stage 1) and a precise re-ranker (Stage 2) for better accuracy.",
            ]
            ingestor.add_documents(sample_kb)

        ranker = AtlasReRanker(model_name=model_name)
        retriever = ChromaRetrieverAdapter(ingestor, n_results=n_results)
        reranker = CrossEncoderRerankerAdapter(ranker, top_n=top_n)
        llm = OllamaLLMWrapper(model_name=llm_model_name)

        return cls(retriever, reranker, llm)

    def ask(self, query):
        candidates = self.retriever.get_relevant_documents_with_ids(query)
        if not candidates:
            return {
                "answer": "I couldn't find any relevant documents in the database.",
                "source_documents": [],
            }

        ranked_results = self.reranker.rerank(query, candidates)
        context_docs = [res["document"] for res in ranked_results]
        prompt = f"Context:\n{'\n\n'.join(context_docs)}\n\nQuestion: {query}\n\nAnswer concisely based on the context:"

        answer = self.llm.chat(prompt)
        return {
            "answer": answer,
            "source_documents": ranked_results,
        }
