from typing import Any

import chromadb
from chromadb.utils import embedding_functions


def _chunk_text(
    text: str,
    chunk_size: int = 512,
    chunk_overlap: int = 64,
) -> list[str]:
    if chunk_size <= 0:
        return [text]
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = start + chunk_size
        chunk = " ".join(words[start:end])
        chunks.append(chunk)
        start += chunk_size - chunk_overlap
        if start >= len(words):
            break
    return chunks if chunks else [text]


class AtlasIngestor:
    def __init__(
        self,
        db_path: str = "./atlas_db",
        embedding_model_name: str = "all-MiniLM-L6-v2",
        collection_name: str = "documents",
    ) -> None:
        self.client = chromadb.PersistentClient(path=db_path)
        self.emb_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=embedding_model_name
        )
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            embedding_function=self.emb_fn,
            metadata={"hnsw:space": "cosine"},
        )

    def add_documents(
        self,
        text_list: list[str],
        metadata_list: list[dict[str, Any]] | None = None,
        ids: list[str] | None = None,
        chunk_size: int | None = None,
        chunk_overlap: int = 64,
    ) -> None:
        if chunk_size is not None and chunk_size > 0:
            chunked_texts: list[str] = []
            chunked_metadata: list[dict[str, Any]] | None = [] if metadata_list is not None else None
            chunked_ids: list[str] = []

            for i, text in enumerate(text_list):
                meta = metadata_list[i] if metadata_list is not None else None
                doc_id = ids[i] if ids is not None else f"id_{i}"
                chunks = _chunk_text(text, chunk_size, chunk_overlap)
                for j, chunk in enumerate(chunks):
                    chunked_texts.append(chunk)
                    if chunked_metadata is not None:
                        chunked_metadata.append({**(meta or {}), "_chunk_index": j, "_parent_id": doc_id})
                    chunked_ids.append(f"{doc_id}_chunk_{j}")

            text_list = chunked_texts
            metadata_list = chunked_metadata
            ids = chunked_ids

        if ids is None:
            ids = [f"id_{i}" for i in range(len(text_list))]

        self.collection.add(documents=text_list, metadatas=metadata_list, ids=ids)

    def search(
        self,
        query: str,
        n_results: int = 10,
        where: dict[str, Any] | None = None,
    ) -> list[str]:
        results = self.collection.query(
            query_texts=[query],
            n_results=n_results,
            where=where,
        )
        return results["documents"][0]

    def search_with_ids(
        self,
        query: str,
        n_results: int = 10,
        where: dict[str, Any] | None = None,
    ) -> list[tuple[str, str]]:
        results = self.collection.query(
            query_texts=[query],
            n_results=n_results,
            where=where,
        )
        return list(zip(results["ids"][0], results["documents"][0]))


if __name__ == "__main__":
    ingestor = AtlasIngestor()

    sample_kb = [
        "PyTorch is an open source machine learning framework based on the Torch library.",
        "TensorFlow is a free and open-source software library for machine learning and artificial intelligence.",
        "RAG stands for Retrieval-Augmented Generation, a technique to provide external data to LLMs.",
        "A Cross-Encoder is a type of deep learning model that processes pairs of inputs simultaneously.",
        "Vector databases like ChromaDB store high-dimensional embeddings for fast similarity search.",
        "Gradient descent is an optimization algorithm used to minimize the loss function in ML models.",
        "Transformers are a type of neural network architecture that has revolutionized NLP.",
        "Ollama allows you to run large language models locally on your machine.",
    ]

    ingestor.add_documents(sample_kb)

    query = "What is RAG and why use a vector DB?"
    candidates = ingestor.search(query, n_results=3)

    print(f"\nQuery: {query}")
    print("Top Candidate Matches:")
    for i, doc in enumerate(candidates):
        print(f"{i+1}. {doc}")
