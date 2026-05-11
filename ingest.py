import chromadb
from chromadb.utils import embedding_functions
import os

class AtlasIngestor:
    def __init__(self, db_path="./atlas_db"):
        """
        Initializes the Vector Database (ChromaDB)
        """
        self.client = chromadb.PersistentClient(path=db_path)
        # Using a standard sentence-transformer model for embeddings
        self.emb_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name="all-MiniLM-L6-v2"
        )
        self.collection = self.client.get_or_create_collection(
            name="documents", 
            embedding_function=self.emb_fn
        )

    def add_documents(self, text_list, metadata_list=None):
        """
        Chunks and adds documents to the vector store.
        """
        ids = [f"id_{i}" for i in range(len(text_list))]
        self.collection.add(
            documents=text_list,
            metadatas=metadata_list,
            ids=ids
        )
        print(f"Successfully added {len(text_list)} documents to the vector store.")

    def search(self, query, n_results=10):
        """
        Fast Vector Search (Similarity Search).
        Returns top-N candidate documents.
        """
        results = self.collection.query(
            query_texts=[query],
            n_results=n_results
        )
        # Flatten the results
        return results['documents'][0]

if __name__ == "__main__":
    ingestor = AtlasIngestor()
    
    # Test knowledge base
    sample_kb = [
        "PyTorch is an open source machine learning framework based on the Torch library.",
        "TensorFlow is a free and open-source software library for machine learning and artificial intelligence.",
        "RAG stands for Retrieval-Augmented Generation, a technique to provide external data to LLMs.",
        "A Cross-Encoder is a type of deep learning model that processes pairs of inputs simultaneously.",
        "Vector databases like ChromaDB store high-dimensional embeddings for fast similarity search.",
        "Gradient descent is an optimization algorithm used to minimize the loss function in ML models.",
        "Transformers are a type of neural network architecture that has revolutionized NLP.",
        "Ollama allows you to run large language models locally on your machine."
    ]
    
    ingestor.add_documents(sample_kb)
    
    # Test block
    query = "What is RAG and why use a vector DB?"
    candidates = ingestor.search(query, n_results=3)
    
    print(f"\nQuery: {query}")
    print("Top Candidate Matches:")
    for i, doc in enumerate(candidates):
        print(f"{i+1}. {doc}")
