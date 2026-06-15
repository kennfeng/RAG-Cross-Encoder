import torch
from sentence_transformers import CrossEncoder
import numpy as np

class AtlasReRanker:
    def __init__(self, model_name='BAAI/bge-reranker-base'):
        """
        Initializes a PyTorch-based Cross-Encoder.
        This model takes a Query, Document pair and outputs a relevancy score.
        """
        print(f"Loading PyTorch Re-ranker model: {model_name}...")
        # Check for GPU availability
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = CrossEncoder(model_name, device=self.device)
        print(f"Model loaded on {self.device}")

    def rerank(self, query, documents, top_n=3):
        """
        Re-ranks a list of documents based on their relevance to the query.
        """
        if not documents:
            return []

        # Prepare pairs for the Cross-Encoder
        pairs = [[query, doc] for doc in documents]
        
        # Predict relevancy scores
        scores = self.model.predict(pairs)
        
        # Sort documents by scores in descending order
        ranked_indices = np.argsort(scores)[::-1]
        
        results = [
            {"document": documents[i], "score": float(scores[i])} 
            for i in ranked_indices[:top_n]
        ]
        
        return results
    
    def rerank_with_ids(self, query, candidates, top_n=3):
        """
        Same as rerank() but with (id, document tuples)
        """
        if not candidates:
            return []
        
        pairs = [[query, doc] for _, doc in candidates]
        scores = self.model.predict(pairs)
        ranked_indices = np.argsort(scores)[::-1]

        results = [
            {
                "id": candidates[i][0],
                "document": candidates[i][1],
                "score": float(scores[i]),
            }
            for i in ranked_indices[:top_n]
        ]
        return results

if __name__ == "__main__":
    # Test block
    ranker = AtlasReRanker()
    test_query = "How do I build a RAG system?"
    test_docs = [
        "To build a RAG system, you need a vector database and an LLM.",
        "Making a sandwich requires bread, cheese, and ham.",
        "Retrieval-Augmented Generation (RAG) combines search with LLM generation for better accuracy.",
        "The weather today is sunny with a chance of rain."
    ]
    
    print("\nOriginal Documents Count:", len(test_docs))
    ranked = ranker.rerank(test_query, test_docs)
    
    print("\nTop Ranked Results:")
    for i, res in enumerate(ranked):
        print(f"{i+1}. [Score: {res['score']:.4f}] {res['document']}")
