import ollama


class ChromaRetrieverAdapter:
    def __init__(self, ingestor, n_results=10):
        self.ingestor = ingestor
        self.n_results = n_results

    def get_relevant_documents(self, query):
        return self.ingestor.search(query, n_results=self.n_results)

    def get_relevant_documents_with_ids(self, query):
        return self.ingestor.search_with_ids(query, n_results=self.n_results)


class CrossEncoderRerankerAdapter:
    def __init__(self, ranker, top_n=3):
        self.ranker = ranker
        self.top_n = top_n

    def rerank(self, query, candidates):
        return self.ranker.rerank_with_ids(query, candidates, top_n=self.top_n)


class OllamaLLMWrapper:
    def __init__(self, model_name="llama3.2:1b", client=None):
        self.model_name = model_name
        self.client = client or ollama

    def chat(self, prompt):
        response = self.client.chat(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
        )
        return response["message"]["content"]
