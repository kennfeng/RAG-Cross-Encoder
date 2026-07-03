from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage


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
        try:
            self.client = client or ChatOllama(model=self.model_name)
        except Exception as exc:
            raise RuntimeError(
                f"Failed to initialize Ollama client for model '{self.model_name}'. "
                "Ensure Ollama is running and the model name is valid."
            ) from exc

    def chat(self, prompt):
        try:
            response = self.client.invoke([HumanMessage(content=prompt)])
        except Exception as exc:
            raise RuntimeError(
                "Ollama invocation failed. Ensure Ollama is reachable and the prompt is valid."
            ) from exc
        return response.content
