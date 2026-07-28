import sys

from langchain_core.messages import HumanMessage


def create_llm(provider="ollama", model_name="llama3.2:1b", **kwargs):
    """Factory that returns a LangChain BaseChatModel for the given provider.

    Supported providers:
        - "ollama"  (langchain_ollama.ChatOllama)
    """
    if provider == "ollama":
        from langchain_ollama import ChatOllama

        return ChatOllama(model=model_name, **kwargs)

    raise ValueError(
        f"Unknown provider: {provider!r}. Supported: 'ollama'."
    )


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
