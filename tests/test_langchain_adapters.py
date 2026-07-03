from unittest.mock import MagicMock

from langchain_core.messages import HumanMessage

from langchain_adapters import (
    ChromaRetrieverAdapter,
    CrossEncoderRerankerAdapter,
    OllamaLLMWrapper,
)


def test_chroma_retriever_adapter_uses_ingestor_search():
    ingestor = MagicMock()
    ingestor.search.return_value = ["doc1", "doc2"]

    adapter = ChromaRetrieverAdapter(ingestor, n_results=5)
    results = adapter.get_relevant_documents("query")

    ingestor.search.assert_called_once_with("query", n_results=5)
    assert results == ["doc1", "doc2"]


def test_chroma_retriever_adapter_uses_ingestor_search_with_ids():
    ingestor = MagicMock()
    ingestor.search_with_ids.return_value = [("id_1", "doc1")]

    adapter = ChromaRetrieverAdapter(ingestor, n_results=7)
    results = adapter.get_relevant_documents_with_ids("query")

    ingestor.search_with_ids.assert_called_once_with("query", n_results=7)
    assert results == [("id_1", "doc1")]


def test_cross_encoder_reranker_adapter_reranks_candidates():
    ranker = MagicMock()
    ranker.rerank_with_ids.return_value = [
        {"id": "id_2", "document": "high", "score": 0.9}
    ]

    adapter = CrossEncoderRerankerAdapter(ranker, top_n=1)
    results = adapter.rerank("query", [("id_1", "low"), ("id_2", "high")])

    ranker.rerank_with_ids.assert_called_once_with(
        "query", [("id_1", "low"), ("id_2", "high")], top_n=1
    )
    assert results == [{"id": "id_2", "document": "high", "score": 0.9}]


def test_ollama_llm_wrapper_uses_invoke_and_returns_content():
    response = MagicMock()
    response.content = "answer text"

    client = MagicMock()
    client.invoke.return_value = response

    wrapper = OllamaLLMWrapper(model_name="test-model", client=client)
    result = wrapper.chat("prompt")

    client.invoke.assert_called_once()
    msgs = client.invoke.call_args.args[0]
    assert isinstance(msgs[0], HumanMessage)
    assert msgs[0].content == "prompt"
    assert result == "answer text"
