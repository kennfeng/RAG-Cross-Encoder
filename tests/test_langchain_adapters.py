import sys
from unittest.mock import MagicMock, patch

import pytest
from langchain_core.messages import HumanMessage

from langchain_adapters import (
    ChromaRetrieverAdapter,
    CrossEncoderRerankerAdapter,
    create_llm,
)


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


class TestCreateLLM:
    def test_create_llm_ollama_returns_chat_ollama(self):
        with patch("langchain_ollama.ChatOllama") as mock_cls:
            mock_cls.return_value = MagicMock()
            llm = create_llm(provider="ollama", model_name="llama3.2:1b")
            mock_cls.assert_called_once_with(model="llama3.2:1b")
            assert llm is not None

    def test_create_llm_default_provider_is_ollama(self):
        with patch("langchain_ollama.ChatOllama") as mock_cls:
            mock_cls.return_value = MagicMock()
            create_llm(model_name="llama3.2:1b")
            mock_cls.assert_called_once_with(model="llama3.2:1b")

    def test_create_llm_openai_import_error(self):
        with (
            patch.dict(sys.modules, {"langchain_openai": None}),
            pytest.raises(ImportError, match="langchain-openai"),
        ):
            create_llm(provider="openai", model_name="gpt-4o")

    def test_create_llm_openai_when_installed(self):
        mock_openai_cls = MagicMock()
        mock_module = MagicMock()
        mock_module.ChatOpenAI = mock_openai_cls
        with patch.dict("sys.modules", {"langchain_openai": mock_module}):
            llm = create_llm(provider="openai", model_name="gpt-4o")
            mock_openai_cls.assert_called_once_with(model="gpt-4o")
            assert llm is not None

    def test_create_llm_unknown_provider_raises(self):
        with pytest.raises(ValueError, match="Unknown provider"):
            create_llm(provider="nonexistent", model_name="model")

    def test_create_llm_passes_extra_kwargs(self):
        with patch("langchain_ollama.ChatOllama") as mock_cls:
            mock_cls.return_value = MagicMock()
            create_llm(provider="ollama", model_name="llama3", temperature=0.7)
            mock_cls.assert_called_once_with(model="llama3", temperature=0.7)

    def test_create_llm_invoke_returns_content(self):
        response = MagicMock()
        response.content = "hello"
        with patch("langchain_ollama.ChatOllama") as mock_cls:
            mock_cls.return_value.invoke.return_value = response
            llm = create_llm(provider="ollama", model_name="test")
            mock_cls.assert_called_once_with(model="test")
            result = llm.invoke([HumanMessage(content="hi")])
            assert result.content == "hello"
