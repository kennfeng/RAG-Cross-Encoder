import main
import pytest
from unittest.mock import MagicMock
from main import AtlasRAG


@pytest.fixture
def mock_ingestor(monkeypatch):
    ingestor = MagicMock()
    ingestor.collection.count.return_value = 1
    ingestor.search.return_value = []
    ingestor.search_with_ids.return_value = []
    monkeypatch.setattr(main, "AtlasIngestor", lambda *args, **kwargs: ingestor)
    return ingestor


@pytest.fixture
def mock_ranker(monkeypatch):
    ranker = MagicMock()
    ranker.rerank.return_value = []
    ranker.rerank_with_ids.return_value = []
    monkeypatch.setattr(main, "AtlasReRanker", lambda *args, **kwargs: ranker)
    return ranker


@pytest.fixture
def mock_llm(monkeypatch):
    llm = MagicMock()
    llm.invoke.return_value = MagicMock(content="This is a concise answer.")
    monkeypatch.setattr(main, "create_llm", lambda *args, **kwargs: llm)
    return llm


def test_ask_returns_message_when_no_candidates(
    mock_ingestor, mock_ranker, mock_llm
):
    mock_ingestor.search_with_ids.return_value = []

    rag = AtlasRAG()
    result = rag.ask("nonexistent query")

    assert result["answer"] == "I couldn't find any relevant documents in the database."
    assert result["source_documents"] == []
    mock_llm.invoke.assert_not_called()


def test_init_loads_sample_data_when_db_is_empty(
    mock_ingestor, mock_ranker, mock_llm
):
    mock_ingestor.collection.count.return_value = 0

    rag = AtlasRAG()

    mock_ingestor.add_documents.assert_called_once()
    assert rag.ingestor is mock_ingestor


def test_init_skips_sample_data_when_db_is_not_empty(
    mock_ingestor, mock_ranker, mock_llm
):
    mock_ingestor.collection.count.return_value = 1

    rag = AtlasRAG()

    mock_ingestor.add_documents.assert_not_called()
    assert rag.ingestor is mock_ingestor


def test_ask_reranks_candidates_and_returns_llm_response(
    mock_ingestor, mock_ranker, mock_llm
):
    mock_ingestor.search_with_ids.return_value = [
        ("id_1", "low"),
        ("id_2", "high"),
        ("id_3", "mid"),
    ]
    mock_ranker.rerank_with_ids.return_value = [
        {"id": "id_2", "document": "high", "score": 0.95},
        {"id": "id_3", "document": "mid", "score": 0.85},
        {"id": "id_1", "document": "low", "score": 0.75},
    ]
    mock_llm.invoke.return_value = MagicMock(content="This is a concise answer.")

    rag = AtlasRAG()
    result = rag.ask("query")

    assert result["answer"] == "This is a concise answer."
    assert result["source_documents"] == mock_ranker.rerank_with_ids.return_value
    mock_ranker.rerank_with_ids.assert_called_once_with(
        "query",
        [("id_1", "low"), ("id_2", "high"), ("id_3", "mid")],
        top_n=3,
    )
    mock_llm.invoke.assert_called_once()


def test_atlasrag_pipeline_integration_uses_real_adapters(mock_llm):
    dummy_ingestor = MagicMock()
    dummy_ingestor.collection.count.return_value = 1
    dummy_ingestor.search_with_ids.return_value = [
        ("id_1", "first document"),
        ("id_2", "second document"),
    ]

    dummy_ranker = MagicMock()
    dummy_ranker.rerank_with_ids.return_value = [
        {"id": "id_2", "document": "second document", "score": 0.92},
        {"id": "id_1", "document": "first document", "score": 0.81},
    ]

    rag = AtlasRAG(
        ingestor=dummy_ingestor,
        ranker=dummy_ranker,
        model="test-model",
    )

    result = rag.ask("integrated query")

    assert result["answer"] == "This is a concise answer."
    assert result["source_documents"] == dummy_ranker.rerank_with_ids.return_value
    dummy_ranker.rerank_with_ids.assert_called_once_with(
        "integrated query",
        [("id_1", "first document"), ("id_2", "second document")],
        top_n=3,
    )
    mock_llm.invoke.assert_called_once()


def test_ask_returns_error_message_when_llm_fails(
    mock_ingestor, mock_ranker, mock_llm
):
    mock_ingestor.search_with_ids.return_value = [
        ("id_1", "doc1"),
        ("id_2", "doc2"),
    ]
    mock_ranker.rerank_with_ids.return_value = [
        {"id": "id_1", "document": "doc1", "score": 0.8}
    ]
    mock_llm.invoke.side_effect = Exception("Ollama unavailable")

    rag = AtlasRAG()
    result = rag.ask("query")

    assert result["answer"].startswith("ERROR: Could not connect to Ollama")
    assert result["source_documents"] == []
    mock_llm.invoke.assert_called_once()


def test_atlasrag_default_provider_is_ollama(mock_ingestor, mock_ranker, monkeypatch):
    mock_create_llm = MagicMock()
    mock_create_llm.return_value = MagicMock()
    monkeypatch.setattr(main, "create_llm", mock_create_llm)

    rag = AtlasRAG()
    rag.ask("query")

    mock_create_llm.assert_called_once_with(provider="ollama", model_name="llama3.2:1b")


def test_atlasrag_accepts_custom_provider(mock_ingestor, mock_ranker, monkeypatch):
    mock_create_llm = MagicMock()
    mock_create_llm.return_value = MagicMock()
    monkeypatch.setattr(main, "create_llm", mock_create_llm)

    rag = AtlasRAG(provider="openai", model="gpt-4o")
    rag.ask("query")

    mock_create_llm.assert_called_once_with(provider="openai", model_name="gpt-4o")
