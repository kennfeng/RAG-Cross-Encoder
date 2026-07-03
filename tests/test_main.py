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
def mock_ollama(monkeypatch):
    ollama = MagicMock()
    monkeypatch.setattr(main, "ollama", ollama)
    return ollama


def test_ask_returns_message_when_no_candidates(
    mock_ingestor, mock_ranker, mock_ollama
):
    mock_ingestor.search_with_ids.return_value = []

    rag = AtlasRAG()
    result = rag.ask("nonexistent query")

    assert result["answer"] == "I couldn't find any relevant documents in the database."
    assert result["source_documents"] == []
    mock_ollama.chat.assert_not_called()


def test_init_loads_sample_data_when_db_is_empty(
    mock_ingestor, mock_ranker, mock_ollama
):
    mock_ingestor.collection.count.return_value = 0

    rag = AtlasRAG()

    mock_ingestor.add_documents.assert_called_once()
    assert rag.ingestor is mock_ingestor


def test_init_skips_sample_data_when_db_is_not_empty(
    mock_ingestor, mock_ranker, mock_ollama
):
    mock_ingestor.collection.count.return_value = 1

    rag = AtlasRAG()

    mock_ingestor.add_documents.assert_not_called()
    assert rag.ingestor is mock_ingestor


def test_ask_reranks_candidates_and_returns_ollama_response(
    mock_ingestor, mock_ranker, mock_ollama
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
    mock_ollama.chat.return_value = {
        "message": {"content": "This is a concise answer."}
    }

    rag = AtlasRAG()
    result = rag.ask("query")

    assert result["answer"] == "This is a concise answer."
    assert result["source_documents"] == mock_ranker.rerank_with_ids.return_value
    mock_ranker.rerank_with_ids.assert_called_once_with(
        "query",
        [("id_1", "low"), ("id_2", "high"), ("id_3", "mid")],
        top_n=3,
    )
    mock_ollama.chat.assert_called_once()


def test_atlasrag_pipeline_integration_uses_real_adapters():
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

    client = MagicMock()
    client.chat.return_value = {"message": {"content": "Integrated answer."}}

    rag = AtlasRAG(
        ingestor=dummy_ingestor,
        ranker=dummy_ranker,
        llm_client=client,
        model="test-model",
    )

    result = rag.ask("integrated query")

    assert result["answer"] == "Integrated answer."
    assert result["source_documents"] == dummy_ranker.rerank_with_ids.return_value
    dummy_ranker.rerank_with_ids.assert_called_once_with(
        "integrated query",
        [("id_1", "first document"), ("id_2", "second document")],
        top_n=3,
    )
    client.chat.assert_called_once()
    assert "Context:" in client.chat.call_args.kwargs["messages"][0]["content"]


def test_ask_returns_error_message_when_ollama_fails(
    mock_ingestor, mock_ranker, mock_ollama
):
    mock_ingestor.search_with_ids.return_value = [
        ("id_1", "doc1"),
        ("id_2", "doc2"),
    ]
    mock_ranker.rerank_with_ids.return_value = [
        {"id": "id_1", "document": "doc1", "score": 0.8}
    ]
    mock_ollama.chat.side_effect = Exception("Ollama unavailable")

    rag = AtlasRAG()
    result = rag.ask("query")

    assert result["answer"].startswith("ERROR: Could not connect to Ollama")
    assert result["source_documents"] == []
    mock_ollama.chat.assert_called_once()
