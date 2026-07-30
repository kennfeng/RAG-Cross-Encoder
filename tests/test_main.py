from unittest.mock import MagicMock

import pytest

from main import AtlasRAG


@pytest.fixture
def mock_pipeline():
    pipeline = MagicMock()
    pipeline.ask.return_value = {
        "answer": "This is a concise answer.",
        "source_documents": [
            {"id": "id_2", "document": "high", "score": 0.95},
            {"id": "id_3", "document": "mid", "score": 0.85},
        ],
    }
    return pipeline


def test_ask_returns_message_when_no_candidates(mock_pipeline):
    mock_pipeline.ask.return_value = {
        "answer": "I couldn't find any relevant documents in the database.",
        "source_documents": [],
    }
    rag = AtlasRAG(pipeline=mock_pipeline)
    result = rag.ask("nonexistent query")

    assert result["answer"] == "I couldn't find any relevant documents in the database."
    assert result["source_documents"] == []


def test_ask_reranks_candidates_and_returns_llm_response(mock_pipeline):
    rag = AtlasRAG(pipeline=mock_pipeline)
    result = rag.ask("query")

    assert result["answer"] == "This is a concise answer."
    assert (
        result["source_documents"] == mock_pipeline.ask.return_value["source_documents"]
    )


def test_ask_returns_error_message_when_llm_fails(mock_pipeline):
    mock_pipeline.ask.side_effect = Exception("Ollama unavailable")
    rag = AtlasRAG(pipeline=mock_pipeline)
    result = rag.ask("query")

    assert result["answer"].startswith("ERROR: Could not connect to Ollama")
    assert result["source_documents"] == []


def test_init_defaults():
    rag = AtlasRAG(pipeline=MagicMock())
    assert rag.pipeline is not None


def test_ask_forwards_to_pipeline(mock_pipeline):
    rag = AtlasRAG(pipeline=mock_pipeline)
    rag.ask("hello")
    mock_pipeline.ask.assert_called_once_with("hello")
