from unittest.mock import MagicMock

import httpx
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


def test_ask_returns_contract_keys_when_no_candidates(mock_pipeline):
    mock_pipeline.ask.return_value = {
        "answer": "I couldn't find any relevant documents in the database.",
        "source_documents": [],
    }
    rag = AtlasRAG(pipeline=mock_pipeline)
    result = rag.ask("nonexistent query")

    assert set(result) == {"answer", "source_documents"}
    assert result["source_documents"] == []


def test_ask_reranks_candidates_and_returns_llm_response(mock_pipeline):
    rag = AtlasRAG(pipeline=mock_pipeline)
    result = rag.ask("query")

    assert result["answer"] == "This is a concise answer."
    assert (
        result["source_documents"] == mock_pipeline.ask.return_value["source_documents"]
    )


def test_ask_returns_error_message_on_httpx_connect_error(mock_pipeline):
    mock_pipeline.ask.side_effect = httpx.ConnectError("Ollama unavailable")
    rag = AtlasRAG(pipeline=mock_pipeline)
    result = rag.ask("query")

    assert result["answer"].startswith(
        "ERROR: Could not connect to Ollama (ConnectError:"
    )
    assert result["source_documents"] == []


def test_ask_returns_error_message_on_builtin_connection_error(mock_pipeline):
    mock_pipeline.ask.side_effect = ConnectionError("Ollama unavailable")
    rag = AtlasRAG(pipeline=mock_pipeline)
    result = rag.ask("query")

    assert result["answer"].startswith(
        "ERROR: Could not connect to Ollama (ConnectionError:"
    )
    assert result["source_documents"] == []


def test_ask_propagates_unexpected_errors(mock_pipeline):
    mock_pipeline.ask.side_effect = RuntimeError("bug in reranker")
    rag = AtlasRAG(pipeline=mock_pipeline)

    with pytest.raises(RuntimeError, match="bug in reranker"):
        rag.ask("query")


def test_init_stores_injected_pipeline():
    pipeline = MagicMock()
    rag = AtlasRAG(pipeline=pipeline)

    assert rag.pipeline is pipeline


def test_ask_forwards_to_pipeline(mock_pipeline):
    rag = AtlasRAG(pipeline=mock_pipeline)
    rag.ask("hello")
    mock_pipeline.ask.assert_called_once_with("hello")
