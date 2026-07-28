from unittest.mock import MagicMock

from rag_pipeline import LangChainRAG


def test_langchainrag_uses_chain_with_context_and_query():
    retriever = MagicMock()
    retriever.get_relevant_documents_with_ids.return_value = [
        ("id_1", "doc1"),
        ("id_2", "doc2"),
    ]

    reranker = MagicMock()
    reranker.rerank.return_value = [
        {"id": "id_2", "document": "doc2", "score": 0.9},
        {"id": "id_1", "document": "doc1", "score": 0.8},
    ]

    llm = MagicMock()
    llm.invoke.return_value = MagicMock(content="final answer")

    pipeline = LangChainRAG(retriever, reranker, llm)
    result = pipeline.ask("test query")

    reranker.rerank.assert_called_once_with(
        "test query", [("id_1", "doc1"), ("id_2", "doc2")]
    )
    assert result["answer"] == "final answer"
    assert result["source_documents"] == reranker.rerank.return_value

    llm.invoke.assert_called_once()


def test_langchainrag_returns_no_documents_message_when_empty():
    retriever = MagicMock()
    retriever.get_relevant_documents_with_ids.return_value = []
    reranker = MagicMock()
    llm = MagicMock()

    pipeline = LangChainRAG(retriever, reranker, llm)
    result = pipeline.ask("missing query")

    assert result["answer"] == "I couldn't find any relevant documents in the database."
    assert result["source_documents"] == []
    llm.invoke.assert_not_called()
