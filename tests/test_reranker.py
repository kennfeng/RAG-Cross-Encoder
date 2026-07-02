from reranker import AtlasReRanker
import numpy as np


def make_reranker():
    return AtlasReRanker()


def test_rerank_orders_documents_by_descending():
    reranker = make_reranker()
    reranker.model.predict = lambda pairs: np.array([0.2, 0.5, 0.3])

    documents = ["low", "high", "mid"]
    results = reranker.rerank("query", documents, top_n=3)

    assert results[0]["document"] == "high"
    assert results[1]["document"] == "mid"
    assert results[2]["document"] == "low"
    assert results[0]["score"] == 0.5
    assert results[1]["score"] == 0.3
    assert results[2]["score"] == 0.2


def test_rerank_with_ids_orders_candidates_and_includes_ids():
    reranker = make_reranker()
    reranker.model.predict = lambda pairs: np.array([0.1, 0.9, 0.5])

    candidates = [("id_1", "low"), ("id_2", "high"), ("id_3", "mid")]
    results = reranker.rerank_with_ids("query", candidates, top_n=2)

    assert len(results) == 2
    assert results[0]["id"] == "id_2"
    assert results[0]["document"] == "high"
    assert results[0]["score"] == 0.9
    assert results[1]["id"] == "id_3"
    assert results[1]["document"] == "mid"
    assert results[1]["score"] == 0.5


def test_rerank_returns_empty_list_for_no_documents():
    reranker = make_reranker()
    results = reranker.rerank("query", [], top_n=3)
    assert results == []


def test_rerank_with_ids_returns_empty_list_for_no_candidates():
    reranker = make_reranker()
    results = reranker.rerank_with_ids("query", [], top_n=3)
    assert results == []
