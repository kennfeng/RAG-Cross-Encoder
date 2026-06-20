from reranker import AtlasReRanker
import numpy as np

def make_reranker():
    return AtlasReRanker()

def test_rerank_orders_docuemnts_by_descending():
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