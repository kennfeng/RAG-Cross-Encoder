import pytest
from unittest.mock import MagicMock

from eval import run_eval


def test_precision_at_k_handles_zero_k():
    assert run_eval.precision_at_k(["id_1", "id_2"], ["id_2"], 0) == 0.0


def test_precision_at_k_counts_relevant_documents():
    retrieved = ["id_1", "id_2", "id_3"]
    relevant = ["id_2", "id_3"]

    assert run_eval.precision_at_k(retrieved, relevant, 3) == pytest.approx(2 / 3)


def test_hit_rate_at_k_returns_one_when_any_relevant_in_top_k():
    retrieved = ["id_5", "id_2", "id_3"]
    relevant = ["id_2"]

    assert run_eval.hit_rate_at_k(retrieved, relevant, 2) == 1.0
    assert run_eval.hit_rate_at_k(retrieved, relevant, 1) == 0.0


def test_reciprocal_rank_returns_inverse_position_of_first_relevant_item():
    retrieved = ["id_a", "id_b", "id_c"]
    relevant = ["id_b", "id_c"]

    assert run_eval.reciprocal_rank(retrieved, relevant) == pytest.approx(0.5)
    assert run_eval.reciprocal_rank(["id_x", "id_y"], ["id_z"]) == 0.0


def test_run_retrieval_only_returns_ids_and_latency():
    ingestor = MagicMock()
    ingestor.search_with_ids.return_value = [("id_1", "doc1"), ("id_2", "doc2")]

    ids, latency = run_eval.run_retrieval_only(ingestor, "query", n_results=5)

    ingestor.search_with_ids.assert_called_once_with("query", n_results=5)
    assert ids == ["id_1", "id_2"]
    assert isinstance(latency, float)
    assert latency >= 0.0


def test_run_retrieval_plus_rerank_returns_reranked_ids_and_latency():
    ingestor = MagicMock()
    ingestor.search_with_ids.return_value = [("id_1", "doc1"), ("id_2", "doc2")]

    ranker = MagicMock()
    ranker.rerank_with_ids.return_value = [
        {"id": "id_2", "document": "doc2", "score": 0.9},
        {"id": "id_1", "document": "doc1", "score": 0.8},
    ]

    ids, latency = run_eval.run_retrieval_plus_rerank(
        ingestor, ranker, "query", n_results=5, top_n=2
    )

    ingestor.search_with_ids.assert_called_once_with("query", n_results=5)
    ranker.rerank_with_ids.assert_called_once_with(
        "query", [("id_1", "doc1"), ("id_2", "doc2")], top_n=2
    )
    assert ids == ["id_2", "id_1"]
    assert isinstance(latency, float)
    assert latency >= 0.0


def test_summarize_averages_metrics():
    per_query_results = [
        {"hit_rate": 1.0, "precision": 0.5, "mrr": 1.0, "latency_ms": 100.0},
        {"hit_rate": 0.0, "precision": 0.0, "mrr": 0.0, "latency_ms": 90.0},
    ]

    summary = run_eval.summarize("test", per_query_results, k=3)

    assert summary["name"] == "test"
    assert summary["avg_hit_rate"] == pytest.approx(0.5)
    assert summary["avg_precision"] == pytest.approx(0.25)
    assert summary["avg_mrr"] == pytest.approx(0.5)
    assert summary["avg_latency_ms"] == pytest.approx(95.0)
    assert summary["k"] == 3


def test_print_table_outputs_rows(capsys):
    rows = [
        {"query": "a", "score": 1},
        {"query": "b", "score": 2},
    ]

    run_eval.print_table(rows)
    captured = capsys.readouterr()

    assert "query" in captured.out
    assert "score" in captured.out
    assert "a" in captured.out
    assert "2" in captured.out


def test_summarize_handles_empty_results():
    summary = run_eval.summarize("empty_test", [], k=5)
    assert summary["name"] == "empty_test"
    assert summary["avg_hit_rate"] == 0.0
    assert summary["avg_precision"] == 0.0
    assert summary["avg_mrr"] == 0.0
    assert summary["avg_latency_ms"] == 0.0
    assert summary["k"] == 5
