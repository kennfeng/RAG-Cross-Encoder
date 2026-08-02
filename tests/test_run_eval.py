from pathlib import Path
from unittest.mock import MagicMock

import pandas as pd
import pytest

from eval import run_eval
from eval.analyzer import EvalReporter

RESULTS_JSON = Path(__file__).resolve().parent.parent / "eval" / "results.json"


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


class TestE2ERealResults:
    """E2E tests that load the actual eval/results.json and exercise
    EvalReporter end-to-end without any model loading."""

    @pytest.fixture(autouse=True)
    def _load_real_results(self):
        if not RESULTS_JSON.exists():
            pytest.skip("results.json not found")
        self.reporter = EvalReporter.from_file(RESULTS_JSON)

    def test_loads_correct_config(self):
        assert self.reporter.config["k"] == 3
        assert self.reporter.config["retrieve_n"] == 10
        assert self.reporter.config["num_queries"] == 15

    def test_per_query_df_has_30_rows(self):
        df = self.reporter.per_query_df
        assert len(df) == 30

    def test_per_query_df_has_both_strategies(self):
        df = self.reporter.per_query_df
        assert set(df["strategy"].unique()) == {
            "retrieval_only",
            "retrieval_plus_rerank",
        }

    def test_summary_df_has_two_rows(self):
        df = self.reporter.summary_df
        assert len(df) == 2

    def test_summary_df_mrr_improvement(self):
        df = self.reporter.summary_df
        mrr_only = df[df["name"] == "Retrieval Only"]["mean_mrr"].iloc[0]
        mrr_rerank = df[df["name"] == "Retrieval + Re-rank"]["mean_mrr"].iloc[0]
        assert mrr_rerank > mrr_only

    def test_latency_percentiles_retrieval_only(self):
        df = self.reporter.latency_percentiles("retrieval_only")
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 5
        p50 = df[df["quantile"] == 0.5]["latency_ms"].iloc[0]
        p90 = df[df["quantile"] == 0.9]["latency_ms"].iloc[0]
        assert p50 < p90
        assert (df["latency_ms"] > 0).all()

    def test_latency_percentiles_rerank(self):
        df = self.reporter.latency_percentiles("retrieval_plus_rerank")
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 5
        p50 = df[df["quantile"] == 0.5]["latency_ms"].iloc[0]
        p90 = df[df["quantile"] == 0.9]["latency_ms"].iloc[0]
        assert p50 < p90
        assert (df["latency_ms"] > 0).all()

    def test_worst_queries_returns_sorted(self):
        df = self.reporter.worst_queries("retrieval_only", metric="mrr", n=5)
        assert len(df) == 5
        assert df["mrr"].is_monotonic_increasing

    def test_difficulty_breakdown_counts(self):
        df = self.reporter.difficulty_breakdown("retrieval_only")
        assert df["count"].sum() == 15

    def test_compare_same_reporter(self):
        df = self.reporter.compare(self.reporter)
        assert len(df) == 2
        delta_cols = [col for col in df.columns if col.endswith("_delta")]
        assert len(delta_cols) > 0
        assert (df[delta_cols] == 0).all().all()

    def test_export_csv_summary(self, tmp_path):
        self.reporter.export_csv(tmp_path / "summary.csv", which="summary")
        assert (tmp_path / "summary.csv").exists()
        loaded = pd.read_csv(tmp_path / "summary.csv")
        assert len(loaded) == 2

    def test_export_csv_per_query(self, tmp_path):
        self.reporter.export_csv(tmp_path / "pq.csv", which="per_query")
        assert (tmp_path / "pq.csv").exists()
        loaded = pd.read_csv(tmp_path / "pq.csv")
        assert len(loaded) == 30
