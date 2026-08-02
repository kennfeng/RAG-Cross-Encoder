import copy
import json
from pathlib import Path

import pandas as pd
import pytest

from eval.analyzer import EvalReporter

SAMPLE_RESULTS = {
    "config": {"k": 3, "retrieve_n": 25, "num_queries": 4},
    "summary": [
        {
            "name": "Retrieval Only",
            "avg_hit_rate": 1.0,
            "avg_precision": 0.5,
            "avg_mrr": 0.4167,
            "avg_latency_ms": 100.0,
            "k": 3,
        },
        {
            "name": "Retrieval + Re-rank",
            "avg_hit_rate": 1.0,
            "avg_precision": 0.4167,
            "avg_mrr": 0.75,
            "avg_latency_ms": 6000.0,
            "k": 3,
        },
    ],
    "per_query": {
        "retrieval_only": [
            {
                "query": "What is RAG?",
                "retrieved_ids": ["doc_2", "hn_0", "doc_10"],
                "precision": 0.6667,
                "hit_rate": 1.0,
                "mrr": 0.5,
                "latency_ms": 120.0,
            },
            {
                "query": "How does a cross-encoder work?",
                "retrieved_ids": ["hn_1", "hn_14", "doc_3"],
                "precision": 0.3333,
                "hit_rate": 1.0,
                "mrr": 0.3333,
                "latency_ms": 200.0,
            },
            {
                "query": "Why chunk documents?",
                "retrieved_ids": ["hn_5", "doc_11", "hn_4"],
                "precision": 0.3333,
                "hit_rate": 1.0,
                "mrr": 0.5,
                "latency_ms": 70.0,
            },
            {
                "query": "What is quantization?",
                "retrieved_ids": ["hn_13", "doc_29", "hn_3"],
                "precision": 0.3333,
                "hit_rate": 1.0,
                "mrr": 0.5,
                "latency_ms": 80.0,
            },
        ],
        "retrieval_plus_rerank": [
            {
                "query": "What is RAG?",
                "retrieved_ids": ["doc_2", "hn_0", "hn_12"],
                "precision": 0.3333,
                "hit_rate": 1.0,
                "mrr": 1.0,
                "latency_ms": 5500.0,
            },
            {
                "query": "How does a cross-encoder work?",
                "retrieved_ids": ["hn_1", "doc_8", "doc_3"],
                "precision": 0.6667,
                "hit_rate": 1.0,
                "mrr": 0.5,
                "latency_ms": 7000.0,
            },
            {
                "query": "Why chunk documents?",
                "retrieved_ids": ["hn_5", "doc_11", "doc_22"],
                "precision": 0.3333,
                "hit_rate": 1.0,
                "mrr": 0.5,
                "latency_ms": 6000.0,
            },
            {
                "query": "What is quantization?",
                "retrieved_ids": ["doc_29", "hn_13", "hn_3"],
                "precision": 0.3333,
                "hit_rate": 1.0,
                "mrr": 1.0,
                "latency_ms": 5000.0,
            },
        ],
    },
}


DELTA_COLUMNS = [
    "mean_hit_rate_delta",
    "mean_precision_delta",
    "mean_mrr_delta",
    "mean_latency_ms_delta",
]


@pytest.fixture
def sample_reporter():
    return EvalReporter.from_dict(SAMPLE_RESULTS)


@pytest.fixture
def tmp_results_file(tmp_path, sample_reporter):
    path = tmp_path / "results.json"
    with open(path, "w") as f:
        json.dump(SAMPLE_RESULTS, f)
    return path


@pytest.fixture
def sample_reporter_2():
    results_2 = {
        "config": {"k": 5, "retrieve_n": 25, "num_queries": 4},
        "summary": [
            {
                "name": "Retrieval Only",
                "avg_hit_rate": 1.0,
                "avg_precision": 0.6,
                "avg_mrr": 0.55,
                "avg_latency_ms": 100.0,
                "k": 5,
            },
            {
                "name": "Retrieval + Re-rank",
                "avg_hit_rate": 1.0,
                "avg_precision": 0.5,
                "avg_mrr": 0.8,
                "avg_latency_ms": 6500.0,
                "k": 5,
            },
        ],
        "per_query": {
            "retrieval_only": [
                {
                    "query": "What is RAG?",
                    "retrieved_ids": [],
                    "precision": 0.8,
                    "hit_rate": 1.0,
                    "mrr": 0.6,
                    "latency_ms": 110.0,
                },
                {
                    "query": "How does a cross-encoder work?",
                    "retrieved_ids": [],
                    "precision": 0.4,
                    "hit_rate": 1.0,
                    "mrr": 0.4,
                    "latency_ms": 210.0,
                },
                {
                    "query": "Why chunk documents?",
                    "retrieved_ids": [],
                    "precision": 0.6,
                    "hit_rate": 1.0,
                    "mrr": 0.6,
                    "latency_ms": 75.0,
                },
                {
                    "query": "What is quantization?",
                    "retrieved_ids": [],
                    "precision": 0.6,
                    "hit_rate": 1.0,
                    "mrr": 0.6,
                    "latency_ms": 85.0,
                },
            ],
            "retrieval_plus_rerank": [
                {
                    "query": "What is RAG?",
                    "retrieved_ids": [],
                    "precision": 0.6,
                    "hit_rate": 1.0,
                    "mrr": 1.0,
                    "latency_ms": 5600.0,
                },
                {
                    "query": "How does a cross-encoder work?",
                    "retrieved_ids": [],
                    "precision": 0.4,
                    "hit_rate": 1.0,
                    "mrr": 0.6,
                    "latency_ms": 7200.0,
                },
                {
                    "query": "Why chunk documents?",
                    "retrieved_ids": [],
                    "precision": 0.6,
                    "hit_rate": 1.0,
                    "mrr": 0.6,
                    "latency_ms": 6100.0,
                },
                {
                    "query": "What is quantization?",
                    "retrieved_ids": [],
                    "precision": 0.4,
                    "hit_rate": 1.0,
                    "mrr": 1.0,
                    "latency_ms": 5100.0,
                },
            ],
        },
    }
    return EvalReporter.from_dict(results_2)


# --- Construction ---


class TestConstruction:
    def test_from_dict_creates_reporter(self, sample_reporter):
        assert sample_reporter is not None

    def test_from_file_loads_results(self, tmp_results_file):
        reporter = EvalReporter.from_file(tmp_results_file)
        assert reporter.config["k"] == 3
        assert reporter.config["num_queries"] == 4

    def test_from_file_raises_on_missing_file(self):
        with pytest.raises(FileNotFoundError):
            EvalReporter.from_file(Path("/nonexistent/results.json"))

    def test_from_dict_empty_results(self):
        reporter = EvalReporter.from_dict({})
        assert reporter.config == {}
        assert len(reporter.per_query_df) == 0

    def test_from_dict_populates_reporter_fields(self):
        reporter = EvalReporter.from_dict(SAMPLE_RESULTS)
        assert reporter.config["k"] == 3
        assert len(reporter.summary_df) == 2
        assert len(reporter.per_query_df) == 8
        assert set(reporter.per_query_df["strategy"]) == {
            "retrieval_only",
            "retrieval_plus_rerank",
        }


# --- Per-query DataFrame ---


class TestPerQueryDataFrame:
    def test_has_expected_columns(self, sample_reporter):
        df = sample_reporter.per_query_df
        expected = {"query", "strategy", "precision", "hit_rate", "mrr", "latency_ms"}
        assert expected.issubset(set(df.columns))

    def test_rows_count(self, sample_reporter):
        df = sample_reporter.per_query_df
        assert len(df) == 8

    def test_strategy_labels(self, sample_reporter):
        df = sample_reporter.per_query_df
        strategies = set(df["strategy"].unique())
        assert strategies == {"retrieval_only", "retrieval_plus_rerank"}

    def test_is_dataframe(self, sample_reporter):
        assert isinstance(sample_reporter.per_query_df, pd.DataFrame)


# --- Summary DataFrame ---


class TestSummaryDataFrame:
    def test_summary_df_shape(self, sample_reporter):
        df = sample_reporter.summary_df
        assert len(df) == 2
        assert set(df["name"]) == {"Retrieval Only", "Retrieval + Re-rank"}

    def test_summary_df_has_mean_metrics(self, sample_reporter):
        df = sample_reporter.summary_df
        row = df[df["name"] == "Retrieval Only"].iloc[0]
        assert row["mean_precision"] == pytest.approx(0.5)
        assert row["mean_mrr"] == pytest.approx(0.4167, abs=0.01)

    def test_summary_df_includes_config(self, sample_reporter):
        df = sample_reporter.summary_df
        assert "k" in df.columns
        assert df["k"].iloc[0] == 3

    def test_summary_empty_results(self):
        reporter = EvalReporter.from_dict({})
        assert len(reporter.summary_df) == 0


# --- Latency Percentiles ---


class TestLatencyPercentiles:
    def test_percentiles_returns_dataframe(self, sample_reporter):
        df = sample_reporter.latency_percentiles("retrieval_only")
        assert isinstance(df, pd.DataFrame)

    def test_percentiles_default_quantiles(self, sample_reporter):
        df = sample_reporter.latency_percentiles("retrieval_only")
        assert set(df.columns) == {"quantile", "latency_ms"}
        assert len(df) == 5

    def test_percentiles_p50_p90_p99(self, sample_reporter):
        df = sample_reporter.latency_percentiles(
            "retrieval_only", quantiles=[0.5, 0.9, 0.99]
        )
        assert len(df) == 3
        assert 0.5 in df["quantile"].values
        assert 0.9 in df["quantile"].values
        assert 0.99 in df["quantile"].values

    def test_percentiles_values_are_non_negative(self, sample_reporter):
        df = sample_reporter.latency_percentiles("retrieval_plus_rerank")
        assert (df["latency_ms"] >= 0).all()

    def test_percentiles_p50_less_than_p90(self, sample_reporter):
        df = sample_reporter.latency_percentiles("retrieval_only", quantiles=[0.5, 0.9])
        assert df.iloc[0]["latency_ms"] <= df.iloc[1]["latency_ms"]

    def test_percentiles_unknown_strategy_raises(self, sample_reporter):
        with pytest.raises(ValueError, match="Unknown strategy"):
            sample_reporter.latency_percentiles("nonexistent")


# --- Worst Queries ---


class TestWorstQueries:
    def test_worst_queries_returns_dataframe(self, sample_reporter):
        df = sample_reporter.worst_queries("retrieval_only", metric="mrr", n=2)
        assert isinstance(df, pd.DataFrame)

    def test_worst_queries_n_results(self, sample_reporter):
        df = sample_reporter.worst_queries("retrieval_only", metric="mrr", n=2)
        assert len(df) == 2

    def test_worst_queries_ascending_order(self, sample_reporter):
        df = sample_reporter.worst_queries("retrieval_only", metric="mrr", n=3)
        values = df["mrr"].tolist()
        assert values == sorted(values)

    def test_worst_queries_has_query_column(self, sample_reporter):
        df = sample_reporter.worst_queries("retrieval_only", metric="precision", n=1)
        assert "query" in df.columns

    def test_worst_queries_n_larger_than_available(self, sample_reporter):
        df = sample_reporter.worst_queries("retrieval_only", metric="mrr", n=100)
        assert len(df) == 4


# --- Difficulty Breakdown ---


class TestDifficultyBreakdown:
    def test_difficulty_breakdown_returns_dataframe(self, sample_reporter):
        df = sample_reporter.difficulty_breakdown("retrieval_only")
        assert isinstance(df, pd.DataFrame)

    def test_difficulty_breakdown_has_groups(self, sample_reporter):
        df = sample_reporter.difficulty_breakdown("retrieval_only")
        assert "precision_bucket" in df.columns
        assert "count" in df.columns

    def test_difficulty_breakdown_counts_add_up(self, sample_reporter):
        df = sample_reporter.difficulty_breakdown("retrieval_only")
        assert df["count"].sum() == 4


# --- Compare ---


class TestCompare:
    def test_compare_returns_dataframe(self, sample_reporter, sample_reporter_2):
        df = sample_reporter.compare(sample_reporter_2)
        assert isinstance(df, pd.DataFrame)

    def test_compare_has_both_configs(self, sample_reporter, sample_reporter_2):
        df = sample_reporter.compare(sample_reporter_2)
        assert "k" in df.columns
        assert set(df["k"]) == {3, 5}
        for column in DELTA_COLUMNS:
            assert column in df.columns
        assert df["mean_mrr_delta"].isna().all()

    def test_compare_preserves_all_strategies(self, sample_reporter, sample_reporter_2):
        df = sample_reporter.compare(sample_reporter_2)
        assert set(df["name"]) == {"Retrieval Only", "Retrieval + Re-rank"}

    def test_compare_same_reporter(self, sample_reporter):
        df = sample_reporter.compare(sample_reporter)
        assert len(df) == 2
        assert (df[DELTA_COLUMNS] == 0).all().all()

    def test_compare_same_k_deltas(self, sample_reporter):
        modified = copy.deepcopy(SAMPLE_RESULTS)
        for entry in modified["summary"]:
            entry["avg_mrr"] += 0.1
            entry["avg_precision"] += 0.05
            entry["avg_latency_ms"] -= 10.0
        other = EvalReporter.from_dict(modified)
        df = sample_reporter.compare(other)
        assert len(df) == 2
        by_name = df.set_index("name")
        assert by_name.loc["Retrieval Only", "mean_mrr_delta"] == pytest.approx(0.1)
        assert by_name.loc[
            "Retrieval + Re-rank", "mean_precision_delta"
        ] == pytest.approx(0.05)
        assert by_name.loc["Retrieval Only", "mean_latency_ms_delta"] == pytest.approx(
            -10.0
        )
        assert by_name.loc["Retrieval Only", "mean_hit_rate_delta"] == pytest.approx(
            0.0
        )

    def test_compare_nan_delta_when_side_missing(self, sample_reporter):
        modified = copy.deepcopy(SAMPLE_RESULTS)
        modified["summary"].append(
            {
                "name": "Third Strategy",
                "avg_hit_rate": 0.9,
                "avg_precision": 0.4,
                "avg_mrr": 0.6,
                "avg_latency_ms": 50.0,
                "k": 3,
            }
        )
        other = EvalReporter.from_dict(modified)
        df = sample_reporter.compare(other)
        assert len(df) == 3
        third = df[df["name"] == "Third Strategy"].iloc[0]
        assert pd.isna(third["mean_mrr_delta"])
        assert pd.isna(third["mean_precision_delta"])
        assert third["mean_hit_rate"] == pytest.approx(0.9)
        assert third["mean_precision"] == pytest.approx(0.4)
        assert third["mean_mrr"] == pytest.approx(0.6)
        assert third["mean_latency_ms"] == pytest.approx(50.0)


# --- Export CSV ---


class TestExportCSV:
    def test_export_per_query_csv(self, sample_reporter, tmp_path):
        path = tmp_path / "per_query.csv"
        sample_reporter.export_csv(path, which="per_query")
        assert path.exists()
        loaded = pd.read_csv(path)
        assert len(loaded) == 8

    def test_export_summary_csv(self, sample_reporter, tmp_path):
        path = tmp_path / "summary.csv"
        sample_reporter.export_csv(path, which="summary")
        assert path.exists()
        loaded = pd.read_csv(path)
        assert len(loaded) == 2

    def test_export_all_csv_creates_two_files_inside_dir(
        self, sample_reporter, tmp_path
    ):
        out_dir = tmp_path / "export"
        sample_reporter.export_csv(out_dir, which="all")
        assert (out_dir / "summary.csv").exists()
        assert (out_dir / "per_query.csv").exists()
        loaded_summary = pd.read_csv(out_dir / "summary.csv")
        loaded_per_query = pd.read_csv(out_dir / "per_query.csv")
        assert len(loaded_summary) == 2
        assert len(loaded_per_query) == 8

    def test_export_all_writes_nothing_to_parent(self, sample_reporter, tmp_path):
        out_dir = tmp_path / "exports"
        sample_reporter.export_csv(out_dir, which="all")
        assert set(tmp_path.iterdir()) == {out_dir}

    def test_export_invalid_which_raises(self, sample_reporter, tmp_path):
        with pytest.raises(ValueError, match="Unknown which"):
            sample_reporter.export_csv(tmp_path / "x.csv", which="invalid")
