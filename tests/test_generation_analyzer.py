import json
from pathlib import Path

import pandas as pd
import pytest

from eval.generation_analyzer import GenerationReporter

SAMPLE_GENERATION_RESULTS = {
    "config": {"n_results": 10, "top_n": 3, "num_queries": 4},
    "summary": {
        "avg_faithfulness": 0.75,
        "avg_relevance": 0.8,
        "num_queries": 4,
        "num_judge_errors": 1,
        "avg_generation_latency_ms": 250.0,
        "avg_judge_latency_ms": 100.0,
    },
    "per_query": [
        {
            "query": "q1",
            "answer": "a1",
            "context": "ctx1",
            "source_ids": ["d1"],
            "faithfulness": 0.9,
            "relevance": 0.8,
            "rationale": "good",
            "judge_error": False,
            "generation_latency_ms": 200.0,
            "judge_latency_ms": 100.0,
        },
        {
            "query": "q2",
            "answer": "a2",
            "context": "ctx2",
            "source_ids": ["d1", "d2"],
            "faithfulness": 0.6,
            "relevance": 0.7,
            "rationale": "ok",
            "judge_error": False,
            "generation_latency_ms": 300.0,
            "judge_latency_ms": 90.0,
        },
        {
            "query": "q3",
            "answer": "a3",
            "context": "ctx3",
            "source_ids": [],
            "faithfulness": None,
            "relevance": None,
            "rationale": None,
            "judge_error": True,
            "generation_latency_ms": 250.0,
            "judge_latency_ms": 100.0,
        },
        {
            "query": "q4",
            "answer": "a4",
            "context": "ctx4",
            "source_ids": ["d3"],
            "faithfulness": 0.75,
            "relevance": 0.9,
            "rationale": "good",
            "judge_error": False,
            "generation_latency_ms": 250.0,
            "judge_latency_ms": 110.0,
        },
    ],
}


@pytest.fixture
def sample_reporter():
    return GenerationReporter.from_dict(SAMPLE_GENERATION_RESULTS)


@pytest.fixture
def tmp_results_file(tmp_path):
    path = tmp_path / "generation_results.json"
    with open(path, "w") as f:
        json.dump(SAMPLE_GENERATION_RESULTS, f)
    return path


class TestConstruction:
    def test_from_dict_creates_reporter(self, sample_reporter):
        assert sample_reporter is not None

    def test_from_file_loads_results(self, tmp_results_file):
        reporter = GenerationReporter.from_file(tmp_results_file)
        assert reporter.config["n_results"] == 10
        assert reporter.config["num_queries"] == 4

    def test_from_file_raises_on_missing_file(self):
        with pytest.raises(FileNotFoundError):
            GenerationReporter.from_file(Path("/nonexistent/generation_results.json"))

    def test_from_dict_empty_results(self):
        reporter = GenerationReporter.from_dict({"config": {}})
        assert reporter.config == {}
        assert len(reporter.per_query_df) == 0
        assert len(reporter.summary_df) == 0
        assert len(reporter.worst_queries()) == 0


class TestPerQueryDataFrame:
    def test_per_query_df_has_expected_columns(self, sample_reporter):
        df = sample_reporter.per_query_df
        expected = {
            "query",
            "answer",
            "context",
            "source_ids",
            "faithfulness",
            "relevance",
            "rationale",
            "judge_error",
            "generation_latency_ms",
            "judge_latency_ms",
        }
        assert set(df.columns) == expected

    def test_per_query_df_rows_count(self, sample_reporter):
        assert len(sample_reporter.per_query_df) == 4

    def test_per_query_df_is_dataframe(self, sample_reporter):
        assert isinstance(sample_reporter.per_query_df, pd.DataFrame)


class TestSummaryDataFrame:
    def test_summary_df_single_row(self, sample_reporter):
        assert len(sample_reporter.summary_df) == 1
        assert sample_reporter.summary["num_queries"] == 4

    def test_summary_df_has_avg_metrics(self, sample_reporter):
        row = sample_reporter.summary_df.iloc[0]
        assert row["avg_faithfulness"] == pytest.approx(0.75)
        assert row["avg_relevance"] == pytest.approx(0.8)
        assert row["num_judge_errors"] == 1


class TestWorstQueries:
    def test_worst_queries_default_metric(self, sample_reporter):
        df = sample_reporter.worst_queries(n=2)
        assert len(df) == 2
        assert set(df["query"]) == {"q2", "q4"}

    def test_worst_queries_ascending_order(self, sample_reporter):
        df = sample_reporter.worst_queries(metric="faithfulness", n=4)
        values = [v for v in df["faithfulness"].tolist() if pd.notna(v)]
        assert values == sorted(values)

    def test_worst_queries_nan_last(self, sample_reporter):
        df = sample_reporter.worst_queries(metric="faithfulness", n=4)
        assert pd.isna(df["faithfulness"].iloc[-1])
        assert df["query"].iloc[-1] == "q3"


class TestExportCSV:
    def test_export_csv_modes(self, sample_reporter, tmp_path):
        summary_path = tmp_path / "summary.csv"
        sample_reporter.export_csv(summary_path, which="summary")
        assert summary_path.exists()
        assert len(pd.read_csv(summary_path)) == 1

        per_query_path = tmp_path / "per_query.csv"
        sample_reporter.export_csv(per_query_path, which="per_query")
        assert per_query_path.exists()
        assert len(pd.read_csv(per_query_path)) == 4

        out_dir = tmp_path / "export"
        sample_reporter.export_csv(out_dir, which="all")
        assert (out_dir / "summary.csv").exists()
        assert (out_dir / "per_query.csv").exists()
        assert len(pd.read_csv(out_dir / "summary.csv")) == 1
        assert len(pd.read_csv(out_dir / "per_query.csv")) == 4
