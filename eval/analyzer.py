import json
from pathlib import Path

import numpy as np
import pandas as pd


class EvalReporter:
    """Pandas-backed evaluation reporter for RAG retrieval analysis.

    Uses DataFrames as the core data structure for per-query analysis,
    latency percentiles, difficulty breakdowns, and multi-run comparison.
    """

    def __init__(self, results: dict):
        self._results = results
        self._config = results.get("config", {})
        self._summary_raw = results.get("summary", [])
        self._per_query = results.get("per_query", {})

        self._per_query_df = self._build_per_query_df()
        self._summary_df = self._build_summary_df()

    @classmethod
    def from_dict(cls, results: dict) -> "EvalReporter":
        return cls(results)

    @classmethod
    def from_file(cls, path: Path) -> "EvalReporter":
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Results file not found: {path}")
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls(data)

    @property
    def config(self) -> dict:
        return self._config

    @property
    def per_query_df(self) -> pd.DataFrame:
        return self._per_query_df

    @property
    def summary_df(self) -> pd.DataFrame:
        return self._summary_df

    def _build_per_query_df(self) -> pd.DataFrame:
        frames = []
        for strategy, rows in self._per_query.items():
            if not rows:
                continue
            df = pd.DataFrame(rows)
            df["strategy"] = strategy
            frames.append(df)
        if not frames:
            return pd.DataFrame(
                columns=["query", "strategy", "precision", "hit_rate", "mrr", "latency_ms"]
            )
        return pd.concat(frames, ignore_index=True)

    def _build_summary_df(self) -> pd.DataFrame:
        if not self._summary_raw:
            return pd.DataFrame()

        k = self._config.get("k")
        records = []
        for entry in self._summary_raw:
            records.append(
                {
                    "name": entry["name"],
                    "k": entry.get("k", k),
                    "mean_hit_rate": entry.get("avg_hit_rate", 0.0),
                    "mean_precision": entry.get("avg_precision", 0.0),
                    "mean_mrr": entry.get("avg_mrr", 0.0),
                    "mean_latency_ms": entry.get("avg_latency_ms", 0.0),
                }
            )
        return pd.DataFrame(records)

    def latency_percentiles(
        self, strategy: str, quantiles=None
    ) -> pd.DataFrame:
        if quantiles is None:
            quantiles = [0.0, 0.25, 0.5, 0.75, 0.9]

        df = self._per_query_df
        subset = df[df["strategy"] == strategy]
        if subset.empty:
            raise ValueError(f"Unknown strategy: {strategy}")

        values = subset["latency_ms"].values
        result = np.quantile(values, quantiles)
        return pd.DataFrame(
            {"quantile": quantiles, "latency_ms": [float(v) for v in result]}
        )

    def worst_queries(
        self, strategy: str, metric: str = "mrr", n: int = 5
    ) -> pd.DataFrame:
        df = self._per_query_df
        subset = df[df["strategy"] == strategy].copy()
        sorted_df = subset.sort_values(by=metric, ascending=True)
        return sorted_df.head(n)

    def difficulty_breakdown(self, strategy: str) -> pd.DataFrame:
        df = self._per_query_df
        subset = df[df["strategy"] == strategy].copy()

        bins = [0.0, 0.33, 0.67, 1.0]
        labels = ["low", "medium", "high"]
        subset["precision_bucket"] = pd.cut(
            subset["precision"], bins=bins, labels=labels, include_lowest=True
        )
        grouped = (
            subset.groupby("precision_bucket", observed=False)
            .size()
            .reset_index(name="count")
        )
        return grouped

    def compare(self, other: "EvalReporter") -> pd.DataFrame:
        left = self.summary_df.copy()
        right = other.summary_df.copy()

        combined = pd.concat([left, right], ignore_index=True)
        combined = combined.drop_duplicates(subset=["name", "k"], keep="first")
        return combined.reset_index(drop=True)

    def export_csv(self, path: Path, which: str = "all") -> None:
        path = Path(path)

        if which == "per_query":
            self._per_query_df.to_csv(path, index=False)
        elif which == "summary":
            self._summary_df.to_csv(path, index=False)
        elif which == "all":
            parent = path
            parent.mkdir(parents=True, exist_ok=True)
            stem = parent.name
            self._summary_df.to_csv(parent.parent / f"{stem}_summary.csv", index=False)
            self._per_query_df.to_csv(parent.parent / f"{stem}_per_query.csv", index=False)
        else:
            raise ValueError(f"Unknown which: {which!r}. Use 'per_query', 'summary', or 'all'.")
