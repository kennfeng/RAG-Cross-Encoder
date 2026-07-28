# Atlas

Atlas is a local, two-stage Retrieval-Augmented Generation (RAG) system.

### How It Works
```
User Query
    │
    ▼
[Stage 1] Bi-encoder Vector Search
    │  Returns top-N candidate documents
    ▼
[Stage 2] Cross-Encoder Re-ranking
    │  Scores each (query, document) pair jointly, selects top-K
    ▼
[Stage 3] LLM Generation
    │  Generates an answer from the re-ranked context
    ▼
Answer
```

### Prerequisites
- [Ollama](https://ollama.com/) installed and running.
- Python 3.10 or higher.

### Installation
1. Clone the repository and navigate to the directory.
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Pull the required LLM:
   ```bash
   ollama pull llama3.2:1b
   ```

### Running the Project
```bash
python main.py
```

### LLM Provider Swapping

Atlas uses a `create_llm` factory to configure the LLM provider:

```python
from langchain_adapters import create_llm

llm = create_llm(provider="ollama", model_name="llama3.2:1b")
```

Pass a provider to `AtlasRAG` or `LangChainRAG.from_defaults()`:

```python
rag = AtlasRAG(provider="ollama", model="llama3.2:1b")
```

### Testing

Run the full test suite:
```bash
python -m pytest tests/ -v
```

The test suite uses mocked ML dependencies (torch, sentence-transformers, chromadb) so tests run fast without GPU or model downloads. Tests cover:

- **Unit tests**: ingest, reranker, adapters, pipeline, eval metrics
- **E2E tests**: real `results.json` loaded through `EvalReporter` for analytics validation
- **Integration tests**: full `AtlasRAG` wiring with mock LLM

### Evaluation

```bash
python eval/run_eval.py
```

#### CLI Flags

| Flag | Description |
|---|---|
| `--k` | Number of top results to evaluate (default: 3) |
| `--retrieve-n` | Candidates to retrieve before re-ranking (default: 10) |
| `--output` | Save results as JSON |
| `--export-csv` | Export summary and per_query CSVs via EvalReporter |
| `--compare` | Path to another results JSON for side-by-side comparison |
| `--keep-db` | Keep the ChromaDB database after evaluation |

#### Evaluation Results

| Metric   | Retrieval Only | Retrieval + Re-rank |
| -------- | -------------- | ------------------- |
| Hit Rate | 100%           | 100%                |
| MRR      | 0.467          | 0.733 (+57%)        |
| Latency  | ~57 ms         | ~2.1 s (CPU)        |

The cross-encoder improved Mean Reciprocal Rank (MRR) by 57%, proving it is significantly better at putting the most factual document at the #1 spot for the LLM.

### Eval Analytics

The `EvalReporter` class (`eval/analyzer.py`) uses pandas DataFrames for rich evaluation analysis:

```python
from eval.analyzer import EvalReporter

reporter = EvalReporter.from_file("eval/results.json")

# Per-query DataFrame with strategy labels
reporter.per_query_df

# Summary comparison
reporter.summary_df

# Latency percentiles
reporter.latency_percentiles("retrieval_only", quantiles=[0.5, 0.9, 0.99])

# Worst-performing queries
reporter.worst_queries("retrieval_plus_rerank", metric="mrr", n=5)

# Difficulty breakdown by precision buckets
reporter.difficulty_breakdown("retrieval_only")

# Compare two evaluation runs
reporter.compare(other_reporter)

# Export to CSV
reporter.export_csv("exports/", which="all")
```

### File Structure

```
RAG-Cross-Encoder/
├── main.py
├── ingest.py
├── reranker.py
├── rag_pipeline.py
├── langchain_adapters.py
├── requirements.txt
│
├── eval/
│   ├── run_eval.py
│   ├── analyzer.py
│   ├── eval_dataset.json
│   └── results.json
│
└── tests/
    ├── conftest.py
    ├── test_ingest.py
    ├── test_reranker.py
    ├── test_langchain_adapters.py
    ├── test_rag_pipeline.py
    ├── test_main.py
    ├── test_run_eval.py
    └── test_analyzer.py
```
