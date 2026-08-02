# AGENTS.md

## Project

Atlas: local two-stage RAG — ChromaDB bi-encoder retrieval (`ingest.py`) -> `BAAI/bge-reranker-base` cross-encoder re-rank (`reranker.py`) -> Ollama LLM generation (`rag_pipeline.py`). `main.py` is the `AtlasRAG` CLI wrapper, `langchain_adapters.py` holds the `create_llm` factory and adapter classes, `eval/` holds `run_eval.py` + pandas-backed `EvalReporter` (`analyzer.py`). Code style: fully type-annotated, no comments (commits enforce this), ruff-formatted.

## Commands

- Run app: `python main.py` — needs Ollama running with `ollama pull llama3.2:1b`. On startup `LangChainRAG.from_defaults` seeds `./atlas_db` (gitignored) with sample docs if the collection is empty (rag_pipeline.py:66)
- Tests: `python -m pytest tests/ -q` from repo root — 102 tests, ~1 min, no GPU/network needed
- Eval: `python eval/run_eval.py` — downloads bge-reranker-base from HF; refuses to reuse an existing `eval/eval_db` without `--yes` (deletes it first). Flags documented in README
- Lint/format: `ruff check --fix` + `ruff format`; pre-commit config mirrors both plus trailing-whitespace/eof/check-yaml etc. (`pre-commit run --all-files`)

## Testing quirks

- `tests/conftest.py` pre-injects MagicMocks for `torch`, `sentence_transformers`, `chromadb` (incl. `chromadb.utils`) into `sys.modules` — heavy deps never load. Any new heavy import must be added there or tests will attempt real downloads/model loads
- Tests do need real `langchain-core`, `pandas`, and `numpy` (numpy is declared in requirements-dev.txt; it also comes transitively via pandas/torch)
- `test_run_eval.py` E2E tests assert structural invariants of the committed `eval/results.json` (config k=3, retrieve_n=10, 15 queries, 30 per-query rows, both strategies, latency ordering, mrr improvement) — regenerating results.json with different settings breaks them
- `eval/` is a package (`__init__.py`); tests import `from eval import run_eval`

## Gotchas

- `requirements.txt` declares all runtime deps, including `langchain-core` (imported directly by rag_pipeline.py/tests) and the `langchain-ollama`/`langchain-openai` adapters (lazily imported by `create_llm`). Dev tools (`pytest`, `ruff`) and `numpy` (imported directly by tests) live in `requirements-dev.txt`; install both with `pip install -r requirements.txt -r requirements-dev.txt`
- `AtlasRAG.ask` swallows only connection-class exceptions (`ConnectionError`/`httpx.TransportError`) into the answer string `"ERROR: Could not connect to Ollama (...)"`; other errors propagate (main.py:30-38)
- On Windows, ChromaDB holds file handles in `eval/eval_db`; run_eval deletes it via `del ingestor; gc.collect()` then `rmtree`, catching PermissionError with a warning — a leftover DB is normal, not a bug
- Imports are repo-root-relative (conftest.py and run_eval.py insert repo root into `sys.path`); always run from repo root
