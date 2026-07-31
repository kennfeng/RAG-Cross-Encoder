# AGENTS.md

## Project

Atlas: local two-stage RAG — ChromaDB bi-encoder retrieval (`ingest.py`) -> `BAAI/bge-reranker-base` cross-encoder re-rank (`reranker.py`) -> Ollama LLM generation (`rag_pipeline.py`). `main.py` is the `AtlasRAG` CLI wrapper, `langchain_adapters.py` holds the `create_llm` factory and adapter classes, `eval/` holds `run_eval.py` + pandas-backed `EvalReporter` (`analyzer.py`). Code style: fully type-annotated, no comments (commits enforce this), ruff-formatted.

## Commands

- Run app: `python main.py` — needs Ollama running with `ollama pull llama3.2:1b`. On startup `LangChainRAG.from_defaults` seeds `./atlas_db` (gitignored) with sample docs if the collection is empty (rag_pipeline.py:65)
- Tests: `python -m pytest tests/ -q` from repo root — 98 tests, ~1 min, no GPU/network needed
- Eval: `python eval/run_eval.py` — downloads bge-reranker-base from HF; refuses to reuse an existing `eval/eval_db` without `--yes` (deletes it first). Flags documented in README
- Lint/format: `ruff check --fix` + `ruff format`; pre-commit config mirrors both plus trailing-whitespace/eof/check-yaml etc. (`pre-commit run --all-files`)

## Testing quirks

- `tests/conftest.py` pre-injects MagicMocks for `torch`, `sentence_transformers`, `chromadb` (incl. `chromadb.utils`), `ollama` into `sys.modules` — heavy deps never load. Any new heavy import must be added there or tests will attempt real downloads/model loads
- Tests do need real `langchain-core`, `pandas`, and `numpy` (numpy isn't in requirements; comes transitively via pandas/torch)
- `test_run_eval.py` E2E tests assert exact contents of `eval/results.json` (config k=3, retrieve_n=10, 15 queries, 30 per-query rows, rerank p50 latency > 1000ms) — regenerating results.json with different settings breaks them
- `eval/` is a package (`__init__.py`); tests import `from eval import run_eval`

## Gotchas

- `requirements.txt` is incomplete for the default path: `langchain-ollama` (needed by `create_llm(provider="ollama")` via lazy import) is NOT listed, and neither is `langchain-openai` (needed for the "openai" provider; install with `pip install langchain-openai`). A fresh `pip install -r requirements.txt` will fail at runtime on `create_llm`
- `AtlasRAG.ask` swallows exceptions into the answer string `"ERROR: Could not connect to Ollama (...)"` (main.py:36-39)
- On Windows, ChromaDB holds file handles in `eval/eval_db`; run_eval deletes it via `del ingestor; gc.collect()` then `rmtree`, catching PermissionError with a warning — a leftover DB is normal, not a bug
- Imports are repo-root-relative (conftest.py and run_eval.py insert repo root into `sys.path`); always run from repo root
