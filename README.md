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

### Running the Evaluation
```bash
python eval/run_eval.py
```

### Evaluation Results

| Metric   | Retrieval Only | Retrieval + Re-rank |
| -------- | -------------- | ------------------- |
| Hit Rate | 100%           | 100%                |
| MRR      | 0.466          | 0.733 (+57%)        |
| Latency  | ~95 ms         | ~6.5 s (CPU)        |

The cross-encoder improved our Mean Reciprocal Rank (MRR) by 57%, proving it is significantly better at putting the single most factual document at the #1 spot for the LLM.
