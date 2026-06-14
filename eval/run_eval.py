import json
import time
import argparse
from pathlib import Path
import os
import shutil
from ingest import AtlasIngestor

def load_dataset(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data

def precision_at_k(retrieved_ids, relevant_ids, k):
    top_k = retrieved_ids[:k]
    relevant_set = set(relevant_ids)

    if k == 0:
        return 0.0
    
    hits = sum(1 for rid in top_k if rid in relevant_set)
    return hits / k

def hit_rate_at_k(retrieved_ids, relevant_ids, k):
    top_k = retrieved_ids[:k]
    return 1.0 if any(rid in relevant_ids for rid in top_k) else 0.0

def reciprocal_rank(retrieved_ids, relevant_ids):
    relevant_set = set(relevant_ids)
    for idx, rid in enumerate(retrieved_ids):
        if rid in relevant_set:
            return 1.0 / (idx + 1)
    return 0.0

def run_retrieval_only(ingestor, query, n_results):
    start = time.perf_counter()
    candidates = ingestor.search_with_ids(query, n_results=n_results)

    elapsed_ms = (time.perf_counter() - start) * 1000
    retrieved_ids =  [doc_id for doc_id, _ in candidates]

    return retrieved_ids, elapsed_ms

def summarize(name, per_query_results, k):
    n = len(per_query_results)
    avg_hit_rate = sum(result['hit_rate'] for result in per_query_results) / n
    avg_precision = sum(result['precision'] for result in per_query_results) / n
    avg_mrr = sum(result['mrr'] for result in per_query_results) / n
    avg_latency = sum(result['latency_ms'] for result in per_query_results) / n

    return {
        "name": name,
        "avg_hit_rate": avg_hit_rate,
        "avg_precision": avg_precision,
        "avg_mrr": avg_mrr,
        "avg_latency_ms": avg_latency,
        "k": k
    }

def print_table(rows):
    if not rows:
        return
    
    headers = list(rows[0].keys())
    widths = {header: max(len(header), max(len(str(row[header])) for row in rows)) for header in headers}

    def fmt_row(values):
        return " | ".join(f"{str(value):<{widths[header]}}" for header, value in zip(headers, values))
    
    print(fmt_row(headers))
    print("-+-".join("-" * widths[header] for header in headers))
    for row in rows:
        print(fmt_row([row[header] for header in headers]))

def main():
    parser = argparse.ArgumentParser(description="Evaluate RAG retrieval vs retrieval+re-ranking")
    parser.add_argument("--dataset", default=Path(__file__).parent / "eval_dataset.json")
    parser.add_argument("--k", type=int, default=3, help="Number of top results to evaluate")
    parser.add_argument("--retrieve-n", type=int, default=10, help="Number of candidates to retrieve before re-ranking")
    parser.add_argument("--db-path", default=Path(__file__).parent / "eval_db", help="Path to ChromaDB database directory")
    parser.add_argument("--output", default=None, help="Path to save evaluation results as JSON")
    parser.add_argument("--keep-db", action="store_true", help="Keep the ChromaDB database after evaluation")
    args = parser.parse_args()

    data = load_dataset(args.dataset)
    corpus = data["corpus"]
    queries = data["queries"]

    if os.path.exists(args.db_path) and not args.keep_db:
        print(f"Removing existing database at {args.db_path}...")
        shutil.rmtree(args.db_path)

    print("Initializing Ingestor and ingesting corpus...")
    ingestor = AtlasIngestor(db_path=args.db_path)
    ingestor.add_documents(
        text_list=[doc['text'] for doc in corpus],
        ids=[doc['id'] for doc in corpus]
    )
