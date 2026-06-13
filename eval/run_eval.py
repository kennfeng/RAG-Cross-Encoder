import json

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


