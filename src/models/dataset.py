# src/models/dataset.py
import json
from datasets import Dataset

def load_jsonl(path: str) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f]

def format_example(record: dict) -> dict:
    completion = json.dumps(record["target"], ensure_ascii=False)
    return {
        "text": record["prompt"] + "\n" + completion,
        "doc_id": record["doc_id"],
    }

def build_dataset(jsonl_path: str, tokenizer, max_seq_length: int = 4096) -> Dataset:
    records = load_jsonl(jsonl_path)
    kept, n_dropped = [], 0
    for r in records:
        ex = format_example(r)
        n_tokens = len(tokenizer(ex["text"])["input_ids"])
        if n_tokens <= max_seq_length:
            kept.append(ex)
        else:
            n_dropped += 1
    print(f"{jsonl_path}: kept {len(kept)}, dropped {n_dropped} (> {max_seq_length} tokens)")
    return Dataset.from_list(kept)