# src/models/dataset.py
import json
import torch
from datasets import Dataset

def load_jsonl(path: str) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f]

def build_dataset(jsonl_path: str, tokenizer, max_seq_length: int = 4096) -> Dataset:
    records = load_jsonl(jsonl_path)
    kept, n_dropped = [], 0

    for r in records:
        completion = json.dumps(r["target"], ensure_ascii=False)
        prompt = r["prompt"] + "\n"
        full_text = prompt + completion + tokenizer.eos_token

        prompt_ids = tokenizer(prompt, add_special_tokens=True)["input_ids"]
        full_ids = tokenizer(full_text, add_special_tokens=True)["input_ids"]

        if len(full_ids) > max_seq_length:
            n_dropped += 1
            continue

        labels = full_ids.copy()
        prompt_len = len(prompt_ids)
        labels[:prompt_len] = [-100] * prompt_len  # không tính loss trên phần document

        kept.append({
            "input_ids": full_ids,
            "labels": labels,
            "attention_mask": [1] * len(full_ids),
        })

    print(f"{jsonl_path}: kept {len(kept)}, dropped {n_dropped} (> {max_seq_length} tokens)")
    return Dataset.from_list(kept)


def make_collate_fn(pad_token_id: int):
    def collate_fn(batch):
        max_len = max(len(ex["input_ids"]) for ex in batch)
        input_ids, labels, attention_mask = [], [], []
        for ex in batch:
            pad_len = max_len - len(ex["input_ids"])
            input_ids.append(ex["input_ids"] + [pad_token_id] * pad_len)
            labels.append(ex["labels"] + [-100] * pad_len)
            attention_mask.append(ex["attention_mask"] + [0] * pad_len)
        return {
            "input_ids": torch.tensor(input_ids),
            "labels": torch.tensor(labels),
            "attention_mask": torch.tensor(attention_mask),
        }
    return collate_fn