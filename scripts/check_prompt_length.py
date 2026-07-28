# scripts/check_prompt_lengths.py
import json
from transformers import AutoTokenizer

MODEL_NAME = "meta-llama/Meta-Llama-3-8B-Instruct"  # đổi nếu bạn dùng model khác

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

lengths = []
with open("data/processed/train.jsonl") as f:
    for line in f:
        rec = json.loads(line)
        full_text = rec["prompt"] + json.dumps(rec["target"])
        lengths.append(len(tokenizer(full_text)["input_ids"]))

lengths.sort()
n = len(lengths)
print(f"n={n}")
print(f"min={lengths[0]}, max={lengths[-1]}")
print(f"p50={lengths[n//2]}, p90={lengths[int(n*0.9)]}, p99={lengths[int(n*0.99)]}")