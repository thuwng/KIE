# evaluate.py
import json, re
from collections import defaultdict
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
from src.extraction.postprocess import parse_llm_output
from src.preprocessing.docile_loader import load_document

MODEL_NAME = "meta-llama/Meta-Llama-3-8B-Instruct"
ADAPTER_PATH = "outputs/lora_v0"
VAL_PATH = "data/processed/val.jsonl"
DATA_ROOT = "data/docile"

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True, bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16,
)
tokenizer = AutoTokenizer.from_pretrained(ADAPTER_PATH)
base_model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, quantization_config=bnb_config, device_map="auto")
model = PeftModel.from_pretrained(base_model, ADAPTER_PATH)
model.eval()


def generate(prompt: str, max_new_tokens: int = 400) -> str:
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False, pad_token_id=tokenizer.pad_token_id)
    return tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)


def to_id_set(value):
    return frozenset(value.split()) if value else frozenset()


def normalize_text(s: str) -> str:
    s = s.lower().strip()
    s = re.sub(r"[^\w\s]", "", s)
    s = re.sub(r"\s+", " ", s)
    return s


def ids_to_text(id_tags: frozenset, id_map: dict) -> str:
    blocks = []
    for tag in id_tags:
        block_id = tag.replace("ID_", "")
        if block_id in id_map:
            blocks.append(id_map[block_id])
    blocks.sort(key=lambda b: (b.page, round(b.bbox[1], 3), b.bbox[0]))
    return " ".join(b.text for b in blocks)


class MetricAccumulator:
    def __init__(self, name):
        self.name = name
        self.tp = defaultdict(int)
        self.fp = defaultdict(int)
        self.fn = defaultdict(int)

    def add(self, field, tp=0, fp=0, fn=0):
        self.tp[field] += tp
        self.fp[field] += fp
        self.fn[field] += fn

    def report(self):
        print(f"\n=== {self.name} ===")
        print(f"{'field':30s} {'P':>6s} {'R':>6s} {'F1':>6s} {'support':>8s}")
        fields = sorted(set(list(self.tp) + list(self.fp) + list(self.fn)))
        all_tp = all_fp = all_fn = 0
        for f in fields:
            t, p, n = self.tp[f], self.fp[f], self.fn[f]
            prec = t / (t + p) if (t + p) else 0.0
            rec = t / (t + n) if (t + n) else 0.0
            f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
            print(f"{f:30s} {prec:6.3f} {rec:6.3f} {f1:6.3f} {t+n:8d}")
            all_tp += t; all_fp += p; all_fn += n
        mp = all_tp / (all_tp + all_fp) if (all_tp + all_fp) else 0.0
        mr = all_tp / (all_tp + all_fn) if (all_tp + all_fn) else 0.0
        mf1 = 2 * mp * mr / (mp + mr) if (mp + mr) else 0.0
        print(f"Micro-average: P={mp:.3f} R={mr:.3f} F1={mf1:.3f}")


def evaluate():
    records = [json.loads(l) for l in open(VAL_PATH)]

    exact = MetricAccumulator("1. Exact-match F1 (toàn field phải khớp tuyệt đối)")
    token = MetricAccumulator("2. Token-level F1 (so từng ID riêng lẻ, partial credit)")
    text_norm = MetricAccumulator("3. Text-normalized F1 (so text sau back-map + normalize)")
    presence = MetricAccumulator("4. Presence F1 (chỉ đo có tìm thấy field hay không)")
    n_parse_errors = 0

    doc_cache = {}
    for rec in tqdm(records, desc="evaluating"):
        raw_output = generate(rec["prompt"])
        try:
            parsed = parse_llm_output(raw_output)
        except Exception:
            parsed = {}
            n_parse_errors += 1

        doc_id = rec["doc_id"]
        if doc_id not in doc_cache:
            doc = load_document(doc_id, DATA_ROOT)
            doc_cache[doc_id] = {b.id: b for b in doc.blocks}
        id_map = doc_cache[doc_id]

        gt = rec["target"]
        for field, gt_value in gt.items():
            gt_ids = to_id_set(gt_value)
            pred_ids = to_id_set(parsed.get(field))

            # 1. Exact-match
            if gt_ids:
                exact.add(field, tp=1) if pred_ids == gt_ids else exact.add(field, fn=1)
            elif pred_ids:
                exact.add(field, fp=1)

            # 2. Token-level
            tp = len(gt_ids & pred_ids)
            fp = len(pred_ids - gt_ids)
            fn = len(gt_ids - pred_ids)
            token.add(field, tp=tp, fp=fp, fn=fn)

            # 3. Text-normalized
            gt_text = normalize_text(ids_to_text(gt_ids, id_map)) if gt_ids else ""
            pred_text = normalize_text(ids_to_text(pred_ids, id_map)) if pred_ids else ""
            if gt_text:
                text_norm.add(field, tp=1) if gt_text == pred_text else text_norm.add(field, fn=1)
            elif pred_text:
                text_norm.add(field, fp=1)

            # 4. Presence
            gt_present, pred_present = bool(gt_ids), bool(pred_ids)
            if gt_present:
                presence.add(field, tp=1) if pred_present else presence.add(field, fn=1)
            elif pred_present:
                presence.add(field, fp=1)

    print(f"\nParse errors: {n_parse_errors}/{len(records)}")
    exact.report()
    token.report()
    text_norm.report()
    presence.report()


if __name__ == "__main__":
    evaluate()