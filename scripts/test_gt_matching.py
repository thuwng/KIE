# scripts/test_gt_matching.py
from src.preprocessing.docile_loader import load_document, load_split_ids
from src.structuring.gt_matcher import match_field_to_blocks, build_training_target

train_ids = load_split_ids("data/docile/train.json")
doc = load_document(train_ids[0])

print(f"Doc {doc.doc_id}\n")
for gt in doc.gt_fields:
    matched = match_field_to_blocks(gt, doc.blocks)
    matched_text = " ".join(b.text for b in matched)
    ok = "OK" if matched_text.strip() else "MISS"
    print(f"[{ok}] {gt.fieldtype}: GT='{gt.text}' | matched='{matched_text}' | ids={[b.id for b in matched]}")

print("\n=== Training target JSON ===")
import json
target = build_training_target(doc.gt_fields, doc.blocks)
print(json.dumps(target, indent=2, ensure_ascii=False))