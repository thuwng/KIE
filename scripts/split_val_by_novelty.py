# scripts/split_val_by_novelty.py
import json
from src.preprocessing.docile_loader import load_document, load_split_ids

def get_vendor_names(split_ids, data_root="data/docile"):
    doc_to_vendor = {}
    for doc_id in split_ids:
        doc = load_document(doc_id, data_root)
        names = [f.text.strip().lower() for f in doc.gt_fields if f.fieldtype == "vendor_name"]
        doc_to_vendor[doc_id] = names[0] if names else None
    return doc_to_vendor

train_ids = load_split_ids("data/docile/train.json")
val_ids = load_split_ids("data/docile/val.json")

train_vendors = set(v for v in get_vendor_names(train_ids).values() if v)
val_doc_to_vendor = get_vendor_names(val_ids)

seen_ids = [d for d, v in val_doc_to_vendor.items() if v in train_vendors]
novel_ids = [d for d, v in val_doc_to_vendor.items() if v not in train_vendors]

print(f"Seen-vendor docs: {len(seen_ids)}")
print(f"Novel-vendor docs: {len(novel_ids)}")

with open("data/processed/val_seen_ids.json", "w") as f:
    json.dump(seen_ids, f)
with open("data/processed/val_novel_ids.json", "w") as f:
    json.dump(novel_ids, f)