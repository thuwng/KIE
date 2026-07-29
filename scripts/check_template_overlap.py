# scripts/check_template_overlap.py
import json
from src.preprocessing.docile_loader import load_document, load_split_ids

def get_vendor_names(split_ids, data_root="data/docile"):
    names = set()
    for doc_id in split_ids:
        doc = load_document(doc_id, data_root)
        for f in doc.gt_fields:
            if f.fieldtype == "vendor_name":
                names.add(f.text.strip().lower())
    return names

train_ids = load_split_ids("data/docile/train.json")
val_ids = load_split_ids("data/docile/val.json")

train_vendors = get_vendor_names(train_ids)
val_vendors = get_vendor_names(val_ids)

overlap = train_vendors & val_vendors
print(f"Số vendor unique trong train: {len(train_vendors)}")
print(f"Số vendor unique trong val: {len(val_vendors)}")
print(f"Số vendor TRÙNG giữa train và val: {len(overlap)} ({len(overlap)/len(val_vendors)*100:.1f}% của val)")