# scripts/test_load_one_doc.py
from src.preprocessing.docile_loader import load_document, load_split_ids

train_ids = load_split_ids("data/docile/train.json")
print("Số doc trong train:", len(train_ids))
print("3 id đầu:", train_ids[:3])

doc = load_document(train_ids[0])
print(f"\nDoc {doc.doc_id}: {len(doc.blocks)} blocks, {len(doc.gt_fields)} gt fields")
print("3 blocks đầu:", doc.blocks[:3])
print("3 gt fields đầu:", doc.gt_fields[:3])