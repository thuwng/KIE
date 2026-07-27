# scripts/test_serialize_one_doc.py
from src.preprocessing.docile_loader import load_document, load_split_ids
from src.structuring.serializer import serialize, build_prompt

train_ids = load_split_ids("data/docile/train.json")
doc = load_document(train_ids[0])

serialized, id_map = serialize(doc.blocks)
print("=== SERIALIZED (30 dòng đầu) ===")
print("\n".join(serialized.split("\n")[:30]))

schema = sorted(set(f.fieldtype for f in doc.gt_fields))
prompt = build_prompt(schema, serialized)
print("\n=== PROMPT (1500 ký tự đầu) ===")
print(prompt[:1500])

print(f"\nTổng số token ước lượng (theo whitespace split): {len(prompt.split())}")