# scripts/build_training_data.py
import json
from tqdm import tqdm
from src.preprocessing.docile_loader import load_document, load_split_ids
from src.structuring.serializer import serialize, build_prompt
from src.structuring.gt_matcher import build_training_target

DATA_ROOT = "data/docile"
OUT_DIR = "data/processed"

def build_split(split_name: str, schema: list[str]):
    ids = load_split_ids(f"{DATA_ROOT}/{split_name}.json")
    out_path = f"{OUT_DIR}/{split_name}.jsonl"
    n_ok, n_fail = 0, 0

    with open(out_path, "w") as f_out:
        for doc_id in tqdm(ids, desc=split_name):
            try:
                doc = load_document(doc_id, DATA_ROOT)
                serialized, id_map = serialize(doc.blocks)
                prompt = build_prompt(schema, serialized)
                target = build_training_target(doc.gt_fields, doc.blocks)

                record = {
                    "doc_id": doc.doc_id,
                    "prompt": prompt,
                    "target": target,
                    "n_blocks": len(doc.blocks),
                }
                f_out.write(json.dumps(record, ensure_ascii=False) + "\n")
                n_ok += 1
            except FileNotFoundError as e:
                print(f"[SKIP] {doc_id}: {e}")
                n_fail += 1

    print(f"{split_name}: {n_ok} ok, {n_fail} fail -> {out_path}")


if __name__ == "__main__":
    import os
    os.makedirs(OUT_DIR, exist_ok=True)

    # Schema cố định toàn bộ dataset (không lấy theo từng doc như trước,
    # vì mỗi doc chỉ có 1 phần field xuất hiện -> schema phải là union toàn bộ
    # tập train để model học được field nào "không có" -> null)
    train_ids = load_split_ids(f"{DATA_ROOT}/train.json")
    all_fieldtypes = set()
    for doc_id in tqdm(train_ids, desc="scanning schema"):
        doc = load_document(doc_id, DATA_ROOT)
        all_fieldtypes.update(f.fieldtype for f in doc.gt_fields)
    schema = sorted(all_fieldtypes)
    print(f"Schema ({len(schema)} fields): {schema}")

    with open(f"{OUT_DIR}/schema.json", "w") as f:
        json.dump(schema, f, indent=2)

    for split in ["train", "val", "test"]:
        build_split(split, schema)