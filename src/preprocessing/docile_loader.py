# src/preprocessing/docile_loader.py
import json
from dataclasses import dataclass

@dataclass
class TextBlock:
    id: str
    text: str
    bbox: tuple   # (x0, y0, x1, y1) normalized 0-1
    page: int
    # đại diện cho từ: id, text, bbox, page 

@dataclass
class GTField:
    fieldtype: str
    text: str
    bbox: tuple
    page: int
    # lưu trữ annotation: fieldtype, text, bbox, page

@dataclass
class Document:
    doc_id: str
    blocks: list[TextBlock]
    gt_fields: list[GTField]
    # tài liệu hoàn chỉnh


def load_ocr_blocks(ocr_path: str) -> list[TextBlock]:
    with open(ocr_path) as f:
        ocr = json.load(f)
    blocks = []
    counter = 1
    for page in ocr["pages"]:
        page_idx = page["page_idx"]
        for block in page["blocks"]:
            for line in block["lines"]:
                for word in line["words"]:
                    (x0, y0), (x1, y1) = word["geometry"]
                    blocks.append(TextBlock(
                        id=f"{counter:04d}",
                        text=word["value"],
                        bbox=(x0, y0, x1, y1),
                        page=page_idx,
                    ))
                    counter += 1
    return blocks


def load_gt_fields(ann_path: str) -> list[GTField]:
    with open(ann_path) as f:
        ann = json.load(f)
    return [
        GTField(
            fieldtype=fe["fieldtype"],
            text=fe["text"],
            bbox=tuple(fe["bbox"]),
            page=fe["page"],
        )
        for fe in ann["field_extractions"]
    ]


def load_document(doc_id: str, data_root: str = "data/docile") -> Document:
    ocr_path = f"{data_root}/ocr/{doc_id}.json"
    ann_path = f"{data_root}/annotations/{doc_id}.json"
    return Document(
        doc_id=doc_id,
        blocks=load_ocr_blocks(ocr_path),
        gt_fields=load_gt_fields(ann_path),
    )


def load_split_ids(split_path: str) -> list[str]:
    """train.json/val.json/test.json — chưa rõ định dạng chính xác
    (list[str] hay dict), nên xử lý cả 2 trường hợp."""
    with open(split_path) as f:
        data = json.load(f)
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        # phòng trường hợp có key kiểu {"docs": [...]}
        for v in data.values():
            if isinstance(v, list):
                return v
    raise ValueError(f"Không nhận diện được format của {split_path}: {type(data)}")