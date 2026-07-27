# src/structuring/gt_matcher.py
from src.preprocessing.docile_loader import TextBlock, GTField


def _intersection_area(a: tuple, b: tuple) -> float:
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    ix0, iy0 = max(ax0, bx0), max(ay0, by0)
    ix1, iy1 = min(ax1, bx1), min(ay1, by1)
    if ix1 <= ix0 or iy1 <= iy0:
        return 0.0
    return (ix1 - ix0) * (iy1 - iy0)


def _area(bbox: tuple) -> float:
    x0, y0, x1, y1 = bbox
    return max(0.0, x1 - x0) * max(0.0, y1 - y0)


def match_field_to_blocks(
    gt: GTField, blocks: list[TextBlock], containment_threshold: float = 0.5
) -> list[TextBlock]:
    """Trả về các block OCR nằm phần lớn (>threshold diện tích) trong bbox của GT field,
    trên cùng page, đã sort theo thứ tự đọc (top-to-bottom, left-to-right)."""
    candidates = []
    for b in blocks:
        if b.page != gt.page:
            continue
        b_area = _area(b.bbox)
        if b_area == 0:
            continue
        ratio = _intersection_area(b.bbox, gt.bbox) / b_area
        if ratio >= containment_threshold:
            candidates.append(b)
    candidates.sort(key=lambda b: (b.bbox[1], b.bbox[0]))  # y0, rồi x0
    return candidates


def build_training_target(gt_fields: list[GTField], blocks: list[TextBlock]) -> dict:
    """Sinh target JSON dạng {"fieldtype": "ID_xxxx ID_yyyy", ...} để train LoRA.
    Nếu không match được block nào -> None (field bị OCR miss hoặc threshold chưa đúng)."""
    target = {}
    for gt in gt_fields:
        matched = match_field_to_blocks(gt, blocks)
        if not matched:
            target[gt.fieldtype] = None
            continue
        tag_str = " ".join(f"ID_{b.id}" for b in matched)
        target[gt.fieldtype] = tag_str
    return target