# src/extraction/postprocess.py
import json


def _extract_json_block(raw_output: str) -> str:
    """Tìm khối JSON đầu tiên bằng cách đếm ngoặc cân bằng,
    thay vì regex greedy \{.*\} (dễ bắt sai nếu có { } thừa trong output)."""
    start = raw_output.find("{")
    if start == -1:
        raise ValueError(f"Không tìm thấy '{{' trong output: {raw_output[:200]!r}")

    depth = 0
    for i in range(start, len(raw_output)):
        if raw_output[i] == "{":
            depth += 1
        elif raw_output[i] == "}":
            depth -= 1
            if depth == 0:
                return raw_output[start:i + 1]

    raise ValueError(f"JSON không đóng ngoặc (bị cắt bởi max_new_tokens?): {raw_output[:200]!r}")


def parse_llm_output(raw_output: str) -> dict:
    json_str = _extract_json_block(raw_output)
    return json.loads(json_str)


def back_map(parsed_json: dict, id_map: dict) -> dict:
    """Dùng cho main.py (inference thật) — trả về text/bbox/page thay vì ID thô.
    evaluate.py không dùng hàm này vì tự làm so khớp theo ID-set/text riêng."""
    result = {}
    for field, id_tag_str in parsed_json.items():
        if not id_tag_str:
            result[field] = None
            continue
        blocks = []
        for tag in id_tag_str.split():
            block_id = tag.replace("ID_", "").strip()
            block = id_map.get(block_id)
            if block:
                blocks.append(block)
        if not blocks:
            result[field] = None
            continue
        blocks.sort(key=lambda b: (b.page, round(b.bbox[1], 3), b.bbox[0]))
        result[field] = {
            "text": " ".join(b.text for b in blocks),
            "bbox": [b.bbox for b in blocks],
            "page": blocks[0].page,
        }
    return result