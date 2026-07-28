# src/structuring/serializer.py
from src.preprocessing.docile_loader import TextBlock

def serialize(blocks: list[TextBlock]) -> tuple[str, dict]:
    # nhận đầu vào là danh sahcs các từ. bọc mõi từ bằng thẻ id. ví dụ [ID_0001] "Tổng" \n [ID_0002] "tiền" \n [ID_0003] "là" \n [ID_0004] "100" \n [ID_0005] "USD".
    # tra từ điển ngược
    lines = []
    id_map = {}
    for b in blocks:
        lines.append(f'[ID_{b.id}] "{b.text}"')
        id_map[b.id] = b
    return "\n".join(lines), id_map

RESPONSE_MARKER = 'Output JSON (values must be ID tags, e.g. "document_id": "ID_0042"):'

def build_prompt(schema: list[str], serialized_text: str) -> str:
    schema_str = ", ".join(schema)
    return f"""Extract the following fields as JSON, using ONLY the [ID_xxx] tags as values (not raw text). If a field is not present, use null.
Fields: {schema_str}

Document:
{serialized_text}

{RESPONSE_MARKER}"""