# src/structuring/serializer.py
from src.preprocessing.docile_loader import TextBlock

def serialize(blocks: list[TextBlock]) -> tuple[str, dict]:
    lines = []
    id_map = {}
    for b in blocks:
        lines.append(f'[ID_{b.id}] "{b.text}"')
        id_map[b.id] = b
    return "\n".join(lines), id_map

def build_prompt(schema: list[str], serialized_text: str) -> str:
    schema_str = ", ".join(schema)
    return f"""Extract the following fields as JSON, using ONLY the [ID_xxx] tags as values (not raw text). If a field is not present, use null.
Fields: {schema_str}

Document:
{serialized_text}

Output JSON (values must be ID tags, e.g. "document_id": "ID_0042"):"""