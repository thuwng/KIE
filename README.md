# KIE

# Pipeline đơn giản - base ban đầu

data/raw/ -> input PDF/ảnh
src/preprocessing/ -> Bước 1: OCR + layout
src/structuring/ -> Bước 2: Serialize + build prompt
src/models/ -> Bước 3: LLM wrapper (Llama + LoRA)
src/extraction/ -> Bước 4: Parse JSON + back-mapping
main.py -> nối 4 bước lại
