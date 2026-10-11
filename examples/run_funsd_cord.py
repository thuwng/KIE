#!/usr/bin/env python
# coding=utf-8
"""
run_funsd_cord.py GỐC của LayoutLMv3 (cùng pipeline, cùng Trainer, cùng metric seqeval, do_eval chấm trên test
như bản gốc), chỉ BỔ SUNG phần ghi log + visualize. Không có LSS.

Khác bản gốc đúng 3 chỗ kỹ thuật (đều để chạy được/đúng trên môi trường hiện tại, KHÔNG đổi kết quả):
  1. Metric: gọi thẳng thư viện seqeval (seqeval_compute) thay cho load_metric("seqeval") - cùng thư viện,
     cùng chế độ mặc định nên ra cùng con số, nhưng không cần tải script metric từ Hub.
  2. .map(..., load_from_cache_file=False): hàm tokenize không hash được nên datasets đặt fingerprint "ngẫu
     nhiên" theo RNG đã bị set_seed cố định -> cùng seed sẽ nạp nhầm cache của lượt chạy khác (lỗi đã gặp).
  3. Thêm cột doc_id / word_idx (Trainer tự bỏ vì model không nhận) để ghép lại theo tài liệu khi vẽ.

FILE TRONG output_dir:
  train.log, run_args.json, data_stats.json, log_history.jsonl
  train_results.json / eval_results.json / test_results.json   (metric GỐC, như bản gốc)
  test_predictions.txt          (như bản gốc)
  test_predictions.jsonl        mỗi tài liệu: từ, nhãn dự đoán, nhãn gold, F1 tài liệu
  test_report.txt               classification report (seqeval) cấp tài liệu
  test_error_breakdown.json     lỗi SAI LOẠI / LỆCH BIÊN / THỪA / SÓT, ma trận nhầm loại, tài liệu khó nhất
  vis/<doc>_<ảnh>.png           (--visualize) mỗi tài liệu 1 ảnh
  vis_all.pdf                   (--visualize) trang tổng hợp + toàn bộ ảnh trong 1 file PDF
"""
import dataclasses
import json
import logging
import math
import os
import platform
import socket
import subprocess
import sys
import time
from collections import Counter
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from datasets import ClassLabel, load_dataset

import transformers

from layoutlmft.data import DataCollatorForKeyValueExtraction
from transformers import (
    AutoConfig,
    AutoModelForTokenClassification,
    AutoTokenizer,
    HfArgumentParser,
    PreTrainedTokenizerFast,
    Trainer,
    TrainingArguments,
    set_seed,
)
from transformers.trainer_utils import get_last_checkpoint, is_main_process
from transformers.utils import check_min_version

# Will error if the minimal version of Transformers is not installed. Remove at your own risks.
check_min_version("4.5.0")

logger = logging.getLogger(__name__)
from layoutlmft.data.image_utils import RandomResizedCropAndInterpolationWithTwoPic, pil_loader, Compose

from timm.data.constants import \
    IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD, IMAGENET_INCEPTION_MEAN, IMAGENET_INCEPTION_STD
from torchvision import transforms
import torch


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where do you want to store the pretrained models downloaded from huggingface.co"},
    )
    model_revision: str = field(
        default="main",
        metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."},
    )
    use_auth_token: bool = field(
        default=False,
        metadata={
            "help": "Will use the token generated when running `transformers-cli login` (necessary to use this script "
            "with private models)."
        },
    )


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    task_name: Optional[str] = field(default="ner", metadata={"help": "The name of the task (ner, pos...)."})
    dataset_name: Optional[str] = field(
        default='funsd', metadata={"help": "The name of the dataset to use (via the datasets library)."}
    )
    dataset_config_name: Optional[str] = field(
        default=None, metadata={"help": "The configuration name of the dataset to use (None = mặc định của funsd.py)."}
    )
    train_file: Optional[str] = field(
        default=None, metadata={"help": "The input training data file (a csv or JSON file)."}
    )
    validation_file: Optional[str] = field(
        default=None,
        metadata={"help": "An optional input evaluation data file to evaluate on (a csv or JSON file)."},
    )
    test_file: Optional[str] = field(
        default=None,
        metadata={"help": "An optional input test data file to predict on (a csv or JSON file)."},
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )
    pad_to_max_length: bool = field(
        default=True,
        metadata={
            "help": "Whether to pad all samples to model maximum sentence length. "
            "If False, will pad the samples dynamically when batching to the maximum length in the batch. More "
            "efficient on GPU but very bad for TPU."
        },
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of training examples to this "
            "value if set."
        },
    )
    max_val_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of validation examples to this "
            "value if set."
        },
    )
    max_test_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of test examples to this "
            "value if set."
        },
    )
    label_all_tokens: bool = field(
        default=False,
        metadata={
            "help": "Whether to put the label for one word on all tokens of generated by that word or just on the "
            "one (in which case the other tokens will have a padding index)."
        },
    )
    return_entity_level_metrics: bool = field(
        default=False,
        metadata={"help": "Whether to return all the entity levels during evaluation or just the overall ones."},
    )
    segment_level_layout: bool = field(default=True)
    visual_embed: bool = field(default=True)
    data_dir: Optional[str] = field(default=None)
    input_size: int = field(default=224, metadata={"help": "images input size for backbone"})
    second_input_size: int = field(default=112, metadata={"help": "images input size for discrete vae"})
    train_interpolation: str = field(
        default='bicubic', metadata={"help": "Training interpolation (random, bilinear, bicubic)"})
    second_interpolation: str = field(
        default='lanczos', metadata={"help": "Interpolation for discrete vae (random, bilinear, bicubic)"})
    imagenet_default_mean_and_std: bool = field(default=False, metadata={"help": ""})
    # ---- BỔ SUNG: visualize ----
    visualize: bool = field(default=False, metadata={"help": "Vẽ kết quả test: vis/*.png + vis_all.pdf"})
    visualize_max_docs: Optional[int] = field(default=None, metadata={"help": "Chỉ vẽ N tài liệu đầu (None = tất cả)"})
    visualize_pdf_order: str = field(default="doc", metadata={"help": "Thứ tự trang trong vis_all.pdf: doc | f1 (tệ nhất trước)"})


# =============================================================================
# LOG HELPERS
# =============================================================================
def dump_json(obj, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, default=str)


def environment_info():
    info = {"time": time.strftime("%Y-%m-%d %H:%M:%S"), "host": socket.gethostname(),
            "python": platform.python_version(), "torch": torch.__version__,
            "transformers": transformers.__version__, "command": " ".join(sys.argv)}
    try:
        import datasets as _ds
        info["datasets"] = _ds.__version__
    except Exception:
        pass
    if torch.cuda.is_available():
        info["gpu"] = torch.cuda.get_device_name(0)
        info["cuda"] = torch.version.cuda
    try:
        info["git_commit"] = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL).decode().strip()
        info["git_dirty"] = bool(subprocess.check_output(
            ["git", "status", "--porcelain"], stderr=subprocess.DEVNULL).decode().strip())
    except Exception:
        info["git_commit"] = None
    return info


def seqeval_compute(predictions, references):
    """Cùng đầu ra với load_metric("seqeval").compute() (seqeval, chế độ mặc định), gọi thẳng thư viện."""
    from seqeval.metrics import accuracy_score, classification_report
    report = classification_report(references, predictions, output_dict=True, zero_division=0)
    report.pop("macro avg", None)
    report.pop("weighted avg", None)
    overall = report.pop("micro avg")
    out = {t: {"precision": float(v["precision"]), "recall": float(v["recall"]), "f1": float(v["f1-score"]),
               "number": int(v["support"])} for t, v in report.items()}
    out.update(overall_precision=float(overall["precision"]), overall_recall=float(overall["recall"]),
               overall_f1=float(overall["f1-score"]), overall_accuracy=float(accuracy_score(references, predictions)))
    return out


# =============================================================================
# KHỚP ENTITY (dùng chung cho đếm lỗi + visualize)
# =============================================================================
def match_entities(y_pred, y_true):
    """(gold, pred, tp, errors). Mỗi pred sai thuộc đúng 1 loại TYPE/SPAN/FP; mỗi gold sai tính đúng 1 lần."""
    from seqeval.metrics.sequence_labeling import get_entities
    gold, pred = set(get_entities(y_true)), set(get_entities(y_pred))
    tp = gold & pred
    errors, used = [], set()
    for t, s, e in sorted(pred - tp, key=lambda x: x[1]):
        same = [g for g in gold - tp if g[1] == s and g[2] == e]
        if same:
            used.add(same[0])
            errors.append({"kind": "TYPE", "span": (s, e), "pred": t, "gold": same[0][0], "gold_spans": []})
            continue
        ov = sorted((g for g in gold - tp if not (g[2] < s or g[1] > e)), key=lambda g: g[1])
        if ov:
            used.update(ov)
            errors.append({"kind": "SPAN", "span": (s, e), "pred": t, "gold": ov[0][0],
                           "gold_spans": [(g[1], g[2]) for g in ov], "gold_len": ov[0][2] - ov[0][1] + 1})
        else:
            errors.append({"kind": "FP", "span": (s, e), "pred": t, "gold": None, "gold_spans": []})
    for g in sorted(gold - tp - used, key=lambda x: x[1]):
        errors.append({"kind": "FN", "span": (g[1], g[2]), "pred": None, "gold": g[0], "gold_spans": []})
    return gold, pred, tp, errors


def _prf(tp, n_pred, n_gold):
    p = tp / n_pred if n_pred else 0.0
    r = tp / n_gold if n_gold else 0.0
    return p, r, (2 * p * r / (p + r) if p + r else 0.0)


def error_breakdown(docs):
    """docs: list (doc_id, y_pred, y_true) cấp tài liệu."""
    total = Counter()
    per_type, confusion, per_doc = {}, Counter(), []
    for d, yp, yt in docs:
        gold, pred, tp, errs = match_entities(yp, yt)
        total["n_gold"] += len(gold)
        total["n_pred"] += len(pred)
        total["TP"] += len(tp)
        for t, _, _ in gold:
            per_type.setdefault(t, Counter())["gold"] += 1
        for t, _, _ in pred:
            per_type.setdefault(t, Counter())["pred"] += 1
        for t, _, _ in tp:
            per_type[t]["TP"] += 1
        dc = Counter()
        for e in errs:
            total[e["kind"]] += 1
            dc[e["kind"]] += 1
            per_type.setdefault(e["pred"] if e["kind"] == "FP" else e["gold"], Counter())[e["kind"]] += 1
            if e["kind"] == "TYPE":
                confusion[f'{e["gold"]}->{e["pred"]}'] += 1
        per_doc.append({"doc_id": d, "n_gold": len(gold), "n_pred": len(pred), "TP": len(tp),
                        "f1": _prf(len(tp), len(pred), len(gold))[2],
                        **{k: dc[k] for k in ("TYPE", "SPAN", "FP", "FN")}})
    p, r, f = _prf(total["TP"], total["n_pred"], total["n_gold"])
    overall = {**{k: total[k] for k in ("n_gold", "n_pred", "TP", "TYPE", "SPAN", "FP", "FN")},
               "precision": p, "recall": r, "f1": f}
    pt = {t: {**dict(c), "f1": _prf(c["TP"], c["pred"], c["gold"])[2]} for t, c in sorted(per_type.items())}
    worst = sorted(per_doc, key=lambda x: x["f1"])[:15]
    return {"overall": overall, "per_type": pt, "type_confusion": dict(confusion.most_common()),
            "worst_docs": worst}, per_doc


# =============================================================================
# VISUALIZE
# =============================================================================
# Bảng màu Okabe-Ito (an toàn với người mù màu đỏ-lục) + khác biệt nét đứt/liền
VIS_COLORS = {
    "TP":   (0, 158, 115),    # xanh lục lam  - đúng
    "TYPE": (230, 159, 0),    # cam           - đúng biên, SAI LOẠI
    "SPAN": (204, 121, 167),  # tím hồng      - LỆCH BIÊN (nét đứt = biên gold)
    "FP":   (213, 94, 0),     # đỏ son        - THỪA
    "FN":   (0, 114, 178),    # xanh dương    - SÓT (nét đứt)
}
VIS_NAMES = {"TP": "ĐÚNG", "TYPE": "SAI LOẠI", "SPAN": "LỆCH BIÊN", "FP": "THỪA", "FN": "SÓT"}
_SHORT = {"QUESTION": "Q", "ANSWER": "A", "HEADER": "H"}


def _short(t):
    return _SHORT.get(t, t)


def _error_note(e):
    s, t = e["span"], e["kind"]
    if t == "TYPE":
        return f"gold {_short(e['gold'])} → pred {_short(e['pred'])}"
    if t == "SPAN":
        note = f"pred {_short(e['pred'])} {s[1] - s[0] + 1} từ / gold {_short(e['gold'])} {e['gold_len']} từ"
        return note + (f" (gộp {len(e['gold_spans'])} entity)" if len(e["gold_spans"]) > 1 else "")
    if t == "FP":
        return f"dư {_short(e['pred'])}"
    return f"thiếu {_short(e['gold'])}"


def _load_font(size):
    from PIL import ImageFont
    for f in ("DejaVuSans.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
              "/usr/share/fonts/dejavu/DejaVuSans.ttf", "arial.ttf"):
        try:
            return ImageFont.truetype(f, size)
        except OSError:
            pass
    return ImageFont.load_default()


def _dashed_rect(draw, box, color, width, dash=8, gap=5):
    x0, y0, x1, y1 = box
    for a, b, c, d in ((x0, y0, x1, y0), (x1, y0, x1, y1), (x1, y1, x0, y1), (x0, y1, x0, y0)):
        length = max(math.hypot(c - a, d - b), 1e-6)
        pos = 0.0
        while pos < length:
            s, e = pos / length, min(pos + dash, length) / length
            draw.line([(a + (c - a) * s, b + (d - b) * s), (a + (c - a) * e, b + (d - b) * e)],
                      fill=color, width=width)
            pos += dash + gap


def visualize_document(image_path, words, boxes, y_pred, y_true, out_path, title=""):
    """words/boxes/y_pred/y_true cùng độ dài, cùng thứ tự; box chuẩn hoá 0..1000."""
    from PIL import Image, ImageDraw

    img = Image.open(image_path).convert("RGBA")
    W, H = img.size

    def ent_box(s, e):
        bs = boxes[s:e + 1]
        return (min(b[0] for b in bs) * W / 1000.0, min(b[1] for b in bs) * H / 1000.0,
                max(b[2] for b in bs) * W / 1000.0, max(b[3] for b in bs) * H / 1000.0)

    def ent_text(s, e, n=30):
        t = " ".join(words[s:e + 1])
        return t if len(t) <= n else t[:n - 1] + "…"

    gold, pred, tp, errs = match_entities(y_pred, y_true)
    items = [(e["kind"], ent_box(*e["span"]), [ent_box(*g) for g in e["gold_spans"]],
              _error_note(e), ent_text(*e["span"])) for e in errs]
    items.sort(key=lambda it: (round(it[1][1] / 10), it[1][0]))

    lw = max(2, round(min(W, H) / 400))
    overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
    od = ImageDraw.Draw(overlay)
    for _, s, e in tp:
        c = VIS_COLORS["TP"]
        od.rectangle(ent_box(s, e), fill=c + (35,), outline=c + (255,), width=lw)
    for kind, b, golds, _, _ in items:
        c = VIS_COLORS[kind]
        if kind == "FN":
            od.rectangle(b, fill=c + (30,))
            _dashed_rect(od, b, c + (255,), lw + 1)
        else:
            od.rectangle(b, fill=c + (60,), outline=c + (255,), width=lw + 1)
        for gb in golds:
            _dashed_rect(od, gb, c + (255,), lw)
    img = Image.alpha_composite(img, overlay)

    d = ImageDraw.Draw(img)
    f_tag = _load_font(max(11, H // 75))
    for k, (kind, b, _, _, _) in enumerate(items, 1):
        tag = str(k)
        _, _, tw, th = d.textbbox((0, 0), tag, font=f_tag)
        x, y = b[0], max(0, b[1] - th - 5)
        d.rectangle([x, y, x + tw + 6, y + th + 4], fill=VIS_COLORS[kind])
        d.text((x + 3, y + 1), tag, fill="white", font=f_tag)

    PW = max(460, int(W * 0.6))
    canvas = Image.new("RGB", (W + PW, H), "white")
    canvas.paste(img.convert("RGB"), (0, 0))
    d = ImageDraw.Draw(canvas)
    f_title, f = _load_font(max(14, H // 55)), _load_font(max(11, H // 80))
    lh = d.textbbox((0, 0), "Ág", font=f)[3] + 6
    x, y = W + 16, 12
    nP, nG, nT = len(pred), len(gold), len(tp)
    f1 = _prf(nT, nP, nG)[2]
    d.text((x, y), title, fill="black", font=f_title)
    y += d.textbbox((0, 0), "Ág", font=f_title)[3] + 10
    d.text((x, y), f"gold {nG} | pred {nP} | đúng {nT} | F1 tài liệu {f1 * 100:.1f}", fill="black", font=f)
    y += lh + 4
    counts = Counter(it[0] for it in items)
    counts["TP"] = nT
    for kind in ("TP", "TYPE", "SPAN", "FP", "FN"):
        c, sq = VIS_COLORS[kind], lh - 8
        if kind == "FN":
            _dashed_rect(d, (x, y + 2, x + sq, y + 2 + sq), c, 2, dash=4, gap=3)
        else:
            d.rectangle([x, y + 2, x + sq, y + 2 + sq], fill=c)
        d.text((x + sq + 8, y), f"{VIS_NAMES[kind]}: {counts[kind]}", fill="black", font=f)
        y += lh
    d.line([(x, y + 4), (W + PW - 16, y + 4)], fill=(180, 180, 180), width=1)
    y += 12
    for k, (kind, _, _, note, txt) in enumerate(items, 1):
        if y + 2 * lh > H - 8:
            d.text((x, y), f"… còn {len(items) - k + 1} lỗi (xem test_predictions.jsonl)", fill="gray", font=f)
            break
        d.rectangle([x, y + 2, x + 6, y + 2 * lh - 8], fill=VIS_COLORS[kind])
        d.text((x + 12, y), f"#{k} {VIS_NAMES[kind]}: {note}", fill=(20, 20, 20), font=f)
        d.text((x + 24, y + lh - 2), f"\"{txt}\"", fill=(70, 70, 70), font=f)
        y += 2 * lh
    canvas.save(out_path)


def make_vis_pdf(pages, out_pdf, title, official, overall, order="doc"):
    """pages: list dict(path, doc_id, image, f1, TYPE, SPAN, FP, FN). Trang 1 = tổng hợp, sau đó mỗi tài liệu 1 trang."""
    from PIL import Image, ImageDraw
    if order == "f1":
        pages = sorted(pages, key=lambda p: p["f1"])
    W, H = 1240, 1754                                    # A4 ở 150 dpi
    cover = Image.new("RGB", (W, H), "white")
    d = ImageDraw.Draw(cover)
    f_big, f_mid, f = _load_font(34), _load_font(22), _load_font(18)
    y = 60
    d.text((70, y), "Tổng hợp kết quả visualize", fill="black", font=f_big)
    y += 56
    d.text((70, y), title, fill=(60, 60, 60), font=f_mid)
    y += 46
    d.text((70, y), "Metric chính (như bản gốc, seqeval):  "
                    f"P {official.get('precision', 0) * 100:.2f}   R {official.get('recall', 0) * 100:.2f}   "
                    f"F1 {official.get('f1', 0) * 100:.2f}", fill="black", font=f_mid)
    y += 40
    d.text((70, y), f"Cấp tài liệu: gold {overall['n_gold']} | pred {overall['n_pred']} | đúng {overall['TP']} | "
                    f"F1 {overall['f1'] * 100:.2f}", fill="black", font=f)
    y += 40
    x = 70
    for kind in ("TP", "TYPE", "SPAN", "FP", "FN"):          # chú giải + số lỗi
        n = overall["TP"] if kind == "TP" else overall[kind]
        if kind == "FN":
            _dashed_rect(d, (x, y + 3, x + 22, y + 25), VIS_COLORS[kind], 3, dash=5, gap=3)
        else:
            d.rectangle([x, y + 3, x + 22, y + 25], fill=VIS_COLORS[kind])
        label = f"{VIS_NAMES[kind]}: {n}"
        d.text((x + 30, y + 2), label, fill="black", font=f)
        x += 50 + d.textbbox((0, 0), label, font=f)[2]
    y += 56
    head = ["Trang", "Tài liệu", "F1", "Sai loại", "Lệch biên", "Thừa", "Sót"]
    cols = [70, 160, 560, 680, 800, 940, 1040]
    for c, t in zip(cols, head):
        d.text((c, y), t, fill="black", font=f_mid)
    y += 34
    d.line([(70, y), (W - 70, y)], fill=(150, 150, 150), width=2)
    y += 8
    rh = max(18, min(28, (H - y - 60) // max(len(pages), 1)))
    fr = _load_font(max(12, rh - 10))
    for i, p in enumerate(pages, 2):
        if y + rh > H - 40:
            d.text((70, y), f"… còn {len(pages) - i + 2} tài liệu", fill="gray", font=fr)
            break
        if p["f1"] < 0.7:
            d.rectangle([66, y - 2, W - 66, y + rh - 4], fill=(253, 236, 226))   # tô nền tài liệu F1 < 70
        row = [str(i), f"{p['doc_id']} {p['image']}"[:34], f"{p['f1'] * 100:.1f}",
               str(p["TYPE"]), str(p["SPAN"]), str(p["FP"]), str(p["FN"])]
        for c, t in zip(cols, row):
            d.text((c, y), t, fill=(30, 30, 30), font=fr)
        y += rh
    imgs = [Image.open(p["path"]).convert("RGB") for p in pages]
    cover.save(out_pdf, "PDF", resolution=150.0, save_all=True, append_images=imgs)
    for im in imgs:
        im.close()


def main():
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # Detecting last checkpoint.
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif last_checkpoint is not None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )

    out_dir = training_args.output_dir
    os.makedirs(out_dir, exist_ok=True)
    main_proc = is_main_process(training_args.local_rank)

    # Setup logging (+ BỔ SUNG: ghi ra train.log)
    fmt = "%(asctime)s - %(levelname)s - %(name)s -   %(message)s"
    logging.basicConfig(format=fmt, datefmt="%m/%d/%Y %H:%M:%S", handlers=[logging.StreamHandler(sys.stdout)])
    logger.setLevel(logging.INFO if main_proc else logging.WARN)
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    if main_proc:
        transformers.utils.logging.set_verbosity_info()
        transformers.utils.logging.enable_default_handler()
        transformers.utils.logging.enable_explicit_format()
        fh = logging.FileHandler(os.path.join(out_dir, "train.log"), mode="w", encoding="utf-8")
        fh.setFormatter(logging.Formatter(fmt, datefmt="%m/%d/%Y %H:%M:%S"))
        logging.getLogger().addHandler(fh)
        if hasattr(transformers.utils.logging, "add_handler"):
            transformers.utils.logging.add_handler(fh)
        dump_json({"env": environment_info(), "model_args": dataclasses.asdict(model_args),
                   "data_args": dataclasses.asdict(data_args), "training_args": training_args.to_dict()},
                  os.path.join(out_dir, "run_args.json"))
    logger.info(f"Training/evaluation parameters {training_args}")

    # Set seed before initializing model.
    set_seed(training_args.seed)

    if data_args.dataset_name == 'funsd':
        import layoutlmft.data.funsd
        datasets = load_dataset(os.path.abspath(layoutlmft.data.funsd.__file__), name=data_args.dataset_config_name,
                                cache_dir=model_args.cache_dir)
    elif data_args.dataset_name == 'cord':
        import layoutlmft.data.cord
        datasets = load_dataset(os.path.abspath(layoutlmft.data.cord.__file__), name=data_args.dataset_config_name,
                                cache_dir=model_args.cache_dir)
    else:
        raise NotImplementedError()

    if training_args.do_train:
        column_names = datasets["train"].column_names
        features = datasets["train"].features
    else:
        column_names = datasets["test"].column_names
        features = datasets["test"].features

    text_column_name = "words" if "words" in column_names else "tokens"

    label_column_name = (
        f"{data_args.task_name}_tags" if f"{data_args.task_name}_tags" in column_names else column_names[1]
    )

    remove_columns = column_names

    def get_label_list(labels):
        unique_labels = set()
        for label in labels:
            unique_labels = unique_labels | set(label)
        label_list = list(unique_labels)
        label_list.sort()
        return label_list

    if isinstance(features[label_column_name].feature, ClassLabel):
        label_list = features[label_column_name].feature.names
        label_to_id = {i: i for i in range(len(label_list))}
    else:
        label_list = get_label_list(datasets["train"][label_column_name])
        label_to_id = {l: i for i, l in enumerate(label_list)}
    num_labels = len(label_list)

    config = AutoConfig.from_pretrained(
        model_args.config_name if model_args.config_name else model_args.model_name_or_path,
        num_labels=num_labels,
        finetuning_task=data_args.task_name,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        input_size=data_args.input_size,
        use_auth_token=True if model_args.use_auth_token else None,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
        tokenizer_file=None,  # avoid loading from a cached file of the pre-trained model in another machine
        cache_dir=model_args.cache_dir,
        use_fast=True,
        add_prefix_space=True,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )
    model = AutoModelForTokenClassification.from_pretrained(
        model_args.model_name_or_path,
        from_tf=bool(".ckpt" in model_args.model_name_or_path),
        config=config,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )

    if not isinstance(tokenizer, PreTrainedTokenizerFast):
        raise ValueError(
            "This example script only works for models that have a fast tokenizer. Checkout the big table of models "
            "at https://huggingface.co/transformers/index.html#bigtable to find the model types that meet this "
            "requirement"
        )

    padding = "max_length" if data_args.pad_to_max_length else False

    if data_args.visual_embed:
        imagenet_default_mean_and_std = data_args.imagenet_default_mean_and_std
        mean = IMAGENET_INCEPTION_MEAN if not imagenet_default_mean_and_std else IMAGENET_DEFAULT_MEAN
        std = IMAGENET_INCEPTION_STD if not imagenet_default_mean_and_std else IMAGENET_DEFAULT_STD
        common_transform = Compose([
            RandomResizedCropAndInterpolationWithTwoPic(
                size=data_args.input_size, interpolation=data_args.train_interpolation),
        ])

        patch_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                mean=torch.tensor(mean),
                std=torch.tensor(std))
        ])

    # Tokenize all texts and align the labels with them.
    def tokenize_and_align_labels(examples, augmentation=False):
        tokenized_inputs = tokenizer(
            examples[text_column_name],
            padding=False,
            truncation=True,
            return_overflowing_tokens=True,
            is_split_into_words=True,
        )

        labels = []
        bboxes = []
        images = []
        doc_ids, word_idxs = [], []                       # BỔ SUNG: để ghép theo tài liệu khi vẽ
        for batch_index in range(len(tokenized_inputs["input_ids"])):
            word_ids = tokenized_inputs.word_ids(batch_index=batch_index)
            org_batch_index = tokenized_inputs["overflow_to_sample_mapping"][batch_index]

            label = examples[label_column_name][org_batch_index]
            bbox = examples["bboxes"][org_batch_index]
            previous_word_idx = None
            label_ids = []
            bbox_inputs = []
            widx = []
            for word_idx in word_ids:
                if word_idx is None:
                    label_ids.append(-100)
                    bbox_inputs.append([0, 0, 0, 0])
                    widx.append(-1)
                elif word_idx != previous_word_idx:
                    label_ids.append(label_to_id[label[word_idx]])
                    bbox_inputs.append(bbox[word_idx])
                    widx.append(word_idx)
                else:
                    label_ids.append(label_to_id[label[word_idx]] if data_args.label_all_tokens else -100)
                    bbox_inputs.append(bbox[word_idx])
                    widx.append(-1)
                previous_word_idx = word_idx
            labels.append(label_ids)
            bboxes.append(bbox_inputs)
            doc_ids.append(examples["id"][org_batch_index])
            word_idxs.append(widx)

            if data_args.visual_embed:
                ipath = examples["image_path"][org_batch_index]
                img = pil_loader(ipath)
                for_patches, _ = common_transform(img, augmentation=augmentation)
                patch = patch_transform(for_patches)
                images.append(patch)

        tokenized_inputs["labels"] = labels
        tokenized_inputs["bbox"] = bboxes
        tokenized_inputs["doc_id"] = doc_ids          # Trainer tự bỏ (model không nhận)
        tokenized_inputs["word_idx"] = word_idxs      # Trainer tự bỏ
        if data_args.visual_embed:
            tokenized_inputs["images"] = images

        return tokenized_inputs

    def prep(ds, max_n):
        if max_n is not None:
            ds = ds.select(range(max_n))
        n_docs = len(ds)
        out = ds.map(tokenize_and_align_labels, batched=True, remove_columns=remove_columns,
                     num_proc=data_args.preprocessing_num_workers,
                     load_from_cache_file=False)          # KHÔNG dùng cache (xem đầu file, mục 2)
        if len(set(out["doc_id"])) != n_docs:
            raise RuntimeError(f"Sau tokenize có {len(set(out['doc_id']))} tài liệu, đầu vào {n_docs} -> dừng.")
        return out

    if training_args.do_train:
        if "train" not in datasets:
            raise ValueError("--do_train requires a train dataset")
        train_dataset = prep(datasets["train"], data_args.max_train_samples)

    if training_args.do_eval:
        validation_name = "test"                          # như bản gốc
        if validation_name not in datasets:
            raise ValueError("--do_eval requires a validation dataset")
        eval_dataset = prep(datasets[validation_name], data_args.max_val_samples)

    if training_args.do_predict:
        if "test" not in datasets:
            raise ValueError("--do_predict requires a test dataset")
        test_dataset = prep(datasets["test"], data_args.max_test_samples)

    # BỔ SUNG: data_stats.json
    if main_proc:
        stats = {"dataset": data_args.dataset_name, "config": data_args.dataset_config_name}

        def _st(tok):
            chunks = Counter(tok["doc_id"])
            return {"docs": len(chunks), "chunks_512": len(tok),
                    "docs_over_512_tokens": sum(1 for v in chunks.values() if v > 1)}

        if training_args.do_train:
            stats["train"] = _st(train_dataset)
        if training_args.do_predict:
            stats["test"] = _st(test_dataset)
        elif training_args.do_eval:
            stats["test"] = _st(eval_dataset)
        dump_json(stats, os.path.join(out_dir, "data_stats.json"))
        logger.info(f"Data stats: {json.dumps(stats, ensure_ascii=False)}")

    data_collator = DataCollatorForKeyValueExtraction(
        tokenizer,
        pad_to_multiple_of=8 if training_args.fp16 else None,
        padding=padding,
        max_length=512,
    )

    # Metrics (như bản gốc: seqeval trên từng đoạn, sub-token đầu)
    def compute_metrics(p):
        predictions, labels = p
        predictions = np.argmax(predictions, axis=2)

        true_predictions = [
            [label_list[p] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]
        true_labels = [
            [label_list[l] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]

        results = seqeval_compute(predictions=true_predictions, references=true_labels)
        if data_args.return_entity_level_metrics:
            final_results = {}
            for key, value in results.items():
                if isinstance(value, dict):
                    for n, v in value.items():
                        final_results[f"{key}_{n}"] = v
                else:
                    final_results[key] = value
            return final_results
        else:
            return {
                "precision": results["overall_precision"],
                "recall": results["overall_recall"],
                "f1": results["overall_f1"],
                "accuracy": results["overall_accuracy"],
            }

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=eval_dataset if training_args.do_eval else None,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    def write_log_history():
        if trainer.is_world_process_zero():
            with open(os.path.join(out_dir, "log_history.jsonl"), "w", encoding="utf-8") as w:
                for rec in trainer.state.log_history:
                    w.write(json.dumps(rec, ensure_ascii=False) + "\n")

    # Training
    if training_args.do_train:
        checkpoint = last_checkpoint if last_checkpoint else None
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        metrics = train_result.metrics
        trainer.save_model()

        max_train_samples = (
            data_args.max_train_samples if data_args.max_train_samples is not None else len(train_dataset)
        )
        metrics["train_samples"] = min(max_train_samples, len(train_dataset))

        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()
        write_log_history()

    # Evaluation
    if training_args.do_eval:
        logger.info("*** Evaluate ***")

        metrics = trainer.evaluate()

        max_val_samples = data_args.max_val_samples if data_args.max_val_samples is not None else len(eval_dataset)
        metrics["eval_samples"] = min(max_val_samples, len(eval_dataset))

        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    # Predict
    if training_args.do_predict:
        logger.info("*** Predict ***")

        raw_predictions, labels, metrics = trainer.predict(test_dataset)
        predictions = np.argmax(raw_predictions, axis=2)

        true_predictions = [
            [label_list[p] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]

        trainer.log_metrics("test", metrics)
        trainer.save_metrics("test", metrics)

        output_test_predictions_file = os.path.join(training_args.output_dir, "test_predictions.txt")
        if trainer.is_world_process_zero():
            with open(output_test_predictions_file, "w") as writer:
                for prediction in true_predictions:
                    writer.write(" ".join(prediction) + "\n")

        # ================= BỔ SUNG: phân tích cấp tài liệu + visualize =================
        if trainer.is_world_process_zero():
            doc_ids, word_idx = test_dataset["doc_id"], test_dataset["word_idx"]
            docs, order = {}, []
            for i in range(len(predictions)):
                d = doc_ids[i]
                if d not in docs:
                    docs[d] = {}
                    order.append(d)
                for j, w in enumerate(word_idx[i]):
                    if w < 0 or labels[i][j] == -100 or w in docs[d]:
                        continue
                    docs[d][w] = (label_list[predictions[i][j]], label_list[labels[i][j]])
            seqs = []
            for d in order:
                it = sorted(docs[d].items())
                seqs.append((d, [w for w, _ in it], [p for _, (p, _) in it], [g for _, (_, g) in it]))

            breakdown, per_doc = error_breakdown([(d, yp, yt) for d, _, yp, yt in seqs])
            dump_json(breakdown, os.path.join(out_dir, "test_error_breakdown.json"))
            from seqeval.metrics import classification_report
            with open(os.path.join(out_dir, "test_report.txt"), "w", encoding="utf-8") as w:
                w.write(f"{data_args.dataset_name} / {data_args.dataset_config_name} | seed {training_args.seed}\n")
                w.write("seqeval (mặc định), ghép theo TÀI LIỆU (metric chính ở test_results.json tính theo đoạn)\n\n")
                w.write(classification_report([s[3] for s in seqs], [s[2] for s in seqs], digits=4))
            logger.info(f"Error breakdown: {json.dumps(breakdown['overall'], ensure_ascii=False)}")

            raw = datasets["test"]
            if "image" in raw.column_names:
                raw = raw.remove_columns("image")
            id2row = {ex_id: i for i, ex_id in enumerate(raw["id"])}
            doc_stat = {r["doc_id"]: r for r in per_doc}
            with open(os.path.join(out_dir, "test_predictions.jsonl"), "w", encoding="utf-8") as w:
                for d, wl, yp, yt in seqs:
                    ex = raw[id2row[d]]
                    w.write(json.dumps({"doc_id": d, "image": os.path.basename(ex["image_path"]),
                                        "doc_f1": doc_stat[d]["f1"], "words": [ex[text_column_name][i] for i in wl],
                                        "pred": yp, "gold": yt}, ensure_ascii=False) + "\n")

            if data_args.visualize:
                vis_dir = os.path.join(out_dir, "vis")
                os.makedirs(vis_dir, exist_ok=True)
                n_vis = len(seqs) if data_args.visualize_max_docs is None else data_args.visualize_max_docs
                run_title = f"{data_args.dataset_config_name or data_args.dataset_name} | seed {training_args.seed}"
                pages = []
                for d, wl, yp, yt in seqs[:n_vis]:
                    ex = raw[id2row[d]]
                    name = os.path.splitext(os.path.basename(ex["image_path"]))[0]
                    path = os.path.join(vis_dir, f"{d}_{name}.png")
                    try:
                        visualize_document(ex["image_path"], [ex[text_column_name][w] for w in wl],
                                           [ex["bboxes"][w] for w in wl], yp, yt, path,
                                           title=f"{name} | {run_title}")
                        s = doc_stat[d]
                        pages.append({"path": path, "doc_id": d, "image": name, "f1": s["f1"],
                                      **{k: s[k] for k in ("TYPE", "SPAN", "FP", "FN")}})
                    except Exception as e:
                        logger.warning(f"Không vẽ được tài liệu {d}: {e}")
                logger.info(f"Đã lưu {len(pages)}/{n_vis} ảnh visualize vào {vis_dir}")
                if pages:
                    pdf_path = os.path.join(out_dir, "vis_all.pdf")
                    try:
                        official = {k: metrics.get(f"test_{k}", 0.0) for k in ("precision", "recall", "f1")}
                        make_vis_pdf(pages, pdf_path, os.path.basename(os.path.abspath(out_dir)) + " | " + run_title,
                                     official, breakdown["overall"], order=data_args.visualize_pdf_order)
                        logger.info(f"Đã lưu PDF tổng hợp {len(pages) + 1} trang: {pdf_path}")
                    except Exception as e:
                        logger.warning(f"Không tạo được PDF tổng hợp: {e}")

    write_log_history()


def _mp_fn(index):
    # For xla_spawn (TPUs)
    main()


if __name__ == "__main__":
    main()