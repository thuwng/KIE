#!/usr/bin/env python
# coding=utf-8
"""
Fine-tune LayoutLMv3 (baseline) hoặc LayoutLMv3 + Latent Soft Segment (LSS) trên FUNSD / CORD.

METRIC CHÍNH (duy nhất): seqeval entity-level micro F1, IOB2, chế độ mặc định (như LayoutLMv3 gốc),
chấm trên sub-token đầu của mỗi từ, gộp các đoạn 512 token về NGUYÊN tài liệu, mỗi từ đúng 1 lần.

FILE LOG trong output_dir (mỗi lượt chạy):
  train.log                 log Python (logger + transformers)
  run_args.json             toàn bộ tham số + môi trường (phiên bản, GPU, git commit, lệnh chạy)
  data_stats.json           số tài liệu / từ / đoạn 512 / entity theo loại cho từng split
  model_stats.json          số tham số backbone / tham số mới
  log_history.jsonl         lịch sử log theo bước (loss, ce_loss, aff_loss, tf_prob, alpha, lambda, lr, dev F1...)
  train_results.json        thời gian, tốc độ, loss cuối, bộ nhớ GPU
  eval_results.json         (nếu có dev) F1 + theo loại + chất lượng gom nhóm
  test_results.json         F1 chính + P/R + theo loại + gom nhóm + đếm lỗi + kiểm tra nhất quán
  test_report.txt           classification report của seqeval
  test_error_breakdown.json lỗi theo loại (SAI LOẠI/LỆCH BIÊN/THỪA/SÓT), ma trận nhầm loại, tài liệu khó nhất
  test_predictions.jsonl    mỗi tài liệu: từ, nhãn dự đoán, nhãn gold, F1 tài liệu
  test_raw_predictions.npz  id nhãn + độ tự tin theo token, word_idx, doc_id, meta
  vis/                      (nếu --visualize) mỗi tài liệu 1 ảnh PNG
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
import torch
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

check_min_version("4.5.0")

logger = logging.getLogger(__name__)
from layoutlmft.data.image_utils import RandomResizedCropAndInterpolationWithTwoPic, pil_loader, Compose

from timm.data.constants import \
    IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD, IMAGENET_INCEPTION_MEAN, IMAGENET_INCEPTION_STD
from torchvision import transforms


@dataclass
class ModelArguments:
    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    config_name: Optional[str] = field(default=None)
    tokenizer_name: Optional[str] = field(default=None)
    cache_dir: Optional[str] = field(default=None)
    model_revision: str = field(default="main")
    use_auth_token: bool = field(default=False)


@dataclass
class DataTrainingArguments:
    task_name: Optional[str] = field(default="ner")
    dataset_name: Optional[str] = field(default='funsd', metadata={"help": "funsd | cord"})
    dataset_config_name: Optional[str] = field(
        default=None,
        metadata={"help": "funsd | funsd_word | funsd_word_ro | cord | cord_word (mặc định = config gốc)"},
    )
    overwrite_cache: bool = field(default=False)
    preprocessing_num_workers: Optional[int] = field(default=None)
    pad_to_max_length: bool = field(default=True)
    max_train_samples: Optional[int] = field(default=None)
    max_val_samples: Optional[int] = field(default=None)
    max_test_samples: Optional[int] = field(default=None)
    label_all_tokens: bool = field(default=False)
    return_entity_level_metrics: bool = field(default=True, metadata={"help": "Log thêm P/R/F1 theo từng loại entity"})
    # tập dev để chọn siêu tham số. 0 = không tách dev (lượt chạy cuối, số bước cố định).
    dev_ratio: float = field(
        default=0.0,
        metadata={"help": "Tỉ lệ train tách làm dev (FUNSD không có dev). Dataset có sẵn validation thì dùng luôn."},
    )
    dev_split_seed: int = field(default=42, metadata={"help": "Seed CỐ ĐỊNH cho việc tách dev, độc lập seed train."})
    visual_embed: bool = field(default=True)
    # ---- Latent Soft Segment ----
    use_latent_segment: bool = field(default=False, metadata={"help": "Dùng LayoutLMv3ForLatentSegmentTokenClassification"})
    latent_layer: int = field(default=6, metadata={"help": "Số lớp chạy ở lượt 1 để tính affinity"})
    latent_soft_box: bool = field(default=True, metadata={"help": "Ablation: tắt box mềm"})
    latent_attn_bias: bool = field(default=True, metadata={"help": "Ablation: tắt bias attention"})
    latent_tf_ratio: float = field(default=0.5, metadata={"help": "Teacher forcing giảm 1->0 trong tỉ lệ này của tổng bước; 0 = tắt"})
    latent_oracle_eval: bool = field(default=False, metadata={"help": "CẬN TRÊN: dùng nhóm gold lúc test (không phải kết quả chính)"})
    affinity_loss_weight: float = field(default=1.0)
    head_learning_rate: float = field(default=5e-4, metadata={"help": "LR cho các tham số latent_* mới"})
    input_size: int = field(default=224)
    second_input_size: int = field(default=112)
    train_interpolation: str = field(default='bicubic')
    second_interpolation: str = field(default='lanczos')
    imagenet_default_mean_and_std: bool = field(default=False)
    # ---- visualize ----
    visualize: bool = field(default=False, metadata={"help": "Vẽ kết quả test: mỗi tài liệu 1 file PNG trong output_dir/vis"})
    visualize_max_docs: Optional[int] = field(default=None, metadata={"help": "Chỉ vẽ N tài liệu đầu (None = tất cả)"})


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


def seqeval_compute(y_pred, y_true):
    """Tính y hệt metric "seqeval" của HF datasets/evaluate (mode mặc định, IOB2), nhưng gọi thẳng
    thư viện seqeval -> không cần cài `evaluate`, không cần tải script metric từ Hub."""
    from seqeval.metrics import accuracy_score, classification_report
    report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
    report.pop("macro avg", None)
    report.pop("weighted avg", None)
    overall = report.pop("micro avg")
    # ép về float/int Python: np.int64 làm json.dump (save_metrics) báo lỗi
    out = {t: {"precision": float(v["precision"]), "recall": float(v["recall"]), "f1": float(v["f1-score"]),
               "number": int(v["support"])} for t, v in report.items()}
    out.update(overall_precision=float(overall["precision"]), overall_recall=float(overall["recall"]),
               overall_f1=float(overall["f1-score"]), overall_accuracy=float(accuracy_score(y_true, y_pred)))
    return out


def gpu_mem_gb():
    return round(torch.cuda.max_memory_allocated() / 1e9, 2) if torch.cuda.is_available() else 0.0


# =============================================================================
# KHỚP ENTITY (dùng chung cho đếm lỗi + visualize, cùng get_entities với seqeval)
# =============================================================================
def match_entities(y_pred, y_true):
    """Trả về (gold, pred, tp, errors). Mỗi pred sai thuộc đúng 1 loại TYPE/SPAN/FP;
    mỗi gold sai được tính đúng 1 lần (qua TYPE, SPAN hoặc FN) -> tổng khớp với P/R của seqeval."""
    from seqeval.metrics.sequence_labeling import get_entities
    gold, pred = set(get_entities(y_true)), set(get_entities(y_pred))
    tp = gold & pred
    errors, used = [], set()
    for t, s, e in sorted(pred - tp, key=lambda x: x[1]):
        same = [g for g in gold - tp if g[1] == s and g[2] == e]
        if same:                                    # đúng biên, sai loại
            used.add(same[0])
            errors.append({"kind": "TYPE", "span": (s, e), "pred": t, "gold": same[0][0], "gold_spans": []})
            continue
        ov = sorted((g for g in gold - tp if not (g[2] < s or g[1] > e)), key=lambda g: g[1])
        if ov:                                      # chồng lấn nhưng lệch biên
            used.update(ov)
            errors.append({"kind": "SPAN", "span": (s, e), "pred": t, "gold": ov[0][0],
                           "gold_spans": [(g[1], g[2]) for g in ov], "gold_len": ov[0][2] - ov[0][1] + 1})
        else:                                       # không chạm gold nào
            errors.append({"kind": "FP", "span": (s, e), "pred": t, "gold": None, "gold_spans": []})
    for g in sorted(gold - tp - used, key=lambda x: x[1]):
        errors.append({"kind": "FN", "span": (g[1], g[2]), "pred": None, "gold": g[0], "gold_spans": []})
    return gold, pred, tp, errors


def _prf(tp, n_pred, n_gold):
    p = tp / n_pred if n_pred else 0.0
    r = tp / n_gold if n_gold else 0.0
    return p, r, (2 * p * r / (p + r) if p + r else 0.0)


def error_breakdown(docs):
    """docs: list (doc_id, y_pred, y_true). Đếm lỗi toàn tập, theo loại entity, ma trận nhầm loại, theo tài liệu."""
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
        f1 = _prf(len(tp), len(pred), len(gold))[2]
        per_doc.append({"doc_id": d, "n_gold": len(gold), "n_pred": len(pred), "TP": len(tp),
                        "f1": f1, **{k: dc[k] for k in ("TYPE", "SPAN", "FP", "FN")}})
    p, r, f = _prf(total["TP"], total["n_pred"], total["n_gold"])
    overall = {**{k: total[k] for k in ("n_gold", "n_pred", "TP", "TYPE", "SPAN", "FP", "FN")},
               "precision": p, "recall": r, "f1": f}
    pt = {}
    for t, c in sorted(per_type.items()):
        tp_, pr_, gd_ = c["TP"], c["pred"], c["gold"]
        pt[t] = {**dict(c), "f1": _prf(tp_, pr_, gd_)[2]}
    per_doc.sort(key=lambda x: x["f1"])
    return {"overall": overall, "per_type": pt, "type_confusion": dict(confusion.most_common()),
            "worst_docs": per_doc[:15]}, per_doc


# =============================================================================
# VISUALIZE: mỗi tài liệu 1 ảnh, chấm ở CẤP ENTITY đúng như metric seqeval
# =============================================================================
# Bảng màu Okabe-Ito: phân biệt được cả với người mù màu đỏ-lục; thêm khác biệt nét (đứt/liền)
VIS_COLORS = {
    "TP":   (0, 158, 115),    # xanh lục lam  - đúng (cả biên lẫn loại)
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
    """words/boxes/y_pred/y_true cùng độ dài, cùng thứ tự với chuỗi đã chấm điểm; box chuẩn hoá 0..1000."""
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
    items.sort(key=lambda it: (round(it[1][1] / 10), it[1][0]))    # đánh số theo thứ tự đọc

    # ---- lớp phủ: TP mảnh, lỗi đậm hơn; FN và biên gold dùng nét đứt ----
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
    for k, (kind, b, _, _, _) in enumerate(items, 1):              # nhãn số ở góc box
        tag = str(k)
        _, _, tw, th = d.textbbox((0, 0), tag, font=f_tag)
        x, y = b[0], max(0, b[1] - th - 5)
        d.rectangle([x, y, x + tw + 6, y + th + 4], fill=VIS_COLORS[kind])
        d.text((x + 3, y + 1), tag, fill="white", font=f_tag)

    # ---- bảng ghi chú bên phải ----
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
    for kind in ("TP", "TYPE", "SPAN", "FP", "FN"):               # chú giải
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
        d.rectangle([x, y + 2, x + 6, y + 2 * lh - 8], fill=VIS_COLORS[kind])   # vạch màu, chữ tối cho dễ đọc
        d.text((x + 12, y), f"#{k} {VIS_NAMES[kind]}: {note}", fill=(20, 20, 20), font=f)
        d.text((x + 24, y + lh - 2), f"\"{txt}\"", fill=(70, 70, 70), font=f)
        y += 2 * lh
    canvas.save(out_path)


# =============================================================================
# MAIN
# =============================================================================
def main():
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )

    out_dir = training_args.output_dir
    os.makedirs(out_dir, exist_ok=True)
    main_proc = is_main_process(training_args.local_rank)

    # ---------------- logging: stdout + train.log ----------------
    fmt = "%(asctime)s - %(levelname)s - %(name)s -   %(message)s"
    logging.basicConfig(format=fmt, datefmt="%m/%d/%Y %H:%M:%S", handlers=[logging.StreamHandler(sys.stdout)])
    logger.setLevel(logging.INFO if main_proc else logging.WARN)
    if main_proc:
        transformers.utils.logging.set_verbosity_info()
        transformers.utils.logging.enable_default_handler()
        transformers.utils.logging.enable_explicit_format()
        fh = logging.FileHandler(os.path.join(out_dir, "train.log"), mode="w", encoding="utf-8")
        fh.setFormatter(logging.Formatter(fmt, datefmt="%m/%d/%Y %H:%M:%S"))
        logging.getLogger().addHandler(fh)                  # logger của script (propagate lên root)
        if hasattr(transformers.utils.logging, "add_handler"):   # bản transformers cũ không có hàm này
            transformers.utils.logging.add_handler(fh)          # logger của transformers
        dump_json({"env": environment_info(),
                   "model_args": dataclasses.asdict(model_args),
                   "data_args": dataclasses.asdict(data_args),
                   "training_args": training_args.to_dict()},
                  os.path.join(out_dir, "run_args.json"))
    logger.info(f"Training/evaluation parameters {training_args}")
    if data_args.latent_oracle_eval:
        logger.warning("!!! latent_oracle_eval=True: dùng nhóm GOLD lúc test -> CHỈ là cận trên, không báo cáo như kết quả chính.")

    set_seed(training_args.seed)

    # ------------------------------------------------------------ dataset
    if data_args.dataset_name == 'funsd':
        import layoutlmft.data.funsd as ds_module
    elif data_args.dataset_name == 'cord':
        import layoutlmft.data.cord as ds_module
    else:
        raise NotImplementedError()
    datasets = load_dataset(os.path.abspath(ds_module.__file__), name=data_args.dataset_config_name,
                            cache_dir=model_args.cache_dir)

    # tách dev cố định (không bao giờ chọn mô hình trên test)
    dev_raw = None
    if training_args.do_eval:
        if "validation" in datasets:
            dev_raw = datasets["validation"]
        elif data_args.dev_ratio > 0:
            split = datasets["train"].train_test_split(test_size=data_args.dev_ratio, seed=data_args.dev_split_seed)
            datasets["train"], dev_raw = split["train"], split["test"]
        else:
            raise ValueError("--do_eval cần tập dev: đặt --dev_ratio > 0 (KHÔNG đánh giá chọn mô hình trên test).")

    split_for_meta = "train" if training_args.do_train else "test"
    column_names = datasets[split_for_meta].column_names
    features = datasets[split_for_meta].features
    text_column_name = "words" if "words" in column_names else "tokens"
    label_column_name = (
        f"{data_args.task_name}_tags" if f"{data_args.task_name}_tags" in column_names else column_names[1]
    )
    remove_columns = column_names

    if isinstance(features[label_column_name].feature, ClassLabel):
        label_list = features[label_column_name].feature.names
        label_to_id = {i: i for i in range(len(label_list))}
    else:
        unique = set()
        for l in datasets["train"][label_column_name]:
            unique |= set(l)
        label_list = sorted(unique)
        label_to_id = {l: i for i, l in enumerate(label_list)}
    num_labels = len(label_list)

    # ------------------------------------------------------------ model
    config = AutoConfig.from_pretrained(
        model_args.config_name if model_args.config_name else model_args.model_name_or_path,
        num_labels=num_labels,
        finetuning_task=data_args.task_name,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        input_size=data_args.input_size,
        visual_embed=data_args.visual_embed,
        use_auth_token=True if model_args.use_auth_token else None,
    )
    config.id2label = {i: l for i, l in enumerate(label_list)}
    config.label2id = {l: i for i, l in enumerate(label_list)}
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
        tokenizer_file=None,
        cache_dir=model_args.cache_dir,
        use_fast=True,
        add_prefix_space=True,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )
    model_kwargs = dict(
        from_tf=bool(".ckpt" in model_args.model_name_or_path),
        config=config,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )
    if data_args.use_latent_segment:
        from layoutlmft.models.layoutlmv3.modeling_layoutlmv3_segment import (
            LayoutLMv3ForLatentSegmentTokenClassification,
        )
        config.latent_layer = data_args.latent_layer
        config.latent_soft_box = data_args.latent_soft_box
        config.latent_attn_bias = data_args.latent_attn_bias
        config.latent_oracle_eval = data_args.latent_oracle_eval
        config.affinity_loss_weight = data_args.affinity_loss_weight
        model = LayoutLMv3ForLatentSegmentTokenClassification.from_pretrained(
            model_args.model_name_or_path, **model_kwargs)
    else:
        model = AutoModelForTokenClassification.from_pretrained(model_args.model_name_or_path, **model_kwargs)

    if not isinstance(tokenizer, PreTrainedTokenizerFast):
        raise ValueError("This example script only works for models that have a fast tokenizer.")

    if main_proc:
        n_all = sum(p.numel() for p in model.parameters())
        n_new = sum(p.numel() for n, p in model.named_parameters() if n.startswith("latent_"))
        dump_json({"class": type(model).__name__, "params_total": n_all, "params_latent_new": n_new,
                   "params_latent_ratio": n_new / max(n_all, 1)}, os.path.join(out_dir, "model_stats.json"))

    padding = "max_length" if data_args.pad_to_max_length else False

    if data_args.visual_embed:
        mean = IMAGENET_INCEPTION_MEAN if not data_args.imagenet_default_mean_and_std else IMAGENET_DEFAULT_MEAN
        std = IMAGENET_INCEPTION_STD if not data_args.imagenet_default_mean_and_std else IMAGENET_DEFAULT_STD
        common_transform = Compose([
            RandomResizedCropAndInterpolationWithTwoPic(
                size=data_args.input_size, interpolation=data_args.train_interpolation),
        ])
        patch_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=torch.tensor(mean), std=torch.tensor(std)),
        ])

    # ------------------------------------------------------------ tokenize
    def tokenize_and_align_labels(examples, augmentation=False):
        tokenized_inputs = tokenizer(
            examples[text_column_name],
            padding=False,
            truncation=True,
            return_overflowing_tokens=True,  # tài liệu > 512 token bị cắt thành nhiều đoạn KHÔNG chồng lấn
            is_split_into_words=True,
        )
        labels, bboxes, images, group_ids, doc_ids, word_idxs = [], [], [], [], [], []
        for batch_index in range(len(tokenized_inputs["input_ids"])):
            word_ids = tokenized_inputs.word_ids(batch_index=batch_index)
            org = tokenized_inputs["overflow_to_sample_mapping"][batch_index]
            label = examples[label_column_name][org]
            bbox = examples["bboxes"][org]

            # id nhóm gold cho từng từ (chỉ dùng làm nhãn lúc train). Nếu dataset chưa có cột
            # group_ids (vd. cord.py chưa sửa) thì suy ra từ box bằng nhau như code cũ.
            if "group_ids" in examples:
                word_gid = examples["group_ids"][org]
            else:
                word_gid, c, prev = [], -1, None
                for wb in bbox:
                    if tuple(wb) != prev:
                        c, prev = c + 1, tuple(wb)
                    word_gid.append(c)

            previous_word_idx = None
            label_ids, bbox_inputs, gid_inputs, widx_inputs = [], [], [], []
            for word_idx in word_ids:
                if word_idx is None:
                    label_ids.append(-100)
                    bbox_inputs.append([0, 0, 0, 0])
                    gid_inputs.append(-1)
                    widx_inputs.append(-1)
                elif word_idx != previous_word_idx:
                    label_ids.append(label_to_id[label[word_idx]])
                    bbox_inputs.append(bbox[word_idx])
                    gid_inputs.append(word_gid[word_idx])
                    widx_inputs.append(word_idx)          # sub-token ĐẦU của từ -> dùng để chấm điểm
                else:
                    label_ids.append(label_to_id[label[word_idx]] if data_args.label_all_tokens else -100)
                    bbox_inputs.append(bbox[word_idx])
                    gid_inputs.append(word_gid[word_idx])
                    widx_inputs.append(-1)
                previous_word_idx = word_idx
            labels.append(label_ids)
            bboxes.append(bbox_inputs)
            group_ids.append(gid_inputs)
            doc_ids.append(examples["id"][org])
            word_idxs.append(widx_inputs)

            if data_args.visual_embed:
                img = pil_loader(examples["image_path"][org])
                for_patches, _ = common_transform(img, augmentation=augmentation)
                images.append(patch_transform(for_patches))

        tokenized_inputs["labels"] = labels
        tokenized_inputs["bbox"] = bboxes
        tokenized_inputs["group_ids"] = group_ids   # Trainer tự bỏ cột này nếu forward() không nhận
        tokenized_inputs["doc_id"] = doc_ids        # chỉ dùng để chấm điểm cấp tài liệu (Trainer tự bỏ)
        tokenized_inputs["word_idx"] = word_idxs    # chỉ dùng để chấm điểm (Trainer tự bỏ)
        if data_args.visual_embed:
            tokenized_inputs["images"] = images
        return tokenized_inputs

    def prep(ds, max_n):
        if max_n is not None:
            ds = ds.select(range(max_n))
        return ds.map(tokenize_and_align_labels, batched=True, remove_columns=remove_columns,
                      num_proc=data_args.preprocessing_num_workers,
                      load_from_cache_file=not data_args.overwrite_cache)

    train_dataset = prep(datasets["train"], data_args.max_train_samples) if training_args.do_train else None
    eval_dataset = prep(dev_raw, data_args.max_val_samples) if training_args.do_eval else None
    test_dataset = prep(datasets["test"], data_args.max_test_samples) if training_args.do_predict else None

    # ---------------- data_stats.json ----------------
    def split_stats(raw, tok, max_n):
        from seqeval.metrics.sequence_labeling import get_entities
        if max_n is not None:
            raw = raw.select(range(max_n))
        ent, n_words = Counter(), 0
        for tags in raw[label_column_name]:
            names = [label_list[label_to_id[t]] for t in tags]
            n_words += len(names)
            ent.update(t for t, _, _ in get_entities(names))
        chunks = Counter(tok["doc_id"])
        return {"docs": len(raw), "words": n_words, "chunks_512": len(tok),
                "docs_over_512_tokens": sum(1 for v in chunks.values() if v > 1),
                "entities": dict(sorted(ent.items())), "entities_total": sum(ent.values())}

    if main_proc:
        stats = {"dataset": data_args.dataset_name, "config": data_args.dataset_config_name}
        if train_dataset is not None:
            stats["train"] = split_stats(datasets["train"], train_dataset, data_args.max_train_samples)
        if eval_dataset is not None:
            stats["dev"] = split_stats(dev_raw, eval_dataset, data_args.max_val_samples)
        if test_dataset is not None:
            stats["test"] = split_stats(datasets["test"], test_dataset, data_args.max_test_samples)
        dump_json(stats, os.path.join(out_dir, "data_stats.json"))
        logger.info(f"Data stats: {json.dumps(stats, ensure_ascii=False)}")

    data_collator = DataCollatorForKeyValueExtraction(
        tokenizer,
        pad_to_multiple_of=8 if training_args.fp16 else None,
        padding=padding,
        max_length=512,
    )

    # ------------------------------------------------------------ METRIC (chốt 1 cách)
    # seqeval entity-level micro P/R/F1, sơ đồ IOB2, chế độ mặc định (giống LayoutLMv3 gốc),
    # chấm trên sub-token ĐẦU của mỗi từ, GỘP các đoạn 512 về NGUYÊN TÀI LIỆU, mỗi từ đúng 1 lần.

    def doc_level_sequences(predictions, label_ids, ds):
        predictions = np.argmax(predictions, axis=2)
        doc_ids, word_idx = ds["doc_id"], ds["word_idx"]
        docs, order = {}, []
        for i in range(len(predictions)):
            d = doc_ids[i]
            if d not in docs:
                docs[d] = {}
                order.append(d)
            seen = docs[d]
            for j, w in enumerate(word_idx[i]):
                if w < 0 or label_ids[i][j] == -100 or w in seen:
                    continue
                seen[w] = (label_list[predictions[i][j]], label_list[label_ids[i][j]])
        y_pred, y_true, widx = [], [], []
        for d in order:
            items = sorted(docs[d].items())
            widx.append([w for w, _ in items])
            y_pred.append([p for _, (p, _) in items])
            y_true.append([g for _, (_, g) in items])
        return order, y_pred, y_true, widx

    def make_compute_metrics(ds):
        def compute_metrics(p):
            _, y_pred, y_true, _ = doc_level_sequences(p.predictions, p.label_ids, ds)
            results = seqeval_compute(y_pred, y_true)
            out = {"precision": results["overall_precision"], "recall": results["overall_recall"],
                   "f1": results["overall_f1"], "accuracy": results["overall_accuracy"]}
            if data_args.return_entity_level_metrics:
                for key, value in results.items():
                    if isinstance(value, dict):
                        for n, v in value.items():
                            out[f"{key}_{n}"] = v
            return out
        return compute_metrics

    # ------------------------------------------------------------ trainer
    class KIETrainer(Trainer):
        """- Baseline: optimizer/logic y hệt Trainer gốc (chỉ thêm log).
           - LSS: LR riêng cho latent_*, cập nhật xác suất teacher forcing mỗi bước,
             log thêm ce_loss / aff_loss / tf_prob / alpha / lambda / chất lượng gom nhóm."""

        def create_optimizer(self):
            if self.optimizer is None and data_args.use_latent_segment:
                base, head = [], []
                for n, p in self.model.named_parameters():
                    if p.requires_grad:
                        (head if n.startswith("latent_") else base).append(p)
                self.optimizer = torch.optim.AdamW(
                    [{"params": base, "lr": self.args.learning_rate, "weight_decay": self.args.weight_decay},
                     {"params": head, "lr": data_args.head_learning_rate, "weight_decay": 0.0}],
                    betas=(self.args.adam_beta1, self.args.adam_beta2),
                    eps=self.args.adam_epsilon,
                )
            return super().create_optimizer()

        def training_step(self, model, inputs, *args, **kwargs):
            if data_args.use_latent_segment:
                r = data_args.latent_tf_ratio
                total = max(1, self.state.max_steps)
                self.model.latent_tf_prob = max(0.0, 1.0 - self.state.global_step / (r * total)) if r > 0 else 0.0
            return super().training_step(model, inputs, *args, **kwargs)

        def log(self, logs, *args, **kwargs):
            m = self.model
            if hasattr(m, "pop_latent_stats"):
                if "loss" in logs:                                   # log huấn luyện định kỳ
                    logs.update(m.pop_latent_stats(""))
                    logs["tf_prob"] = float(m.latent_tf_prob)
                    logs["alpha_word_box"] = float(torch.sigmoid(m.latent_alpha.detach()).item())
                    lam = m.latent_attn_lambda.detach().float()
                    logs["attn_lambda_mean"] = float(lam.mean())
                    logs["attn_lambda_absmax"] = float(lam.abs().max())
                elif any(k.startswith("eval_") for k in logs):       # log đánh giá dev
                    logs.update(m.pop_latent_stats("eval_"))
            if torch.cuda.is_available():
                logs["gpu_mem_gb"] = gpu_mem_gb()
            super().log(logs, *args, **kwargs)

    trainer = KIETrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=make_compute_metrics(eval_dataset) if eval_dataset is not None else None,
    )

    def write_log_history():
        if trainer.is_world_process_zero():
            with open(os.path.join(out_dir, "log_history.jsonl"), "w", encoding="utf-8") as w:
                for rec in trainer.state.log_history:
                    w.write(json.dumps(rec, ensure_ascii=False) + "\n")

    if training_args.do_train:
        train_result = trainer.train(resume_from_checkpoint=last_checkpoint)
        trainer.save_model()
        metrics = train_result.metrics
        metrics["train_samples"] = len(train_dataset)
        metrics["train_gpu_mem_gb"] = gpu_mem_gb()
        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()
        write_log_history()

    if training_args.do_eval:
        logger.info("*** Evaluate on DEV ***")
        if hasattr(model, "pop_latent_stats"):
            model.pop_latent_stats()                                 # xoá thống kê cũ
        metrics = trainer.evaluate()                                 # log() đã gắn eval_group_pair_*
        metrics["eval_samples"] = len(eval_dataset)
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    if training_args.do_predict:
        logger.info("*** Predict on TEST (checkpoint cuối, số bước cố định) ***")
        trainer.compute_metrics = make_compute_metrics(test_dataset)
        if hasattr(model, "pop_latent_stats"):
            model.pop_latent_stats()
        predictions, label_ids, metrics = trainer.predict(test_dataset)
        if hasattr(model, "pop_latent_stats"):
            metrics.update(model.pop_latent_stats("test_"))          # test_group_pair_precision/recall/f1
        metrics["test_gpu_mem_gb"] = gpu_mem_gb()
        metrics["test_samples"] = len(test_dataset)

        order, y_pred, y_true, widx = doc_level_sequences(predictions, label_ids, test_dataset)
        breakdown, per_doc = error_breakdown(list(zip(order, y_pred, y_true)))
        ov = breakdown["overall"]
        for k in ("TYPE", "SPAN", "FP", "FN", "TP", "n_gold", "n_pred"):
            metrics[f"test_err_{k}" if k in ("TYPE", "SPAN", "FP", "FN") else f"test_{k}"] = ov[k]
        # Kiểm tra nhất quán: đếm entity của mình phải cho đúng F1 của seqeval
        consistent = abs(ov["f1"] - metrics["test_f1"]) < 1e-6
        metrics["test_metric_consistent"] = int(consistent)
        if not consistent:
            logger.warning(f"!!! F1 đếm lỗi {ov['f1']:.6f} khác seqeval {metrics['test_f1']:.6f} - kiểm tra lại!")
        trainer.log_metrics("test", metrics)
        trainer.save_metrics("test", metrics)

        if trainer.is_world_process_zero():
            from seqeval.metrics import classification_report
            report = classification_report(y_true, y_pred, digits=4)
            with open(os.path.join(out_dir, "test_report.txt"), "w", encoding="utf-8") as w:
                w.write(f"{data_args.dataset_name} / {data_args.dataset_config_name} | seed {training_args.seed}\n")
                w.write("seqeval (mode mặc định, IOB2), cấp tài liệu, sub-token đầu\n\n" + report)
            logger.info("\n" + report)
            dump_json(breakdown, os.path.join(out_dir, "test_error_breakdown.json"))
            logger.info(f"Error breakdown: {json.dumps(ov, ensure_ascii=False)}")

            raw = datasets["test"]
            if "image" in raw.column_names:            # bỏ cột ảnh 224x224 cho nhẹ, dùng ảnh gốc từ image_path
                raw = raw.remove_columns("image")
            id2row = {ex_id: i for i, ex_id in enumerate(raw["id"])}
            doc_f1 = {r["doc_id"]: r["f1"] for r in per_doc}
            with open(os.path.join(out_dir, "test_predictions.jsonl"), "w", encoding="utf-8") as w:
                for d, yp, yt, wl in zip(order, y_pred, y_true, widx):
                    ex = raw[id2row[d]]
                    w.write(json.dumps({"doc_id": d, "image": os.path.basename(ex["image_path"]),
                                        "doc_f1": doc_f1[d], "words": [ex[text_column_name][i] for i in wl],
                                        "pred": yp, "gold": yt}, ensure_ascii=False) + "\n")

            probs = torch.softmax(torch.from_numpy(np.asarray(predictions, dtype=np.float32)), dim=-1)
            conf, pred_ids = probs.max(-1)
            np.savez_compressed(
                os.path.join(out_dir, "test_raw_predictions.npz"),
                pred_ids=pred_ids.numpy().astype(np.int16), confidence=conf.numpy().astype(np.float16),
                label_ids=np.asarray(label_ids).astype(np.int16),
                word_idx=np.array(test_dataset["word_idx"], dtype=object),
                doc_id=np.array(test_dataset["doc_id"]),
                meta=json.dumps({"label_list": list(label_list), "dataset": data_args.dataset_name,
                                 "config": data_args.dataset_config_name, "seed": training_args.seed,
                                 "oracle_eval": data_args.latent_oracle_eval}),
            )

            # mỗi tài liệu test -> 1 file PNG trong output_dir/vis
            if data_args.visualize:
                vis_dir = os.path.join(out_dir, "vis")
                os.makedirs(vis_dir, exist_ok=True)
                n_vis = len(order) if data_args.visualize_max_docs is None else data_args.visualize_max_docs
                n_ok = 0
                for d, yp, yt, wl in list(zip(order, y_pred, y_true, widx))[:n_vis]:
                    ex = raw[id2row[d]]
                    words = [ex[text_column_name][w] for w in wl]
                    boxes = [ex["bboxes"][w] for w in wl]
                    name = os.path.splitext(os.path.basename(ex["image_path"]))[0]
                    title = f"{name} | {data_args.dataset_config_name or data_args.dataset_name} | seed {training_args.seed}"
                    try:
                        visualize_document(ex["image_path"], words, boxes, yp, yt,
                                           os.path.join(vis_dir, f"{d}_{name}.png"), title=title)
                        n_ok += 1
                    except Exception as e:  # 1 ảnh lỗi không làm hỏng cả lượt chạy
                        logger.warning(f"Không vẽ được tài liệu {d}: {e}")
                logger.info(f"Đã lưu {n_ok}/{n_vis} ảnh visualize vào {vis_dir}")

    write_log_history()


def _mp_fn(index):
    main()


if __name__ == "__main__":
    main()