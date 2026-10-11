#!/bin/bash
# =============================================================================
# LayoutLMv3 GỐC trên FUNSD / CORD, đủ 4 setting đầu vào (chỉ khác cách dựng box + thứ tự, model/train y nguyên).
#
#   DATASET=funsd SIZE=base  SETTING=A bash scripts/run.sh   # tái hiện paper (base 90.29 / large 92.08)
#   DATASET=funsd SIZE=base  SETTING=C bash scripts/run.sh   # setting thực tế (box từ + thứ tự đọc)
#   DATASET=cord  SIZE=large SETTING=B bash scripts/run.sh
#   CURVE=1 ...  : chấm test mỗi 100 bước như README gốc (--do_eval trỏ vào test) để đối chiếu số "best"
#
# SETTING  config        box 2D                         thứ tự từ        dùng khối gold?
#   A      funsd|cord    khối annotation (gốc v3)        annotation       box + thứ tự  (rò rỉ biên entity)
#   B      *_word        từng từ                         annotation       thứ tự
#   C      *_word_ro     từng từ                         đọc (hình học)   không
#   D      *_line_ro     dòng gom hình học (giống OCR)   đọc (hình học)   không
#
# Siêu tham số = paper mục 3.3: 1000 bước, không warmup;
#   FUNSD lr 1e-5, batch 16 | CORD lr 5e-5, batch 64. Paper không nêu riêng cho large -> dùng chung.
# Số chính = test_f1 ở checkpoint CUỐI (không chọn checkpoint trên test), seqeval entity-level như bản gốc.
# Tag: logs/<dataset>-<size>-<setting>-LayoutLMv3-seed<seed>
# =============================================================================
set -eo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH="$(pwd):$PYTHONPATH" TOKENIZERS_PARALLELISM=false WANDB_DISABLED=true

DATASET=${DATASET:-funsd}
SIZE=${SIZE:-base}
SETTING=${SETTING:-A}
SEEDS=(${SEEDS:-42 123 1993})
CURVE=${CURVE:-0}
MODEL_PATH=${MODEL_PATH:-models/layoutlmv3-${SIZE}}      # hoặc microsoft/layoutlmv3-${SIZE}
EXTRA=${EXTRA:-}

case "$SETTING" in
  A) SUFFIX="" ;;  B) SUFFIX="_word" ;;  C) SUFFIX="_word_ro" ;;  D) SUFFIX="_line_ro" ;;
  *) echo "SETTING phải là A|B|C|D"; exit 1 ;;
esac
case "$DATASET" in
  funsd) LR=1e-5; BATCH=16 ;;
  cord)  LR=5e-5; BATCH=64 ;;
  *) echo "DATASET phải là funsd|cord"; exit 1 ;;
esac
case "$SIZE" in
  base)  PER_DEV=${PER_DEV:-2} ;;
  large) PER_DEV=${PER_DEV:-1} ;;
  *) echo "SIZE phải là base|large"; exit 1 ;;
esac
CONFIG="${DATASET}${SUFFIX}"
ACCUM=$(( BATCH / PER_DEV ))                               # 1 GPU: batch hiệu dụng = PER_DEV x ACCUM
TAG="${DATASET}-${SIZE}-${SETTING}-LayoutLMv3"

EVAL_FLAGS=""
[ "$CURVE" = "1" ] && EVAL_FLAGS="--do_eval --evaluation_strategy steps --eval_steps 100"

for SEED in "${SEEDS[@]}"; do
  OUT_DIR="./logs/${TAG}-seed${SEED}"
  if [ "${RERUN:-0}" != "1" ] && [ -f "$OUT_DIR/test_results.json" ]; then
    echo "=== ${TAG} | seed ${SEED}: đã có, bỏ qua (RERUN=1 để chạy lại) ==="; continue
  fi
  mkdir -p "$OUT_DIR"
  VIS=""; [ "$SEED" = "${SEEDS[0]}" ] && VIS="--visualize True"
  echo "=== ${TAG} | config ${CONFIG} | seed ${SEED} | lr ${LR} | batch ${PER_DEV}x${ACCUM} ==="
  python examples/run_funsd_cord.py \
    --dataset_name "$DATASET" --dataset_config_name "$CONFIG" \
    --model_name_or_path "$MODEL_PATH" --output_dir "$OUT_DIR" \
    --do_train --do_predict $EVAL_FLAGS $VIS \
    --segment_level_layout 1 --visual_embed 1 --input_size 224 \
    --max_steps 1000 --learning_rate "$LR" --warmup_ratio 0 \
    --per_device_train_batch_size "$PER_DEV" --gradient_accumulation_steps "$ACCUM" --per_device_eval_batch_size 4 \
    --save_strategy no --logging_steps 20 --dataloader_num_workers 4 --report_to none \
    --seed "$SEED" --overwrite_output_dir \
    $EXTRA 2>&1 | tee "$OUT_DIR/console.log"
done

# ----------------------------------------------------------------------------- tổng hợp
export TAG DATASET SIZE SEEDS_STR="${SEEDS[*]}"
python - <<'PY'
import os, json, numpy as np
tag, ds, size = os.environ["TAG"], os.environ["DATASET"], os.environ["SIZE"]
seeds = os.environ["SEEDS_STR"].split()

def load(t):
    out = {}
    for s in seeds:
        d = f"./logs/{t}-seed{s}"
        if not os.path.exists(f"{d}/test_results.json"):
            continue
        r = json.load(open(f"{d}/test_results.json"))
        best = None
        if os.path.exists(f"{d}/log_history.jsonl"):
            ev = [json.loads(l) for l in open(f"{d}/log_history.jsonl")]
            ev = [e["eval_f1"] for e in ev if "eval_f1" in e]
            best = max(ev) if ev else None
        out[s] = {"f1": r["test_f1"], "p": r["test_precision"], "r": r["test_recall"], "best_on_test": best}
    return out

runs = load(tag)
if not runs:
    raise SystemExit("Không có kết quả.")
ms = lambda v: (100 * np.mean(v), 100 * (np.std(v, ddof=1) if len(v) > 1 else 0.0))
print(f"\n===== {tag} | seed: {', '.join(runs)} =====")
for k, name in (("p", "precision"), ("r", "recall"), ("f1", "F1 (checkpoint cuối)  <- SỐ BÁO CÁO")):
    m, s = ms([x[k] for x in runs.values()])
    print(f"{name:38s} {m:6.2f} ± {s:4.2f}   [{', '.join(f'{100*x[k]:.2f}' for x in runs.values())}]")
b = [x["best_on_test"] for x in runs.values() if x["best_on_test"] is not None]
if b:
    m, s = ms(b)
    print(f"{'F1 best/100 bước trên test (kiểu README)':38s} {m:6.2f} ± {s:4.2f}   (chỉ để đối chiếu, chọn trên test)")
summary = {"tag": tag, "seeds": list(runs), "runs": runs, "f1_mean": float(np.mean([x["f1"] for x in runs.values()]))}
json.dump(summary, open(f"./logs/{tag}_summary.json", "w"), indent=2)

ref_tag = f"{ds}-{size}-A-LayoutLMv3"
if ref_tag != tag:
    ref = load(ref_tag)
    d = [runs[s]["f1"] - ref[s]["f1"] for s in runs if s in ref]
    if d:
        m, s = ms(d)
        print(f"so với setting A (cặp seed): {m:+.2f} ± {s:.2f} điểm")
PY