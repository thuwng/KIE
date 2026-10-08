#!/bin/bash
# =============================================================================
# FUNSD: LayoutLMv3 gốc (B0) và Latent Soft Segment (LSS).
#
#   SETTING=A MODEL=B0                     bash scripts/run_lss.sh   # tái hiện paper (~90.3)
#   SETTING=C MODEL=B0                     bash scripts/run_lss.sh   # baseline đúng cho setting chính
#   SETTING=C MODEL=LSS PROTOCOL=dev LAYER=4 bash scripts/run_lss.sh # tune trên dev
#   SETTING=C MODEL=LSS LAYER=6            bash scripts/run_lss.sh   # số báo cáo
#   SETTING=C MODEL=LSS LAYER=6 ORACLE=1   bash scripts/run_lss.sh   # cận trên, dùng lại checkpoint LSS
#   SETTING=C MODEL=LSS LAYER=6 EXTRA="--latent_attn_bias False" EXTRA_TAG=-noBias bash scripts/run_lss.sh
#
# SETTING : A = box segment gold + thứ tự annotation (chỉ để đối chiếu 90.29, có rò rỉ nhãn)
#           B = box từ + thứ tự annotation
#           C = box từ + thứ tự đọc (trên->dưới, trái->phải)   <- SETTING CHÍNH
# METRIC  : test_f1 = seqeval entity-level (IOB2, mode mặc định như LayoutLMv3), gộp về nguyên tài liệu.
# Chỉ so LSS với B0 CÙNG SETTING + PROTOCOL (script tự in delta nếu đã chạy B0).
# =============================================================================
set -eo pipefail

cd "$(dirname "$0")/.."
export PYTHONPATH="$(pwd):$PYTHONPATH"
export TOKENIZERS_PARALLELISM=false
export WANDB_DISABLED=true

SETTING=${SETTING:-C}
MODEL=${MODEL:-B0}
PROTOCOL=${PROTOCOL:-final}
LAYER=${LAYER:-6}
ORACLE=${ORACLE:-0}
SEEDS=(${SEEDS:-42 123 1993})
MODEL_PATH=${MODEL_PATH:-models/layoutlmv3-base}
EXTRA=${EXTRA:-}
EXTRA_TAG=${EXTRA_TAG:-}

case "$SETTING" in
  A) CONFIG=funsd ;;
  B) CONFIG=funsd_word ;;
  C) CONFIG=funsd_word_ro ;;
  *) echo "SETTING phải là A, B hoặc C"; exit 1 ;;
esac

case "$MODEL" in
  B0)  MODEL_FLAGS="" ;;
  LSS) MODEL_FLAGS="--use_latent_segment True --latent_layer ${LAYER}" ;;
  *) echo "MODEL phải là B0 hoặc LSS"; exit 1 ;;
esac

# dev: tách 10% train làm dev, log dev F1 mỗi 100 bước. final: train toàn bộ, chấm test 1 lần ở bước cuối.
case "$PROTOCOL" in
  dev)   PROTO_FLAGS="--do_train --do_eval --dev_ratio 0.1 --evaluation_strategy steps --eval_steps 100" ;;
  final) PROTO_FLAGS="--do_train --do_predict" ;;
  *) echo "PROTOCOL phải là dev hoặc final"; exit 1 ;;
esac

BASE_TAG="funsd-${PROTOCOL}-${SETTING}-${MODEL}"
[ "$MODEL" = "LSS" ] && BASE_TAG="${BASE_TAG}-L${LAYER}"
BASE_TAG="${BASE_TAG}${EXTRA_TAG}"
TAG="$BASE_TAG"
if [ "$ORACLE" = "1" ]; then
  [ "$MODEL" = "LSS" ] && [ "$PROTOCOL" = "final" ] || { echo "ORACLE chỉ dùng với MODEL=LSS PROTOCOL=final"; exit 1; }
  TAG="${BASE_TAG}-ORACLE"
fi

for SEED in "${SEEDS[@]}"; do
  OUT_DIR="./logs/${TAG}-seed${SEED}"
  # Seed đã có kết quả thì bỏ qua (đặt RERUN=1 để chạy lại tất cả)
  if [ "${RERUN:-0}" != "1" ] && { [ -f "$OUT_DIR/eval_results.json" ] || [ -f "$OUT_DIR/test_results.json" ]; }; then
    echo "=== ${TAG} | seed ${SEED}: đã có kết quả, bỏ qua ==="; continue
  fi
  mkdir -p "$OUT_DIR"

  RUN_FLAGS="$PROTO_FLAGS --model_name_or_path $MODEL_PATH"
  if [ "$ORACLE" = "1" ]; then   # không train lại: nạp checkpoint LSS cùng seed, test với nhóm gold
    RUN_FLAGS="--do_predict --latent_oracle_eval True --model_name_or_path ./logs/${BASE_TAG}-seed${SEED}"
  fi
  VIS=""
  [ "$SEED" = "${SEEDS[0]}" ] && VIS="--visualize True"   # chỉ vẽ ảnh cho seed đầu

  echo "=== ${TAG} | seed ${SEED} ==="
  # Siêu tham số theo paper LayoutLMv3 cho FUNSD: batch 16 (2 x 8 accumulation), lr 1e-5, 1000 bước,
  # không warmup, không chọn checkpoint (giữ checkpoint cuối). EXTRA đặt CUỐI để ghi đè được.
  python examples/run_funsd_cord.py \
    --dataset_name funsd --dataset_config_name "$CONFIG" \
    $RUN_FLAGS $MODEL_FLAGS $VIS \
    --output_dir "$OUT_DIR" \
    --visual_embed True --input_size 224 \
    --max_steps 1000 --learning_rate 1e-5 \
    --per_device_train_batch_size 2 --gradient_accumulation_steps 8 --per_device_eval_batch_size 4 \
    --save_strategy no --logging_steps 20 \
    --dataloader_num_workers 4 --report_to none \
    --seed "$SEED" --overwrite_output_dir \
    $EXTRA 2>&1 | tee "$OUT_DIR/console.log"
done

# ----------------------------------------------------------------------------- tổng hợp
export TAG BASE_TAG PROTOCOL SETTING MODEL ORACLE SEEDS_STR="${SEEDS[*]}"
python - <<'PY'
import os, json, numpy as np
tag, base_tag = os.environ["TAG"], os.environ["BASE_TAG"]
proto, setting = os.environ["PROTOCOL"], os.environ["SETTING"]
seeds = os.environ["SEEDS_STR"].split()
pre, fn = ("eval_", "eval_results.json") if proto == "dev" else ("test_", "test_results.json")
main_key = pre + "f1"

runs = {}
for s in seeds:
    p = f"./logs/{tag}-seed{s}/{fn}"
    if not os.path.exists(p):
        print("[thiếu]", p); continue
    r = json.load(open(p))
    t = f"./logs/{tag}-seed{s}/train_results.json"
    if os.path.exists(t):
        r["train_runtime"] = json.load(open(t)).get("train_runtime", 0.0)
    runs[s] = r
if not runs:
    raise SystemExit("Không có kết quả nào.")

keep = ("f1", "precision", "recall", "err_", "consistent", "runtime")
keys = sorted({k for r in runs.values() for k, v in r.items()
               if isinstance(v, (int, float)) and any(x in k for x in keep)})
summary = {}
for k in keys:
    v = [float(runs[s][k]) for s in runs if k in runs[s]]
    summary[k] = {"values": v, "mean": float(np.mean(v)), "std": float(np.std(v, ddof=1)) if len(v) > 1 else 0.0}
summary["main_f1"] = dict(summary[main_key], seeds=list(runs), source=main_key)

print(f"\n===== {tag} | {len(runs)} seed: {', '.join(runs)} =====")
for k, d in summary.items():
    if k == "main_f1":
        continue
    sc = 100 if any(x in k for x in ("f1", "precision", "recall")) else (1 / 60 if "runtime" in k else 1)
    unit = " phút" if "runtime" in k else ""
    print(f"{k:34s} {sc*d['mean']:8.2f} ± {sc*d['std']:5.2f}{unit}")
m = summary["main_f1"]
print(f"{'>>> main_f1 (' + main_key + ')':34s} {100*m['mean']:8.2f} ± {100*m['std']:5.2f}   "
      f"[{', '.join(f'{100*x:.2f}' for x in m['values'])}]")
json.dump(summary, open(f"./logs/{tag}_summary.json", "w"), indent=2)
print("Saved:", f"./logs/{tag}_summary.json")

def paired_delta(ref_file, name):
    if not os.path.exists(ref_file):
        print(f"(chưa có {ref_file} để so sánh)"); return None
    ref = json.load(open(ref_file))["main_f1"]
    rv = dict(zip(ref["seeds"], ref["values"]))
    d = [runs[s][main_key] - rv[s] for s in runs if s in rv]
    if not d:
        return None
    print(f"so với {name}: {100*np.mean(d):+.2f} ± {100*(np.std(d, ddof=1) if len(d) > 1 else 0):.2f} điểm "
          f"(theo cặp seed, thắng {sum(x > 0 for x in d)}/{len(d)}) [{name} {100*ref['mean']:.2f}]")
    return ref["mean"]

# So với B0 cùng setting + protocol (theo cặp seed)
if os.environ["MODEL"] != "B0":
    b0_mean = paired_delta(f"./logs/funsd-{proto}-{setting}-B0_summary.json", "B0")
    # Cận trên: phần khoảng hở lấy lại được = (LSS - B0) / (ORACLE - B0)
    if os.environ["ORACLE"] == "1":
        lss_mean = paired_delta(f"./logs/{base_tag}_summary.json", "LSS (không oracle)")
        if b0_mean is not None and lss_mean is not None and m["mean"] - b0_mean > 1e-9:
            print(f"Khoảng hở lấy lại được (LSS−B0)/(ORACLE−B0) = {100*(lss_mean-b0_mean)/(m['mean']-b0_mean):.1f}%")
PY